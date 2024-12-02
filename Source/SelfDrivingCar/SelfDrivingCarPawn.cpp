// Copyright Epic Games, Inc. All Rights Reserved.

#include "SelfDrivingCarPawn.h"
#include "SelfDrivingCarWheelFront.h"
#include "SelfDrivingCarWheelRear.h"
#include "Components/SkeletalMeshComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "Camera/CameraComponent.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputActionValue.h"
#include "ChaosWheeledVehicleMovementComponent.h"
#include "Components/ArrowComponent.h"
#include "Components/SplineComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Learning/SdLearningAgentsManager.h"

#define LOCTEXT_NAMESPACE "VehiclePawn"

DEFINE_LOG_CATEGORY(LogTemplateVehicle);

ASelfDrivingCarPawn::ASelfDrivingCarPawn()
{
	// construct the front camera boom
	FrontSpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("Front Spring Arm"));
	FrontSpringArm->SetupAttachment(GetMesh());
	FrontSpringArm->TargetArmLength = 0.0f;
	FrontSpringArm->bDoCollisionTest = false;
	FrontSpringArm->bEnableCameraRotationLag = true;
	FrontSpringArm->CameraRotationLagSpeed = 15.0f;
	FrontSpringArm->SetRelativeLocation(FVector(30.0f, 0.0f, 120.0f));

	FrontCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("Front Camera"));
	FrontCamera->SetupAttachment(FrontSpringArm);
	FrontCamera->bAutoActivate = false;

	// construct the back camera boom
	BackSpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("Back Spring Arm"));
	BackSpringArm->SetupAttachment(GetMesh());
	BackSpringArm->TargetArmLength = 650.0f;
	BackSpringArm->SocketOffset.Z = 150.0f;
	BackSpringArm->bDoCollisionTest = false;
	BackSpringArm->bInheritPitch = false;
	BackSpringArm->bInheritRoll = false;
	BackSpringArm->bEnableCameraRotationLag = true;
	BackSpringArm->CameraRotationLagSpeed = 2.0f;
	BackSpringArm->CameraLagMaxDistance = 50.0f;

	BackCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("Back Camera"));
	BackCamera->SetupAttachment(BackSpringArm);

	// Configure the car mesh
	GetMesh()->SetSimulatePhysics(true);
	GetMesh()->SetCollisionProfileName(FName("Vehicle"));

	// get the Chaos Wheeled movement component
	ChaosVehicleMovement = CastChecked<UChaosWheeledVehicleMovementComponent>(GetVehicleMovement());

}

void ASelfDrivingCarPawn::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	if (UEnhancedInputComponent* EnhancedInputComponent = Cast<UEnhancedInputComponent>(PlayerInputComponent))
	{
		// steering 
		EnhancedInputComponent->BindAction(SteeringAction, ETriggerEvent::Triggered, this, &ASelfDrivingCarPawn::Steering);
		EnhancedInputComponent->BindAction(SteeringAction, ETriggerEvent::Completed, this, &ASelfDrivingCarPawn::Steering);

		// throttle 
		EnhancedInputComponent->BindAction(ThrottleAction, ETriggerEvent::Triggered, this, &ASelfDrivingCarPawn::Throttle);
		EnhancedInputComponent->BindAction(ThrottleAction, ETriggerEvent::Completed, this, &ASelfDrivingCarPawn::Throttle);

		// break 
		EnhancedInputComponent->BindAction(BrakeAction, ETriggerEvent::Triggered, this, &ASelfDrivingCarPawn::Brake);
		EnhancedInputComponent->BindAction(BrakeAction, ETriggerEvent::Started, this, &ASelfDrivingCarPawn::StartBrake);
		EnhancedInputComponent->BindAction(BrakeAction, ETriggerEvent::Completed, this, &ASelfDrivingCarPawn::StopBrake);

		// handbrake 
		EnhancedInputComponent->BindAction(HandbrakeAction, ETriggerEvent::Started, this, &ASelfDrivingCarPawn::StartHandbrake);
		EnhancedInputComponent->BindAction(HandbrakeAction, ETriggerEvent::Completed, this, &ASelfDrivingCarPawn::StopHandbrake);

		// look around 
		EnhancedInputComponent->BindAction(LookAroundAction, ETriggerEvent::Triggered, this, &ASelfDrivingCarPawn::LookAround);

		// toggle camera 
		EnhancedInputComponent->BindAction(ToggleCameraAction, ETriggerEvent::Triggered, this, &ASelfDrivingCarPawn::ToggleCamera);

		// reset the vehicle 
		EnhancedInputComponent->BindAction(ResetVehicleAction, ETriggerEvent::Triggered, this, &ASelfDrivingCarPawn::ResetVehicle);
	}
	else
	{
		UE_LOG(LogTemplateVehicle, Error, TEXT("'%s' Failed to find an Enhanced Input component! This template is built to use the Enhanced Input system. If you intend to use the legacy system, then you will need to update this C++ file."), *GetNameSafe(this));
	}
}

void ASelfDrivingCarPawn::BeginPlay()
{
	Super::BeginPlay();

	TArray<AActor*> LearningManagers;
	UGameplayStatics::GetAllActorsWithTag(this, "LearningAgentsManager", LearningManagers);
	for (AActor* LearningManager : LearningManagers)
	{
		if (auto* ManagerComp = LearningManager->GetComponentByClass<USdLearningAgentsManager>())
		{
			ManagerComp->AddAgent(this);
		}
	}

	for (const auto& [DirType, DirArrows] : DistanceCheckDirections)
	{
		for (const UArrowComponent* DirArrow : DirArrows.Arrows)
		{
			if (DirArrow && DirArrow->ArrowLength > MaxDistanceCheck)
			{
				MaxDistanceCheck = DirArrow->ArrowLength;
			}
		}
	}
}

void ASelfDrivingCarPawn::Tick(float Delta)
{
	Super::Tick(Delta);

	// add some angular damping if the vehicle is in midair
	bool bMovingOnGround = ChaosVehicleMovement->IsMovingOnGround();
	GetMesh()->SetAngularDamping(bMovingOnGround ? 0.0f : 3.0f);

	// realign the camera yaw to face front
	float CameraYaw = BackSpringArm->GetRelativeRotation().Yaw;
	CameraYaw = FMath::FInterpTo(CameraYaw, 0.0f, Delta, 1.0f);

	BackSpringArm->SetRelativeRotation(FRotator(0.0f, CameraYaw, 0.0f));
}

void ASelfDrivingCarPawn::CalcDistances()
{
	ObservedObstacles.Empty();
	for (const auto& [DirType, DirArrows] : DistanceCheckDirections)
	{
		for (const UArrowComponent* DirArrow : DirArrows.Arrows)
		{
			if (!DirArrow)
			{
				continue;
			}

			const FVector RayCastDir = DirArrow->GetForwardVector();
			const FVector Start = DirArrow->GetComponentLocation();
			const FVector End = Start + RayCastDir * DirArrow->ArrowLength;
			FHitResult Hit;
			FCollisionQueryParams Params;
			Params.AddIgnoredActor(this);
			TArray<UPrimitiveComponent*> Components;
			GetComponents(Components);
			Params.AddIgnoredComponents(Components);
			const bool bHit = GetWorld()->LineTraceSingleByObjectType(Hit, Start, End, ECC_GameTraceChannel1, Params);
			const bool bRegisteredHit = bHit && (Hit.Normal.Dot(Hit.Normal.GetSafeNormal2D()) > .707);

			const float DesiredDotValueForDir = -FMath::Abs(RayCastDir.Dot(DirArrow->GetRightVector()));
			if (bRegisteredHit)
			{
				const FSdHitMetaData HitMetaData
				{
					.HitDistance = Hit.Distance / MaxDistanceCheck,
					.HitDotProduct = static_cast<float>(RayCastDir.Dot(Hit.Normal)),
					.DesiredHitDotProduct = DesiredDotValueForDir,
					.HitLocation = Hit.Location,
					.VisualizerLocation = DirArrow->GetComponentLocation()
				};
				ObservedObstacles.Add(HitMetaData);
			}
			else
			{
				const FSdHitMetaData HitMetaData
				{
					.HitDistance = 1.f,
					.HitDotProduct = DesiredDotValueForDir,
					.DesiredHitDotProduct = DesiredDotValueForDir,
					.HitLocation = Hit.Location,
					.VisualizerLocation = DirArrow->GetComponentLocation()
				};
				ObservedObstacles.Add(HitMetaData);
			}

			if (bRegisteredHit)
			{
				DrawDebugLine(GetWorld(), Start, Hit.Location, FColor::Red, false, 0.1f);
				DrawDebugPoint(GetWorld(), Hit.Location, 10.0f, FColor::Red, false, 0.1f);
			}
			else
			{
				DrawDebugLine(GetWorld(), Start, End, FColor::Green, false, 0.1f);
			}
		}
	}
}

void ASelfDrivingCarPawn::ResetToRandomPointOnSpline(USplineComponent* Spline)
{
	const float SplineLength = Spline->GetSplineLength();
	const float RandDistance = FMath::FRandRange(0.0f, SplineLength - 1.0f);

	const FVector Location = Spline->GetLocationAtDistanceAlongSpline(RandDistance, ESplineCoordinateSpace::World);

	FRotator Rotation = Spline->GetRotationAtDistanceAlongSpline(RandDistance, ESplineCoordinateSpace::World);
	Rotation = FRotator(0.0f, Rotation.Yaw, 0.0f);

	DrawDebugSphere(GetWorld(), Location, 20, 20, FColor::Green, false, 10);

	const FTransform NewTransform(Rotation, Location);
	SetActorTransform(NewTransform, false, nullptr, ETeleportType::TeleportPhysics);
	GetMesh()->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
	GetMesh()->SetPhysicsLinearVelocity(FVector::ZeroVector);
}

void ASelfDrivingCarPawn::Steering(const FInputActionValue& Value)
{
	// get the input magnitude for steering
	float SteeringValue = Value.Get<float>();

	// add the input
	ChaosVehicleMovement->SetSteeringInput(SteeringValue);
}

void ASelfDrivingCarPawn::Throttle(const FInputActionValue& Value)
{
	// get the input magnitude for the throttle
	float ThrottleValue = Value.Get<float>();

	// add the input
	ChaosVehicleMovement->SetThrottleInput(ThrottleValue);
}

void ASelfDrivingCarPawn::Brake(const FInputActionValue& Value)
{
	// get the input magnitude for the brakes
	float BreakValue = Value.Get<float>();

	// add the input
	ChaosVehicleMovement->SetBrakeInput(BreakValue);
}

void ASelfDrivingCarPawn::StartBrake(const FInputActionValue& Value)
{
	// call the Blueprint hook for the break lights
	BrakeLights(true);
}

void ASelfDrivingCarPawn::StopBrake(const FInputActionValue& Value)
{
	// call the Blueprint hook for the break lights
	BrakeLights(false);

	// reset brake input to zero
	ChaosVehicleMovement->SetBrakeInput(0.0f);
}

void ASelfDrivingCarPawn::StartHandbrake(const FInputActionValue& Value)
{
	// add the input
	ChaosVehicleMovement->SetHandbrakeInput(true);

	// call the Blueprint hook for the break lights
	BrakeLights(true);
}

void ASelfDrivingCarPawn::StopHandbrake(const FInputActionValue& Value)
{
	// add the input
	ChaosVehicleMovement->SetHandbrakeInput(false);

	// call the Blueprint hook for the break lights
	BrakeLights(false);
}

void ASelfDrivingCarPawn::LookAround(const FInputActionValue& Value)
{
	// get the flat angle value for the input 
	float LookValue = Value.Get<float>();

	// add the input
	BackSpringArm->AddLocalRotation(FRotator(0.0f, LookValue, 0.0f));
}

void ASelfDrivingCarPawn::ToggleCamera(const FInputActionValue& Value)
{
	// toggle the active camera flag
	bFrontCameraActive = !bFrontCameraActive;

	FrontCamera->SetActive(bFrontCameraActive);
	BackCamera->SetActive(!bFrontCameraActive);
}

void ASelfDrivingCarPawn::ResetVehicle(const FInputActionValue& Value)
{
	// reset to a location slightly above our current one
	FVector ResetLocation = GetActorLocation() + FVector(0.0f, 0.0f, 50.0f);

	// reset to our yaw. Ignore pitch and roll
	FRotator ResetRotation = GetActorRotation();
	ResetRotation.Pitch = 0.0f;
	ResetRotation.Roll = 0.0f;
	
	// teleport the actor to the reset spot and reset physics
	SetActorTransform(FTransform(ResetRotation, ResetLocation, FVector::OneVector), false, nullptr, ETeleportType::TeleportPhysics);

	GetMesh()->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
	GetMesh()->SetPhysicsLinearVelocity(FVector::ZeroVector);

	UE_LOG(LogTemplateVehicle, Error, TEXT("Reset Vehicle"));
}

#undef LOCTEXT_NAMESPACE