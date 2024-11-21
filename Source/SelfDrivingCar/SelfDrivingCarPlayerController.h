// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "SelfDrivingCarPlayerController.generated.h"

class UInputMappingContext;
class ASelfDrivingCarPawn;
class USelfDrivingCarUI;

/**
 *  Vehicle Player Controller class
 *  Handles input mapping and user interface
 */
UCLASS(abstract)
class SELFDRIVINGCAR_API ASelfDrivingCarPlayerController : public APlayerController
{
	GENERATED_BODY()

protected:

	/** Input Mapping Context to be used for player input */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input)
	UInputMappingContext* InputMappingContext;

	/** If true, the optional steering wheel input mapping context will be registered */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input)
	bool bUseSteeringWheelControls = false;

	/** Optional Input Mapping Context to be used for steering wheel input.
	 *  This is added alongside the default Input Mapping Context and does not block other forms of input.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = Input, meta=(EditCondition="bUseSteeringWheelControls"))
	UInputMappingContext* SteeringWheelInputMappingContext;

	/** Pointer to the controlled vehicle pawn */
	TObjectPtr<ASelfDrivingCarPawn> VehiclePawn;

	/** Type of the UI to spawn */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = UI)
	TSubclassOf<USelfDrivingCarUI> VehicleUIClass;

	/** Pointer to the UI widget */
	TObjectPtr<USelfDrivingCarUI> VehicleUI;

	

	// Begin Actor interface
protected:

	virtual void BeginPlay() override;
	virtual void SetupInputComponent() override;

public:

	virtual void Tick(float Delta) override;

	// End Actor interface

	// Begin PlayerController interface
protected:

	virtual void OnPossess(APawn* InPawn) override;

	// End PlayerController interface
};
