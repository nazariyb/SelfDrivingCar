// Fill out your copyright notice in the Description page of Project Settings.


#include "SdSportsCarLearningEnv.h"

#include "LearningAgentsRewards.h"
#include "Tags.h"
#include "Components/SplineComponent.h"
#include "SelfDrivingCar/SelfDrivingCarPawn.h"

using UtilsRewards = ULearningAgentsRewards;

void USdSportsCarLearningEnv::GatherAgentReward_Implementation(float& OutReward, const int32 AgentId)
{
	auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const FTransform& AgentTransform = AgentActor->GetTransform();

	const float LocationReward = UtilsRewards::MakeRewardFromLocationDifference(
		AgentActor->PreviousLocation,
		AgentTransform.GetLocation(),
		100.f,
		2.f,
		Sd::Rewards::Movement_Delta.GetTag().GetTagName(),
		true,
		this,
		AgentId,
		AgentTransform.GetLocation(),
		FColor::Green);
	// AgentActor->PreviousLocation = AgentTransform.GetLocation();

	const float VelocityReward = UtilsRewards::MakeRewardFromVelocityAlongSpline(
		TrackSpline,
		AgentTransform.GetLocation(),
		AgentActor->GetVelocity(),
		1000.f,
		25.f,
		8.f,
		Sd::Rewards::Movement_Velocity.GetTag().GetTagName(),
		true,
		this,
		AgentId,
		AgentTransform.GetLocation(),
		FColor::Green);

	const float CurrentDistance = TrackSpline->GetDistanceAlongSplineAtLocation(AgentTransform.GetLocation(), ESplineCoordinateSpace::World);
	const float DistanceToFinish = CurrentDistance / TrackSpline->GetSplineLength();
	const float CloseToFinishReward = UtilsRewards::MakeRewardFromLocationSimilarity(
		FVector(DistanceToFinish, 0.0f, 0.0f),
		FVector(1.f, 0.f, 0.f),
		1.f,
		10.f,
		Sd::Rewards::DistanceToFinish.GetTag().GetTagName(),
		true,
		this,
		AgentId,
		AgentTransform.GetLocation(),
		FColor::Green);

	const TArray<FSdHitMetaData>& ObservedObstacles = AgentActor->ObservedObstacles;

	float ObstacleDistanceReward = 0.f;
	for (const FSdHitMetaData& HitData : ObservedObstacles)
	{
		if (HitData.HitDistance < 30.f / AgentActor->MaxDistanceCheck)
		{
			ObstacleDistanceReward += UtilsRewards::MakeRewardFromLocationSimilarity(
				FVector(HitData.HitDistance, 0.f, 0.f),
				FVector::ZeroVector,
				1.f,
				-10.f,
				Sd::Rewards::Obstacle_Distance.GetTag().GetTagName(),
				true,
				this,
				AgentId,
				HitData.VisualizerLocation,
				FColor::Green);
		}
	}

	float ObstacleAngleReward = 0.f;
	for (const FSdHitMetaData& HitData : ObservedObstacles)
	{
		const float ObservedAngle = FMath::RadiansToDegrees(FMath::Acos(HitData.HitDotProduct));
		const float ExpectedAngle = FMath::RadiansToDegrees(FMath::Acos(HitData.DesiredHitDotProduct));

		const FVector ObservedAngleHit = HitData.HitLocation;
		const FVector ExpectedAngleHit = FRotator(0.f, ExpectedAngle - ObservedAngle, 0.f).RotateVector(ObservedAngleHit - HitData.VisualizerLocation);

		ObstacleAngleReward += UtilsRewards::MakeRewardFromAngleSimilarity(
			ObservedAngle,
			ExpectedAngle,
			.03f,
			Sd::Rewards::Obstacle_Angle.GetTag().GetTagName(),
			true,
			this,
			AgentId,
			ObservedAngleHit,
			ExpectedAngleHit,
			HitData.VisualizerLocation,
			FColor::Green);
	}

	const float TimeReward = UtilsRewards::MakeRewardFromLocationDifference(
		FVector(GetEpisodeTime(AgentId), 0.f, 0.f),
		FVector::ZeroVector,
		1.f,
		1.f,
		Sd::Rewards::TimeToFinish.GetTag().GetTagName(),
		true,
		this,
		AgentId,
		AgentTransform.GetLocation(),
		FColor::Green);

	OutReward = /*LocationReward + */VelocityReward + CloseToFinishReward + ObstacleDistanceReward + ObstacleAngleReward/* + TimeReward*/;
}

void USdSportsCarLearningEnv::GatherAgentCompletion_Implementation(
	ELearningAgentsCompletion& OutCompletion, const int32 AgentId)
{
	auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const FTransform& AgentTransform = AgentActor->GetTransform();

	const double CurrentTime = AgentActor->GetWorld()->GetTimeSeconds();
	if (FMath::IsNearlyZero(AgentActor->PreviousMeasureTime))
	{
		AgentActor->PreviousLocation = AgentTransform.GetLocation();
		AgentActor->PreviousMeasureTime = CurrentTime;
	}

	if (CurrentTime - AgentActor->PreviousMeasureTime > 5.f)
	{
		OutCompletion = ULearningAgentsCompletions::MakeCompletionOnLocationDifferenceBelowThreshold(
			AgentTransform.GetLocation(),
			AgentActor->PreviousLocation,
			40.f,
			ELearningAgentsCompletion::Truncation,
			"Completion.Movement",
			true,
			this,
			AgentId,
			AgentTransform.GetLocation(),
			FColor::Yellow);

		AgentActor->PreviousLocation = AgentTransform.GetLocation();
		AgentActor->PreviousMeasureTime = CurrentTime;
	}
	else
	{
		OutCompletion = ELearningAgentsCompletion::Running;
	}


	if (OutCompletion != ELearningAgentsCompletion::Running)
	{
		return;
	}

	// if (OutCompletion == ELearningAgentsCompletion::Running)
	// {
	// 	const bool bCloseToObstacle = Algo::AnyOf(AgentActor->ObservedObstacles, [](const FSdHitMetaData& Hit) { return Hit.HitDistance < 40; });
	//
	// 	OutCompletion = ULearningAgentsCompletions::MakeCompletionOnCondition(
	// 		bCloseToObstacle,
	// 		ELearningAgentsCompletion::Truncation,
	// 		"HitObstacle",
	// 		true,
	// 		this,
	// 		AgentId,
	// 		AgentTransform.GetLocation(),
	// 		FColor::Red);
	// }

	const FVector LocationOnSpline = TrackSpline->FindLocationClosestToWorldLocation(
		AgentTransform.GetLocation(), ESplineCoordinateSpace::World);
	OutCompletion = ULearningAgentsCompletions::MakeCompletionOnLocationDifferenceAboveThreshold(
		LocationOnSpline, AgentTransform.GetLocation(), 800.f, ELearningAgentsCompletion::Termination);

	if (OutCompletion != ELearningAgentsCompletion::Running)
	{
		return;
	}

	const float CurrentDistance = TrackSpline->GetDistanceAlongSplineAtLocation(AgentTransform.GetLocation(), ESplineCoordinateSpace::World);
	const float DistanceToFinish = TrackSpline->GetSplineLength() - CurrentDistance;

	OutCompletion = ULearningAgentsCompletions::MakeCompletionOnLocationDifferenceBelowThreshold(
		FVector(DistanceToFinish, 0.f, 0.f),
		FVector::ZeroVector,
		200.f,
		ELearningAgentsCompletion::Truncation,
		"DistanceToFinish",
		true,
		this,
		AgentId,
		AgentTransform.GetLocation(),
		FColor::Yellow);
}

void USdSportsCarLearningEnv::ResetAgentEpisode_Implementation(const int32 AgentId)
{
	auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const int32 PointIndex = FMath::Rand() % RespawnTransforms.Num();
	const FTransform NewTransform = RespawnTransforms[PointIndex];

	AgentActor->SetActorTransform(NewTransform, false, nullptr, ETeleportType::TeleportPhysics);
	AgentActor->GetMesh()->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
	AgentActor->GetMesh()->SetPhysicsLinearVelocity(FVector::ZeroVector);

	// AgentActor->ResetToRandomPointOnSpline(TrackSpline);
}
