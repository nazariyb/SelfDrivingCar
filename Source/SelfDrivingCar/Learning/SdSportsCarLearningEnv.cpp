// Fill out your copyright notice in the Description page of Project Settings.


#include "SdSportsCarLearningEnv.h"

#include "LearningAgentsRewards.h"
#include "Components/SplineComponent.h"
#include "SelfDrivingCar/SelfDrivingCarPawn.h"

using UtilsRewards = ULearningAgentsRewards;

void USdSportsCarLearningEnv::GatherAgentReward_Implementation(float& OutReward, const int32 AgentId)
{
	const auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const FTransform& AgentTransform = AgentActor->GetTransform();

	const float VelocityReward = UtilsRewards::MakeRewardFromVelocityAlongSpline(
		TrackSpline, AgentTransform.GetLocation(), AgentActor->GetVelocity(), 1000.f, 1.f, 10.f);
	const float LocationReward = UtilsRewards::MakeRewardOnLocationDifferenceAboveThreshold(
		AgentTransform.GetLocation(),
		TrackSpline->FindLocationClosestToWorldLocation(AgentTransform.GetLocation(), ESplineCoordinateSpace::World),
		800.f, -10.f);

	OutReward = LocationReward + VelocityReward;
}

void USdSportsCarLearningEnv::GatherAgentCompletion_Implementation(
	ELearningAgentsCompletion& OutCompletion, const int32 AgentId)
{
	const auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const FTransform& AgentTransform = AgentActor->GetTransform();

	const FVector LocationOnSpline = TrackSpline->FindLocationClosestToWorldLocation(
		AgentTransform.GetLocation(), ESplineCoordinateSpace::World);

	OutCompletion = ULearningAgentsCompletions::MakeCompletionOnLocationDifferenceAboveThreshold(
		LocationOnSpline, AgentTransform.GetLocation(), 800.f, ELearningAgentsCompletion::Termination);
}

void USdSportsCarLearningEnv::ResetAgentEpisode_Implementation(const int32 AgentId)
{
	auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	AgentActor->ResetToRandomPointOnSpline(TrackSpline);
}
