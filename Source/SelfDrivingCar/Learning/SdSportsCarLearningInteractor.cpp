// Fill out your copyright notice in the Description page of Project Settings.


#include "SdSportsCarLearningInteractor.h"

#include "ChaosVehicleMovementComponent.h"
#include "Components/SplineComponent.h"
#include "SelfDrivingCar/SelfDrivingCarPawn.h"

using UtilsObservations = ULearningAgentsObservations;
using UtilsActions = ULearningAgentsActions;

void USdSportsCarLearningInteractor::SpecifyAgentObservation_Implementation(
	FLearningAgentsObservationSchemaElement& OutObservationSchemaElement,
	ULearningAgentsObservationSchema* InObservationSchema)
{
	const auto LocationObs = UtilsObservations::SpecifyLocationAlongSplineObservation(InObservationSchema);
	const auto DirectionObs = UtilsObservations::SpecifyDirectionAlongSplineObservation(InObservationSchema);
	const TMap<FName, FLearningAgentsObservationSchemaElement> SplineObservations =
		{
			{ "Location", LocationObs },
			{ "Direction", DirectionObs }
		};

	const auto TrackObservation = UtilsObservations::SpecifyStructObservation(InObservationSchema, SplineObservations);
	const auto TrackObservationSamples = UtilsObservations::SpecifyStaticArrayObservation(
		InObservationSchema, TrackObservation, 7/*TrackDistanceSamples.Num()*/);

	const auto VelocityObs = UtilsObservations::SpecifyLocationObservation(InObservationSchema);

	const TMap<FName, FLearningAgentsObservationSchemaElement> Observations =
		{
			{ "Track", TrackObservationSamples },
			{ "Car", VelocityObs }
		};

	OutObservationSchemaElement = UtilsObservations::SpecifyStructObservation(InObservationSchema, Observations);
}

void USdSportsCarLearningInteractor::SpecifyAgentAction_Implementation(
	FLearningAgentsActionSchemaElement& OutActionSchemaElement,
	ULearningAgentsActionSchema* InActionSchema)
{
	const auto Steering = UtilsActions::SpecifyFloatAction(InActionSchema, 1.f, "Steering");
	const auto ThrottleBrake = UtilsActions::SpecifyFloatAction(InActionSchema, 1.f, "ThrottleBreak");

	const TMap<FName, FLearningAgentsActionSchemaElement> Actions =
		{
			{ "Steering", Steering },
			{ "ThrottleBreak", ThrottleBrake }
		};

	OutActionSchemaElement = UtilsActions::SpecifyStructAction(InActionSchema, Actions);
}

void USdSportsCarLearningInteractor::GatherAgentObservation_Implementation(
	FLearningAgentsObservationObjectElement& OutObservationObjectElement,
	ULearningAgentsObservationObject* InObservationObject, const int32 AgentId)
{
	const auto* AgentActor = Cast<AActor>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const FTransform& AgentTransform = AgentActor->GetActorTransform();
	const float AgentDistAlongSpline = TrackSpline->GetDistanceAlongSplineAtLocation(
		AgentTransform.GetLocation(), ESplineCoordinateSpace::World);

	TArray<FLearningAgentsObservationObjectElement> TrackObservations;
	TrackObservations.Reserve(TrackDistanceSamples.Num());

	for (float Distance : TrackDistanceSamples)
	{
		const float DistAlongSpline = AgentDistAlongSpline + Distance;

		const auto LocationObs = UtilsObservations::MakeLocationAlongSplineObservation(
			InObservationObject,
			TrackSpline,
			DistAlongSpline,
			AgentTransform,
			"LocationAlongSplineObservation",
			true,
			this,
			AgentId,
			TrackSpline->GetLocationAtDistanceAlongSpline(DistAlongSpline, ESplineCoordinateSpace::World),
			FColor::Cyan);
		const auto DirectionObs = UtilsObservations::MakeDirectionAlongSplineObservation(
			InObservationObject,
			TrackSpline,
			DistAlongSpline,
			AgentTransform,
			"DirectionAlongSplineObservation",
			true,
			this,
			AgentId,
			TrackSpline->GetLocationAtDistanceAlongSpline(DistAlongSpline, ESplineCoordinateSpace::World),
			100.f,
			FColor::Yellow);

		const TMap<FName, FLearningAgentsObservationObjectElement> SplineObservations =
			{
			{ "Location", LocationObs },
			{ "Direction", DirectionObs }
			};

		const auto TrackObservation = UtilsObservations::MakeStructObservation(InObservationObject, SplineObservations);
		TrackObservations.Add(TrackObservation);
	}

	const auto TrackObservationSamples = UtilsObservations::MakeStaticArrayObservation(
		InObservationObject, TrackObservations);

	const auto VelocityObs = UtilsObservations::MakeLocationObservation(
		InObservationObject,
		AgentTransform.GetLocation(),
		AgentTransform,
		"LocationObservation",
		true,
		this,
		AgentId,
		AgentTransform.GetLocation(),
		FColor::Green);

	const TMap<FName, FLearningAgentsObservationObjectElement> Observations =
		{
			{ "Track", TrackObservationSamples },
			{ "Car", VelocityObs }
		};

	OutObservationObjectElement = UtilsObservations::MakeStructObservation(InObservationObject, Observations);
}

void USdSportsCarLearningInteractor::PerformAgentAction_Implementation(
	const ULearningAgentsActionObject* InActionObject, const FLearningAgentsActionObjectElement& InActionObjectElement,
	const int32 AgentId)
{
	const auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	TMap<FName, FLearningAgentsActionObjectElement> Actions;
	UtilsActions::GetStructAction(Actions, InActionObject, InActionObjectElement);

	float SteeringValue;
	UtilsActions::GetFloatAction(SteeringValue, InActionObject, Actions["Steering"], "Steering", true, nullptr, AgentId);

	AgentActor->GetVehicleMovement()->SetSteeringInput(SteeringValue);

	float ThrottleBreak;
	UtilsActions::GetFloatAction(ThrottleBreak, InActionObject, Actions["ThrottleBreak"], "ThrottleBreak", true, nullptr, AgentId);

	if (ThrottleBreak > 0.f)
	{
		AgentActor->GetVehicleMovement()->SetThrottleInput(ThrottleBreak);
		AgentActor->GetVehicleMovement()->SetBrakeInput(0.f);
	}
	else if (ThrottleBreak < 0.f)
	{
		AgentActor->GetVehicleMovement()->SetThrottleInput(0.f);
		AgentActor->GetVehicleMovement()->SetBrakeInput(-ThrottleBreak);
	}
}
