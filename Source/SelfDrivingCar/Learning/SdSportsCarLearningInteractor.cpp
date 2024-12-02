// Fill out your copyright notice in the Description page of Project Settings.


#include "SdSportsCarLearningInteractor.h"

#include "ChaosVehicleMovementComponent.h"
#include "Tags.h"
#include "Components/SplineComponent.h"
#include "SelfDrivingCar/SelfDrivingCarPawn.h"

using UtilsObservations = ULearningAgentsObservations;
using UtilsActions = ULearningAgentsActions;

void USdSportsCarLearningInteractor::SpecifyAgentObservation_Implementation(
	FLearningAgentsObservationSchemaElement& OutObservationSchemaElement,
	ULearningAgentsObservationSchema* InObservationSchema)
{
	TMap<FName, FLearningAgentsObservationSchemaElement> Observations;

	/// #############################################
	/// === === ===  SPLINE OBSERVATIONS  === === ===
	/// #############################################
	if (ObservationsToUse.HasTag(Sd::Observations::Spline))
	{
		TMap<FName, FLearningAgentsObservationSchemaElement> SplineObservations;

		if (ObservationsToUse.HasTag(Sd::Observations::Spline_Location))
		{
			const auto LocationObs = UtilsObservations::SpecifyLocationAlongSplineObservation(
				InObservationSchema, 100.f, Sd::Observations::Spline_Location.GetTag().GetTagName());
			SplineObservations.Add("Location", LocationObs);
		}

		if (ObservationsToUse.HasTag(Sd::Observations::Spline_Direction))
		{
			const auto DirectionObs = UtilsObservations::SpecifyDirectionAlongSplineObservation(
				InObservationSchema, Sd::Observations::Spline_Direction.GetTag().GetTagName());
			SplineObservations.Add("Direction", DirectionObs);
		}

		const auto TrackObservation = UtilsObservations::SpecifyStructObservation(InObservationSchema, SplineObservations);
		const auto TrackObservationSamples = UtilsObservations::SpecifyStaticArrayObservation(
			InObservationSchema, TrackObservation, TrackDistanceSamples.Num());

		Observations.Add("Track", TrackObservationSamples);
	}

	/// ##########################################
	/// === === ===  CAR OBSERVATIONS  === === ===
	/// ##########################################
	if (ObservationsToUse.HasTag(Sd::Observations::Car))
	{
		TMap<FName, FLearningAgentsObservationSchemaElement> CarObservations;

		if (ObservationsToUse.HasTag(Sd::Observations::Car_Location))
		{
			const auto LocationObs = UtilsObservations::SpecifyLocationObservation(
				InObservationSchema, 100, Sd::Observations::Car_Location.GetTag().GetTagName());
			CarObservations.Add("Location", LocationObs);
		}

		if (ObservationsToUse.HasTag(Sd::Observations::Car_Velocity))
		{
			const auto VelocityObs = UtilsObservations::SpecifyVelocityObservation(
				InObservationSchema, 200., Sd::Observations::Car_Velocity.GetTag().GetTagName());
			CarObservations.Add("Velocity", VelocityObs);
		}

		const auto CarObservation = UtilsObservations::SpecifyStructObservation(InObservationSchema, CarObservations);

		Observations.Add("Car", CarObservation);
	}

	/// ###############################################
	/// === === ===  OBSTACLE OBSERVATIONS  === === ===
	/// ###############################################
	if (ObservationsToUse.HasTag(Sd::Observations::Obstacle))
	{
		TMap<FName, FLearningAgentsObservationSchemaElement> ObstacleObservations;

		if (ObservationsToUse.HasTag(Sd::Observations::Obstacle_Distance))
		{
			const auto DistanceObs = UtilsObservations::SpecifyFloatObservation(
				InObservationSchema, 1.f, Sd::Observations::Obstacle_Distance.GetTag().GetTagName());
			ObstacleObservations.Add("Distance", DistanceObs);
		}

		if (ObservationsToUse.HasTag(Sd::Observations::Obstacle_Angle))
		{
			const auto AngleObs = UtilsObservations::SpecifyFloatObservation(
				InObservationSchema, 1.f, Sd::Observations::Obstacle_Angle.GetTag().GetTagName());
			ObstacleObservations.Add("Angle", AngleObs);
		}

		const auto ObstacleObservation = UtilsObservations::SpecifyStructObservation(InObservationSchema, ObstacleObservations);
		const auto ObstacleObservationSamples = UtilsObservations::SpecifyStaticArrayObservation(
			InObservationSchema, ObstacleObservation, ObstacleDistanceObservationsNum);

		Observations.Add("Obstacle", ObstacleObservationSamples);
	}

	OutObservationSchemaElement = UtilsObservations::SpecifyStructObservation(InObservationSchema, Observations);
}

void USdSportsCarLearningInteractor::GatherAgentObservation_Implementation(
	FLearningAgentsObservationObjectElement& OutObservationObjectElement,
	ULearningAgentsObservationObject* InObservationObject, const int32 AgentId)
{
	auto* AgentActor = Cast<ASelfDrivingCarPawn>(GetAgent(AgentId));
	if (!AgentActor || !TrackSpline)
	{
		return;
	}

	const FTransform& AgentTransform = AgentActor->GetActorTransform();

	TMap<FName, FLearningAgentsObservationObjectElement> Observations;

	/// #############################################
	/// === === ===  SPLINE OBSERVATIONS  === === ===
	/// #############################################
	if (ObservationsToUse.HasTag(Sd::Observations::Spline))
	{
		const float AgentDistAlongSpline = TrackSpline->GetDistanceAlongSplineAtLocation(
			AgentTransform.GetLocation(), ESplineCoordinateSpace::World);

		TArray<FLearningAgentsObservationObjectElement> TrackObservations;
		TrackObservations.Reserve(TrackDistanceSamples.Num());

		for (float Distance : TrackDistanceSamples)
		{
			TMap<FName, FLearningAgentsObservationObjectElement> SplineObservations;
			const float DistAlongSpline = AgentDistAlongSpline + Distance;

			if (ObservationsToUse.HasTag(Sd::Observations::Spline_Location))
			{
				const auto LocationObs = UtilsObservations::MakeLocationAlongSplineObservation(
					InObservationObject,
					TrackSpline,
					DistAlongSpline,
					AgentTransform,
					Sd::Observations::Spline_Location.GetTag().GetTagName(),
					true,
					this,
					AgentId,
					TrackSpline->GetLocationAtDistanceAlongSpline(DistAlongSpline, ESplineCoordinateSpace::World),
					FColor::Cyan);
				SplineObservations.Add("Location", LocationObs);
			}

			if (ObservationsToUse.HasTag(Sd::Observations::Spline_Direction))
			{
				const auto DirectionObs = UtilsObservations::MakeDirectionAlongSplineObservation(
					InObservationObject,
					TrackSpline,
					DistAlongSpline,
					AgentTransform,
					Sd::Observations::Spline_Direction.GetTag().GetTagName(),
					true,
					this,
					AgentId,
					TrackSpline->GetLocationAtDistanceAlongSpline(DistAlongSpline, ESplineCoordinateSpace::World),
					100.f,
					FColor::Yellow);
				SplineObservations.Add("Direction", DirectionObs);
			}

			const auto TrackObservation = UtilsObservations::MakeStructObservation(InObservationObject, SplineObservations);
			TrackObservations.Add(TrackObservation);
		}

		const auto TrackObservationSamples = UtilsObservations::MakeStaticArrayObservation(
			InObservationObject, TrackObservations);
		Observations.Add("Track", TrackObservationSamples);
	}

	/// ##########################################
	/// === === ===  CAR OBSERVATIONS  === === ===
	/// ##########################################
	if (ObservationsToUse.HasTag(Sd::Observations::Car))
	{
		TMap<FName, FLearningAgentsObservationObjectElement> CarObservationsSample;

		if (ObservationsToUse.HasTag(Sd::Observations::Car_Location))
		{
			const auto LocationObs = UtilsObservations::MakeLocationObservation(
				InObservationObject,
				AgentTransform.GetLocation(),
				AgentTransform,
				Sd::Observations::Car_Location.GetTag().GetTagName(),
				true,
				this,
				AgentId,
				AgentTransform.GetLocation(),
				FColor::Green);

			CarObservationsSample.Add("Location", LocationObs);
		}

		if (ObservationsToUse.HasTag(Sd::Observations::Car_Velocity))
		{
			const auto VelocityObs = UtilsObservations::MakeVelocityObservation(
				InObservationObject,
				AgentActor->GetVelocity(),
				AgentTransform,
				Sd::Observations::Car_Velocity.GetTag().GetTagName(),
				true,
				this,
				AgentId,
				AgentTransform.GetLocation(),
				AgentTransform.GetLocation(),
				FColor::Purple);

			CarObservationsSample.Add("Velocity", VelocityObs);
		}

		const auto CarObservationStruct = UtilsObservations::MakeStructObservation(InObservationObject, CarObservationsSample);
		Observations.Add("Car", CarObservationStruct);
	}

	/// ###############################################
	/// === === ===  OBSTACLE OBSERVATIONS  === === ===
	/// ###############################################
	if (ObservationsToUse.HasTag(Sd::Observations::Obstacle))
	{
		TArray<FLearningAgentsObservationObjectElement> ObstacleObservations;
		AgentActor->CalcDistances();

		for (const FSdHitMetaData& HitData : AgentActor->ObservedObstacles)
		{
			TMap<FName, FLearningAgentsObservationObjectElement> ObstacleObservationsSample;

			if (ObservationsToUse.HasTag(Sd::Observations::Obstacle_Distance))
			{
				const auto DistanceObs = UtilsObservations::MakeFloatObservation(
					InObservationObject,
					HitData.HitDistance,
					Sd::Observations::Obstacle_Distance.GetTag().GetTagName(),
					true,
					this,
					AgentId,
					HitData.VisualizerLocation,
					FColor::Purple);

				ObstacleObservationsSample.Add("Distance", DistanceObs);
			}

			if (ObservationsToUse.HasTag(Sd::Observations::Obstacle_Angle))
			{
				const auto AngleObs = UtilsObservations::MakeFloatObservation(
					InObservationObject,
					HitData.HitDotProduct,
					Sd::Observations::Obstacle_Angle.GetTag().GetTagName(),
					true,
					this,
					AgentId,
					HitData.VisualizerLocation,
					FColor::Purple);

				ObstacleObservationsSample.Add("Angle", AngleObs);
			}

			const auto ObstacleObservation = UtilsObservations::MakeStructObservation(InObservationObject, ObstacleObservationsSample);
			ObstacleObservations.Add(ObstacleObservation);
		}

		const auto ObstacleObservationSamples = UtilsObservations::MakeStaticArrayObservation(
			InObservationObject, ObstacleObservations);
		Observations.Add("Obstacle", ObstacleObservationSamples);
	}

	OutObservationObjectElement = UtilsObservations::MakeStructObservation(InObservationObject, Observations);
}

void USdSportsCarLearningInteractor::SpecifyAgentAction_Implementation(
	FLearningAgentsActionSchemaElement& OutActionSchemaElement,
	ULearningAgentsActionSchema* InActionSchema)
{
	const auto Steering = UtilsActions::SpecifyFloatAction(InActionSchema, 1.f, Sd::Actions::Steering.GetTag().GetTagName());
	const auto ThrottleBrake = UtilsActions::SpecifyFloatAction(InActionSchema, 1.f, Sd::Actions::ThrottleBreak.GetTag().GetTagName());

	const TMap<FName, FLearningAgentsActionSchemaElement> Actions =
		{
			{ Sd::Actions::Steering.GetTag().GetTagName(), Steering },
			{ Sd::Actions::ThrottleBreak.GetTag().GetTagName(), ThrottleBrake }
		};

	OutActionSchemaElement = UtilsActions::SpecifyStructAction(InActionSchema, Actions);
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
	UtilsActions::GetFloatAction(
		SteeringValue,
		InActionObject,
		Actions[Sd::Actions::Steering.GetTag().GetTagName()],
		Sd::Actions::Steering.GetTag().GetTagName(),
		true,
		this,
		AgentId,
		AgentActor->GetActorLocation());

	AgentActor->GetVehicleMovement()->SetSteeringInput(SteeringValue);

	float ThrottleBreak;
	UtilsActions::GetFloatAction(
		ThrottleBreak,
		InActionObject,
		Actions[Sd::Actions::ThrottleBreak.GetTag().GetTagName()],
		Sd::Actions::ThrottleBreak.GetTag().GetTagName(),
		true,
		this,
		AgentId,
		AgentActor->GetActorLocation());

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
