// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameplayTagContainer.h"
#include "LearningAgentsInteractor.h"
#include "SdSportsCarLearningInteractor.generated.h"

class USplineComponent;

UCLASS()
class SELFDRIVINGCAR_API USdSportsCarLearningInteractor : public ULearningAgentsInteractor
{
	GENERATED_BODY()

	virtual void SpecifyAgentObservation_Implementation(
		FLearningAgentsObservationSchemaElement& OutObservationSchemaElement,
		ULearningAgentsObservationSchema* InObservationSchema) override;

	virtual void GatherAgentObservation_Implementation(
		FLearningAgentsObservationObjectElement& OutObservationObjectElement,
		ULearningAgentsObservationObject* InObservationObject,
		const int32 AgentId) override;

	virtual void SpecifyAgentAction_Implementation(
		FLearningAgentsActionSchemaElement& OutActionSchemaElement,
		ULearningAgentsActionSchema* InActionSchema) override;

	virtual void PerformAgentAction_Implementation(
		const ULearningAgentsActionObject* InActionObject,
		const FLearningAgentsActionObjectElement& InActionObjectElement,
		const int32 AgentId) override;

public:
	UPROPERTY(Transient)
	USplineComponent* TrackSpline = nullptr;

	UPROPERTY(EditAnywhere)
	TArray<float> TrackDistanceSamples;

	UPROPERTY(EditAnywhere)
	FGameplayTagContainer ObservationsToUse;

	UPROPERTY(EditAnywhere)
	int32 ObstacleDistanceObservationsNum = 7;
};
