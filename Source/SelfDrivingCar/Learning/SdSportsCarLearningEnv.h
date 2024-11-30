// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "LearningAgentsTraining/Public/LearningAgentsTrainingEnvironment.h"
#include "SdSportsCarLearningEnv.generated.h"

class USplineComponent;


UCLASS()
class SELFDRIVINGCAR_API USdSportsCarLearningEnv : public ULearningAgentsTrainingEnvironment
{
	GENERATED_BODY()

	virtual void GatherAgentReward_Implementation(float& OutReward, const int32 AgentId) override;
	virtual void GatherAgentCompletion_Implementation(ELearningAgentsCompletion& OutCompletion, const int32 AgentId) override;

	virtual void ResetAgentEpisode_Implementation(const int32 AgentId) override;

public:
	UPROPERTY(Transient)
	USplineComponent* TrackSpline = nullptr;
};
