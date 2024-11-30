// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "LearningAgentsCommunicator.h"
#include "LearningAgentsCritic.h"
#include "LearningAgentsPolicy.h"
#include "LearningAgentsPPOTrainer.h"
#include "LearningAgentsRecorder.h"
#include "GameFramework/Actor.h"
#include "SdRLManager.generated.h"

class ULearningAgentsPolicy;
class USdSportsCarLearningEnv;
class USdSportsCarLearningInteractor;
class USplineComponent;
class ASelfDrivingCarPawn;
class USdLearningAgentsManager;


UCLASS()
class SELFDRIVINGCAR_API ASdRLManager : public AActor
{
	GENERATED_BODY()

public:
	ASdRLManager();

protected:
	virtual void BeginPlay() override;

public:
	virtual void Tick(float DeltaTime) override;

#if WITH_EDITOR
	virtual void PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

protected:
	UPROPERTY(EditAnywhere, Category = "Learning")
	TSubclassOf<AActor> SelfDrivingCarPawnClass = nullptr;

	UPROPERTY(Transient,VisibleAnywhere, Category = "Learning")
	USplineComponent* TrackSpline = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning")
	AActor* TrackActor = nullptr;

	UPROPERTY(Transient)
	TArray<AActor*> CarActors;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Coomponents")
	USdLearningAgentsManager* AgentsManager = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Coomponents")
	USdSportsCarLearningInteractor* SportsCarLearningInteractor = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Coomponents")
	USdSportsCarLearningEnv* SportsCarLearningEnv = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Coomponents")
	ULearningAgentsPolicy* AgentsPolicy = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Coomponents")
	ULearningAgentsCritic* AgentsCritic = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Coomponents")
	ULearningAgentsPPOTrainer* AgentsPPOTrainer = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsPolicySettings AgentsPolicySettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsCriticSettings AgentsCriticSettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsPPOTrainerSettings AgentsPPOTrainerSettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsPPOTrainingSettings AgentsTrainingSettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsTrainingGameSettings AgentsGameSettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsTrainerProcessSettings AgentsTrainerProcessSettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsSharedMemoryCommunicatorSettings AgentsSharedMemorySettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Settings")
	FLearningAgentsRecorderPathSettings AgentsPathSettings;

	UPROPERTY(EditDefaultsOnly, Category = "Learning")
	FLearningAgentsTrainerProcess TrainerProcess;

	UPROPERTY(EditDefaultsOnly, Category = "Learning")
	FLearningAgentsCommunicator AgentsCommunicator;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	bool bReinitializeEncoderNetwork = true;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	bool bReinitializePolicyNetwork = true;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	bool bReinitializeDecoderNetwork = true;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	bool bReinitializeCriticNetwork = true;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* EncoderNeuralNetwork = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* DecoderNeuralNetwork = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* PolicyNeuralNetwork = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* CriticNeuralNetwork = nullptr;

	UPROPERTY(EditDefaultsOnly, Category = "Learning")
	bool bRunInference = false;

	UPROPERTY(EditDefaultsOnly, Category = "Learning|Observations")
	TArray<float> TrackDistanceSamples;
};
