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

	UPROPERTY(EditAnywhere, Category = "Learning|Coomponents")
	USdLearningAgentsManager* AgentsManager = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|Coomponents")
	USdSportsCarLearningInteractor* SportsCarLearningInteractor = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|Coomponents")
	USdSportsCarLearningEnv* SportsCarLearningEnv = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|Coomponents")
	ULearningAgentsPolicy* AgentsPolicy = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|Coomponents")
	ULearningAgentsCritic* AgentsCritic = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|Coomponents")
	ULearningAgentsPPOTrainer* AgentsPPOTrainer = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsPolicySettings AgentsPolicySettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsCriticSettings AgentsCriticSettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsPPOTrainerSettings AgentsPPOTrainerSettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsPPOTrainingSettings AgentsTrainingSettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsTrainingGameSettings AgentsGameSettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsTrainerProcessSettings AgentsTrainerProcessSettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsSharedMemoryCommunicatorSettings AgentsSharedMemorySettings;

	UPROPERTY(EditAnywhere, Category = "Learning|Settings")
	FLearningAgentsRecorderPathSettings AgentsPathSettings;

	UPROPERTY(EditAnywhere, Category = "Learning")
	FLearningAgentsTrainerProcess TrainerProcess;

	UPROPERTY(EditAnywhere, Category = "Learning")
	FLearningAgentsCommunicator AgentsCommunicator;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	bool bReinitializeEncoderNetwork = true;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	bool bReinitializePolicyNetwork = true;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	bool bReinitializeDecoderNetwork = true;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	bool bReinitializeCriticNetwork = true;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* EncoderNeuralNetwork = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* DecoderNeuralNetwork = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* PolicyNeuralNetwork = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning|NNs")
	ULearningAgentsNeuralNetwork* CriticNeuralNetwork = nullptr;

	UPROPERTY(EditAnywhere, Category = "Learning")
	bool bRunInference = false;

	UPROPERTY(EditAnywhere, Category = "Learning|Observations")
	TArray<float> TrackDistanceSamples;
};
