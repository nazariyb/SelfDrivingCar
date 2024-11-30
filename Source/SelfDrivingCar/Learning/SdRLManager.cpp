// Fill out your copyright notice in the Description page of Project Settings.


#include "SdRLManager.h"

#include "LearningAgentsCritic.h"
#include "LearningAgentsInteractor.h"
#include "LearningAgentsPolicy.h"
#include "LearningAgentsPPOTrainer.h"
#include "LearningAgentsTrainingEnvironment.h"
#include "SdLearningAgentsManager.h"
#include "SdSportsCarLearningEnv.h"
#include "SdSportsCarLearningInteractor.h"
#include "SdSportsCarPPOTrainer.h"
#include "Components/SplineComponent.h"
#include "Kismet/GameplayStatics.h"
#include "SelfDrivingCar/SelfDrivingCarPawn.h"


ASdRLManager::ASdRLManager()
{
	PrimaryActorTick.bCanEverTick = true;

	AgentsManager = CreateDefaultSubobject<USdLearningAgentsManager>(TEXT("SdLearningAgentsManager"));
}

void ASdRLManager::BeginPlay()
{
	Super::BeginPlay();

	if (!SelfDrivingCarPawnClass)
	{
		UE_LOG(LogTemp, Error, TEXT("No class set for SelfDrivingCarPawnClass"));
		return;
	}

	if (auto* Spline = TrackActor->FindComponentByClass<USplineComponent>())
	{
		TrackSpline = Spline;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Actor %s doesn't have SplineComponent"), *TrackActor->GetName());
		return;
	}

	UGameplayStatics::GetAllActorsOfClass(this, SelfDrivingCarPawnClass, CarActors);

	for (AActor* CarActor : CarActors)
	{
		CarActor->AddTickPrerequisiteActor(this);
	}

	ULearningAgentsManager* AgentsManagerBase = AgentsManager;

	SportsCarLearningInteractor = Cast<USdSportsCarLearningInteractor>(ULearningAgentsInteractor::MakeInteractor(
		AgentsManagerBase, USdSportsCarLearningInteractor::StaticClass()));
	SportsCarLearningInteractor->TrackSpline = TrackSpline;
	SportsCarLearningInteractor->TrackDistanceSamples = TrackDistanceSamples;

	ULearningAgentsInteractor* InteractorBase = SportsCarLearningInteractor;

	AgentsPolicy = ULearningAgentsPolicy::MakePolicy(
		AgentsManagerBase,
		InteractorBase,
		ULearningAgentsPolicy::StaticClass(),
		"SportsCarPolicy",
		EncoderNeuralNetwork,
		PolicyNeuralNetwork,
		DecoderNeuralNetwork,
		!bRunInference && bReinitializeEncoderNetwork,
		!bRunInference && bReinitializePolicyNetwork,
		!bRunInference && bReinitializeDecoderNetwork,
		AgentsPolicySettings,
		1234);

	AgentsCritic = ULearningAgentsCritic::MakeCritic(
		AgentsManagerBase,
		InteractorBase,
		AgentsPolicy,
		ULearningAgentsCritic::StaticClass(),
		"SportsCarCritic",
		CriticNeuralNetwork,
		!bRunInference && bReinitializeCriticNetwork,
		AgentsCriticSettings,
		1234);

	SportsCarLearningEnv = Cast<USdSportsCarLearningEnv>(ULearningAgentsTrainingEnvironment::MakeTrainingEnvironment(
		AgentsManagerBase,
		USdSportsCarLearningEnv::StaticClass(),
		"SportsCarTrainingEnvironment"));
	SportsCarLearningEnv->TrackSpline = TrackSpline;

	TrainerProcess = ULearningAgentsCommunicatorLibrary::SpawnSharedMemoryTrainingProcess(
		AgentsTrainerProcessSettings, AgentsSharedMemorySettings);
	AgentsCommunicator = ULearningAgentsCommunicatorLibrary::MakeSharedMemoryCommunicator(
		TrainerProcess, AgentsSharedMemorySettings);

	ULearningAgentsTrainingEnvironment* LearningEnvBase = SportsCarLearningEnv;

	AgentsPPOTrainer = ULearningAgentsPPOTrainer::MakePPOTrainer(
		AgentsManagerBase,
		InteractorBase,
		LearningEnvBase,
		AgentsPolicy,
		AgentsCritic,
		AgentsCommunicator,
		USdSportsCarPPOTrainer::StaticClass(),
		"SportsCarPPOTrainer",
		AgentsPPOTrainerSettings);

	if (bRunInference && ensure(TrackSpline))
	{
		for (AActor* CarActor : CarActors)
		{
			Cast<ASelfDrivingCarPawn>(CarActor)->ResetToRandomPointOnSpline(TrackSpline);
		}
	}
}

void ASdRLManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (bRunInference)
	{
		AgentsPolicy->RunInference(0);
	}
	else
	{
		AgentsPPOTrainer->RunTraining(AgentsTrainingSettings, AgentsGameSettings, true, true);
	}
}

void ASdRLManager::PostEditChangeProperty(struct FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	if (PropertyChangedEvent.GetPropertyName() == GET_MEMBER_NAME_CHECKED(ASdRLManager, TrackActor))
	{
		if (auto* Spline = TrackActor->FindComponentByClass<USplineComponent>())
		{
			TrackSpline = Spline;
		}
		else
		{
			UE_LOG(LogTemp, Log, TEXT("Actor %s doesn't have SplineComponent"), *TrackActor->GetName());
		}
	}
}

