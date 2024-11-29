// Fill out your copyright notice in the Description page of Project Settings.


#include "SdRLManager.h"

#include "LearningAgentsInteractor.h"
#include "LearningAgentsPolicy.h"
#include "SdLearningAgentsManager.h"
#include "SdSportsCarLearningInteractor.h"


// Sets default values
ASdRLManager::ASdRLManager()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	auto* LearningManager = CreateDefaultSubobject<USdLearningAgentsManager>(TEXT("SdLearningAgentsManager"));

	CreateDefaultSubobject<USdSportsCarLearningInteractor>(TEXT("LearningAgentsInteractor"));
	CreateDefaultSubobject<ULearningAgentsPolicy>(TEXT("LearningAgentsPolicy"));

}

// Called when the game starts or when spawned
void ASdRLManager::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ASdRLManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

