// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "LearningAgentsManager.h"
#include "SdLearningAgentsManager.generated.h"


UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class SELFDRIVINGCAR_API USdLearningAgentsManager : public ULearningAgentsManager
{
	GENERATED_BODY()

public:
	// Sets default values for this component's properties
	USdLearningAgentsManager();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
								FActorComponentTickFunction* ThisTickFunction) override;
};
