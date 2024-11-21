// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "SelfDrivingCarPawn.h"
#include "SelfDrivingCarSportsCar.generated.h"

/**
 *  Sports car wheeled vehicle implementation
 */
UCLASS(abstract)
class SELFDRIVINGCAR_API ASelfDrivingCarSportsCar : public ASelfDrivingCarPawn
{
	GENERATED_BODY()
	
public:

	ASelfDrivingCarSportsCar();
};
