// Copyright Epic Games, Inc. All Rights Reserved.

#include "SelfDrivingCarGameMode.h"
#include "SelfDrivingCarPlayerController.h"

ASelfDrivingCarGameMode::ASelfDrivingCarGameMode()
{
	PlayerControllerClass = ASelfDrivingCarPlayerController::StaticClass();
}
