// Copyright Epic Games, Inc. All Rights Reserved.

#include "SelfDrivingCarWheelRear.h"
#include "UObject/ConstructorHelpers.h"

USelfDrivingCarWheelRear::USelfDrivingCarWheelRear()
{
	AxleType = EAxleType::Rear;
	bAffectedByHandbrake = true;
	bAffectedByEngine = true;
}