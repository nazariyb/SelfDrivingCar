// Copyright Epic Games, Inc. All Rights Reserved.

#include "SelfDrivingCarWheelFront.h"
#include "UObject/ConstructorHelpers.h"

USelfDrivingCarWheelFront::USelfDrivingCarWheelFront()
{
	AxleType = EAxleType::Front;
	bAffectedBySteering = true;
	MaxSteerAngle = 40.f;
}