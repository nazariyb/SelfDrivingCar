// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class SelfDrivingCar : ModuleRules
{
	public SelfDrivingCar(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "EnhancedInput", "ChaosVehicles", "PhysicsCore" });
	}
}