#include "Tags.h"


namespace Sd::Observations
{
	UE_DEFINE_GAMEPLAY_TAG(Spline, "Observations.Spline");
	UE_DEFINE_GAMEPLAY_TAG(Spline_Location, "Observations.Spline.Location");
	UE_DEFINE_GAMEPLAY_TAG(Spline_Direction, "Observations.Spline.Direction");

	UE_DEFINE_GAMEPLAY_TAG(Car, "Observations.Car");
	UE_DEFINE_GAMEPLAY_TAG(Car_Location, "Observations.Car.Location");
	UE_DEFINE_GAMEPLAY_TAG(Car_Velocity, "Observations.Car.Velocity");

	UE_DEFINE_GAMEPLAY_TAG(Obstacle, "Observations.Obstacle");
	UE_DEFINE_GAMEPLAY_TAG(Obstacle_Distance, "Observations.Obstacle.Distance");
	UE_DEFINE_GAMEPLAY_TAG(Obstacle_Angle, "Observations.Obstacle.Angle");
}

namespace Sd::Actions
{
	UE_DEFINE_GAMEPLAY_TAG(Steering, "Actions.Steering");
	UE_DEFINE_GAMEPLAY_TAG(ThrottleBreak, "Actions.ThrottleBreak");
}

namespace Sd::Rewards
{
	UE_DEFINE_GAMEPLAY_TAG(Movement, "Rewards.Movement");
	UE_DEFINE_GAMEPLAY_TAG(Movement_Delta, "Rewards.Movement.Delta");
	UE_DEFINE_GAMEPLAY_TAG(Movement_Velocity, "Rewards.Movement.Velocity");

	UE_DEFINE_GAMEPLAY_TAG(DistanceToFinish, "Rewards.DistanceToFinish");
	UE_DEFINE_GAMEPLAY_TAG(TimeToFinish, "Rewards.TimeToFinish");

	UE_DEFINE_GAMEPLAY_TAG(Obstacle, "Rewards.Obstacle");
	UE_DEFINE_GAMEPLAY_TAG(Obstacle_Distance, "Rewards.Obstacle.Distance");
	UE_DEFINE_GAMEPLAY_TAG(Obstacle_Angle, "Rewards.Obstacle.Angle");
}
