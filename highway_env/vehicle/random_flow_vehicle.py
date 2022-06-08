from typing import List, Tuple, Union, Optional

import random
import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


class RandomFlowVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        self.counter = 0
  
        #self.internal_episode_length = steps * frames, where frames = (sim_freq / policy_freq)    # determines after how many time steps speed and steering is changed
        self.internal_episode_length = 30 * 50   # same action is taken for 15 steps (number of frames=15; 2nd parameter)
        #so if we take steps to be larger than  steps in the interaction loop, 
        #speed and angle won't change in one episode
        #self.internal_episode_length = 15
        self.steer_angle = 0 
        self.heading_ref = 0

    @classmethod
    def create_from(cls, vehicle: "RandomFlowVehicle") -> "RandomFlowVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        if self.counter % self.internal_episode_length == 0:
            self.target_speed = random.uniform(10, 40)  # 23 is initial speed and 40 is max speed
            #self.target_speed = 3
        action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.speed_control(self.target_speed)}
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        self.counter += 1
        super().act(action)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
       
        if self.counter % self.internal_episode_length == 0:
            self.heading_ref = random.uniform(-10,10)

            # Heading control
            heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(self.heading_ref - self.heading)
           
            # Heading rate to steering angle
            slip_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command, -1, 1))

            steering_angle = np.arctan(2 * np.tan(slip_angle))
            steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
            self.steer_angle = steering_angle
            #self.steer_angle = random.uniform(-self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
          
        else:
            if random.random() >= 0.7:
                if self.steer_angle >= 0:
                    if random.random() >= 0.7:
                        self.steer_angle =  random.uniform(0,1)
                    else:
                        self.steer_angle =  random.uniform(-1,0)
                else:
                    if random.random() >= 0.7:
                        self.steer_angle =  random.uniform(-1,0)
                    else:
                        self.steer_angle =  random.uniform(0,1)
            else:
                self.steer_angle = 0
        #steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(self.steer_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                     for t in times]))