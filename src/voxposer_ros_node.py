#!/usr/bin/env python

from voxposer_ros.srv import NLManipulation, NLManipulationRequest, NLManipulationResponse
import rospy
from envs.ros_env import VoxPoserROS
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import numpy as np
import rospkg

def handle_nl_manipulation(req: NLManipulationRequest):
    instruction = req.instruction
    env.reset_to_default_pose()
    env.init_object_names(object_names=req.object_names)
    set_lmp_objects(lmps, env.get_object_names())
    voxposer_ui(instruction)
    return NLManipulationResponse()

if __name__ == '__main__':
    rospy.init_node("voxposer_ros", anonymous=True)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("voxposer_ros")
    voxposer_config = get_config(config_path=f"{pkg_path}/src/configs/ros_config.yaml")
    visualizer = ValueMapVisualizer(voxposer_config["visualizer"])
    env = VoxPoserROS(
        arm_move_group=rospy.get_param("~arm_move_group", default="arm"),
        gripper_move_group=rospy.get_param("~gripper_move_group", default="gripper"),
        gripper_open_pose=rospy.get_param("~gripper_open_pose", default="Open"),
        gripper_closed_pose=rospy.get_param("~gripper_closed_pose", default="Closed"),
        ns=rospy.get_param("~move_group_ns", default="/doris_arm"),
        robot_description=rospy.get_param("~robot_description", default="/doris_arm/robot_description"),
        arm_pose_reference_frame=rospy.get_param("~arm_group_reference_frame", default="doris_arm/base_link"),
        visualizer=visualizer,
        workspace_bounds_min=rospy.get_param("~workspace_bounds_min", default=[-1.0, -1.0, -1.0]),
        workspace_bounds_max=rospy.get_param("~workspace_bounds_max", default=[1.0, 1.0, 1.0]),
        recognitions_topic=rospy.get_param("~recognitions_topic", default="/butia_vision/br/object_recognition3d"),
        full_scene_cloud_topic=rospy.get_param("~full_scene_cloud_topic", default="/camera/depth/points"),
        arm_home_pose=rospy.get_param("~arm_home_pose", default="Home")
    )
    lmps, lmp_env = setup_LMP(env, voxposer_config)
    voxposer_ui = lmps["plan_ui"]
    nl_manipulation_service = rospy.Service("/voxposer_ros/nl_manipulation_service", NLManipulation, handler=handle_nl_manipulation)
    rospy.spin()