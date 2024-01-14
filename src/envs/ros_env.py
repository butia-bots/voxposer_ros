import os
import numpy as np
import open3d as o3d
import json
import moveit_commander
from utils import normalize_vector, bcolors
import rospy
from butia_vision_msgs.msg import Recognitions3D, Description3D
import tf
import base64
from io import BytesIO
from PIL import Image as PILImage
from ros_numpy import numpify
from ros_numpy.point_cloud2 import split_rgb_field
from threading import Event
import tf2_sensor_msgs
import tf2_msgs
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_matrix, quaternion_from_euler
from copy import deepcopy
from sensor_msgs.msg import PointCloud2
import openai
import re
from Levenshtein import distance
import weaviate
import weaviate.classes as wvc


def toBase64(img: PILImage.Image):
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())#.decode('utf-8')
  return img_str

class VoxPoserROS():
    def __init__(self, arm_move_group="arm", gripper_move_group="gripper", gripper_open_pose="Open", gripper_closed_pose="Closed", ns="/doris_arm", robot_description="/doris_arm/robot_description", arm_pose_reference_frame="doris_arm/base_link", visualizer=None, workspace_bounds_min=[-1.0 ,-1.0, -1.0], workspace_bounds_max=[1.0, 1.0, 1.0], recognitions_topic="/butia_vision/br/object_recognition3d", full_scene_cloud_topic="/camera/depth/points", arm_home_pose="Home"):
        """
        Initializes the VoxPoserROS environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        self.arm = moveit_commander.MoveGroupCommander(name=arm_move_group, ns=ns, robot_description=robot_description, wait_for_servers=30.0)
        self.gripper = moveit_commander.MoveGroupCommander(name=gripper_move_group, ns=ns, robot_description=robot_description, wait_for_servers=30.0)
        self.arm.set_pose_reference_frame(arm_pose_reference_frame)
        self.gripper_open_pose = gripper_open_pose
        self.gripper_closed_pose = gripper_closed_pose
        self.arm_home_pose = arm_home_pose
        self._last_action = None

        self.workspace_bounds_min = np.array(workspace_bounds_min)
        self.workspace_bounds_max = np.array(workspace_bounds_max)
        #self.arm.set_workspace(np.concatenate([self.workspace_bounds_min, self.workspace_bounds_max]).tolist())
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        
        self.openai_connection = openai.OpenAI()
        self._object_names = []

        self.vector_client = weaviate.connect_to_local()

        self.tfl = tf.TransformListener()
        self.vision_subscriber = rospy.Subscriber(recognitions_topic, Recognitions3D, self._update_recognitions)

        self.full_cloud_subscriber = rospy.Subscriber(full_scene_cloud_topic, PointCloud2, callback=self._update_full_scene_cloud)

    def _update_recognitions(self, msg):
        self._recognitions: Recognitions3D = msg

    def _update_full_scene_cloud(self, msg):
        self._full_scene_cloud = msg

    def get_vision_collection(self):
        recognitions = deepcopy(self._recognitions)
        if self.vector_client.collections.exists("VisionObjects"):
            self.vector_client.collections.delete("VisionObjects")
        vision_collection = self.vector_client.collections.create(
            name="VisionObjects",
            vectorizer_config=wvc.Configure.Vectorizer.multi2vec_clip(image_fields=["image"])
        )
        for i, description in enumerate(recognitions.descriptions):
            if description.bbox.size.x > 0 and description.bbox.center.position.z < 1.5:
                vision_collection.data.insert(
                    properties={
                        "image": toBase64(PILImage.fromarray(numpify(recognitions.image_rgb)[int(description.bbox2D.center.y-description.bbox2D.size_y//2):int(description.bbox2D.center.y+description.bbox2D.size_y//2),int(description.bbox2D.center.x-description.bbox2D.size_x//2):int(description.bbox2D.center.x+description.bbox2D.size_x//2)][:,:,::-1])).decode(),
                        "descriptionId": i
                    },
                )
        return vision_collection, recognitions

    def init_object_names(self, object_names):
        self._object_names = object_names
    
    def init_lmm(self, instruction):
        recognitions: Recognitions3D = deepcopy(self._recognitions)
        set_of_marks = toBase64(PILImage.fromarray(numpify(recognitions.image_set_of_marks)))
        completion = self.openai_connection.chat.completions.create(messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""I have marked the objects in this image with bounding boxes containing a numerical id for each object instance in it, labeled on the top of the bounding box as `id:$id object $confidence`.
Your task is to parse the image into a structured JSON containing a more specific class name for each object, as well as it's numerical $id.
The object list will be used by a robotic arm to perform the following task: {instruction}
""" +
"""DO IGNORE the `object` label in the bounding box, and provide a new class for the object instead.
DO IGNORE shadows and other visual artifacts.
Provide class names that are relevant to the task to be executed by the robotic arm, but, if an object is not relevant to the task, do not include it in the list.
Think step-by-step and provide your answer in the following format:

Thought: ...
Object List:
```json
{
  "objects": [
    {
      "object_class_name": "...",
      "object_id": $id
    },
    ...
  ]
}
```
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{set_of_marks}"
                    }
                ]
            }
        ], model="gpt-4-vision-preview" , temperature=0, max_tokens=512)
        response_text = completion.choices[0].message.content
        pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)
        found = pattern.search(response_text)
        scene_description = found.group(1)
        scene_description = json.loads(scene_description.strip())
        class_names = list(set([obj["object_class_name"] for obj in scene_description["objects"]]))
        ids = dict([(class_name, 0) for class_name in class_names])
        name2id = {}
        for obj in scene_description["objects"]:
            idx = ids[obj["object_class_name"]]
            name = f"{obj['object_class_name']}_{idx}"
            global_id = int(obj["object_id"])
            name2id[name] = global_id
            ids[obj["object_class_name"]] += 1
        self.name2id = name2id
        self.id2name = dict([(v,k) for k,v in self.name2id.items()])
        previous_descriptions = {}
        for description in recognitions.descriptions:
            if description.global_id in self.id2name:
                previous_descriptions[description.global_id] = description
        self.previous_descriptions = previous_descriptions

    
    
    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        #return list(self.name2id.keys())
        return self._object_names

    def load_task(self, task):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.

        Args:
            task (str or rlbench.tasks.Task): Name of the task class or a task object.
        """
        raise NotImplementedError()

    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        """object_names = self.get_object_names()
        object_names.sort(key=lambda e: distance(query_name, e))
        query_name = object_names[0]"""
        description: Description3D = None
        #recognitions = deepcopy(self._recognitions)
        """for current_description in recognitions.descriptions:
            idx = self.name2id[query_name]
            if current_description.global_id == idx:
                description = current_description
            self.previous_descriptions[current_description.global_id] = current_description
        if description == None:
            description = self.previous_descriptions[self.name2id[query_name]]"""
        vision_collection, recognitions = self.get_vision_collection()
        query_return = vision_collection.query.near_text(query=query_name, limit=1)
        description_id = query_return.objects[0].properties['descriptionId']
        description = recognitions.descriptions[int(description_id)]
        assert len(description.filtered_cloud.data) > 0
        self.tfl.waitForTransform(self.arm.get_pose_reference_frame(), description.header.frame_id, rospy.Time(), rospy.Duration(10.0))
        position, orientation = self.tfl.lookupTransform(self.arm.get_pose_reference_frame(), description.header.frame_id, rospy.Time())
        transform = TransformStamped()
        transform.header.frame_id = self.arm.get_pose_reference_frame()
        transform.child_frame_id = description.header.frame_id
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        point_cloud = tf2_sensor_msgs.do_transform_cloud(description.filtered_cloud, transform)
        points = numpify(point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(list(zip(points['x'], points['y'], points['z']))))
        pcd.remove_non_finite_points()
        pcd.estimate_normals()
        rotation_matrix = quaternion_matrix(orientation)
        camera_extrinsics = rotation_matrix
        forward_vector = np.asarray([0.0, 0.0, 1.0])
        lookat_vector = camera_extrinsics[:3,:3] @ forward_vector
        lookat_vector = normalize_vector(lookat_vector)
        cam_normals = np.asarray(pcd.normals)
        flip_indices = np.dot(cam_normals, lookat_vector) > 0
        cam_normals[flip_indices] *= -1
        pcd.normals = o3d.utility.Vector3dVector(cam_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        return obj_points, obj_normals

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        full_scene_cloud: PointCloud2 = deepcopy(self._full_scene_cloud)
        self.tfl.waitForTransform(self.arm.get_pose_reference_frame(), full_scene_cloud.header.frame_id, rospy.Time(), rospy.Duration(10.0))
        position, orientation = self.tfl.lookupTransform(self.arm.get_pose_reference_frame(), full_scene_cloud.header.frame_id, rospy.Time())
        transform = TransformStamped()
        transform.header.frame_id = self.arm.get_pose_reference_frame()
        transform.child_frame_id = full_scene_cloud.header.frame_id
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        point_cloud = tf2_sensor_msgs.do_transform_cloud(full_scene_cloud, transform)
        points = numpify(point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(list(zip(points['x'], points['y'], points['z']))))
        rgb_arr = split_rgb_field(points)
        pcd.colors = o3d.utility.Vector3dVector(np.array(list(zip(rgb_arr['r'], rgb_arr['g'], rgb_arr['b']))))
        pcd.remove_non_finite_points()
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
        points = np.asarray(pcd_downsampled.points)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        raise NotImplementedError()

    def apply_action(self, action):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """

        action = self._process_action(action)
        pose = action[:7]
        gripper_cmd = action[-1]
        self.arm.set_pose_target(pose=pose.tolist())
        self.arm.go(wait=True)
        if gripper_cmd > 0.0:
            self.gripper.set_named_target(self.gripper_open_pose)
        else:
            self.gripper.set_named_target(self.gripper_closed_pose)
        self.gripper.go(wait=True)
        self._last_action = action

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        self.apply_action(np.concatenate([pose, [self.get_last_gripper_action() if isinstance (self._last_action, np.ndarray) else 0.0]]))
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        self.apply_action(np.concatenate([self.get_ee_pose(), [1.0]]))

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        self.apply_action(np.concatenate([self.get_ee_pose(), [0.0]]))

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        raise NotImplementedError()

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        self.arm.set_named_target(self.arm_home_pose)
        self.arm.go(wait=True)
        #joint_values = self.arm.get_current_joint_values()
        #joint_values[4] = np.pi/2
        #self.arm.set_joint_value_target(joint_values)
        #self.arm.go(wait=True)
        self.open_gripper()

    def get_ee_pose(self):
        pose = self.arm.get_current_pose()
        return np.asarray([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
        ])

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        return self._last_action[-1] if isinstance(self._last_action, np.ndarray) else 1.0

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        raise NotImplementedError()

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
            self.visualizer.update_scene_points(points, colors)
    
    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        raise NotImplementedError()

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        quat_wxyz = action[3:7]
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        action[3:7] = quat_xyzw
        return action