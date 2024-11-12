"""
Visualize and manually control the robot in an environment, useful for debugging purposes.

Controls:
xyz: "i": +x, "k": -x, "j": +y, "l": -y, "u": +z, "o": -z
rotation rpy: "1": +r, "2": -r, "3": +p, "4": -p, "5": +y, "6": -y
reset environment: "r"
gripper open: "f", gripper close: "g"

If --enable-sapien-viewer, press "0" to switch to sapien viewer; 
    In the viewer, press "coordinate axes" option on the left twice to activate it;
    You can click on an object / articulation link and press "f" to focus on it; 
    Use right mouse button to rotate; middle-mouth-button + shift to translate;
    Under "scene hierarchy" on the bottom left, you can select different actors and articulation links;
    When an articulated object is selected (e.g., robot / cabinet), then under "articulation" on the bottom right, you can move the scrollbar to change each of its joint positions / angles; 
    Press "pause" on the top left to pause the simulation;
    Press "g" to grab object; "g" + "x"/"y"/"z" to move object along x/y/z axis;
    
If rgb_overlay_path is given, Press "v" to visualize the "greenscreened" image overlayed on top of simulation observation;
    this visualization can be used to debug e.g., the alignment of the real table and the simulation proxy table

To debug an environment, you can modify the "env_reset_options" in the main function to change the initial state of the environment.


**Example Scripts:**

cd {this_repo}/ManiSkill2_real2sim

# Prepackaged Envs
# Hint: "prepackaged_config @True" means using the default visual matching environment configs;
#       Press "v" to visualize policy observation under visual matching

python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e GraspSingleOpenedCokeCanInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner -o rgbd \
    --enable-sapien-viewer     prepackaged_config @True     robot google_robot_static
# replace "GraspSingleOpenedCokeCanInScene-v0" with "MoveNearGoogleBakedTexInScene-v1", "OpenDrawerCustomInScene-v0", "CloseDrawerCustomInScene-v0" to test other envs
# for "MoveNearGoogleBakedTexInScene-v1", we also support "-v0". "-v0" uses assets with textures baked onto a single view of each object, 
# whereas "-v1" applies baked textures to all views of objects using Zero123++.

python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e PlaceIntoClosedDrawerCustomInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner -o rgbd \
    --enable-sapien-viewer     prepackaged_config @True     robot google_robot_static   model_ids baked_apple_v2

python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e PutCarrotOnPlateInScene-v0 --enable-sapien-viewer \
    -c arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos -o rgbd --enable-sapien-viewer     prepackaged_config @True     robot widowx
# replace "PutCarrotOnPlateInScene-v0" with "PutSpoonOnTableClothInScene-v0", "StackGreenCubeOnYellowCubeBakedTexInScene-v0", 
#         "PutEggplantInBasketScene-v0" to test other Bridge environments

# Envs constructed through manual config setup
# "rgb_overlay_mode debug" means to visualize 0.5*real image + 0.5*sim image, helpful for examining the alignment of the real table and the simulation proxy table

python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e GraspSingleOpenedCokeCanInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner -o rgbd --enable-sapien-viewer \
    robot google_robot_static sim_freq @501 control_freq @3 scene_name google_pick_coke_can_1_v4 \
    rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/google_coke_can_real_eval_1.png rgb_overlay_cameras overhead_camera
    
python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e MoveNearGoogleInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner --enable-sapien-viewer \
    -o rgbd robot google_robot_static sim_freq @501 control_freq @3 scene_name google_pick_coke_can_1_v4 \
    rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/google_move_near_real_eval_1.png rgb_overlay_cameras overhead_camera
    
python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e MoveNearGoogleBakedTexInScene-v1 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner --enable-sapien-viewer \
    -o rgbd robot google_robot_static sim_freq @501 control_freq @3 scene_name google_pick_coke_can_1_v4 \
    rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/google_move_near_real_eval_1.png rgb_overlay_cameras overhead_camera urdf_version recolor_tabletop_visual_matching_2
    
python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e OpenDrawerCustomInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner --enable-sapien-viewer \
    -o rgbd robot google_robot_static sim_freq @501 control_freq @3 scene_name frl_apartment_stage_simple \
    rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/open_drawer_b0.png rgb_overlay_cameras overhead_camera
    
python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e PlaceIntoClosedDrawerCustomInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner --enable-sapien-viewer \
    -o rgbd robot google_robot_static sim_freq @501 control_freq @3 scene_name frl_apartment_stage_simple \
    rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/open_drawer_b0.png rgb_overlay_cameras overhead_camera model_ids baked_apple_v2
    
python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e PutCarrotOnPlateInScene-v0 --enable-sapien-viewer \
    -c arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos -o rgbd robot widowx sim_freq @500 control_freq @5 \
    scene_name bridge_table_1_v1  rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/bridge_real_eval_1.png rgb_overlay_cameras 3rd_view_camera
# replace "PutCarrotOnPlateInScene-v0" with "PutSpoonOnTableClothInScene-v0", "StackGreenCubeOnYellowCubeInScene-v0" to test other Bridge environments
    
python mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py -e PutEggplantInBasketScene-v0 --enable-sapien-viewer \
    -c arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos -o rgbd robot widowx_sink_camera_setup sim_freq @500 control_freq @5 \
    scene_name bridge_table_1_v2  rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/bridge_sink.png rgb_overlay_cameras 3rd_view_camera

"""

import argparse

import gymnasium as gym
import numpy as np
from transforms3d.quaternions import qmult

from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2_real2sim.utils.sapien_utils import look_at, normalize_vector
from sapien.core import Pose

MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--add-segmentation", action="store_true")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if args.env_id in MS1_ENV_IDS:
        if args.control_mode is not None and not args.control_mode.startswith("base"):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

    if "robot" in args.env_kwargs:
        if "google_robot" in args.env_kwargs["robot"]:
            pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
            args.env_kwargs["render_camera_cfgs"] = {
                "render_camera": dict(p=pose.p, q=pose.q)
            }
        elif "widowx" in args.env_kwargs["robot"]:
            pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
            args.env_kwargs["render_camera_cfgs"] = {
                "render_camera": dict(p=pose.p, q=pose.q)
            }

    from transforms3d.euler import euler2quat

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        camera_cfgs={"add_segmentation": args.add_segmentation},
        **args.env_kwargs
    )

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    env_reset_options = {}
    if (not hasattr(env, "prepackaged_config")) or (not env.prepackaged_config):
        """
        Change the following reset options as you want to debug the environment
        """
        names_in_env_id_fxn = lambda name_list: any(
            name in args.env_id for name in name_list
        )
        if names_in_env_id_fxn(["GraspSingle"]):
            init_rot_quat = (Pose(q=[0, 0, 0, 1])).q
            env_reset_options = {
                "obj_init_options": {"init_xy": [-0.12, 0.2]},
                "robot_init_options": {
                    "init_xy": [0.35, 0.20],
                    "init_rot_quat": init_rot_quat,
                },
            }
        elif names_in_env_id_fxn(["MoveNear"]):
            # data/real_inpainting/google_move_near_real_eval_1.png
            init_rot_quat = (Pose(q=euler2quat(0, 0, -0.09)) * Pose(q=[0, 0, 0, 1])).q
            env_reset_options = {
                "obj_init_options": {},
                "robot_init_options": {
                    "init_xy": [0.35, 0.21],
                    "init_rot_quat": init_rot_quat,
                },
            }
            # data/real_inpainting/google_move_near_real_eval_2.png
            # init_rot_quat = (Pose(q=euler2quat(0, 0, -0.028)) * Pose(q=[0, 0, 0, 1])).q
            # env_reset_options={'obj_init_options': {},
            #                    'robot_init_options': {'init_xy': [0.36, 0.22], 'init_rot_quat': init_rot_quat}}

            env_reset_options["obj_init_options"]["episode_id"] = 0
        elif names_in_env_id_fxn(["Drawer"]):
            init_rot_quat = [0, 0, 0, 1]
            if not names_in_env_id_fxn(["PlaceInClosedDrawer"]):
                init_xy = [0.851, 0.035]
            else:
                init_xy = [0.652, 0.009]
            env_reset_options = {
                "obj_init_options": {"init_xy": [0.0, 0.0]},
                "robot_init_options": {
                    "init_xy": init_xy,
                    "init_rot_quat": init_rot_quat,
                },
            }
        elif names_in_env_id_fxn(
            ["PutSpoonOnTableCloth", "PutCarrotOnPlate", "StackGreenCubeOnYellowCube", "PutEggplantInBasket"]
        ):
            init_rot_quat = Pose(q=[0, 0, 0, 1]).q
            # env_reset_options={'obj_init_options': {},
            #                    'robot_init_options': {'init_xy': [0.147, 0.028], 'init_rot_quat': init_rot_quat}}
            # 'robot_init_options': {'init_xy': [0.147, 0.028], 'init_height': 0.860, 'init_rot_quat': init_rot_quat}}
            # init_rot_quat = (Pose(q=euler2quat(0, 0, 0.03)) * Pose(q=[0, 0, 0, 1])).q
            if env.robot_uid == "widowx":
                env_reset_options = {
                    "obj_init_options": {},
                    "robot_init_options": {
                        "init_xy": [0.147, 0.028],
                        "init_rot_quat": init_rot_quat,
                    },
                }
            elif env.robot_uid == "widowx_camera_setup2":
                env_reset_options = {
                    "obj_init_options": {},
                    "robot_init_options": {
                        "init_xy": [0.147, 0.070],
                        "init_rot_quat": init_rot_quat,
                    },
                }
            elif env.robot_uid == "widowx_sink_camera_setup":
                env_reset_options = {
                    "obj_init_options": {},
                    "robot_init_options": {
                        "init_xy": [0.127, 0.060],
                        "init_rot_quat": init_rot_quat,
                    },
                }
            env_reset_options["obj_init_options"]["episode_id"] = 0

    obs, info = env.reset(options=env_reset_options)
    print("Reset info:", info)
    print("Instruction:", env.get_language_instruction())
    after_reset = True

    if "google_robot" in env.agent.robot.name:
        print(
            "overhead camera pose",
            env.unwrapped._cameras["overhead_camera"].camera.pose,
        )
        print(
            "overhead camera pose wrt robot base",
            env.agent.robot.pose.inv()
            * env.unwrapped._cameras["overhead_camera"].camera.pose,
        )
    elif "wx250s" in env.agent.robot.name:
        print(
            "3rd view camera pose",
            env.unwrapped._cameras["3rd_view_camera"].camera.pose,
        )
        print(
            "3rd view camera pose wrt robot base",
            env.agent.robot.pose.inv()
            * env.unwrapped._cameras["3rd_view_camera"].camera.pose,
        )
    print("robot pose", env.agent.robot.pose)
    # env.obj.get_collision_shapes()[0].get_physical_material().static_friction / dynamic_friction / restitution # object material properties

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.render_human()
            sapien_viewer = env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    is_google_robot = "google_robot" in env.agent.robot.name
    is_widowx = "wx250s" in env.agent.robot.name
    is_gripper_delta_target_control = (
        env.agent.controller.controllers["gripper"].config.use_target
        and env.agent.controller.controllers["gripper"].config.use_delta
    )

    def get_reset_gripper_action():
        # open gripper at initialization
        if not is_google_robot:
            return 1
        else:
            # for google robot, open-and-close actions are reversed
            return -1

    gripper_action = get_reset_gripper_action()

    EE_ACTION = (
        0.1 if not (is_google_robot or is_widowx) else 0.02
    )  # google robot and widowx use unnormalized action space
    EE_ROT_ACTION = (
        1.0 if not (is_google_robot or is_widowx) else 0.1
    )  # google robot and widowx use unnormalized action space

    # print("obj pose", env.obj.pose, "tcp pose", env.tcp.pose)
    print("qpos", env.agent.robot.get_qpos())

    while True:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.render_human()

        render_frame = env.render()

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer(exit_on_esc=False)

        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        key = opencv_viewer.imshow(render_frame)

        if has_base:
            base_action = np.zeros([4])  # hardcoded
        else:
            base_action = np.zeros([0])

        # Parse end-effector action
        if (
            "pd_ee_delta_pose" in args.control_mode
            or "pd_ee_target_delta_pose" in args.control_mode
        ):
            ee_action = np.zeros([6])
        elif (
            "pd_ee_delta_pos" in args.control_mode
            or "pd_ee_target_delta_pos" in args.control_mode
        ):
            ee_action = np.zeros([3])
        else:
            raise NotImplementedError(args.control_mode)

        # Base
        if has_base:
            if key == "w":  # forward
                base_action[0] = 1
            elif key == "s":  # backward
                base_action[0] = -1
            elif key == "a":  # left
                base_action[1] = 1
            elif key == "d":  # right
                base_action[1] = -1
            elif key == "q" and len(base_action > 2):  # rotate counter
                base_action[2] = 1
            elif key == "e" and len(base_action > 2):  # rotate clockwise
                base_action[2] = -1
            elif key == "z" and len(base_action > 2):  # lift
                base_action[3] = 1
            elif key == "x" and len(base_action > 2):  # lower
                base_action[3] = -1

        # End-effector
        if num_arms > 0:
            # Position
            if key == "i":  # +x
                ee_action[0] = EE_ACTION
            elif key == "k":  # -x
                ee_action[0] = -EE_ACTION
            elif key == "j":  # +y
                ee_action[1] = EE_ACTION
            elif key == "l":  # -y
                ee_action[1] = -EE_ACTION
            elif key == "u":  # +z
                ee_action[2] = EE_ACTION
            elif key == "o":  # -z
                ee_action[2] = -EE_ACTION

            # Rotation (axis-angle)
            if key == "1":
                ee_action[3:6] = (EE_ROT_ACTION, 0, 0)
            elif key == "2":
                ee_action[3:6] = (-EE_ROT_ACTION, 0, 0)
            elif key == "3":
                ee_action[3:6] = (0, EE_ROT_ACTION, 0)
            elif key == "4":
                ee_action[3:6] = (0, -EE_ROT_ACTION, 0)
            elif key == "5":
                ee_action[3:6] = (0, 0, EE_ROT_ACTION)
            elif key == "6":
                ee_action[3:6] = (0, 0, -EE_ROT_ACTION)

        # Gripper
        if has_gripper:
            if not is_google_robot:
                if key == "f":  # open gripper
                    gripper_action = 1
                elif key == "g":  # close gripper
                    gripper_action = -1
            else:
                if key == "f":  # open gripper
                    gripper_action = -1
                elif key == "g":  # close gripper
                    gripper_action = 1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            obs, info = env.reset(options=env_reset_options)
            print("Reset info:", info)
            print("Instruction:", env.get_language_instruction())
            gripper_action = get_reset_gripper_action()
            after_reset = True
            continue
        elif key == None:  # exit
            break

        # Visualize observation
        if key == "v":
            if "rgbd" in env.obs_mode:
                from itertools import chain

                from mani_skill2_real2sim.utils.visualization.misc import (
                    observations_to_images,
                    tile_images,
                )

                images = list(
                    chain(*[observations_to_images(x) for x in obs["image"].values()])
                )
                render_frame = tile_images(images)
                opencv_viewer.imshow(render_frame)
            elif "pointcloud" in env.obs_mode:
                import trimesh

                xyzw = obs["pointcloud"]["xyzw"]
                mask = xyzw[..., 3] > 0
                rgb = obs["pointcloud"]["rgb"]
                if "robot_seg" in obs["pointcloud"]:
                    robot_seg = obs["pointcloud"]["robot_seg"]
                    rgb = np.uint8(robot_seg * [11, 61, 127])
                trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        if args.env_id in MS1_ENV_IDS:
            if is_google_robot:
                raise NotImplementedError()
            action_dict = dict(
                base=base_action,
                right_arm=ee_action,
                right_gripper=gripper_action,
                left_arm=np.zeros_like(ee_action),
                left_gripper=np.zeros_like(gripper_action),
            )
            action = env.agent.controller.from_action_dict(action_dict)
        else:
            action_dict = dict(base=base_action, arm=ee_action)
            if has_gripper:
                action_dict["gripper"] = gripper_action
            action = env.agent.controller.from_action_dict(action_dict)

        print("action", action)
        obs, reward, terminated, truncated, info = env.step(action)

        if is_gripper_delta_target_control:
            gripper_action = 0

        # print("obj pose", env.obj.pose, "tcp pose", env.tcp.pose)
        print("tcp pose wrt robot base", env.agent.robot.pose.inv() * env.tcp.pose)
        print("qpos", env.agent.robot.get_qpos())
        print("reward", reward)
        print("terminated", terminated, "truncated", truncated)
        print("info", info)

    env.close()


if __name__ == "__main__":
    main()
