from copy import deepcopy
import numpy as np

from mani_skill2_real2sim.agents.controllers import *
from mani_skill2_real2sim.sensors.camera import CameraConfig
from mani_skill2_real2sim.utils.sapien_utils import look_at


class WidowXDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = (
            "{PACKAGE_ASSET_DIR}/descriptions/widowx_description/wx250s.urdf"
        )

        finger_min_patch_radius = 0.01  # used to calculate torsional friction
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                left_finger_link=dict(
                    material="gripper",
                    patch_radius=finger_min_patch_radius,
                    min_patch_radius=finger_min_patch_radius,
                ),
                right_finger_link=dict(
                    material="gripper",
                    patch_radius=finger_min_patch_radius,
                    min_patch_radius=finger_min_patch_radius,
                ),
            ),
        )

        self.arm_joint_names = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]
        self.gripper_joint_names = ["left_finger", "right_finger"]

        # # arm_pd_ee_delta_pose_align_interpolate_gripper_pd_joint_pos, 3hz
        # self.arm_stiffness = [1193.2765654645982, 800.0, 784.3309604605763, 1250.3737197881153, 1392.0546244178072, 1038.3774360126893]
        # self.arm_damping = [75.5250991585983, 20.0, 23.646570105574618, 23.825760721440837, 67.97737990215525, 78.14407359073823]
        # # arm_pd_ee_delta_pose_align_gripper_pd_joint_pos, 3hz
        # self.arm_stiffness = [1214.6340906847158, 804.5146660467828, 801.9841311029891, 1110.0, 1310.0, 988.4499396558518]
        # self.arm_damping = [175.18652498291488, 73.04563998424553, 62.47429885911165, 104.4069151351231, 108.3230540691408, 136.87526713617873]
        # # arm_pd_ee_target_delta_pose_align_gripper_pd_joint_pos, 5hz non-blocking
        # self.arm_stiffness = [1215.4327150032293, 730.0, 860.0, 1133.9675494142102, 1413.3815895525422, 930.0]
        # self.arm_damping = [285.5831564748846, 118.83365148810542, 126.05256283235089, 142.0533158479584, 142.85223328752122, 96.00503592486184]

        # arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos, 5hz non-blocking
        self.arm_stiffness = [
            1169.7891719504198,
            730.0,
            808.4601346394447,
            1229.1299089624076,
            1272.2760456418862,
            1056.3326605132252,
        ]
        self.arm_damping = [
            330.0,
            180.0,
            152.12036565582588,
            309.6215302722146,
            201.04998711007383,
            269.51458932695414,
        ]

        self.arm_force_limit = [200, 200, 100, 100, 100, 100]
        self.arm_friction = 0.0
        self.arm_vel_limit = 1.5
        self.arm_acc_limit = 2.0

        self.gripper_stiffness = 1000
        self.gripper_damping = 200
        self.gripper_pid_stiffness = 1000
        self.gripper_pid_damping = 200
        self.gripper_pid_integral = 300
        self.gripper_force_limit = 60
        self.gripper_vel_limit = 0.12
        self.gripper_acc_limit = 0.50
        self.gripper_jerk_limit = 5.0

        self.ee_link_name = "ee_gripper_link"

    @property
    def controllers(self):
        _C = {}

        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_common_args = [
            self.arm_joint_names,
            -1.0,  # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
        ]
        arm_common_kwargs = dict(
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            normalize_action=False,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee", **arm_common_kwargs
        )
        arm_pd_ee_delta_pose_align2 = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align2", **arm_common_kwargs
        )
        arm_pd_ee_delta_pose_align2_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align2",
            interpolate=True,
            interpolate_by_planner=True,
            interpolate_planner_vlim=self.arm_vel_limit,
            interpolate_planner_alim=self.arm_acc_limit,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee", use_target=True, **arm_common_kwargs
        )
        arm_pd_ee_target_delta_pose_align = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align", use_target=True, **arm_common_kwargs
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align2", use_target=True, **arm_common_kwargs
        )
        arm_pd_ee_target_delta_pose_align2_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align2",
            use_target=True,
            interpolate=True,
            delta_target_from_last_drive_target=True,
            interpolate_by_planner=True,
            interpolate_planner_vlim=self.arm_vel_limit,
            interpolate_planner_alim=self.arm_acc_limit,
            **arm_common_kwargs,
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_delta_pose_align2=arm_pd_ee_delta_pose_align2,
            arm_pd_ee_delta_pose_align2_interpolate_by_planner=arm_pd_ee_delta_pose_align2_interpolate_by_planner,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
            arm_pd_ee_target_delta_pose_align=arm_pd_ee_target_delta_pose_align,
            arm_pd_ee_target_delta_pose_align2=arm_pd_ee_target_delta_pose_align2,
            arm_pd_ee_target_delta_pose_align2_interpolate_by_planner=arm_pd_ee_target_delta_pose_align2_interpolate_by_planner,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        extra_gripper_clearance = (
            0.001
        )  # since real gripper is PID, we use extra clearance to mitigate PD small errors; also a trick to have force when grasping
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.015 - extra_gripper_clearance,
            0.037 + extra_gripper_clearance,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.015 - extra_gripper_clearance,
            0.037 + extra_gripper_clearance,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_target=True,
            clip_target=True,
            clip_target_thres=extra_gripper_clearance,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -(0.037 - 0.015) - extra_gripper_clearance,
            0.037 - 0.015 + extra_gripper_clearance,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_delta_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -(0.037 - 0.015) - extra_gripper_clearance,
            0.037 - 0.015 + extra_gripper_clearance,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=extra_gripper_clearance,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pid_joint_pos = PIDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.015,
            0.037,
            self.gripper_pid_stiffness,
            self.gripper_pid_damping,
            self.gripper_force_limit,
            integral=self.gripper_pid_integral,
            normalize_action=True,
            drive_mode="force",
        )
        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
            gripper_pd_joint_target_pos=gripper_pd_joint_target_pos,
            gripper_pd_joint_delta_pos=gripper_pd_joint_delta_pos,
            gripper_pd_joint_target_delta_pos=gripper_pd_joint_target_delta_pos,
            gripper_pid_joint_pos=gripper_pid_joint_pos,
        )

        controller_configs = {}
        for arm_controller_name in _C["arm"]:
            for gripper_controller_name in _C["gripper"]:
                c = {}
                c["arm"] = _C["arm"][arm_controller_name]
                c["gripper"] = _C["gripper"][gripper_controller_name]
                combined_name = arm_controller_name + "_" + gripper_controller_name
                controller_configs[combined_name] = c

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        # Table width: about 36cm

        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation
                p=[0.0, -0.16, 0.36],
                # this rotation allows simulation proxy table to align almost perfectly with real table for bridge_real_eval_1.png
                # when calling env.reset(options={'robot_init_options': {'init_xy': [0.147, 0.028], 'init_rot_quat': [0, 0, 0, 1]}})
                q=look_at([0, 0, 0], [1, 0.553, -1.085]).q,
                width=640,
                height=480,
                actor_uid="base_link",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
        ]
        
        
class WidowXBridgeDatasetCameraSetupConfig(WidowXDefaultConfig):
    @property
    def cameras(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used in the Bridge dataset
                p=[0.00, -0.16, 0.336],
                q=[0.909182, -0.0819809, 0.347277, 0.214629],
                width=640,
                height=480,
                actor_uid="base_link",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
        ]



class WidowXSinkCameraSetupConfig(WidowXDefaultConfig):
    @property
    def cameras(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation for the sink setup
                # p=[0.13, 0.27, 1.24],
                # q=look_at([0, 0, 0], [-1, -0.45, -1.05], [0, 0, 1]).q,
                # actor_uid=None,
                p=[-0.00300001, -0.21, 0.39],
                q=[-0.907313, 0.0782, -0.36434, -0.194741],
                actor_uid="base_link",
                width=640,
                height=480,
                fov=1.5,  # ignored if intrinsic is passed
                near=0.01,
                far=10,
                intrinsic = np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]])
            )
        ]