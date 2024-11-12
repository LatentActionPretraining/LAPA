from copy import deepcopy
import numpy as np

from mani_skill2_real2sim.agents.controllers import *
from mani_skill2_real2sim.sensors.camera import CameraConfig


class GoogleRobotDefaultConfig:
    def __init__(self, mobile_base=False, finger_friction=2.0, base_arm_drive_mode="force") -> None:
        if mobile_base:
            self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/googlerobot_description/google_robot_meta_sim_fix_fingertip.urdf"
        else:
            self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"

        finger_min_patch_radius = 0.1  # used to calculate torsional friction
        finger_nail_min_patch_radius = 0.01
        # standard urdf does not support <contact> tag, so we manually define friction here
        self.urdf_config = dict(
            _materials=dict(
                finger_mat=dict(
                    static_friction=finger_friction, dynamic_friction=finger_friction, restitution=0.0
                ),
                finger_tip_mat=dict(
                    static_friction=finger_friction, dynamic_friction=finger_friction, restitution=0.0
                ),
                finger_nail_mat=dict(
                    static_friction=0.1, dynamic_friction=0.1, restitution=0.0
                ),
                base_mat=dict(
                    static_friction=0.1, dynamic_friction=0.0, restitution=0.0
                ),
                wheel_mat=dict(
                    static_friction=1.0, dynamic_friction=0.0, restitution=0.0
                ),
            ),
            link=dict(
                link_base=dict(
                    material="base_mat", patch_radius=0.1, min_patch_radius=0.1
                ),
                link_wheel_left=dict(
                    material="wheel_mat", patch_radius=0.1, min_patch_radius=0.1
                ),
                link_wheel_right=dict(
                    material="wheel_mat", patch_radius=0.1, min_patch_radius=0.1
                ),
                link_finger_left=dict(
                    material="finger_mat",
                    patch_radius=finger_min_patch_radius,
                    min_patch_radius=finger_min_patch_radius,
                ),
                link_finger_right=dict(
                    material="finger_mat",
                    patch_radius=finger_min_patch_radius,
                    min_patch_radius=finger_min_patch_radius,
                ),
                link_finger_tip_left=dict(
                    material="finger_tip_mat",
                    patch_radius=finger_min_patch_radius,
                    min_patch_radius=finger_min_patch_radius,
                ),
                link_finger_tip_right=dict(
                    material="finger_tip_mat",
                    patch_radius=finger_min_patch_radius,
                    min_patch_radius=finger_min_patch_radius,
                ),
                link_finger_nail_left=dict(
                    material="finger_nail_mat",
                    patch_radius=finger_nail_min_patch_radius,
                    min_patch_radius=finger_nail_min_patch_radius,
                ),
                link_finger_nail_right=dict(
                    material="finger_nail_mat",
                    patch_radius=finger_nail_min_patch_radius,
                    min_patch_radius=finger_nail_min_patch_radius,
                ),
            ),
        )

        self.base_joint_names = ["joint_wheel_left", "joint_wheel_right"]
        self.base_damping = 1e3
        self.base_force_limit = 500
        self.mobile_base = mobile_base  # whether the robot base is mobile
        self.base_arm_drive_mode = base_arm_drive_mode  # 'force' or 'acceleration'

        self.arm_joint_names = [
            "joint_torso",
            "joint_shoulder",
            "joint_bicep",
            "joint_elbow",
            "joint_forearm",
            "joint_wrist",
            "joint_gripper",
            "joint_head_pan",
            "joint_head_tilt",
        ]
        self.gripper_joint_names = ["joint_finger_right", "joint_finger_left"]

        if self.base_arm_drive_mode == "acceleration":
            raise NotImplementedError(
                "PD parameters not yet tuned for acceleration drive mode."
            )
        elif self.base_arm_drive_mode == "force":
            # arm_pd_ee_delta_pose_align_interpolate_by_planner
            self.arm_stiffness = [
                1700.0,
                1737.0471680861954,
                979.975871856535,
                930.0,
                1212.154500274304,
                432.96500923932535,
                468.0013365498738,
                2000,
                2000,
            ]
            self.arm_damping = [
                1059.9791902443303,
                1010.4720585373592,
                767.2803161582076,
                680.0,
                674.9946964336588,
                274.613381336198,
                340.532560578637,
                900,
                900,
            ]
            self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
        else:
            raise NotImplementedError()
        self.arm_friction = 0.0
        self.arm_vel_limit = 1.5
        self.arm_acc_limit = 2.0
        self.arm_jerk_limit = 50.0

        self.gripper_stiffness = 200
        self.gripper_damping = 8
        self.gripper_force_limit = 60
        self.gripper_vel_limit = 1.0
        self.gripper_acc_limit = 7.0
        self.gripper_jerk_limit = 50.0

        self.ee_link_name = "link_gripper_tcp"  # end-effector link name

    @property
    def controllers(self):
        _C = {}

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        if self.mobile_base:
            _C["base"] = dict(
                # PD ego-centric joint velocity
                base_pd_joint_vel=PDBaseVelControllerConfig(
                    self.base_joint_names,
                    lower=[-0.5, -0.5],
                    upper=[0.5, 0.5],
                    damping=self.base_damping,
                    force_limit=self.base_force_limit,
                    drive_mode=self.base_arm_drive_mode,
                )
            )
        else:
            _C["base"] = [None]

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
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee", **arm_common_kwargs
        )
        arm_pd_ee_delta_pose_align = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align", **arm_common_kwargs
        )
        arm_pd_ee_delta_pose_align_interpolate = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align", interpolate=True, **arm_common_kwargs
        )
        arm_pd_ee_delta_pose_align_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            interpolate=True,
            interpolate_by_planner=True,
            interpolate_planner_vlim=self.arm_vel_limit,
            interpolate_planner_alim=self.arm_acc_limit,
            interpolate_planner_jerklim=self.arm_jerk_limit,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee", use_target=True, **arm_common_kwargs
        )
        arm_pd_ee_target_delta_pose_align = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align", use_target=True, **arm_common_kwargs
        )
        arm_pd_ee_target_delta_pose_align_interpolate = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            use_target=True,
            interpolate=True,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose_align_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            use_target=True,
            interpolate=True,
            delta_target_from_last_drive_target=True,
            interpolate_by_planner=True,
            interpolate_planner_vlim=self.arm_vel_limit,
            interpolate_planner_alim=self.arm_acc_limit,
            interpolate_planner_jerklim=self.arm_jerk_limit,
            **arm_common_kwargs,
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_delta_pose_align=arm_pd_ee_delta_pose_align,
            arm_pd_ee_delta_pose_align_interpolate=arm_pd_ee_delta_pose_align_interpolate,
            arm_pd_ee_delta_pose_align_interpolate_by_planner=arm_pd_ee_delta_pose_align_interpolate_by_planner,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
            arm_pd_ee_target_delta_pose_align=arm_pd_ee_target_delta_pose_align,
            arm_pd_ee_target_delta_pose_align_interpolate=arm_pd_ee_target_delta_pose_align_interpolate,
            arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_common_args = [
            self.gripper_joint_names,
            -1.3 - 0.01,
            1.3 + 0.01,  # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        ]
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            normalize_action=True,  # if normalize_action==True, then action 0 maps to qpos=0, and action 1 maps to qpos=1.31
            drive_mode="force",
        )
        gripper_pd_joint_target_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_pos_interpolate_by_planner = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
            interpolate=True,
            interpolate_by_planner=True,
            interpolate_planner_exec_set_target_vel=True,
            interpolate_planner_vlim=self.gripper_vel_limit,
            interpolate_planner_alim=self.gripper_acc_limit,
            interpolate_planner_jerklim=self.gripper_jerk_limit,
        )
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_delta_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_delta_pos_interpolate_by_planner = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
            interpolate=True,
            interpolate_by_planner=True,
            interpolate_planner_exec_set_target_vel=True,
            delta_target_from_last_drive_target=True,
            small_action_repeat_last_target=True,
            interpolate_planner_vlim=self.gripper_vel_limit,
            interpolate_planner_alim=self.gripper_acc_limit,
            interpolate_planner_jerklim=self.gripper_jerk_limit,
        )
        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
            gripper_pd_joint_target_pos=gripper_pd_joint_target_pos,
            gripper_pd_joint_target_pos_interpolate_by_planner=gripper_pd_joint_target_pos_interpolate_by_planner,
            gripper_pd_joint_delta_pos=gripper_pd_joint_delta_pos,
            gripper_pd_joint_target_delta_pos=gripper_pd_joint_target_delta_pos,
            gripper_pd_joint_target_delta_pos_interpolate_by_planner=gripper_pd_joint_target_delta_pos_interpolate_by_planner,
        )

        controller_configs = {}
        for base_controller_name in _C["base"]:
            for arm_controller_name in _C["arm"]:
                for gripper_controller_name in _C["gripper"]:
                    c = {}
                    if base_controller_name is not None:
                        c = {"base": _C["base"][base_controller_name]}
                    c["arm"] = _C["arm"][arm_controller_name]
                    c["gripper"] = _C["gripper"][gripper_controller_name]
                    combined_name = arm_controller_name + "_" + gripper_controller_name
                    if base_controller_name is not None:
                        combined_name = base_controller_name + "_" + combined_name
                    controller_configs[combined_name] = c

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="overhead_camera",
            p=[0, 0, 0],
            q=[
                0.5,
                0.5,
                -0.5,
                0.5,
            ],  # SAPIEN uses ros camera convention; the rotation matrix of link_camera's pose is in opencv convention, so we need to transform it to ros convention
            width=640,
            height=512,
            actor_uid="link_camera",
            intrinsic=np.array([[425.0, 0, 320.0], [0, 425.0, 256.0], [0, 0, 1]]),
        )


class GoogleRobotManualTunedIntrinsicConfig(GoogleRobotDefaultConfig):
    @property
    def cameras(self):
        return CameraConfig(
            uid="overhead_camera",
            p=[0, 0, 0],
            q=[0.5, 0.5, -0.5, 0.5],
            width=640,
            height=512,
            actor_uid="link_camera",
            intrinsic=np.array([[425.0, 0, 305.0], [0, 413.1, 233.0], [0, 0, 1]]),
        )


class GoogleRobotStaticBaseConfig(GoogleRobotDefaultConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(mobile_base=False, **kwargs)
        
        
class GoogleRobotStaticBaseHalfFingerFrictionConfig(GoogleRobotStaticBaseConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(finger_friction=1.0, **kwargs)
        

class GoogleRobotStaticBaseQuarterFingerFrictionConfig(GoogleRobotStaticBaseConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(finger_friction=0.5, **kwargs)
        
        
class GoogleRobotStaticBaseOneEighthFingerFrictionConfig(GoogleRobotStaticBaseConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(finger_friction=0.25, **kwargs)
        
        
class GoogleRobotStaticBaseTwiceFingerFrictionConfig(GoogleRobotStaticBaseConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(finger_friction=4.0, **kwargs)


class GoogleRobotStaticBaseManualTunedIntrinsicConfig(
    GoogleRobotStaticBaseConfig, GoogleRobotManualTunedIntrinsicConfig
):
    pass


class GoogleRobotStaticBaseWorseControl1Config(GoogleRobotDefaultConfig):
    def __init__(self) -> None:
        super().__init__(mobile_base=False)
        self.arm_stiffness = [
            1542.4844516168355,
            1906.9938992819923,
            1605.8611345378665,
            1400.0,
            630.0,
            730.0,
            583.6446104792196,
            2000,
            2000,
        ]
        self.arm_damping = [
            513.436152107585,
            504.0051814405743,
            455.6134557131383,
            408.36436883104705,
            253.94979108395967,
            156.7912085424362,
            138.8619324972991,
            900,
            900,
        ]


class GoogleRobotStaticBaseWorseControl2Config(GoogleRobotDefaultConfig):
    def __init__(self) -> None:
        super().__init__(mobile_base=False)
        self.arm_stiffness = [
            2000.0,
            2000.0,
            1500.0,
            1500.0,
            1500.0,
            800.0,
            800.0,
            2000,
            2000,
        ]
        self.arm_damping = [300.0, 300.0, 200.0, 200.0, 150.0, 100.0, 80.0, 900, 900]


class GoogleRobotStaticBaseWorseControl3Config(GoogleRobotDefaultConfig):
    def __init__(self) -> None:
        super().__init__(mobile_base=False)
        self.arm_stiffness = [
            1200.0,
            1200.0,
            800.0,
            800.0,
            600.0,
            300.0,
            300.0,
            2000,
            2000,
        ]
        self.arm_damping = [1200.0, 1200.0, 800.0, 800.0, 800.0, 400.0, 400.0, 900, 900]


class GoogleRobotMobileBaseConfig(GoogleRobotDefaultConfig):
    def __init__(self) -> None:
        super().__init__(mobile_base=True)
