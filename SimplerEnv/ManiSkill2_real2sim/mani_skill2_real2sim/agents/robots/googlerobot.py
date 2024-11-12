import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.configs.google_robot import defaults
from mani_skill2_real2sim.utils.common import compute_angle_between
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class GoogleRobot(BaseAgent):

    """
        Google Robot with the following modifications from the original :
            - "joint_finger_{left/right}" are set as fixed joints with a fixed rotation from their corresponding "link_finger_{left/right}" (see urdf), in order to simulate the actual gripper during real evaluation
            - Manually-specified friction on the finger, finger tip and finger nail (see defaults.py)
            - Wheels are fixed if the robot is static (see urdf)
        robot.qpos is 13-dimensional if the robot is mobile, 11-dimensional if the robot is static
        When the robot is mobile, robot.get_active_joints() returns a list of 13 joints:
            ['joint_wheel_left', 'joint_wheel_right', 'joint_torso', 'joint_shoulder', 
            'joint_bicep', 'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper', 
            'joint_finger_right', 'joint_finger_left', 
            'joint_head_pan', 'joint_head_tilt']
        If the robot is static, the first two joints are removed from the list of active joints.
    """

    def _after_init(self):
        super()._after_init()

        self.base_link = [x for x in self.robot.get_links() if x.name == "link_base"][0]
        self.base_inertial_link = [
            x for x in self.robot.get_links() if x.name == "link_base_inertial"
        ][0]

        self.finger_right_joint = get_entity_by_name(
            self.robot.get_joints(), "joint_finger_right"
        )
        self.finger_left_joint = get_entity_by_name(
            self.robot.get_joints(), "joint_finger_left"
        )

        self.finger_right_link = get_entity_by_name(
            self.robot.get_links(), "link_finger_right"
        )
        self.finger_right_tip_link = get_entity_by_name(
            self.robot.get_links(), "link_finger_tip_right"
        )
        self.finger_left_link = get_entity_by_name(
            self.robot.get_links(), "link_finger_left"
        )
        self.finger_left_tip_link = get_entity_by_name(
            self.robot.get_links(), "link_finger_tip_left"
        )

    def get_gripper_closedness(self):
        finger_qpos = self.robot.get_qpos()[-4:-2]
        finger_qlim = self.robot.get_qlimits()[-4:-2, -1]
        return np.maximum(np.mean(finger_qpos / finger_qlim), 0.0)

    def get_fingers_info(self):
        finger_right_pos = self.finger_right_link.get_global_pose().p
        finger_left_pos = self.finger_left_link.get_global_pose().p
        finger_right_tip_pos = self.finger_right_tip_link.get_global_pose().p
        finger_left_tip_pos = self.finger_left_tip_link.get_global_pose().p

        finger_right_vel = self.finger_right_link.get_velocity()
        finger_left_vel = self.finger_left_link.get_velocity()
        finger_right_tip_vel = self.finger_right_tip_link.get_velocity()
        finger_left_tip_vel = self.finger_left_tip_link.get_velocity()

        return {
            "finger_right_pos": finger_right_pos,
            "finger_left_pos": finger_left_pos,
            "finger_right_tip_pos": finger_right_tip_pos,
            "finger_left_tip_pos": finger_left_tip_pos,
            "finger_right_vel": finger_right_vel,
            "finger_left_vel": finger_left_vel,
            "finger_right_tip_vel": finger_right_tip_vel,
            "finger_left_tip_vel": finger_left_tip_vel,
        }

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=80):
        # check if the actor is grasped by the gripper

        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_tip = get_pairwise_contact_impulse(
            contacts, self.finger_left_tip_link, actor
        )
        limpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_left_link, actor
        )
        rimpulse_tip = get_pairwise_contact_impulse(
            contacts, self.finger_right_tip_link, actor
        )
        rimpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_right_link, actor
        )

        # direction to open the gripper
        ldirection_tip = self.finger_left_tip_link.pose.to_transformation_matrix()[
            :3, 1
        ]
        ldirection_finger = self.finger_left_link.pose.to_transformation_matrix()[:3, 1]
        rdirection_tip = self.finger_right_tip_link.pose.to_transformation_matrix()[
            :3, 1
        ]
        rdirection_finger = self.finger_right_link.pose.to_transformation_matrix()[
            :3, 1
        ]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection_tip, limpulse_tip)
        langle = min(langle, compute_angle_between(ldirection_finger, limpulse_finger))
        rangle = compute_angle_between(rdirection_tip, rimpulse_tip)
        rangle = min(rangle, compute_angle_between(rdirection_finger, rimpulse_finger))

        lflag = (
            max(np.linalg.norm(limpulse_tip), np.linalg.norm(limpulse_finger))
            >= min_impulse
        ) and np.rad2deg(langle) <= max_angle
        rflag = (
            max(np.linalg.norm(rimpulse_tip), np.linalg.norm(rimpulse_finger))
            >= min_impulse
        ) and np.rad2deg(rangle) <= max_angle
        # print(np.linalg.norm(limpulse_tip), np.linalg.norm(limpulse_finger), np.linalg.norm(rimpulse_tip), np.linalg.norm(rimpulse_finger), langle, rangle)

        return all([lflag, rflag])

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_tip = get_pairwise_contact_impulse(
            contacts, self.finger_left_tip_link, actor
        )
        limpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_left_link, actor
        )
        rimpulse_tip = get_pairwise_contact_impulse(
            contacts, self.finger_right_tip_link, actor
        )
        rimpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_right_link, actor
        )

        return (
            max(np.linalg.norm(limpulse_tip), np.linalg.norm(limpulse_finger))
            >= min_impulse,
            max(np.linalg.norm(rimpulse_tip), np.linalg.norm(rimpulse_finger))
            >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """
            Build a grasp pose (Google Robot hand).
            From link_gripper's frame, z=approaching, y=closing
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    def get_proprioception(self):
        state_dict = super().get_proprioception()
        qpos, qvel = state_dict["qpos"], state_dict["qvel"]
        if self.config.mobile_base:
            base_pos, arm_qpos = qpos[:2], qpos[3:]
            base_vel, arm_qvel = qvel[:2], qvel[3:]
            state_dict["qpos"] = arm_qpos
            state_dict["qvel"] = arm_qvel
            state_dict["base_pos"] = base_pos
            state_dict["base_vel"] = base_vel
        else:
            state_dict["qpos"] = qpos
            state_dict["qvel"] = qvel
        return state_dict

    @property
    def base_pose(self):
        return self.base_link.get_pose()

    def set_base_pose(self, xy):
        # set the x and y coordinates of the robot base
        robot_pose = self.robot.get_pose()
        p, q = robot_pose.p, robot_pose.q
        p = np.concatenate([xy, p[2:]])
        self.robot.set_pose(Pose(p, q))


class GoogleRobotStaticBase(GoogleRobot):
    _config: defaults.GoogleRobotStaticBaseConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseConfig()

    def __init__(
        self, scene, control_freq, control_mode=None, fix_root_link=True, config=None
    ):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        super()._after_init()

        # Sanity check
        active_joints = self.robot.get_active_joints()
        assert active_joints[0].name == "joint_torso"


class GoogleRobotStaticBaseHalfFingerFriction(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseHalfFingerFrictionConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseHalfFingerFrictionConfig()
    
class GoogleRobotStaticBaseQuarterFingerFriction(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseQuarterFingerFrictionConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseQuarterFingerFrictionConfig()
    
class GoogleRobotStaticBaseOneEighthFingerFriction(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseOneEighthFingerFrictionConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseOneEighthFingerFrictionConfig()
    
class GoogleRobotStaticBaseTwiceFingerFriction(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseTwiceFingerFrictionConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseTwiceFingerFrictionConfig()



class GoogleRobotStaticBaseManualTunedIntrinsic(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseManualTunedIntrinsicConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseManualTunedIntrinsicConfig()
    
    
class GoogleRobotStaticBaseWorseControl1(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseWorseControl1Config

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseWorseControl1Config()


class GoogleRobotStaticBaseWorseControl2(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseWorseControl2Config

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseWorseControl2Config()


class GoogleRobotStaticBaseWorseControl3(GoogleRobotStaticBase):
    _config: defaults.GoogleRobotStaticBaseWorseControl3Config

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotStaticBaseWorseControl3Config()


class GoogleRobotMobileBase(GoogleRobot):
    _config: defaults.GoogleRobotMobileBaseConfig

    @classmethod
    def get_default_config(cls):
        return defaults.GoogleRobotMobileBaseConfig()

    def __init__(
        self, scene, control_freq, control_mode=None, fix_root_link=True, config=None
    ):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "base_pd_joint_vel_arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        super()._after_init()

        # Sanity check
        active_joints = self.robot.get_active_joints()
        assert active_joints[0].name == "joint_wheel_left"
        assert active_joints[1].name == "joint_wheel_right"
