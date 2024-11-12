import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.configs.widowx import defaults
from mani_skill2_real2sim.utils.common import compute_angle_between
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class WidowX(BaseAgent):
    _config: defaults.WidowXDefaultConfig

    """
        WidowX250 6DoF robot
        links:
            [Actor(name="base_link", id="2"), Actor(name="shoulder_link", id="3"), Actor(name="upper_arm_link", id="4"), Actor(name="upper_forearm_link", id="5"), 
            Actor(name="lower_forearm_link", id="6"), Actor(name="wrist_link", id="7"), Actor(name="gripper_link", id="8"), Actor(name="ee_arm_link", id="9"), 
            Actor(name="gripper_prop_link", id="15"), Actor(name="gripper_bar_link", id="10"), Actor(name="fingers_link", id="11"), 
            Actor(name="left_finger_link", id="14"), Actor(name="right_finger_link", id="13"), Actor(name="ee_gripper_link", id="12")]
        active_joints: 
            ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'left_finger', 'right_finger']
        joint_limits:
            [[-3.1415927  3.1415927]
            [-1.8849556  1.9896754]
            [-2.146755   1.6057029]
            [-3.1415827  3.1415827]
            [-1.7453293  2.146755 ]
            [-3.1415827  3.1415827]
            [ 0.015      0.037    ]
            [ 0.015      0.037    ]]
    """

    @classmethod
    def get_default_config(cls):
        return defaults.WidowXDefaultConfig()

    def __init__(
        self, scene, control_freq, control_mode=None, fix_root_link=True, config=None
    ):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        super()._after_init()
        
        # ignore collision between gripper bar link and two gripper fingers
        gripper_bar_link = get_entity_by_name(self.robot.get_links(), "gripper_bar_link")
        left_finger_link = get_entity_by_name(self.robot.get_links(), "left_finger_link")
        right_finger_link = get_entity_by_name(self.robot.get_links(), "right_finger_link")
        for l in gripper_bar_link.get_collision_shapes():
            l.set_collision_groups(1, 1, 0b11, 0)
        for l in left_finger_link.get_collision_shapes():
            l.set_collision_groups(1, 1, 0b01, 0)
        for l in right_finger_link.get_collision_shapes():
            l.set_collision_groups(1, 1, 0b10, 0)
        
        self.base_link = [x for x in self.robot.get_links() if x.name == "base_link"][0]

        self.finger_right_joint = get_entity_by_name(
            self.robot.get_joints(), "right_finger"
        )
        self.finger_left_joint = get_entity_by_name(
            self.robot.get_joints(), "left_finger"
        )

        self.finger_right_link = get_entity_by_name(
            self.robot.get_links(), "right_finger_link"
        )
        self.finger_left_link = get_entity_by_name(
            self.robot.get_links(), "left_finger_link"
        )

    def get_gripper_closedness(self):
        finger_qpos = self.robot.get_qpos()[-2:]
        finger_qlim = self.robot.get_qlimits()[-2:]
        closedness_left = (finger_qlim[0, 1] - finger_qpos[0]) / (
            finger_qlim[0, 1] - finger_qlim[0, 0]
        )
        closedness_right = (finger_qlim[1, 1] - finger_qpos[1]) / (
            finger_qlim[1, 1] - finger_qlim[1, 0]
        )
        return np.maximum(np.mean([closedness_left, closedness_right]), 0.0)

    def get_fingers_info(self):
        finger_right_pos = self.finger_right_link.get_global_pose().p
        finger_left_pos = self.finger_left_link.get_global_pose().p

        finger_right_vel = self.finger_right_link.get_velocity()
        finger_left_vel = self.finger_left_link.get_velocity()

        return {
            "finger_right_pos": finger_right_pos,
            "finger_left_pos": finger_left_pos,
            "finger_right_vel": finger_right_vel,
            "finger_left_vel": finger_left_vel,
        }

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=60):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_left_link, actor
        )
        rimpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_right_link, actor
        )

        # direction to open the gripper
        ldirection_finger = self.finger_left_link.pose.to_transformation_matrix()[:3, 1]
        rdirection_finger = self.finger_right_link.pose.to_transformation_matrix()[
            :3, 1
        ]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection_finger, limpulse_finger)
        rangle = compute_angle_between(-rdirection_finger, rimpulse_finger)

        lflag = (np.linalg.norm(limpulse_finger) >= min_impulse) and np.rad2deg(
            langle
        ) <= max_angle
        rflag = (np.linalg.norm(rimpulse_finger) >= min_impulse) and np.rad2deg(
            rangle
        ) <= max_angle

        return all([lflag, rflag])

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_left_link, actor
        )
        rimpulse_finger = get_pairwise_contact_impulse(
            contacts, self.finger_right_link, actor
        )

        return (
            np.linalg.norm(limpulse_finger) >= min_impulse,
            np.linalg.norm(rimpulse_finger) >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """
            Build a grasp pose (WidowX gripper).
            From link_gripper's frame, x=approaching, -y=closing
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    @property
    def base_pose(self):
        return self.base_link.get_pose()


class WidowXBridgeDatasetCameraSetup(WidowX):
    _config: defaults.WidowXBridgeDatasetCameraSetupConfig

    @classmethod
    def get_default_config(cls):
        return defaults.WidowXBridgeDatasetCameraSetupConfig()


class WidowXSinkCameraSetup(WidowX):
    _config: defaults.WidowXSinkCameraSetupConfig

    @classmethod
    def get_default_config(cls):
        return defaults.WidowXSinkCameraSetupConfig()