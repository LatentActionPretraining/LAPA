from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv


class MoveNearInSceneEnv(CustomSceneEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str

    def __init__(
        self,
        original_lighting: bool = False,
        slightly_darker_lighting: bool = False,
        slightly_brighter_lighting: bool = False,
        ambient_only_lighting: bool = False,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        self.episode_objs = [None] * 3
        self.episode_model_ids = [None] * 3
        self.episode_model_scales = [None] * 3
        self.episode_model_bbox_sizes = [None] * 3
        self.episode_model_init_xyzs = [None] * 3
        self.episode_obj_heights_after_settle = [None] * 3
        self.episode_source_obj = None
        self.episode_target_obj = None
        self.episode_source_obj_bbox_world = None
        self.episode_target_obj_bbox_world = None
        self.episode_obj_xyzs_after_settle = [None] * 3
        self.episode_source_obj_xyz_after_settle = None
        self.episode_target_obj_xyz_after_settle = None
        self.episode_stats = None

        self.obj_init_options = {}

        self.original_lighting = original_lighting
        self.slightly_darker_lighting = slightly_darker_lighting
        self.slightly_brighter_lighting = slightly_brighter_lighting
        self.ambient_only_lighting = ambient_only_lighting

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())

        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_move_near_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["overhead_camera"]

        return ret

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = (
            0.005
        )  # important to avoid "false-positive" collisions with other objects
        return scene_config

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        if self.original_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
        elif self.slightly_darker_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1],
                [0.8, 0.8, 0.8],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])
        elif self.slightly_brighter_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1],
                [3.6, 3.6, 3.6],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], [1.3, 1.3, 1.3])
            self._scene.add_directional_light([1, 1, -1], [1.3, 1.3, 1.3])
        elif self.ambient_only_lighting:
            self._scene.set_ambient_light([1.0, 1.0, 1.0])
        else:
            # Default lighting
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1],
                [2.2, 2.2, 2.2],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
            self._scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_actors(self):
        self._load_arena_helper()
        self._load_model()
        for obj in self.episode_objs:
            obj.set_damping(0.1, 0.1)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        self.set_episode_rng(seed)
        model_scales = options.get("model_scales", None)
        model_ids = options.get("model_ids", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_ids, model_scales)
        reconfigure = _reconfigure or reconfigure

        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        options["reconfigure"] = reconfigure

        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update(
            {
                "episode_model_ids": self.episode_model_ids,
                "episode_model_scales": self.episode_model_scales,
                "episode_source_obj_name": self.episode_source_obj.name,
                "episode_target_obj_name": self.episode_target_obj.name,
                "episode_source_obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.episode_source_obj.pose,
                "episode_target_obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.episode_target_obj.pose,
            }
        )
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.21],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, -0.09)) * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            all_obj_keep_height=False,
            moved_correct_obj=False,
            moved_wrong_obj=False,
            near_tgt_obj=False,
            is_closest_to_tgt=False,
        )

    @staticmethod
    def _list_equal(l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True

    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose a triplet randomly from self.model_ids."""
        reconfigure = False

        # model ids
        if model_ids is None:
            model_ids = []
            for _ in range(3):
                model_ids.append(random_choice(self.model_ids, self._episode_rng))
        if not self._list_equal(model_ids, self.episode_model_ids):
            self.episode_model_ids = model_ids
            reconfigure = True

        # model scales
        if model_scales is None:
            model_scales = []
            for model_id in self.episode_model_ids:
                this_available_model_scales = self.model_db[model_id].get(
                    "scales", None
                )
                if this_available_model_scales is None:
                    model_scales.append(1.0)
                else:
                    model_scales.append(
                        random_choice(this_available_model_scales, self._episode_rng)
                    )
        if not self._list_equal(model_scales, self.episode_model_scales):
            self.episode_model_scales = model_scales
            reconfigure = True

        # model bbox sizes
        model_bbox_sizes = []
        for model_id, model_scale in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            model_info = self.model_db[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes.append(bbox_size * model_scale)
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    def _initialize_actors(self):
        source_obj_id: int = self.obj_init_options.get("source_obj_id", None)
        target_obj_id: int = self.obj_init_options.get("target_obj_id", None)
        assert source_obj_id is not None and target_obj_id is not None
        self.episode_source_obj = self.episode_objs[source_obj_id]
        self.episode_target_obj = self.episode_objs[target_obj_id]
        self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[
            source_obj_id
        ]  # bbox xyz extents in the world frame at timestep=0
        self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[
            target_obj_id
        ]

        # Objects will fall from a certain initial height onto the table
        obj_init_xys = self.obj_init_options.get("init_xys", None)
        assert obj_init_xys is not None
        obj_init_xys = np.array(obj_init_xys)  # [n_objects, 2]
        assert obj_init_xys.shape == (len(self.episode_objs), 2)

        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5 # let object fall onto the table

        obj_init_rot_quats = self.obj_init_options.get("init_rot_quats", None)
        if obj_init_rot_quats is not None:
            obj_init_rot_quats = np.array(obj_init_rot_quats)
            assert obj_init_rot_quats.shape == (len(self.episode_objs), 4)
        else:
            obj_init_rot_quats = np.zeros((len(self.episode_objs), 4))
            obj_init_rot_quats[:, 0] = 1.0

        for i, obj in enumerate(self.episode_objs):
            p = np.hstack([obj_init_xys[i], obj_init_z])
            q = obj_init_rot_quats[i]
            obj.set_pose(sapien.Pose(p, q))
            # Lock rotation around x and y
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self._settle(0.5)
        
        # Unlock motion
        for obj in self.episode_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel, ang_vel = 0.0, 0.0
        for obj in self.episode_objs:
            lin_vel += np.linalg.norm(obj.velocity)
            ang_vel += np.linalg.norm(obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        self.episode_obj_xyzs_after_settle = []
        for obj in self.episode_objs:
            self.episode_obj_xyzs_after_settle.append(obj.pose.p)
        self.episode_source_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[
            source_obj_id
        ]
        self.episode_target_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[
            target_obj_id
        ]
        self.episode_source_obj_bbox_world = (
            quat2mat(self.episode_source_obj.pose.q)
            @ self.episode_source_obj_bbox_world
        )
        self.episode_target_obj_bbox_world = (
            quat2mat(self.episode_target_obj.pose.q)
            @ self.episode_target_obj_bbox_world
        )

    @property
    def source_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_source_obj.pose.transform(
            self.episode_source_obj.cmass_local_pose
        )

    @property
    def target_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_target_obj.pose.transform(
            self.episode_target_obj.cmass_local_pose
        )

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                source_obj_pose=vectorize_pose(self.source_obj_pose),
                target_obj_pose=vectorize_pose(self.target_obj_pose),
                tcp_to_source_obj_pos=self.source_obj_pose.p - self.tcp.pose.p,
            )
        return obs

    def evaluate(self, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # Check if objects are knocked down or knocked off table
        other_obj_ids = [
            i
            for (i, obj) in enumerate(self.episode_objs)
            if (obj.name != self.episode_source_obj.name)
            and (obj.name != self.episode_target_obj.name)
        ]
        other_obj_heights = [self.episode_objs[i].pose.p[2] for i in other_obj_ids]
        other_obj_heights_after_settle = [
            self.episode_obj_xyzs_after_settle[i][2] for i in other_obj_ids
        ]
        other_obj_diff_heights = [
            x - y for (x, y) in zip(other_obj_heights, other_obj_heights_after_settle)
        ]
        other_obj_keep_height = all(
            [x > -0.02 for x in other_obj_diff_heights]
        )  # require other objects to not be knocked down on the table
        source_obj_diff_height = (
            source_obj_pose.p[2] - self.episode_source_obj_xyz_after_settle[2]
        )  # source object should not be knocked off the table
        target_obj_diff_height = (
            target_obj_pose.p[2] - self.episode_target_obj_xyz_after_settle[2]
        )  # target object should not be knocked off the table
        all_obj_keep_height = (
            other_obj_keep_height
            and (source_obj_diff_height > -0.15)
            and (target_obj_diff_height > -0.15)
        )

        # Check if moving the correct source object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2]
            - self.episode_source_obj.pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
            self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # Check if the source object is near the target object
        dist_to_tgt_obj = np.linalg.norm(source_obj_pose.p[:2] - target_obj_pose.p[:2])
        tgt_obj_bbox_xy_dist = (
            np.linalg.norm(self.episode_target_obj_bbox_world[:2]) / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_bbox_xy_dist = (
            np.linalg.norm(self.episode_source_obj_bbox_world[:2]) / 2
        )
        # print(dist_to_tgt_obj, tgt_obj_bbox_xy_dist, src_obj_bbox_xy_dist)
        near_tgt_obj = (
            dist_to_tgt_obj < tgt_obj_bbox_xy_dist + src_obj_bbox_xy_dist + 0.10
        )

        # Check if the source object is closest to the target object
        dist_to_other_objs = []
        for obj in self.episode_objs:
            if obj.name == self.episode_source_obj.name:
                continue
            dist_to_other_objs.append(
                np.linalg.norm(source_obj_pose.p[:2] - obj.pose.p[:2])
            )
        is_closest_to_tgt = all(
            [dist_to_tgt_obj < x + 0.01 for x in dist_to_other_objs]
        )

        success = (
            all_obj_keep_height
            and moved_correct_obj
            and near_tgt_obj
            and is_closest_to_tgt
        )

        ret_info = dict(
            all_obj_keep_height=all_obj_keep_height,
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            near_tgt_obj=near_tgt_obj,
            is_closest_to_tgt=is_closest_to_tgt,
            success=success,
        )
        for k in self.episode_stats:
            self.episode_stats[k] = ret_info[
                k
            ]  # for this environment, episode stats equal to the current step stats
        ret_info["episode_stats"] = self.episode_stats

        return ret_info

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"move {src_name} near {tgt_name}"


@register_env("MoveNearGoogleInScene-v0", max_episode_steps=80)
class MoveNearGoogleInSceneEnv(MoveNearInSceneEnv, CustomOtherObjectsInSceneEnv):
    def __init__(self, no_distractor=False, **kwargs):
        self.no_distractor = no_distractor
        self._setup_obj_configs()
        super().__init__(**kwargs)

    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.triplets = [
            ("blue_plastic_bottle", "opened_pepsi_can", "orange"),
            ("opened_7up_can", "apple", "sponge"),
            ("opened_coke_can", "opened_redbull_can", "apple"),
            ("sponge", "blue_plastic_bottle", "opened_7up_can"),
            ("orange", "opened_pepsi_can", "opened_redbull_can"),
        ]
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "opened_pepsi_can": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "opened_7up_can": euler2quat(np.pi / 2, 0, 0),
            "apple": [1.0, 0.0, 0.0, 0.0],
            "sponge": euler2quat(0, 0, np.pi / 2),
            "opened_coke_can": euler2quat(np.pi / 2, 0, 0),
            "opened_redbull_can": euler2quat(np.pi / 2, 0, 0),
        }
        self.special_density_dict = {
            "apple": 200,  # toy apple as in real eval
            "orange": 200
            # by default, opened cans have density 50; blue plastic bottle has density 50; sponge has density 150
        }

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        _num_episodes = (
            len(self.triplets)
            * len(self._source_obj_ids)
            * len(self._xy_config_per_triplet)
        )
        episode_id = obj_init_options.get(
            "episode_id", self._episode_rng.randint(_num_episodes)
        )
        triplet = self.triplets[
            episode_id // (len(self._source_obj_ids) * len(self._xy_config_per_triplet))
        ]
        source_obj_id = self._source_obj_ids[episode_id % len(self._source_obj_ids)]
        target_obj_id = self._target_obj_ids[episode_id % len(self._target_obj_ids)]
        xy_config_triplet = self._xy_config_per_triplet[
            (
                episode_id
                % (len(self._source_obj_ids) * len(self._xy_config_per_triplet))
            )
            // len(self._source_obj_ids)
        ]
        quat_config_triplet = [
            self.obj_init_quat_dict[model_id] for model_id in triplet
        ]
        if self.no_distractor:
            triplet = [triplet[source_obj_id], triplet[target_obj_id]]
            xy_config_triplet = [
                xy_config_triplet[source_obj_id],
                xy_config_triplet[target_obj_id],
            ]
            quat_config_triplet = [
                quat_config_triplet[source_obj_id],
                quat_config_triplet[target_obj_id],
            ]
            source_obj_id = 0
            target_obj_id = 1

        options["model_ids"] = triplet
        obj_init_options["source_obj_id"] = source_obj_id
        obj_init_options["target_obj_id"] = target_obj_id
        obj_init_options["init_xys"] = xy_config_triplet
        obj_init_options["init_rot_quats"] = quat_config_triplet
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            if model_id in self.special_density_dict:
                density = self.special_density_dict[model_id]
            else:
                density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)


@register_env("MoveNearGoogleBakedTexInScene-v0", max_episode_steps=80)
class MoveNearGoogleBakedTexInSceneEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v0.json"

    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.triplets = [
            ("blue_plastic_bottle", "baked_opened_pepsi_can", "orange"),
            ("baked_opened_7up_can", "baked_apple", "baked_sponge"),
            ("baked_opened_coke_can", "baked_opened_redbull_can", "baked_apple"),
            ("baked_sponge", "blue_plastic_bottle", "baked_opened_7up_can"),
            ("orange", "baked_opened_pepsi_can", "baked_opened_redbull_can"),
        ]
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "baked_opened_pepsi_can": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_opened_7up_can": euler2quat(np.pi / 2, 0, 0),
            "baked_apple": [1.0, 0.0, 0.0, 0.0],
            "baked_sponge": euler2quat(0, 0, np.pi / 2),
            "baked_opened_coke_can": euler2quat(np.pi / 2, 0, 0),
            "baked_opened_redbull_can": euler2quat(np.pi / 2, 0, 0),
        }
        self.special_density_dict = {"baked_apple": 200, "orange": 200}


@register_env("MoveNearGoogleBakedTexInScene-v1", max_episode_steps=80)
class MoveNearGoogleBakedTexInSceneEnvV1(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"

    def __init__(self, light_mode=None, **kwargs):
        self.light_mode = light_mode
        super().__init__(**kwargs)

    def _setup_lighting(self):
        if (self.light_mode is None) or ("simple" not in self.light_mode):
            super()._setup_lighting()
        elif self.light_mode == "simple":
            self._scene.set_ambient_light([1.0] * 3)
        elif self.light_mode == "simple2":
            self._scene.set_ambient_light([1.0] * 3)
            angle = 90
            self._scene.add_directional_light(
                [-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [0.5] * 3
            )

    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.triplets = [
            ("blue_plastic_bottle", "baked_opened_pepsi_can_v2", "orange"),
            ("baked_opened_7up_can_v2", "baked_apple_v2", "baked_sponge_v2"),
            (
                "baked_opened_coke_can_v2",
                "baked_opened_redbull_can_v2",
                "baked_apple_v2",
            ),
            ("baked_sponge_v2", "blue_plastic_bottle", "baked_opened_7up_can_v2"),
            ("orange", "baked_opened_pepsi_can_v2", "baked_opened_redbull_can_v2"),
        ]
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "baked_opened_pepsi_can_v2": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_opened_7up_can_v2": euler2quat(np.pi / 2, 0, 0),
            "baked_apple_v2": [1.0, 0.0, 0.0, 0.0],
            "baked_sponge_v2": euler2quat(0, 0, np.pi / 2),
            "baked_opened_coke_can_v2": euler2quat(np.pi / 2, 0, 0),
            "baked_opened_redbull_can_v2": euler2quat(np.pi / 2, 0, 0),
        }
        self.special_density_dict = {"baked_apple_v2": 200, "orange": 200}

    def _load_model(self):
        super()._load_model()
        for obj in self.episode_objs:
            for visual in obj.get_visual_bodies():
                for rs in visual.get_render_shapes():
                    mtl = rs.material
                    mtl.set_roughness(1.0)
                    mtl.set_metallic(0.0)
                    mtl.set_specular(0.0)
                    rs.set_material(mtl)


@register_env("MoveNearAltGoogleCameraInScene-v0", max_episode_steps=80)
class MoveNearAltGoogleCameraInSceneEnv(MoveNearGoogleInSceneEnv):
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        if "robot_init_options" not in options:
            options["robot_init_options"] = {}
        options["robot_init_options"]["qpos"] = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.9351361,
            ]
        )

        return super().reset(seed=seed, options=options)


@register_env("MoveNearAltGoogleCamera2InScene-v0", max_episode_steps=80)
class MoveNearAltGoogleCamera2InSceneEnv(MoveNearGoogleInSceneEnv):
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        if "robot_init_options" not in options:
            options["robot_init_options"] = {}
        options["robot_init_options"]["qpos"] = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.6651361,
            ]
        )

        return super().reset(seed=seed, options=options)
