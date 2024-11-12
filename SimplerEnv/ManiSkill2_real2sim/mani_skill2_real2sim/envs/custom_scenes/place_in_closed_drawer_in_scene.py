from collections import OrderedDict
from typing import List, Optional

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.utils.common import random_choice
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult
from mani_skill2_real2sim.utils.sapien_utils import (
    get_pairwise_contacts,
    compute_total_impulse,
)

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv
from .open_drawer_in_scene import OpenDrawerInSceneEnv


class PlaceObjectInClosedDrawerInSceneEnv(OpenDrawerInSceneEnv):

    def __init__(
        self,
        force_advance_subtask_time_steps: int = 100,
        **kwargs,
    ):
        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None
        self.obj = None
        self.obj_init_options = {}

        self.force_advance_subtask_time_steps = force_advance_subtask_time_steps

        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = (
            0.005
        )  # avoid "false-positive" collisions with other objects
        return scene_config
    
    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly from self.model_ids."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)

        self.obj = self._build_actor_helper(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _load_actors(self):
        super()._load_actors()
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            obj_init_xy = self._episode_rng.uniform([-0.10, -0.00], [-0.05, 0.1], [2])
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
            init_rand_axis_rot_range := self.obj_init_options.get(
                "init_rand_axis_rot_range", 0.0
            )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)

        # set objects
        self.obj_init_options = options.get("obj_init_options", {})
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        obs, info = super().reset(seed=self._episode_seed, options=options)
        self.drawer_link: sapien.Link = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.drawer_collision = self.drawer_link.get_collision_shapes()[2]

        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged evaluation configs under visual matching setup
        overlay_ids = ["a0", "b0", "c0"]
        rgb_overlay_paths = [
            str(ASSET_DIR / f"real_inpainting/open_drawer_{i}.png") for i in overlay_ids
        ]
        robot_init_xs = [0.644, 0.652, 0.665]
        robot_init_ys = [-0.179, 0.009, 0.224]
        robot_init_rotzs = [-0.03, 0, 0]
        idx_chosen = self._episode_rng.choice(len(overlay_ids))

        options["robot_init_options"] = {
            "init_xy": [robot_init_xs[idx_chosen], robot_init_ys[idx_chosen]],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, robot_init_rotzs[idx_chosen]))
                * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        self.rgb_overlay_img = (
            cv2.cvtColor(cv2.imread(rgb_overlay_paths[idx_chosen]), cv2.COLOR_BGR2RGB)
            / 255
        )
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
        self.cur_subtask_id = 0 # 0: open drawer, 1: place object into drawer
        self.episode_stats = OrderedDict(
            qpos=0.0, is_drawer_open=False, has_contact=0
        )

    def evaluate(self, **kwargs):
        # Drawer
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = qpos
        is_drawer_open = qpos >= 0.15
        self.episode_stats["is_drawer_open"] = self.episode_stats["is_drawer_open"] or is_drawer_open

        # Check whether the object contacts with the drawer
        contact_infos = get_pairwise_contacts(
            self._scene.get_contacts(),
            self.obj,
            self.drawer_link,
            collision_shape1=self.drawer_collision,
        )
        total_impulse = compute_total_impulse(contact_infos)
        has_contact = np.linalg.norm(total_impulse) > 1e-6
        self.episode_stats["has_contact"] += has_contact

        success = (self.cur_subtask_id == 1) and (qpos >= 0.05) and (self.episode_stats["has_contact"] >= 1)

        return dict(success=success, episode_stats=self.episode_stats)

    def advance_to_next_subtask(self):
        self.cur_subtask_id = 1

    def step(self, action):
        if self._elapsed_steps >= self.force_advance_subtask_time_steps:
            # force advance to the next subtask
            self.advance_to_next_subtask()
        return super().step(action)
    
    def get_language_instruction(self, **kwargs):
        if self.cur_subtask_id == 0:
            return f"open {self.drawer_id} drawer"
        else:
            model_name = self._get_instruction_obj_name(self.model_id)
            return f"place {model_name} into {self.drawer_id} drawer"
        
    def is_final_subtask(self):
        return self.cur_subtask_id == 1


@register_env("PlaceIntoClosedDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedDrawerCustomInSceneEnv(
    PlaceObjectInClosedDrawerInSceneEnv, CustomOtherObjectsInSceneEnv
):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"
    drawer_ids = ["top", "middle", "bottom"]


@register_env("PlaceIntoClosedTopDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedTopDrawerCustomInSceneEnv(PlaceIntoClosedDrawerCustomInSceneEnv):
    drawer_ids = ["top"]


@register_env("PlaceIntoClosedMiddleDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedMiddleDrawerCustomInSceneEnv(
    PlaceIntoClosedDrawerCustomInSceneEnv
):
    drawer_ids = ["middle"]


@register_env("PlaceIntoClosedBottomDrawerCustomInScene-v0", max_episode_steps=200)
class PlaceIntoClosedBottomDrawerCustomInSceneEnv(
    PlaceIntoClosedDrawerCustomInSceneEnv
):
    drawer_ids = ["bottom"]
