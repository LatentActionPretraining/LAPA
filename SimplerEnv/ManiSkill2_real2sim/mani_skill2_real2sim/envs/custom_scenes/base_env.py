from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import os
import sapien.core as sapien
from sapien.core import Pose
import cv2

from mani_skill2_real2sim import ASSET_DIR, format_path
from mani_skill2_real2sim.utils.io_utils import load_json
from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.robots.googlerobot import (
    GoogleRobotStaticBase,
    GoogleRobotStaticBaseWorseControl1, GoogleRobotStaticBaseWorseControl2, GoogleRobotStaticBaseWorseControl3,
    GoogleRobotStaticBaseHalfFingerFriction, GoogleRobotStaticBaseQuarterFingerFriction, GoogleRobotStaticBaseOneEighthFingerFriction, GoogleRobotStaticBaseTwiceFingerFriction
)
from mani_skill2_real2sim.agents.robots.widowx import WidowX, WidowXBridgeDatasetCameraSetup, WidowXSinkCameraSetup
from mani_skill2_real2sim.agents.robots.panda import Panda
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.sensors.camera import CameraConfig
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)

class CustomSceneEnv(BaseEnv):
    SUPPORTED_ROBOTS = {"google_robot_static": GoogleRobotStaticBase, 
                        "widowx": WidowX,
                        "widowx_bridge_dataset_camera_setup": WidowXBridgeDatasetCameraSetup,
                        "widowx_sink_camera_setup": WidowXSinkCameraSetup,
                        "panda": Panda,
                        # configs for ablation studies
                        "google_robot_static_worse_control1": GoogleRobotStaticBaseWorseControl1,
                        "google_robot_static_worse_control2": GoogleRobotStaticBaseWorseControl2,
                        "google_robot_static_worse_control3": GoogleRobotStaticBaseWorseControl3,
                        "google_robot_static_half_finger_friction": GoogleRobotStaticBaseHalfFingerFriction,
                        "google_robot_static_quarter_finger_friction": GoogleRobotStaticBaseQuarterFingerFriction,
                        "google_robot_static_one_eighth_finger_friction": GoogleRobotStaticBaseOneEighthFingerFriction,
                        "google_robot_static_twice_finger_friction": GoogleRobotStaticBaseTwiceFingerFriction,
    }
    agent: Union[GoogleRobotStaticBase, WidowX, Panda]
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str
    
    def __init__(
            self, 
            robot: str = "google_robot_static", 
            rgb_overlay_path: Optional[str] = None, 
            rgb_overlay_cameras: list = [], 
            rgb_overlay_mode: str = 'background',
            rgb_always_overlay_objects: List[str] = [],
            disable_bad_material: bool = False,
            asset_root: Optional[str] = None,
            scene_root: Optional[str] = None,
            scene_name: Optional[str] = None,
            scene_offset: Optional[List[float]] = None,
            scene_pose: Optional[List[float]] = None,
            scene_table_height: float = 0.87,
            model_json: Optional[str] = None,
            model_ids: List[str] = (),
            model_db_override: Dict[str, Dict] = {},
            urdf_version: str = "",
            **kwargs
        ):
        # Assets and scene
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))
        
        if scene_root is None:
            scene_root = self.DEFAULT_SCENE_ROOT
        self.scene_root = Path(format_path(scene_root))
        self.scene_name = scene_name
        self.scene_offset = scene_offset
        self.scene_pose = scene_pose
        self.scene_table_height = scene_table_height

        # Load object model database
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        # NOTE(jigu): absolute path will overwrite asset_root
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found. "
                "If you installed this repo through 'pip install .', or if you stored the assets outside of ManiSkill2_real2sim/data, "
                "you need to set the following environment variable: export MS2_REAL2SIM_ASSET_DIR={path_to_your_ManiSkill2_real2sim_assets} . "
                "(for example, you can download this directory https://github.com/simpler-env/ManiSkill2_real2sim/tree/main/data and set the env variable to the downloaded directory). "
                "Additionally, for assets in the original ManiSkill2 repo, you can copy the assets into the directory that corresponds to MS2_REAL2SIM_ASSET_DIR."
            )
        self.model_db: Dict[str, Dict] = load_json(model_json)
        self.model_db.update(model_db_override)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids
        self._check_assets()
        
        # Load the "greenscreen" image, which is used to overlay the background portions of simulation observation
        if rgb_overlay_path is not None:
            if not os.path.exists(rgb_overlay_path):
                raise FileNotFoundError(
                    f"rgb_overlay_path {rgb_overlay_path} is not found."
                    "If you installed this repo through 'pip install .' , "
                    "you can download this directory https://github.com/simpler-env/ManiSkill2_real2sim/tree/main/data to get the real-world image overlay assets. "
                )
            self.rgb_overlay_img = cv2.cvtColor(cv2.imread(rgb_overlay_path), cv2.COLOR_BGR2RGB) / 255 # (H, W, 3); float32
        else:
            self.rgb_overlay_img = None
        if not isinstance(rgb_overlay_cameras, list):
            rgb_overlay_cameras = [rgb_overlay_cameras]
        self.rgb_overlay_path = rgb_overlay_path 
        self.rgb_overlay_cameras = rgb_overlay_cameras # perform "greenscreen" on the specified camera(s) observations
        self.rgb_overlay_mode = rgb_overlay_mode # 'background' or 'object' or 'debug' or combinations of them
        self.rgb_always_overlay_objects = rgb_always_overlay_objects # always overlay / greenscreen these objects regardless of the rgb_overlay_mode
        assert ('background' in self.rgb_overlay_mode) or ('debug' in self.rgb_overlay_mode), 'Invalid rgb_overlay_mode'

        self.arena = None
        self.robot_init_options = {}
        self.robot_uid = robot
        if urdf_version is None or urdf_version == "None":
            urdf_version = ""
        self.urdf_version = urdf_version
        self.disable_bad_material = disable_bad_material
        
        super().__init__(**kwargs)
    
    def _check_assets(self):
        """Check whether the assets exist."""
        pass
    
    def _load_arena_helper(self, add_collision=True):
        builder = self._scene.create_actor_builder()
        # scene path
        if self.scene_name is None:
            if 'google_robot_static' in self.robot_uid:
                scene_path = str(self.scene_root / "stages/google_pick_coke_can_1_v4.glb") # hardcoded for now
            elif 'widowx' in self.robot_uid:
                scene_path = str(self.scene_root / "stages/bridge_table_1_v1.glb") # hardcoded for now
            else:
                raise NotImplementedError(f"Default scene path for {self.robot_uid} is not yet set")
        elif "dummy" in self.scene_name:
            scene_path = None  # no scene; we will add a dummy scene with ground and optionally a fake tabletop
        else:
            scene_path = str(self.scene_root / "stages" / f"{self.scene_name}.glb")
        
        # scene offset and pose
        if self.scene_offset is None:
            if 'google_robot_static' in self.robot_uid:
                scene_offset = np.array([-1.6616, -3.0337, 0.0]) # corresponds to the default offset of google_pick_coke_can_1_v4.glb
            elif 'widowx' in self.robot_uid:
                scene_offset = np.array([-2.0634, -2.8313, 0.0])# corresponds to the default offset of bridge_table_1_v1.glb
            else:
                raise NotImplementedError(f"Default scene offset for {self.robot_uid} is not yet set")
        else:
            scene_offset = np.array(self.scene_offset)
            
        if self.scene_pose is None:
            scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        else:
            scene_pose = sapien.Pose(q=self.scene_pose)
            
        # Further process scene offset and scene pose for some specific scenes
        if self.scene_name is not None:
            # Hardcoded for other scenes
            if "modern_bedroom" in self.scene_name:
                scene_offset = np.array([-1.6616, -3.0337, 0.0])
                scene_pose = sapien.Pose([0.178, -2.235, 1.669], [0.007, 0, 0, -1]) * scene_pose
            elif "modern_office" in self.scene_name:
                scene_offset = np.array([-1.6616, -3.0337, 0.0])
                scene_pose = sapien.Pose([-0.192, -1.728, 1.48], [0.709, 0, 0, -0.705]) * scene_pose
            elif self.scene_name == "dummy_tabletop":
                scene_pose = sapien.Pose()
                scene_offset = np.array([0, -0.21, 0])

        # Build scene
        if (self.scene_name is None) or ("dummy" not in self.scene_name):
            # NOTE: use nonconvex collision for static scene
            if add_collision:
                builder.add_nonconvex_collision_from_file(scene_path, scene_pose)
            builder.add_visual_from_file(scene_path, scene_pose)
        else:
            if self.scene_name == "dummy":
                # Should be 0.017 instead of 0.017/2
                builder.add_box_visual(half_size=np.array([10.0, 10.0, 0.017/2]))
            elif self.scene_name == "dummy_drawer":
                builder.add_box_visual(half_size=np.array([10.0, 10.0, 0.017]), color=[1, 1, 1])
                # builder.add_box_visual(half_size=np.array([10.0, 10.0, 0.017]), color=[0.6054843 , 0.34402566, 0.17013837])
            elif self.scene_name == "dummy_tabletop":
                _pose = sapien.Pose([-0.295, 0, 0.017 + 0.865 / 2])
                _half_size = np.array([0.63, 0.615, 0.865]) / 2
                # _color = [0.325, 0.187, 0.1166]
                _color = (np.array([168, 120, 79]) / 255) ** 2.2
                rend_mtl = self._renderer.create_material()
                rend_mtl.base_color = np.hstack([_color, 1.0])
                rend_mtl.metallic = 0.0
                rend_mtl.roughness = 0.3
                rend_mtl.specular = 0.8
                builder.add_box_visual(pose=_pose, half_size=_half_size, material=rend_mtl)
                if add_collision:
                    builder.add_box_collision(pose=_pose, half_size=_half_size)
                # Ground
                _color = (np.array([70, 46, 34]) / 255) ** 2.2
                builder.add_box_visual(half_size=np.array([10.0, 10.0, 0.017]), color=_color)
            else:
                raise NotImplementedError(self.scene_name)
        self.arena = builder.build_static(name="arena")
        # Add offset so that the workspace is next to the table
        
        self.arena.set_pose(sapien.Pose(-scene_offset))
        
    def _settle(self, t):
        # step the simulation and let the scene settle for t seconds
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()
    
    def reset(self, seed=None, options=None):
        self.robot_init_options = options.get("robot_init_options", {})
        obs, info = super().reset(seed=seed, options=options)
        info.update({
            'scene_name': self.scene_name,
            'scene_offset': self.scene_offset,
            'scene_pose': self.scene_pose,
            'scene_table_height': self.scene_table_height,
            'urdf_version': self.urdf_version,
            'rgb_overlay_path': self.rgb_overlay_path,
            'rgb_overlay_cameras': self.rgb_overlay_cameras,
            'rgb_overlay_mode': self.rgb_overlay_mode,
            'disable_bad_material': self.disable_bad_material,
        })
        return obs, info
    
    def _configure_agent(self):
        agent_cls: Type[BaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = agent_cls.get_default_config()
        if self.urdf_version != "":
            self._agent_cfg.urdf_path = self._agent_cfg.urdf_path.replace(
                ".urdf", f"_{self.urdf_version}.urdf"
            )

    def _load_agent(self):
        agent_cls: Type[GoogleRobotStaticBase] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        ) # tool-center point, usually the midpoint between the gripper fingers
        if not self.disable_bad_material:
            set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)
            
    def _initialize_agent(self):
        # initialize agent joint position and 6d pose
        
        if "google_robot_static" in self.robot_uid:
            qpos = np.array(
                [-0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0, 0,
                -0.00285961, 0.7851361]
            )
            robot_init_height = 0.06205 + 0.017 # base height + ground offset in default scene
            robot_init_rot_quat = [0, 0, 0, 1]
        elif 'widowx' in self.robot_uid:
            if self.robot_uid in ['widowx', 'widowx_bridge_dataset_camera_setup']:
                qpos = np.array([-0.01840777,  0.0398835,   0.22242722,  -0.00460194,  1.36524296,  0.00153398, 0.037, 0.037])
            elif self.robot_uid == 'widowx_sink_camera_setup':
                qpos = np.array([-0.2600599, -0.12875618, 0.04461369, -0.00652761, 1.7033415, -0.26983038, 0.037,
                                 0.037])
            else:
                raise NotImplementedError(self.robot_uid)
            
            if self.robot_uid in ['widowx', 'widowx_bridge_dataset_camera_setup']:
                robot_init_height = 0.870
            elif self.robot_uid == 'widowx_sink_camera_setup':
                robot_init_height = 0.85
            else:
                raise NotImplementedError(self.robot_uid)
            robot_init_rot_quat = [0, 0, 0, 1]
        else:
            raise NotImplementedError(self.robot_uid)
        
        if self.robot_init_options.get("qpos", None) is not None:
            qpos = self.robot_init_options["qpos"]
        self.agent.reset(qpos)
        
        if self.robot_init_options.get("init_height", None) is not None:
            robot_init_height = self.robot_init_options["init_height"]
        if self.robot_init_options.get("init_rot_quat", None) is not None:
            robot_init_rot_quat = self.robot_init_options["init_rot_quat"]
        
        if (robot_init_xy := self.robot_init_options.get("init_xy", None)) is not None:
            robot_init_xyz = [robot_init_xy[0], robot_init_xy[1], robot_init_height]
        else:
            if 'google_robot' in self.robot_uid:
                init_x = self._episode_rng.uniform(0.30, 0.40)
                init_y = self._episode_rng.uniform(0.0, 0.2)
            elif 'widowx' in self.robot_uid:
                init_x = 0.147
                if self.robot_uid in ['widowx', 'widowx_bridge_dataset_camera_setup']:
                    init_y = 0.028
                elif self.robot_uid == 'widowx_sink_camera_setup':
                    init_y = 0.070
            else:
                init_x, init_y = 0.0, 0.0
            robot_init_xyz = [init_x, init_y, robot_init_height]
        
        self.agent.robot.set_pose(sapien.Pose(robot_init_xyz, robot_init_rot_quat))
        
    def _register_cameras(self):
        # this camera below is not really used; 
        # the used cameras (mounted with respect to the robot agent) are specified under agents/configs/{robot_name}/defaults.py
        
        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_render_cameras(self):
        # camera for visualization and debugging purposes
        
        pose = look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)
        
    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs
    
    def get_obs(self):
        obs = super().get_obs()
        
        # "greenscreen" process
        if self._obs_mode == "image" and self.rgb_overlay_img is not None:
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            target_object_actor_ids = [x.id for x in self.get_actors() if x.name not in ['ground', 'goal_site', '', 'arena'] + self.rgb_always_overlay_objects]
            target_object_actor_ids = np.array(target_object_actor_ids, dtype=np.int32)

            # get the robot link ids (links are subclass of actors)
            robot_links = self.agent.robot.get_links() # e.g., [Actor(name="root", id="1"), Actor(name="root_arm_1_link_1", id="2"), Actor(name="root_arm_1_link_2", id="3"), ...]
            robot_link_ids = np.array([x.id for x in robot_links], dtype=np.int32)

            # get the link ids of other articulated objects
            other_link_ids = []
            for art_obj in self._scene.get_all_articulations():
                if art_obj is self.agent.robot:
                    continue
                if art_obj.name in self.rgb_always_overlay_objects:
                    continue
                for link in art_obj.get_links():
                    other_link_ids.append(link.id)
            other_link_ids = np.array(other_link_ids, dtype=np.int32)

            for camera_name in self.rgb_overlay_cameras:
                # obtain overlay mask based on segmentation info
                assert 'Segmentation' in obs['image'][camera_name].keys(), 'Image overlay requires segment info in the observation!'
                seg = obs['image'][camera_name]['Segmentation'] # (H, W, 4); [..., 0] is mesh-level; [..., 1] is actor-level; [..., 2:] is zero (unused)
                actor_seg = seg[..., 1]
                mask = np.ones_like(actor_seg, dtype=np.float32)
                if ('background' in self.rgb_overlay_mode) or ('debug' in self.rgb_overlay_mode):
                    if ('object' not in self.rgb_overlay_mode) or ('debug' in self.rgb_overlay_mode):
                        # only overlay the background and keep the foregrounds (robot and target objects) rendered in simulation
                        mask[np.isin(actor_seg, np.concatenate([robot_link_ids, target_object_actor_ids, other_link_ids]))] = 0.0
                    else:
                        # overlay everything except the robot links
                        mask[np.isin(actor_seg, robot_link_ids)] = 0.0
                else:
                    raise NotImplementedError(self.rgb_overlay_mode)
                mask = mask[..., np.newaxis]
                
                # perform overlay on the RGB observation image
                rgb_overlay_img = cv2.resize(self.rgb_overlay_img, (obs['image'][camera_name]['Color'].shape[1], obs['image'][camera_name]['Color'].shape[0]))
                if 'debug' not in self.rgb_overlay_mode:
                    obs['image'][camera_name]['Color'][..., :3] = obs['image'][camera_name]['Color'][..., :3] * (1 - mask) + rgb_overlay_img * mask
                else:
                    # debug
                    # obs['image'][camera_name]['Color'][..., :3] = obs['image'][camera_name]['Color'][..., :3] * (1 - mask) + rgb_overlay_img * mask
                    obs['image'][camera_name]['Color'][..., :3] = obs['image'][camera_name]['Color'][..., :3] * 0.5 + rgb_overlay_img * 0.5
                
        return obs

    def compute_dense_reward(self, info, **kwargs):
        # sparse reward for now
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0
    
    @staticmethod
    def _get_instruction_obj_name(s):
        # given an object name, process its name to be used for language instruction
        s = s.split('_')
        rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked', 'v2']
        cleaned = []
        for w in s:
            if w[-2:] == "cm":
                # object size in object name
                continue
            if w not in rm_list:
                cleaned.append(w)
        return ' '.join(cleaned)
    
    def advance_to_next_subtask(self):
        raise NotImplementedError("advance_to_next_subtask is not implemented for this environment.")

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True
    
    
    
# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #

class CustomOtherObjectsInSceneEnv(CustomSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pick_custom_v0.json"
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in self.model_ids:
            model_dir = models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"{model_dir} is not found."
                    "If you installed this repo through 'pip install .', or if you stored the assets outside of ManiSkill2_real2sim/data, "
                    "you need to set the following environment variable: export MS2_REAL2SIM_ASSET_DIR={path_to_your_ManiSkill2_real2sim_assets} . "
                    "(for example, you can download this directory https://github.com/simpler-env/ManiSkill2_real2sim/tree/main/data and set the env variable to the downloaded directory). "
                    "Additionally, for assets in the original ManiSkill2 repo, you can copy the assets into the directory that corresponds to MS2_REAL2SIM_ASSET_DIR."
                )

            collision_file = model_dir / "collision.obj"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj has been renamed to collision.obj. "
                )
                
    @staticmethod
    def _build_actor_helper(
        model_id: str,
        scene: sapien.Scene,
        scale: float = 1.0,
        physical_material: sapien.PhysicalMaterial = None,
        density: float = 1000.0,
        root_dir: str = ASSET_DIR / "custom",
    ):
        builder = scene.create_actor_builder()
        model_dir = Path(root_dir) / "models" / model_id

        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(model_dir / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(model_dir / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(model_dir / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

        actor = builder.build()
        return actor
                
                

class CustomBridgeObjectsInSceneEnv(CustomOtherObjectsInSceneEnv):
    DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"