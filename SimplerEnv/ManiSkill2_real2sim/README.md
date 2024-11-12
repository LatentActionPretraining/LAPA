# ManiSkill2-Real2Sim

This repository is forked from the [original ManiSkill2 repo](https://github.com/haosulab/ManiSkill2), with the following changes:
- **Environment** (`mani_skill2_real2sim/envs`): We removed all environments irrelevant to real-to-sim evaluation, and we implemented real-to-sim evaluation environments under `mani_skill2_real2sim/envs/custom_scenes`. These custom environments act as an independent component of ManiSkill2, allowing for automatic integration into the original ManiSkill2 repository without necessitating any modifications.
- **Robot agents**: We added new robot implementations in `mani_skill2_real2sim/agents/configs` and `mani_skill2_real2sim/agents/robots`. The corresponding robot assets (URDFs) are in `mani_skill2_real2sim/assets/descriptions`.
- **Controllers**: We modified `pd_joint_pos.py`, `pd_ee_pose.py`, and `__init__.py` under `mani_skill2_real2sim/agents/controllers/`, along with `base_controller.py` and `utils.py` under `ManiSkill2_real2sim/mani_skill2_real2sim/agents/`, to support more controller implementations. These scripts can be automatic integrated into the original ManiSkill2 repository.
- **Object assets**: We added custom objects in `data/custom` and custom scenes in `data/hab2_bench_assets` for real-to-sim evaluation purposes. Additionally, we use `MS2_REAL2SIM_ASSET_DIR` to specify the asset directory for custom objects and scenes (if this environment variable is not set, we will use this repo's `data` directory).
- **Demo manual control script** (`mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py`): The script is modified from `mani_skill2_real2sim/examples/demo_manual_control.py` of the original ManiSkill2 repo to support custom real-to-sim environment creationg and visualization. See the script details for usage.


To install, run `pip install -e .`

(Original ManiSkill2 docs: https://haosulab.github.io/ManiSkill2)

Example in interactive python:

```python
import mani_skill2_real2sim.envs, gymnasium as gym
import numpy as np
from transforms3d.euler import euler2quat
from sapien.core import Pose

env1 = gym.make('GraspSingleOpenedCokeCanInScene-v0', obs_mode='rgbd', prepackaged_config=True)
obs1, reset_info_1 = env1.reset()
instruction1 = env1.get_language_instruction()
image1 = obs1['image']['overhead_camera']['rgb']
obs1_alt, reset_info_1_alt = env1.reset(options={'obj_init_options': {'init_xy': [-0.35, -0.02], 'orientation': 'laid_vertically'}}) 
instruction1_alt = env1.get_language_instruction()
image1_alt = obs1_alt['image']['overhead_camera']['rgb']

env2 = gym.make('MoveNearGoogleBakedTexInScene-v0', obs_mode='rgbd', 
    robot='google_robot_static', sim_freq=513, control_freq=3,
    control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
    max_episode_steps=80, scene_name='google_pick_coke_can_1_v4', camera_cfgs={"add_segmentation": True},
    rgb_overlay_path='./data/real_inpainting/google_move_near_real_eval_1.png',
    rgb_overlay_cameras=['overhead_camera'],
    urdf_version='recolor_tabletop_visual_matching_1'
)
# remove "rgb_overlay_path", "rgb_overlay_cameras", and "urdf_version" if you do not want to overlay the real background
obs2, reset_info_2 = env2.reset(options={
    'robot_init_options': {
        'init_xy': np.array([0.35, 0.21]),
        'init_rot_quat': (Pose(q=euler2quat(0, 0, -0.09)) * Pose(q=[0, 0, 0, 1])).q,
    },
    'obj_init_options': {'episode_id': 0}
})
instruction2 = env2.get_language_instruction()
image2 = obs2['image']['overhead_camera']['rgb']

env3 = gym.make('CloseDrawerCustomInScene-v0', obs_mode='rgbd', 
    robot='google_robot_static', sim_freq=513, control_freq=3, max_episode_steps=113, 
    control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
    scene_name='modern_office_no_roof', 
    station_name='mk_station2', # cabinet model
    shader_dir='rt', # enable raytracing, slow for non RTX gpus
)
obs3, reset_info_3 = env3.reset(options={
    'robot_init_options': {
        'init_xy': np.array([0.75, 0.00]),
    },
})
instruction3 = env3.get_language_instruction()
image3 = obs3['image']['overhead_camera']['rgb']

env4 = gym.make('PutSpoonOnTableClothInScene-v0', obs_mode='rgbd', 
    robot='widowx', sim_freq=500, control_freq=5,
    control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',
    max_episode_steps=60, scene_name='bridge_table_1_v1', camera_cfgs={"add_segmentation": True},
    rgb_overlay_path='./data/real_inpainting/bridge_real_eval_1.png',
    rgb_overlay_cameras=['3rd_view_camera'],
)
# remove "rgb_overlay_path", "rgb_overlay_cameras", and "urdf_version" if you do not want to overlay the real background
obs4, reset_info_4 = env4.reset(options={
    'obj_init_options': {'episode_id': 0}
})
instruction4 = env4.get_language_instruction()
image4 = obs4['image']['3rd_view_camera']['rgb']
```
