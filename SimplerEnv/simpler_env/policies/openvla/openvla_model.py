# from collections import deque
# from typing import Optional, Sequence
# import os

# import jax
# import matplotlib.pyplot as plt
# import numpy as np
# from octo.model.octo_model import OctoModel
# import tensorflow as tf
# from transformers import AutoTokenizer
# from transforms3d.euler import euler2axangle
# from PIL import Image
# import csv
# from absl import flags

# import os
# import sys
# from transformers import AutoModelForVision2Seq, AutoProcessor
# from PIL import Image

# import torch


# class OpenVLAInference:
#     def __init__(
#         self,
#         model: Optional[OctoModel] = None,
#         dataset_id: Optional[str] = None,
#         model_type: str = "octo-base",
#         policy_setup: str = "widowx_bridge",
#         horizon: int = 2,
#         pred_action_horizon: int = 4,
#         exec_horizon: int = 1,
#         image_size: int = 256,
#         action_scale: float = 1.0,
#         init_rng: int = 0,
#         action_scale_file: str = None,
#         **kwargs,
#     ) -> None:
#         # JaxDistributedConfig.initialize(kwargs['jax_distributed'])

#         # self._set_flags_from_kwargs(kwargs)
#         # Load Processor & VLA
#         self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
#         self.vla = AutoModelForVision2Seq.from_pretrained(
#             "openvla/openvla-7b", 
#             attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#             torch_dtype=torch.bfloat16, 
#             low_cpu_mem_usage=True, 
#             trust_remote_code=True).to('cuda')
        
    
        
#         self.action_scale_list = []
#         with open(action_scale_file, 'r') as file:
#             reader = csv.reader(file)
#             next(reader) 
#             for row in reader:
#                 # Convert the string values to float and add them to the csv_data list
#                 self.action_scale_list.append([float(value) for value in row if value.strip()])

        
#         self.image_size = image_size
#         self.action_scale = action_scale
#         self.policy_setup = policy_setup
#         self.tokens_per_delta = kwargs['tokens_per_delta']
        
#         self.rng = jax.random.PRNGKey(init_rng)
#         for _ in range(5):
#             # the purpose of this for loop is just to match octo server's inference seeds
#             self.rng, _key = jax.random.split(self.rng)  # each shape [2,]
#         self.task_description = None

#     def _resize_image(self, image: np.ndarray) -> np.ndarray:
#         image = tf.image.resize(
#             image,
#             size=(self.image_size, self.image_size),
#             method="lanczos3",
#             antialias=True,
#         )
#         image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
#         return image


    

#     def get_averaged_values(self, indices):
#         averaged_values = []
#         for row_idx, idx in enumerate(indices):
#             try:
#                 value1 = self.action_scale_list[row_idx][idx]
#                 value2 = self.action_scale_list[row_idx][idx + 1]
#                 average = (value1 + value2) / 2
#             except: 
#                 print("index out of range")
#                 average = 0
#             averaged_values.append(average)
#         return averaged_values

#     def reset(self, task_description: str) -> None:
#         self.task_description = task_description

#     def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
#         """
#         Input:
#             image: np.ndarray of shape (H, W, 3), uint8
#             task_description: Optional[str], task description; if different from previous task description, policy state is reset
#         Output:
#             raw_action: dict; raw policy action output
#             action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
#                 - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
#                 - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
#                 - 'gripper': np.ndarray of shape (1,), gripper action
#                 - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
#         """
#         if task_description is not None:
#             if task_description != self.task_description:
#                 # task description has changed; reset the policy state
#                 self.reset(task_description)

#         assert image.dtype == np.uint8
#         image = Image.fromarray(image)

#         inputs = self.processor(task_description, image).to("cuda", dtype=torch.bflaot16)
#         action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sampler=False)


        
       
#         print("norm raw actions", action)
#         indices = action
#         raw_actions = self.get_averaged_values(indices)


#         print("raw actions", raw_actions)
       

#         raw_action = {
#             "world_vector": np.array(raw_actions[:3]),
#             "rotation_delta": np.array(raw_actions[3:6]),
#             "open_gripper": np.array(raw_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
#         }

#         # process raw_action to obtain the action to be sent to the maniskill2 environment
#         action = {}
#         action["world_vector"] = raw_action["world_vector"] * self.action_scale
#         action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
#         roll, pitch, yaw = action_rotation_delta
#         action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
#         action_rotation_axangle = action_rotation_ax * action_rotation_angle
#         action["rot_axangle"] = action_rotation_axangle * self.action_scale

#         if self.policy_setup == "google_robot":
#             current_gripper_action = raw_action["open_gripper"]

#             # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
#             # gripper_close_commanded = (current_gripper_action < 0.5)
#             # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

#             # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
#             # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
#             #     self.sticky_action_is_on = True
#             #     self.sticky_gripper_action = relative_gripper_action

#             # if self.sticky_action_is_on:
#             #     self.gripper_action_repeat += 1
#             #     relative_gripper_action = self.sticky_gripper_action

#             # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
#             #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
#             #     self.sticky_action_is_on = False
#             #     self.gripper_action_repeat = 0

#             # action['gripper'] = np.array([relative_gripper_action])

#             # alternative implementation
#             if self.previous_gripper_action is None:
#                 relative_gripper_action = np.array([0])
#             else:
#                 relative_gripper_action = (
#                     self.previous_gripper_action - current_gripper_action
#                 )  # google robot 1 = close; -1 = open
#             self.previous_gripper_action = current_gripper_action

#             if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
#                 self.sticky_action_is_on = True
#                 self.sticky_gripper_action = relative_gripper_action

#             if self.sticky_action_is_on:
#                 self.gripper_action_repeat += 1
#                 relative_gripper_action = self.sticky_gripper_action

#             if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
#                 self.sticky_action_is_on = False
#                 self.gripper_action_repeat = 0
#                 self.sticky_gripper_action = 0.0

#             action["gripper"] = relative_gripper_action

#         elif self.policy_setup == "widowx_bridge":
#             action["gripper"] = (
#                 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
#             )  # binarize gripper action to 1 (open) and -1 (close)
#             # self.gripper_is_closed = (action['gripper'] < 0.0)

#         action["terminate_episode"] = np.array([0.0])

#         print("action",action)

#         return raw_action, action

#     def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
#         images = [self._resize_image(image) for image in images]
#         ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

#         img_strip = np.concatenate(np.array(images[::3]), axis=1)

#         # set up plt figure
#         figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
#         plt.rcParams.update({"font.size": 12})
#         fig, axs = plt.subplot_mosaic(figure_layout)
#         fig.set_size_inches([45, 10])

#         # plot actions
#         pred_actions = np.array(
#             [
#                 np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
#                 for a in predicted_raw_actions
#             ]
#         )
#         for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
#             # actions have batch, horizon, dim, in this example we just take the first action for simplicity
#             axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
#             axs[action_label].set_title(action_label)
#             axs[action_label].set_xlabel("Time in one episode")

#         axs["image"].imshow(img_strip)
#         axs["image"].set_xlabel("Time in one episode (subsampled)")
#         plt.legend()
#         plt.savefig(save_path)


from collections import deque
from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from simpler_env.utils.action.action_ensemble import ActionEnsembler
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', "FLAM", "OpenVLA")))
print(sys.path)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import LLaMa2ChatPromptBuilder

class OpenVLAInference:
    def __init__(
        self,
        saved_model_path: str = "openvla/openvla-7b",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 2,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        # Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            # "openvla/openvla-7b",
            saved_model_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))

        image: Image.Image = Image.fromarray(image)
        prompt = self.task_description
        prompt_builder_fn = LLaMa2ChatPromptBuilder
        prompt_builder = prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {prompt}?"},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        # input_ids = self.base_tokenizer(prompt_builder.get_prompt()
        # print("the prompt is", prompt)
        prompt = prompt_builder.get_prompt()
        # prompt =f'[INST] <<SYS>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\nWhat action should the robot take to {prompt}? [/INST]'
        # print(prompt)
        # predict action (7-dof; un-normalize for bridgev2)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        raw_actions = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]
        print(f"*** raw actions {raw_actions} ***")

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()