from collections import deque
from typing import Optional, Sequence
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from transforms3d.euler import euler2axangle
from PIL import Image
import csv

import os
import sys

# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from latent_pretraining.sampler_latent_action_pretrain import DeltaActionSampler
from latent_pretraining.sampler_action_pretrain import ActionSampler


class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)

class LAPAInference:
    def __init__(
        self,
        policy_setup: str = "widowx_bridge",
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        action_scale_file: str = None,
        **kwargs,
    ) -> None:
        flags = FLAGSClass(kwargs)

        if kwargs['tokens_per_delta'] > 0:
            self.model = DeltaActionSampler(FLAGS=flags)
        else:
            self.model = ActionSampler(FLAGS=flags)

        self.action_scale_list = []
        with open(action_scale_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                # Convert the string values to float and add them to the csv_data list
                self.action_scale_list.append([float(value) for value in row if value.strip()])

        
        self.image_size = image_size
        self.action_scale = action_scale
        self.policy_setup = policy_setup
        self.tokens_per_delta = kwargs['tokens_per_delta']
        
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # the purpose of this for loop is just to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng)  # each shape [2,]
        self.task_description = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image


    

    def get_averaged_values(self, indices):
        averaged_values = []
        for row_idx, idx in enumerate(indices):
            try:
                value1 = self.action_scale_list[row_idx][idx]
                value2 = self.action_scale_list[row_idx][idx + 1]
                average = (value1 + value2) / 2
            except: 
                average = 1
            averaged_values.append(average)
        return averaged_values

    def reset(self, task_description: str) -> None:
        self.task_description = task_description

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = Image.fromarray(image)
        prompts = [{'image': [image], 'question':task_description}]
        

        action_output = self.model(prompts)[0]
        raw_actions = self.get_averaged_values(action_output)


        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(raw_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
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

            # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
            # gripper_close_commanded = (current_gripper_action < 0.5)
            # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

            # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
            # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
            #     self.sticky_action_is_on = True
            #     self.sticky_gripper_action = relative_gripper_action

            # if self.sticky_action_is_on:
            #     self.gripper_action_repeat += 1
            #     relative_gripper_action = self.sticky_gripper_action

            # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
            #     self.sticky_action_is_on = False
            #     self.gripper_action_repeat = 0

            # action['gripper'] = np.array([relative_gripper_action])

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
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
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
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
        plt.savefig(save_path)
