import os
import sys

import numpy as np
import tensorflow as tf
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.policies.rt1.rt1_model import RT1Inference
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)

os.environ["SAPIEN_RENDERER_HEADLESS"] = "1"


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ":1"
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "openvla":
        model = OpenVLAInference(
            unnorm_key=args.unnorm_key,
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr, sub_task_arr, sub_task_moved = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
    print(" " * 10, "Average sub task success", np.mean(sub_task_arr))
    with open("evaluation_results.txt", "a") as f:
        f.write(f"Average success: {np.mean(success_arr)}\n")
        f.write(f"Average sub task grasp success: {np.mean(sub_task_arr)}\n")
        f.write(f"Average sub task move success: {np.mean(sub_task_moved)}\n")
        f.write(f"Args: {args}\n")