# isort: off
from .pd_joint_pos import (
    PDJointPosController,
    PDJointPosControllerConfig,
    PDJointPosMimicController,
    PDJointPosMimicControllerConfig,
    PIDJointPosController,
    PIDJointPosControllerConfig,
    PIDJointPosMimicController,
    PIDJointPosMimicControllerConfig,
)
from .pd_ee_pose import (
    PDEEPosController,
    PDEEPosControllerConfig,
    PDEEPoseController,
    PDEEPoseControllerConfig,
)
from .pd_joint_vel import PDJointVelController, PDJointVelControllerConfig
from .passive_controller import PassiveController, PassiveControllerConfig
from .pd_base_vel import PDBaseVelController, PDBaseVelControllerConfig


def deepcopy_dict(configs: dict):
    """Make a deepcopy of dict.
    The built-in behavior will not copy references to the same value.
    """
    from copy import deepcopy

    assert isinstance(configs, dict), type(configs)
    ret = {}
    for k, v in configs.items():
        if isinstance(v, dict):
            ret[k] = deepcopy_dict(v)
        else:
            ret[k] = deepcopy(v)
    return ret
