"""Replay demonstration trajectories."""

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict

from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.config_utils import (add_robot_config_arguments,
                                       get_default_controller_config)
from deoxys import config_root

logger = get_deoxys_example_logger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor_id",
        type=int,
        default=9583,
    )
    parser.add_argument(
        "--product_id",
        type=int,
        default=50734,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # default="/tmp/deoxys_demo_data/run1/recorded_trajecotry.hdf5",
        # default="/home/mbronars/workspace/deoxys_control/demos/demos_1025.hdf5"
        default="/home/mbronars/workspace/deoxys_control/demos/run1/recorded_trajecotry.hdf5",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the replay trajectory"
    )
    parser.add_argument(
        "--controller_type",
        required=True,
        type=str
    )
    parser.add_argument(
        "--control_freq",
        type=int,
        default=20
    )
    robot_config_parse_args(parser)
    return parser.parse_args()


def main():

    args = parse_args()

    # Load the controller here. NOTE: we can replay traj recorded in OSC with OSC or joint control. Not vice verse.
    # TODO: select the SAME control config that you used to collect demos. OSC_POSE is for Gello demos
    control_freq = args.control_freq
    config={} # control config
    if args.controller_type == "JOINT":
        config["controller_type"] = "JOINT_IMPEDANCE" # "OSC_POSITION"
        config["controller_cfg"] = get_default_controller_config(config["controller_type"])
    elif args.controller_type == "OSC":
        config["controller_type"] = "OSC_POSITION"
        config["controller_cfg"] = get_default_controller_config(config["controller_type"])
    elif args.controller_type == "OSC_POSE":
        config["controller_type"] = "OSC_POSE"
        config["controller_cfg"] = get_default_controller_config(config["controller_type"])
    else:
        raise ValueError("Select controller from JOINT or OSC")
    print("Controller Config: ", config["controller_cfg"])
    
    # Load recorded demonstration trajectories TODO: add support for different demo and chunk
    with open(args.dataset, "r") as f:
        demo_file = h5py.File(args.dataset)
        joint_sequence = demo_file['data']['demo_0']['chunk_0']['obs']['joint_positions']
        if config["controller_cfg"]['is_delta'] is True:
            action_sequence = demo_file['data/demo_0/chunk_0/action']
        else:
            action_sequence = demo_file['data/demo_0/chunk_0/action_absolute']

    # Initialize franka interface
    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    # Franka Interface
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg), control_freq=control_freq) # TODO:

    # Reset to the same initial joint configuration
    logger.info("Resetting to the initial configuration")
    reset_joints_to(robot_interface, joint_sequence[0])

    # Command the same sequence of actions
    data = {"action": [], "ee_states": [], "joint_states": [], "gripper_states": []}
    if "OSC" in config["controller_type"]:
        logger.info("Start replay recorded actions using a OSC-family controller")

        for action in action_sequence:
            # TODO: for testing, add gripper, add twist later
            gripper = action[6:]
            action = np.concatenate((action[:6], np.zeros(6)))

            data['joint_states'].append(np.array(robot_interface.last_q))
            data['ee_states'].append(np.array(robot_interface._state_buffer[-1].O_T_EE))
            data['action'].append(np.array(action))

            robot_interface.control(
                controller_type=config["controller_type"],
                action=action,
                controller_cfg=EasyDict(config["controller_cfg"]),
            )

    elif config["controller_type"] == "JOINT_IMPEDANCE":
        logger.info("Start replay recorded actions using Joint Impedence Controller.")
        data['joint_states'], data["action"], data['ee_states'] = follow_joint_traj(robot_interface, joint_sequence)

    logger.info("Finish replaying.")
    robot_interface.close()

    ######### Mod to log replay traj #############
    if not args.save:
        print("Not saving the trajectory")
        return
    folder = os.path.dirname(args.dataset)
    controller_type = config["controller_type"]
    save_path = f"{folder}/{controller_type}_{control_freq}HZ_replay_trajecotry_1030.hdf5"
    with h5py.File(save_path, "w") as h5py_file: # TODO: change name accordingly
        config_dict = {
            "controller_cfg": EasyDict(config["controller_cfg"]),
            "controller_type": config["controller_type"],
        }
        grp = h5py_file.create_group("data")
        grp.attrs["config"] = json.dumps(config_dict)

        grp.create_dataset("actions", data=np.array(data["action"]))
        grp.create_dataset("ee_states", data=np.array(data["ee_states"]))
        grp.create_dataset("joint_states", data=np.array(data["joint_states"]))
        grp.create_dataset("gripper_states", data=np.array(data["gripper_states"]))
        print(f"Finish replay trajectory saving at {save_path}")
    ###############################################


if __name__ == "__main__":
    main()
