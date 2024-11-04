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
from scipy.spatial.transform import Rotation

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


def compute_omega_base_frame(axis_angles: np.ndarray, dt: float=1.0, max_omega: float=3.14):
    """
    Approximate angular omega from axis_angle array in base frame
    Input: axis_angle is Tx3 
    """
    num_points, coordinate = axis_angles.shape

    if coordinate!=3:
        raise ValueError("Input shape should be 3 with x,y,z")
    if num_points<3:
        raise ValueError("Traj seq too short")
    if dt <= 1e-4:
        raise ValueError("dt too small")

    # 将轴角转换为旋转矩阵
    rotation_mats = Rotation.from_rotvec(axis_angles)

    # 计算角速度
    twists = []
    for i in range(num_points):

        if i==0 or i==(num_points-1):
            omega = np.zeros(3)
        else:
            R_i = rotation_mats[i-1].as_matrix()
            R_ip1 = rotation_mats[i+1].as_matrix()
            # 计算旋转矩阵的时间导数 (forward only here)
            # TODO: how accurate it needs to be? To satisfy the R_dot@R_T + R@R_dot_T=0\
            # TODO: what if using R_{T+1}@R_T
            dR = (R_ip1 - R_i) / dt
            # 计算 omega^x = dR * R_i^T
            omega_mat = dR @ R_i.T
            # 从反对称矩阵提取角速度
            omega = np.array([
                omega_mat[2,1],
                omega_mat[0,2],
                omega_mat[1,0]
            ])

        twists.append(omega)

    print("max omega: ", np.max(np.abs(twists)))
    if np.any(np.abs(twists) > max_omega):
        raise ValueError(f"Omega is larger than {max_omega}") 

    return np.array(twists)


def compute_velocity_base_frame(traj: np.ndarray, dt: float=1.0, max_vel: float=0.8):
    """
    A simple linear velocity approximation in base_frame
    v(t) = [s(t+1)-s(t-1)] / 2

    Input x,y,z position = np.array Tx3
    max_vel: float in m/s. To limit the max vel of end pos.
    Output x,y,z velocity = np.array Tx3
    """
    vel = np.zeros_like(traj, dtype=np.float64)

    time_len, coordinate = vel.shape
    if coordinate!=3:
        raise ValueError("Input shape should be 3 with x,y,z")
    if time_len<3:
        raise ValueError("Traj seq too short")
    if dt <= 1e-4:
        raise ValueError("dt too small")

    vel[1:-1, :] = (traj[2:,:]-traj[:-2, :]) * 0.5 / dt

    print("max linear vel: ", np.max(np.abs(vel)))
    if np.any(np.abs(vel) > max_vel):
        raise ValueError(f"Velocity is larger than {max_vel}") 
    return vel


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


    ################## TEST #################
    # for action in action_sequence[:6]:
    #     print(0.05*np.array(action))
    # action_test = action_sequence
    # dt = 1/args.control_freq
    # xyz_traj, axis_angle_traj = np.array(action_sequence)[:,:3], np.array(action_sequence)[:,3:6]
    # xyz_traj = np.array([[1,2,3], [2,3,4], [4,5,6], [9,9,9]])
    # xyz_vel  = compute_velocity_base_frame(xyz_traj, dt= 1/args.control_freq, max_vel=1.5) # TODO: change freq here
    # print("xyz traj: ", xyz_traj[:10], "xzy_vel: ", xyz_vel[:10])
    # assert False

    ##### Test the angular twist calculation
    # # Parameters
    # N = 10  # Number of time steps
    # delta_t = 0.1  # Time interval in seconds
    # omega = 1.0  # Angular speed in radians per second
    # n_hat = np.array([0, 0, 1])  # Rotation axis (z-axis)
    # # Time vector
    # times = np.linspace(0, (N - 1) * delta_t, N)
    # # Initialize axis-angle array (Nx3)
    # axis_angles = np.zeros((N, 3))
    # # Generate axis-angle data
    # for i, t in enumerate(times):
    #     theta_i = omega * t
    #     axis_angles[i] = theta_i * n_hat  # Axis-angle representation
    # omega = compute_omega_base_frame(axis_angle_traj, dt=dt)
    # print(omega)
    # assert False
    # ######################################

    dt = 1/args.control_freq
    xyz_traj, axis_angle_traj = np.array(action_sequence)[:,:3], np.array(action_sequence)[:,3:6]
    linear_vel  = compute_velocity_base_frame(xyz_traj, dt= 1/args.control_freq, max_vel=1.5)
    omega = compute_omega_base_frame(axis_angle_traj, dt=dt)
    twist_base_frame = np.hstack((linear_vel, omega))

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

        counter = 0
        for action_pos, twist in zip(action_sequence, twist_base_frame.tolist()):
            # TODO: for testing, add gripper, add twist later
            gripper = action_pos[6:]
            # action = np.concatenate((action_pos[:6], np.zeros(6))) # no twist
            action = np.concatenate((action_pos[:6], twist))
            # print("actions: ", np.round(action[:9], decimals=3))

            data['joint_states'].append(np.array(robot_interface.last_q))
            data['ee_states'].append(np.array(robot_interface._state_buffer[-1].O_T_EE))
            data['action'].append(np.array(action))

            robot_interface.control(
                controller_type=config["controller_type"],
                action=action,
                controller_cfg=EasyDict(config["controller_cfg"]),
            )
            counter += 1
            # if counter > 20:
            #     break

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
    save_path = f"{folder}/{controller_type}_{control_freq}HZ_with_twist_replay_trajecotry_1103.hdf5"
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