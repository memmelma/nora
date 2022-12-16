import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import termios
import tty

import pybullet as p

from physics_env import PushEnv

os.environ["PYBULLET_CONNECTION_MODE"] = "gui"
os.environ["PYBULLET_RENDERER"] = "egl"

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def run_robot_env():
    # env = DummyEnv()
    env = PushEnv(assets_root="/home/weirdlab/projects/pybullet/pomdp-baselines/envs/tabletop/assets")
    obs = env.reset()
    time = 0
    while True:
        action = np.zeros(5, np.float32)
        ch = getch()

        if ch == "s":
            action[0] = 1.0
        elif ch == "w":
            action[0] = -1.0
        elif ch == "d":
            action[1] = 1.0
        elif ch == "a":
            action[1] = -1.0
        elif ch == "e":
            action[2] = 1.0
        elif ch == "q":
            action[2] = -1.0
        elif ch == "z":
            action[3] = 1.0
        elif ch == "x":
            action[3] = -1.0
        elif ch == "c":
            action[4] = 1.0
        elif ch == "f":
            action[4] = -1.0
        elif ch == "r":
            env.reset()
        elif ch == "/":
            break
        print(action)
        for block in env.blocks:
            print(block, p.getBaseVelocity(block))
        for joint in env.robot.joints:
            p.enableJointForceTorqueSensor(env.robot.body, joint, 1)
        print(p.getJointStates(env.robot.body, env.robot.joints))
        obs, reward, done, info = env.step(action)
        # plt.imshow(obs)
        # plt.show(block=False)
        # plt.pause(0.01)
        print(f"reward: {reward}, success: {info['success']}")
        # joints = [
        #     np.round((p.getJointState(env.robot.body, j)[0] / np.pi), 6)
        #     for j in env.robot.joints
        # ]
        # print(f"joint pos: {joints}")
        print(f"state: {info['state_obs']}")
        print(f"time {time}")
        time += 1
    env.close()


if __name__ == "__main__":
    run_robot_env()
