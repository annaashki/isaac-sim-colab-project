# MuJoCo 6-Axis Robot Arm Simulation

This example demonstrates a 6-axis robot arm simulation using MuJoCo, a fast and accurate physics engine.

## Robot Configuration

The robot arm has 6 joints with the following configuration:

1. **Joint 1 (Gripper)**: Prismatic joints that open/close the gripper fingers
2. **Joint 2 (Gripper Rotation)**: Rotates the gripper around the Y-axis
3. **Joint 3 (Elbow 1)**: First elbow joint for arm positioning
4. **Joint 4 (Elbow 2)**: Second elbow joint for arm positioning
5. **Joint 5 (Elbow 3)**: Third elbow joint for arm positioning
6. **Joint 6 (Base Rotation)**: Spins the entire arm around the Z-axis

## Installation

MuJoCo is easy to install and works with Python 3.13:

```bash
pip install mujoco
```

## Running the Simulation

Run the simulation with:

```bash
python robot_arm_simulation.py
```

## Controls

Control the robot arm using your keyboard:

- **Q/A**: Base rotation (spin the arm)
- **W/S**: Elbow 1 (first joint)
- **E/D**: Elbow 2 (second joint)
- **R/F**: Elbow 3 (third joint)
- **T/G**: Gripper rotation
- **Y/H**: Gripper open/close
- **ESC**: Exit the simulation

The viewer also supports mouse controls:
- **Left-click + drag**: Rotate view
- **Right-click + drag**: Pan view
- **Scroll**: Zoom in/out

## Features

- Fast and accurate physics simulation with MuJoCo
- Interactive keyboard control for all joints
- Realistic rendering with lighting and shadows
- Color-coded links for easy visualization
- Smooth position control with damping
- Ground plane with checker pattern

## Customization

The robot model is defined in XML format within the script. You can customize:

- **Geometry**: Change geom sizes and types (cylinder, box, sphere, etc.)
- **Colors**: Modify rgba values for each link
- **Joint limits**: Adjust range parameters in joint definitions
- **Mass properties**: Change mass and inertial values
- **Control gains**: Modify kp (stiffness) values in actuators
- **Damping**: Adjust damping values for smoother/stiffer motion

## Why MuJoCo?

- Modern physics engine with excellent Python support
- Works perfectly with Python 3.13
- Faster and more accurate than older alternatives
- Used extensively in robotics research and RL
- Free and open source (as of 2022)
