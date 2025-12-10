# Quick Start Guide

## Installation

Install MuJoCo (works with Python 3.13):

```bash
pip install mujoco
```

## Running the Simulation

```bash
python robot_arm_simulation.py
```

A viewer window should open showing the robot arm on a checkered ground plane.

## Keyboard Controls

### Joint Control
- **Q/A** - Base rotation (spin the entire arm)
- **W/S** - Elbow joint 1
- **E/D** - Elbow joint 2  
- **R/F** - Elbow joint 3
- **T/G** - Gripper rotation
- **Y/H** - Gripper open/close

### View Control (Mouse)
- **Left-click + drag** - Rotate camera view
- **Right-click + drag** - Pan camera
- **Scroll wheel** - Zoom in/out

### Exit
- **ESC** - Close simulation

## Tips

1. Start by rotating the base (Q/A keys) to see the arm spin
2. Try combining movements for complex poses
3. The gripper fingers move symmetrically (Y opens, H closes)
4. All movements are smooth with damping for realistic motion
5. The physics simulation runs at 500 Hz for accuracy

## Troubleshooting

**No window appears:**
- Check that you installed mujoco: `pip show mujoco`
- Make sure you're using the correct Python environment

**Controls not working:**
- Click on the viewer window to ensure it has focus
- Use UPPERCASE letters (Q, not q)

**Simulation too fast/slow:**
- Modify the `time.sleep(0.002)` value in the script
- Lower values = faster, higher values = slower
