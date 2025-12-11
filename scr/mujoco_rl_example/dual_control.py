import mujoco
import mujoco.viewer
import numpy as np
import time
from scr.mujoco_rl_example.setup import gen_model_and_data
import asyncio

import xarm

arm = xarm.Controller("USB")
print("Battery voltage in volts:", arm.getBatteryVoltage())

SERVOS_LIST = [6,5,4,3,2,1]
SERVO_PARITY = np.array([1,1,-1,1,1,-(1.57/0.03)])

def move_arm_to_angles(angles):
    """Move the xArm to the specified joint angles (in radians)."""
    # Convert radians to degrees for xArm
    angles = SERVO_PARITY * angles
    angles_degrees = [np.degrees(angle) for angle in angles]
    zipped_angles = list(zip(SERVOS_LIST, angles_degrees))
    arm.setPosition(zipped_angles, wait=False)

async def main():

    message_gap_s = 0.5
    last_message_time = time.time()
    """Launch interactive MuJoCo simulation with slider controls."""
    
    # Load model from XML string
    model, data = gen_model_and_data()
    
    # Initialize control values to zeros (neutral position)
    data.ctrl[:] = 0.0
    
    # Launch the interactive viewer with built-in sliders
    # The viewer automatically creates sliders for all actuators
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        prev_data_control = None
        while viewer.is_running():
            # Step the simulation forward
            mujoco.mj_step(model, data)
            viewer.sync()

            if time.time() - last_message_time > message_gap_s:
                print(f"Time: {data.time:.2f}s | Control targets: {data.ctrl}, {type(data.ctrl)}")
                if prev_data_control is None or not np.array_equal(prev_data_control, data.ctrl):
                    move_arm_to_angles(data.ctrl)
                    last_message_time = time.time()
                    prev_data_control = data.ctrl.copy()
    
            await asyncio.sleep(model.opt.timestep)

            print(f"Time: {data.time:.2f}s | Control targets: {data.ctrl}")
            viewer.sync()

if __name__ == "__main__":
    asyncio.run(main())