import mujoco
import mujoco.viewer
import numpy as np
from dataclasses import dataclass
from scr.mujoco_rl_example.setup import gen_model_and_data
import asyncio

@dataclass
class Command:
    time: float
    angles: list[float]

commands: list[Command] = []
commands.append(Command(time=1, angles=[+0.00, +0.0, +0.0, +0.0, +0.0, +0.0]))
# commands.append(Command(time=2, angles=[+2.36, +0.3, +0.0, +0.0, +0.0, +0.0]))
# commands.append(Command(time=2, angles=[+2.36, -0.24, +1.3, +1.7, +0.8, +0.026]))



async def main():
    """Launch interactive MuJoCo simulation with slider controls."""
    
    # Load model from XML string
    model, data = gen_model_and_data()
    
    # Initialize control values to zeros (neutral position)
    data.ctrl[:] = 0.0
    
    # Launch the interactive viewer with built-in sliders
    # The viewer automatically creates sliders for all actuators
    current_command = 0 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            # Step the simulation forward
            mujoco.mj_step(model, data)
            viewer.sync()
            await asyncio.sleep(model.opt.timestep)
            # Print current control targets (angles) in real time
            print(f"Time: {data.time:.2f}s | Control targets: {data.ctrl}")
            # After 10 seconds, set joint angles to example values
            if current_command < len(commands):
                command = commands[current_command]
                if data.time >= command.time:
                    print(f"Setting joint angles to command at t={command.time}s: {command.angles}")
                    data.ctrl[:] = np.array(command.angles)
                    current_command += 1
                    mujoco.mj_forward(model, data)
            # Sync the viewer (updates visualization)
            viewer.sync()

if __name__ == "__main__":
    asyncio.run(main())