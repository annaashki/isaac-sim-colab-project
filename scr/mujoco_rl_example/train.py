"""
Training script for robot arm pick-and-place using Reinforcement Learning.
Uses Stable-Baselines3 with PPO algorithm.
"""
import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmPickEnv


def make_env():
    """Create and wrap the environment."""
    env = RobotArmPickEnv(render_mode=None)
    env = Monitor(env)
    return env


def train_rl_agent(
    algorithm="PPO",
    total_timesteps=500_000,
    save_dir="./models",
    tensorboard_log="./logs"
):
    """
    Train an RL agent to pick up the cube.
    
    Args:
        algorithm: "PPO" or "SAC" (PPO is recommended for beginners)
        total_timesteps: Total number of timesteps to train
        save_dir: Directory to save trained models
        tensorboard_log: Directory for tensorboard logs
    """
    print(f"Starting {algorithm} training for robot arm pick-and-place task...")
    print(f"Total timesteps: {total_timesteps:,}")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_{timestamp}"
    
    # Use 8 parallel environments for training and evaluation
    num_envs = 8
    print(f"\nCreating training environment with {num_envs} parallel envs...")
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    print(f"Creating evaluation environment with {num_envs} parallel envs...")
    eval_env = DummyVecEnv([make_env for _ in range(num_envs)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for evaluation
        clip_obs=10.0,
        training=False  # Don't update stats during evaluation
    )
    
    # Setup callbacks
    # Save model every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(save_dir, run_name, "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    # Evaluate model every 10k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, run_name, "best_model"),
        log_path=os.path.join(save_dir, run_name, "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Create the RL model
    print(f"\nInitializing {algorithm} model...")
    
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            verbose=1,
            tensorboard_log=os.path.join(tensorboard_log, run_name)
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=os.path.join(tensorboard_log, run_name)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train the model
    print("\n" + "="*60)
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Model checkpoints will be saved to: {os.path.join(save_dir, run_name)}")
    print(f"Tensorboard logs: {os.path.join(tensorboard_log, run_name)}")
    print("="*60 + "\n")
    
    print("To monitor training with TensorBoard, run:")
    print(f"  tensorboard --logdir {tensorboard_log}")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(save_dir, run_name, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(final_model_path, "vec_normalize.pkl"))
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Best model saved to: {os.path.join(save_dir, run_name, 'best_model')}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_path = os.path.join(save_dir, run_name, "interrupted_model")
        model.save(interrupted_path)
        env.save(os.path.join(interrupted_path, "vec_normalize.pkl"))
        print(f"Model saved to: {interrupted_path}")
    
    finally:
        env.close()
        eval_env.close()
    
    return model, env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent for robot arm pick-and-place")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm to use (default: PPO)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total timesteps to train (default: 500,000)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for tensorboard logs (default: ./logs)"
    )
    
    args = parser.parse_args()
    
    # Train the agent
    train_rl_agent(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        tensorboard_log=args.log_dir
    )
