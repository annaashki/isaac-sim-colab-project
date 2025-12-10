"""
Training script with periodic visualization during evaluation.
Shows the robot every 10k steps so you can see progress!
"""
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from robot_arm_env import RobotArmPickEnv


def make_env(render=False):
    """Create and wrap the environment."""
    render_mode = "human" if render else None
    env = RobotArmPickEnv(render_mode=render_mode)
    env = Monitor(env)
    return env


def train_with_visualization(
    total_timesteps=500_000,
    eval_freq=10_000,
    save_dir="./models",
    tensorboard_log="./logs"
):
    """
    Train with periodic visualization during evaluation.
    Every eval_freq steps, you'll see the robot attempt the task!
    """
    print("🤖 Training Robot Arm with Periodic Visualization")
    print("="*60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Visualization every: {eval_freq:,} steps")
    print("\nNote: Training happens WITHOUT rendering (fast)")
    print("      Evaluation shows visualization (so you can watch)")
    print("="*60)
    print()
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_{timestamp}"
    
    # Training environment (NO rendering - fast!)
    print("Creating training environment (headless)...")
    env = DummyVecEnv([lambda: make_env(render=False)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Evaluation environment (WITH rendering - you can watch!)
    print("Creating evaluation environment (with visualization)...")
    eval_env = DummyVecEnv([lambda: make_env(render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(save_dir, run_name, "checkpoints"),
        name_prefix="rl_model",
        save_vecnormalize=True
    )
    
    # Evaluation callback - THIS WILL SHOW VISUALIZATION!
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, run_name, "best_model"),
        log_path=os.path.join(save_dir, run_name, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=3,  # Show 3 episodes each time
        deterministic=True,
        render=False  # Set to True if you want even slower but continuous rendering
    )
    
    # Create PPO model
    print("Initializing PPO model...")
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
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(tensorboard_log, run_name)
    )
    
    print("\n" + "="*60)
    print("Starting training!")
    print(f"Watch progress: tensorboard --logdir {tensorboard_log}")
    print(f"\n💡 You'll see the robot visualized every {eval_freq:,} steps")
    print("   during evaluation episodes!")
    print("="*60 + "\n")
    
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
        print("Training completed!")
        print(f"Final model: {final_model_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        interrupted_path = os.path.join(save_dir, run_name, "interrupted_model")
        model.save(interrupted_path)
        env.save(os.path.join(interrupted_path, "vec_normalize.pkl"))
        print(f"Model saved to: {interrupted_path}")
    
    finally:
        env.close()
        eval_env.close()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with periodic visualization")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Show visualization every N steps (default: 10,000)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for tensorboard logs"
    )
    
    args = parser.parse_args()
    
    train_with_visualization(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        save_dir=args.save_dir,
        tensorboard_log=args.log_dir
    )
