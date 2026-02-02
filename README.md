# rl-mpc

Minimal baselines for CartPole-v1 using PufferLib vector API + PyTorch:
- DQN
- PPO (on-policy, GAE)
- Random-shooting MPC with known dynamics
- Model-based MPC with learned dynamics + stochastic policy + value

## Setup

pufferlib requires gcc (not clang) to build from source:

```bash
CC=gcc CXX=g++ uv sync
```

Or use the Makefile:

```bash
make sync
```

## Run

Arguments use `key=value` syntax (chz). Nested fields use dotted paths.

DQN (vectorized envs supported):
```bash
python -m rl_mpc.algos.dqn total_steps=200000 env.num_envs=8 env.vec_backend=multiprocessing
```

PPO (on-policy, GAE):
```bash
python -m rl_mpc.algos.ppo total_steps=200000 env.num_envs=4 env.vec_backend=multiprocessing
```

PPO with W&B logging:
```bash
python -m rl_mpc.algos.ppo total_steps=200000 env.num_envs=4 wandb=true wandb_project=rl-mpc
```

Video logging (works for all in-repo algos via Gymnasium `RecordVideo`):
```bash
python -m rl_mpc.algos.ppo total_steps=200000 video.dir=videos/ppo video.every_steps=20000 video.episodes=1
```

MPC (random-shooting over action sequences):
```bash
python -m rl_mpc.algos.mpc episodes=10 planner.horizon=15 planner.num_sequences=1000
```

Model-based MPC (learned dynamics + policy + value; MPC samples from policy):
```bash
python -m rl_mpc.algos.mbmpc total_steps=100000 planner.horizon=15 planner.num_sequences=256 env.num_envs=4
```

## Stable-Baselines3 baseline

Use the lightweight Stable-Baselines3 baseline script (Gymnasium-based):
```bash
python scripts/sb3_cartpole.py algo=dqn total_timesteps=200000
python scripts/sb3_cartpole.py algo=ppo total_timesteps=200000 num_envs=4
```

Optional logging and saving:
```bash
python scripts/sb3_cartpole.py algo=ppo log_dir=runs/sb3 save_path=sb3_cartpole
```

W&B logging (syncs TensorBoard; defaults `log_dir` to `runs/sb3`):
```bash
python scripts/sb3_cartpole.py algo=ppo total_timesteps=200000 wandb=true wandb_project=rl-mpc log_dir=runs/sb3
```

Visualize a few trajectories after training:
```bash
python scripts/sb3_cartpole.py algo=ppo total_timesteps=50000 render_episodes=3 render_delay_ms=20
```

Record periodic videos during training (to see learning progress):
```bash
python scripts/sb3_cartpole.py algo=ppo total_timesteps=200000 video_dir=videos/sb3 video_every_steps=20000 video_episodes=1
```

Note: video recording uses Gymnasium's `RecordVideo` wrapper; you may need `imageio` and `imageio-ffmpeg` installed.

Streamlit viewer (stacks videos vertically):
```bash
streamlit run scripts/streamlit_videos.py
```

## Notes
- PufferLib wraps a Gymnasium CartPole-v1 env; env stepping runs on CPU, while networks can run on GPU via `device=cuda`.
- MPC uses the CartPole dynamics consistent with Gymnasium's implementation for rollouts.
