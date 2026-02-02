# rl-mpc

Minimal baselines for CartPole-v1 using PufferLib vector API + PyTorch:
- DQN
- Policy Gradient (REINFORCE)
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

Policy Gradient:
```bash
python -m rl_mpc.algos.pg total_episodes=1000 env.num_envs=4 env.vec_backend=multiprocessing
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

Visualize a few trajectories after training:
```bash
python scripts/sb3_cartpole.py algo=ppo total_timesteps=50000 render_episodes=3 render_delay_ms=20
```

## Notes
- PufferLib wraps a Gymnasium CartPole-v1 env; env stepping runs on CPU, while networks can run on GPU via `device=cuda`.
- MPC uses the CartPole dynamics consistent with Gymnasium's implementation for rollouts.
