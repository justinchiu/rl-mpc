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

## CleanRL comparison

Exact CleanRL single-file scripts are vendored:
- `third_party/cleanrl/v1.0.0/dqn.py`
- `third_party/cleanrl/v1.0.0/ppo.py`

They use argparse (`--flag` syntax).

Examples:
```bash
python third_party/cleanrl/v1.0.0/dqn.py --env-id CartPole-v1 --total-timesteps 500000
python third_party/cleanrl/v1.0.0/ppo.py --env-id CartPole-v1 --total-timesteps 500000
```

Dependencies for the CleanRL script (in addition to this repo's deps):
- `gymnasium`
- `stable-baselines3`
- `tensorboard` (and optional `wandb` if `--track`)

Quick install:
```bash
uv pip install gymnasium stable-baselines3 tensorboard
```

Integration tests (short runs on CartPole):
```bash
./scripts/test_cleanrl.sh
```

You can override defaults:
```bash
ENV_ID=CartPole-v1 TOTAL_TIMESTEPS=4096 ./scripts/test_cleanrl.sh
```

## Notes
- PufferLib wraps a Gymnasium CartPole-v1 env; env stepping runs on CPU, while networks can run on GPU via `device=cuda`.
- MPC uses the CartPole dynamics consistent with Gymnasium's implementation for rollouts.
