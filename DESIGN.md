# Composable Parts Design (rl-mpc)

This document proposes an extensible, concise architecture that keeps the
project "minimal baselines" while making components interchangeable.

## Goals
- Make MBMPC and MPC composed from reusable parts: policy + dynamics + value + planner.
- Keep code small, easy to read, and runnable via `python -m ...`.
- Avoid heavyweight frameworks; stick to simple PyTorch + NumPy.

## Non-goals
- A full RL framework or dependency injection system.
- Multi-env generalization beyond discrete action Gymnasium tasks.

## High-level Architecture
Each algorithm wires together a small set of components:

- `EnvFactory`: creates vectorized environments.
- `Policy`: maps obs -> action distribution (or logits).
- `Dynamics`: maps (obs, action) -> next_obs (and optional done prediction).
- `Value`: maps obs -> scalar value estimate.
- `Planner`: chooses an action by rolling out a model (optional).
- `ReplayBuffer`: stores transitions for off-policy training.
- `Trainer`: updates components (policy/value/dynamics) from data.

Algorithms are just compositions of these pieces with different wiring.

## Core Interfaces (minimal, informal)

### Policy
```
logits = policy(obs: Tensor) -> Tensor  # shape: [B, A]
```

### Dynamics
```
next_obs = dynamics(obs_action: Tensor) -> Tensor  # obs_action shape: [B, O+A]
```

### Value
```
value = value(obs: Tensor) -> Tensor  # shape: [B, 1]
```

### Planner
```
action = planner.act(
    obs: np.ndarray,
    policy: Policy | None,
    dynamics: Dynamics,
    value: Value | None,
    cfg: PlannerConfig,
    rng: Generator,
) -> int
```

### Trainer
```
trainer.update(batch: dict[str, Tensor]) -> dict[str, float]
```

The interfaces are intentionally small and easy to satisfy with simple MLPs.

## Data Flow
1. `EnvFactory` produces env(s) and initial obs.
2. `Planner` (or policy directly) selects actions.
3. Env step produces transitions; transitions go into `ReplayBuffer`.
4. `Trainer` samples from buffer and updates `policy`, `value`, `dynamics`.

## Composition Diagram
```
      +-----------+        +-----------+        +----------------+
      |   Env(s)  | -----> | Collector | -----> |  Batch (typed) |
      +-----------+        +-----------+        +----------------+
                                   |                      |
                                   v                      v
                               +---------+          +-----------+
                               | Policy  |          |  Trainer  |
                               +---------+          +-----------+
                                   |                      |
                                   v                      v
                              +--------+            +-----------+
                              | Value  |            | Metrics   |
                              +--------+            +-----------+
                                   |
                                   v
                              +----------+
                              | Dynamics |
                              +----------+

Notes:
- Batches are BaseModels (e.g., `RolloutBatch`, `TransitionBatch`) for type safety.
- Trainers return BaseModel metrics; logging consumes `model_dump()`.
- Objectives are additive: total loss = weighted sum of component losses.

## Streaming Collector/Packer/Batcher
The training loop pulls fixed-length streams of steps and turns them into
typed batches for optimization:

```
Trainer (request K steps)
        |
        v
Collector -> StepBatch stream (per step, per env)
        |
        v
Packer -> RolloutBuffer (time-major) + last_obs
        |
        v
Batcher -> RolloutBatch minibatches (flat, shuffled)
```

This mirrors LLM-style data pipelines: *steps* are streamed, *packed* into
fixed-length rollouts, then *batched* for SGD. Episode boundaries are carried
via `done` flags, so rollouts can span multiple episodes safely.

## Unified Data Source (Replay Buffer + Stream)
A single abstraction can serve both on-policy streaming and off-policy replay:

```
            +------------------+
Trainer ---> |   DataSource     | -----> Batcher ----> Trainer.update(...)
            +------------------+
               ^            ^
               |            |
        StreamSource     ReplaySource
        (live steps)     (replay buffer)
```

- **StreamSource** wraps the collector/packer pipeline and yields fresh rollouts.
- **ReplaySource** wraps a replay buffer and yields sampled batches (or packed
  sequences if needed).

This keeps training loops identical across algorithms: they simply ask a
DataSource for the next batch/rollout and then update components.
```

## Example Composition

### Random-shooting MPC (known dynamics)
- policy: None
- dynamics: known analytic model
- value: None
- planner: random-shooting over action sequences
- trainer: None (no learning)

### MBMPC (learned dynamics + policy + value)
- policy: MLP
- dynamics: MLP
- value: MLP
- planner: policy-guided sampling with value bootstrap
- trainer: joint updates for policy/dynamics/value

### DQN (off-policy)
- policy: Q-network (acts as policy via argmax)
- dynamics: None
- value: None
- planner: None
- trainer: DQN update on Q-network

## Suggested Module Layout
```
src/rl_mpc/
  components/
    policy.py
    dynamics.py
    value.py
    planner.py
    trainer.py
  algos/
    dqn.py
    ppo.py
    mpc.py
    mbmpc.py
  envs.py
  replay_buffer.py
  utils.py
```

Each algo module should only be wiring + loop, no deep logic.

## Configuration
Use small dataclasses per component and a top-level config for each algo:

```
@dataclass
class MBMPCConfig:
    policy: PolicyConfig
    dynamics: DynamicsConfig
    value: ValueConfig
    planner: PlannerConfig
    trainer: TrainerConfig
    env: EnvConfig
```

This keeps parameters grouped and makes swapping parts explicit.

## Extensibility Examples
- Swap the planner: random shooting -> CEM -> policy-guided.
- Swap the dynamics: MLP -> ensemble -> analytic model.
- Swap the policy loss: behavioral cloning -> advantage-weighted.

## Diffusion Planner as Variational Inference
A diffusion planner can be framed as inference over action sequences:

```
p(tau | s0) ∝ p0(tau | s0) * exp(beta * R(tau))
```

Where `tau` is an action sequence, `p0` is a diffusion prior, and `R(tau)` is
the return from rolling out dynamics. The planning objective becomes:

```
min_q KL(q(tau|s0) || p(tau|s0))
= -beta * E_q[R(tau)] + KL(q(tau|s0) || p0(tau|s0)) + const
```

**Composable pieces**
- `TrajectoryModel` (diffusion denoiser) defines `q(tau|s0)`.
- `Dynamics` rolls out `tau` to compute `R(tau)`.
- Optional `Value` provides terminal bootstrap for truncated horizons.
- `Planner` runs reverse diffusion with reward/value guidance.

**Minimal interfaces**
```
actions = diffusion.sample(obs, horizon, guidance_fn)
reward = rollout_reward(obs, actions, dynamics, value=None)
action = diffusion_planner.act(obs, diffusion, dynamics, value, cfg)
```

This plugs into the same `Planner` slot as MPC/MCTS: only the sampler differs.

## Minimal Implementation Guidance
- Keep policies and values as simple MLPs (use existing `build_mlp`).
- Provide a tiny `Planner` class for random shooting and policy-guided sampling.
- Provide a tiny `Trainer` class for MBMPC (three losses, three optimizers).
- Keep the main training loop in `algos/mbmpc.py` readable and linear.

## Notes on Conciseness
- A strict cap of ~150 lines per algo keeps the baseline feel.
- Shared helpers belong in `components/` or `utils.py`, not embedded in loops.
