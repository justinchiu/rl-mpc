from __future__ import annotations

import math
from typing import Literal

import chz
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import trange

from rl_mpc.components.env import EnvConfig, make_envs
from rl_mpc.components.policy import PolicyConfig, build_policy
from rl_mpc.components.ppo import PPOTrainer
from rl_mpc.components.rollout import RolloutBuffer, explained_variance
from rl_mpc.components.value import ValueConfig, build_value
from rl_mpc.components.video import VideoConfig, VideoLogger
from rl_mpc.utils import get_device, set_seed


@chz.chz
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    lr: float = 3e-4
    normalize_advantage: bool = True
    max_grad_norm: float = 10.0
    seed: int = 0
    device: str = "auto"
    env: EnvConfig = EnvConfig()
    policy: PolicyConfig = PolicyConfig()
    value: ValueConfig = ValueConfig()
    video: VideoConfig = VideoConfig()
    wandb: bool = False
    wandb_project: str = "rl-mpc"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


def train(cfg: PPOConfig) -> None:
    device = get_device(cfg.device)
    rng = set_seed(cfg.seed)

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is enabled but not installed. Add it to dependencies or run "
                "`uv add wandb`."
            ) from exc
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            config=chz.asdict(cfg),
            mode=cfg.wandb_mode,
        )

    envs = make_envs(cfg.env)
    obs, _ = envs.reset(seed=cfg.seed)
    obs_array = np.asarray(obs)
    if obs_array.ndim == 1:
        obs_array = obs_array[None, :]
    num_envs = obs_array.shape[0]
    obs_dim = int(np.prod(obs_array.shape[1:]))

    driver_env = getattr(envs, "driver_env", envs)
    action_space = getattr(driver_env, "single_action_space", getattr(driver_env, "action_space", None))
    if action_space is None or not hasattr(action_space, "n"):
        raise RuntimeError("Unable to determine discrete action space size")
    n_actions = int(action_space.n)

    policy = build_policy(obs_dim, n_actions, cfg.policy).to(device)
    value = build_value(obs_dim, cfg.value).to(device)
    trainer = PPOTrainer(
        policy=policy,
        value=value,
        lr=cfg.lr,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=device,
    )

    steps_per_update = cfg.rollout_steps * num_envs
    num_updates = max(1, math.ceil(cfg.total_steps / steps_per_update))

    def act_fn(obs: np.ndarray) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor)
            if cfg.video.deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())
        return action

    video_logger = VideoLogger(cfg.env.env_id, cfg.video)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lens = np.zeros(num_envs, dtype=np.int32)
    global_step = 0

    rollout = RolloutBuffer(cfg.rollout_steps, num_envs, obs_dim)

    pbar = trange(num_updates, desc="PPO")
    for update in pbar:
        completed_returns: list[float] = []
        completed_lens: list[int] = []

        for t in range(cfg.rollout_steps):
            obs_batch = obs_array if obs_array.ndim == 2 else obs_array[None, :]
            obs_batch = obs_batch.reshape(num_envs, obs_dim)

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
                logits = policy(obs_tensor)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = value(obs_tensor).squeeze(-1)

            actions_np = actions.cpu().numpy()

            next_obs, rewards, terminals, truncateds, _ = envs.step(actions_np)
            next_obs_array = np.asarray(next_obs)
            if next_obs_array.ndim == 1:
                next_obs_array = next_obs_array[None, :]
            rewards_array = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

            rollout.store(
                t,
                obs_batch,
                actions_np,
                log_probs.cpu().numpy(),
                rewards_array,
                dones.astype(np.float32),
                values.cpu().numpy(),
            )

            episode_returns += rewards_array
            episode_lens += 1
            for i in range(num_envs):
                if dones[i]:
                    completed_returns.append(float(episode_returns[i]))
                    completed_lens.append(int(episode_lens[i]))
                    episode_returns[i] = 0.0
                    episode_lens[i] = 0

            obs_array = next_obs_array
            global_step += num_envs

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
            last_values = value(obs_tensor).squeeze(-1).cpu().numpy()

        advantages, returns = rollout.compute_advantages(
            last_values,
            cfg.gamma,
            cfg.gae_lambda,
        )
        batch = rollout.flatten(advantages, returns)
        metrics = trainer.update(
            batch=batch,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            normalize_advantage=cfg.normalize_advantage,
            rng=rng,
        )
        explained = explained_variance(batch.returns, batch.values)

        mean_policy_loss = metrics.policy_loss
        mean_value_loss = metrics.value_loss
        mean_entropy = metrics.entropy
        mean_total_loss = metrics.loss
        mean_approx_kl = metrics.approx_kl
        mean_clip_frac = metrics.clip_frac
        if mean_policy_loss or mean_value_loss or mean_total_loss:
            pbar.set_postfix(
                policy_loss=f"{mean_policy_loss:.3f}",
                value_loss=f"{mean_value_loss:.3f}",
                entropy=f"{mean_entropy:.3f}",
                approx_kl=f"{mean_approx_kl:.3f}",
                clip_frac=f"{mean_clip_frac:.3f}",
            )

            if wandb_run is not None:
                log_data = {
                    "train/policy_gradient_loss": mean_policy_loss,
                    "train/value_loss": mean_value_loss,
                    "train/entropy_loss": -mean_entropy,
                    "train/approx_kl": mean_approx_kl,
                    "train/clip_fraction": mean_clip_frac,
                    "train/loss": mean_total_loss,
                    "train/explained_variance": explained,
                    "train/learning_rate": cfg.lr,
                    "train/n_updates": update + 1,
                    "train/clip_range": cfg.clip_range,
                    "time/iterations": update + 1,
                    "time/total_timesteps": global_step,
                }
                if completed_returns:
                    log_data["rollout/ep_rew_mean"] = float(np.mean(completed_returns))
                if completed_lens:
                    log_data["rollout/ep_len_mean"] = float(np.mean(completed_lens))
                wandb.log(log_data, step=global_step)

        video_logger.maybe_record(global_step, act_fn)

    pbar.close()
    envs.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train(chz.entrypoint(PPOConfig))
