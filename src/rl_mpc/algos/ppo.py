from __future__ import annotations

import math
from typing import Literal

import chz
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import trange

from rl_mpc.components.env import EnvConfig, make_envs
from rl_mpc.components.policy import PolicyConfig, build_policy
from rl_mpc.components.value import ValueConfig, build_value
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
    wandb: bool = False
    wandb_project: str = "rl-mpc"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rollout_steps, num_envs = rewards.shape
    advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
    last_adv = np.zeros(num_envs, dtype=np.float32)
    for t in reversed(range(rollout_steps)):
        next_values = last_values if t == rollout_steps - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


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
    params = list(policy.parameters()) + list(value.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    steps_per_update = cfg.rollout_steps * num_envs
    num_updates = max(1, math.ceil(cfg.total_steps / steps_per_update))

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lens = np.zeros(num_envs, dtype=np.int32)
    episode = 0
    global_step = 0

    pbar = trange(num_updates, desc="PPO")
    for update in pbar:
        obs_buf = np.zeros((cfg.rollout_steps, num_envs, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((cfg.rollout_steps, num_envs), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((cfg.rollout_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((cfg.rollout_steps, num_envs), dtype=np.float32)
        values_buf = np.zeros((cfg.rollout_steps, num_envs), dtype=np.float32)
        completed_returns: list[float] = []
        completed_lens: list[int] = []

        for t in range(cfg.rollout_steps):
            obs_batch = obs_array if obs_array.ndim == 2 else obs_array[None, :]
            obs_buf[t] = obs_batch

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
                logits = policy(obs_tensor)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = value(obs_tensor).squeeze(-1)

            actions_np = actions.cpu().numpy()
            logp_buf[t] = log_probs.cpu().numpy()
            values_buf[t] = values.cpu().numpy()
            actions_buf[t] = actions_np

            next_obs, rewards, terminals, truncateds, _ = envs.step(actions_np)
            next_obs_array = np.asarray(next_obs)
            if next_obs_array.ndim == 1:
                next_obs_array = next_obs_array[None, :]
            rewards_array = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(terminals, dtype=bool) | np.asarray(truncateds, dtype=bool)

            rewards_buf[t] = rewards_array
            dones_buf[t] = dones.astype(np.float32)

            episode_returns += rewards_array
            episode_lens += 1
            for i in range(num_envs):
                if dones[i]:
                    episode += 1
                    print(
                        f"episode={episode} update={update + 1} return={episode_returns[i]:.1f} "
                        f"len={episode_lens[i]}"
                    )
                    completed_returns.append(float(episode_returns[i]))
                    completed_lens.append(int(episode_lens[i]))
                    episode_returns[i] = 0.0
                    episode_lens[i] = 0

            obs_array = next_obs_array
            global_step += num_envs

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
            last_values = value(obs_tensor).squeeze(-1).cpu().numpy()

        advantages, returns = _compute_gae(
            rewards_buf,
            dones_buf,
            values_buf,
            last_values,
            cfg.gamma,
            cfg.gae_lambda,
        )

        obs_flat = obs_buf.reshape(-1, obs_dim)
        actions_flat = actions_buf.reshape(-1)
        logp_flat = logp_buf.reshape(-1)
        adv_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        if cfg.normalize_advantage and adv_flat.size > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        total_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        batch_size_total = obs_flat.shape[0]
        for _epoch in range(cfg.n_epochs):
            indices = rng.permutation(batch_size_total)
            for start in range(0, batch_size_total, cfg.batch_size):
                mb_idx = indices[start:start + cfg.batch_size]
                if mb_idx.size == 0:
                    continue

                obs_tensor = torch.as_tensor(obs_flat[mb_idx], dtype=torch.float32, device=device)
                actions_tensor = torch.as_tensor(actions_flat[mb_idx], dtype=torch.int64, device=device)
                old_logp_tensor = torch.as_tensor(logp_flat[mb_idx], dtype=torch.float32, device=device)
                adv_tensor = torch.as_tensor(adv_flat[mb_idx], dtype=torch.float32, device=device)
                returns_tensor = torch.as_tensor(returns_flat[mb_idx], dtype=torch.float32, device=device)

                logits = policy(obs_tensor)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp_tensor)
                pg_loss_1 = ratio * adv_tensor
                pg_loss_2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_tensor
                policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

                values_pred = value(obs_tensor).squeeze(-1)
                value_loss = 0.5 * (returns_tensor - values_pred).pow(2).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kls.append((old_logp_tensor - logp).mean().item())
                    clip_fracs.append((torch.abs(ratio - 1.0) > cfg.clip_range).float().mean().item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                total_losses.append(loss.item())

        if policy_losses:
            mean_policy_loss = float(np.mean(policy_losses))
            mean_value_loss = float(np.mean(value_losses))
            mean_entropy = float(np.mean(entropy_losses))
            mean_total_loss = float(np.mean(total_losses))
            mean_approx_kl = float(np.mean(approx_kls))
            mean_clip_frac = float(np.mean(clip_fracs))
            pbar.set_postfix(
                policy_loss=f"{mean_policy_loss:.3f}",
                value_loss=f"{mean_value_loss:.3f}",
                entropy=f"{mean_entropy:.3f}",
                approx_kl=f"{mean_approx_kl:.3f}",
                clip_frac=f"{mean_clip_frac:.3f}",
            )

            if wandb_run is not None:
                values_flat = values_buf.reshape(-1)
                returns_flat = returns.reshape(-1)
                returns_var = float(np.var(returns_flat))
                explained_variance = float("nan")
                if returns_var > 1e-8:
                    explained_variance = 1.0 - float(np.var(returns_flat - values_flat)) / returns_var

                log_data = {
                    "train/policy_gradient_loss": mean_policy_loss,
                    "train/value_loss": mean_value_loss,
                    "train/entropy_loss": -mean_entropy,
                    "train/approx_kl": mean_approx_kl,
                    "train/clip_fraction": mean_clip_frac,
                    "train/loss": mean_total_loss,
                    "train/explained_variance": explained_variance,
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

    pbar.close()
    envs.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train(chz.entrypoint(PPOConfig))
