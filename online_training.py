"""Online gait generation and training loop for the VAE-based controller.

This script streams trajectories from the walking policy back into the
autoencoder so it can adapt while the robot is exploring.  The workflow is:

1. Warm up the replay buffer with random gaits to avoid a cold start.
2. Sample latent codes from the autoencoder to decode nominal gaits.
3. Perturb each decoded gait with Gaussian noise so the robot keeps exploring.
4. Execute each gait in the environment, collect rewards, and immediately
   update the autoencoder using the latest trajectory (online update).
5. Maintain a replay buffer so the network rehearses high-reward motions while
   still prioritising fresh experience.

Usage:

```
python online_training.py \
    --config config/simpleAE.yaml \
    --iterations 50 \
    --perturbations 4
```

The script stores training logs, the best gait that was discovered, and the
updated model weights under `results/online_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import gym
import torch
import yaml

import ant_env.environment  # noqa: F401 - registers CustomAnt-v3 on import
from AE.simple_autoencoder import BasicAutoencoder
from generate_data import evaluate_control_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online training loop for gait discovery")
    parser.add_argument("--config", type=str, default="config/simpleAE.yaml", help="Path to the model configuration YAML")
    parser.add_argument("--env", type=str, default="CustomAnt-v3", help="Gym environment id to use for rollouts")
    parser.add_argument("--config-key", type=str, default="AntEnv_v3", help="Configuration key inside the YAML file")
    parser.add_argument("--iterations", type=int, default=50, help="Number of online optimisation iterations")
    parser.add_argument("--perturbations", type=int, default=4, help="Number of perturbations per nominal gait")
    parser.add_argument("--warmup-rollouts", type=int, default=32, help="Random rollouts to seed the replay buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for replay updates")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Replay buffer capacity")
    parser.add_argument("--updates-per-iter", type=int, default=2, help="Replay updates to run after each iteration")
    parser.add_argument("--perturb-std", type=float, default=0.1, help="Standard deviation for Gaussian gait perturbations")
    parser.add_argument("--max-action", type=float, default=1.0, help="Clamp actions within [-max_action, max_action]")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (auto-detected if omitted)")
    return parser.parse_args()


def load_config(path: str, key: str) -> Dict:
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    if key not in config:
        raise KeyError(f"Configuration key '{key}' not found in {path}")
    return config[key]


@dataclass
class ReplayBuffer:
    capacity: int
    buffer: Deque[Tuple[torch.Tensor, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.capacity)

    def add(self, controls: torch.Tensor, reward: float) -> None:
        self.buffer.append((controls.detach().clone(), float(reward)))

    def sample(self, batch_size: int) -> torch.Tensor:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return torch.stack([item[0] for item in batch], dim=0)

    def __len__(self) -> int:  # pragma: no cover - simple accessor
        return len(self.buffer)


class OnlineGaitTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        config = load_config(args.config, args.config_key)
        self.joints = config["joints"]
        self.timesteps = config["timesteps"]
        self.condition_size = config.get("condition_size", 0)

        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        self.autoencoder = BasicAutoencoder(
            joints=self.joints,
            timesteps=self.timesteps,
            latent_size=config["latent_size"],
            layer_sizes=config["layer_sizes"],
            condition_size=self.condition_size,
        ).to(self.device)

        self.env = gym.make(args.env)
        self.replay = ReplayBuffer(args.buffer_size)
        self.batch_size = args.batch_size
        self.num_perturbations = args.perturbations
        self.updates_per_iter = args.updates_per_iter
        self.perturb_std = args.perturb_std
        self.max_action = args.max_action
        self.warmup_rollouts = args.warmup_rollouts

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("results", f"online_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

        self.training_log: List[Dict] = []
        self.best_reward = float("-inf")
        self.best_gait: Optional[torch.Tensor] = None

    def clamp_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return torch.clamp(controls, -self.max_action, self.max_action)

    def rollout(self, controls: torch.Tensor) -> Tuple[float, Dict]:
        controls = self.clamp_controls(controls).cpu().float()
        flat = controls.view(-1)
        info = evaluate_control_sequence(flat, self.env, self.joints)
        rewards = info.get("rewards_over_time", [])
        total_reward = float(sum(rewards)) if rewards else float(info.get("reward", 0.0))
        return total_reward, info

    def warmup(self) -> None:
        if self.warmup_rollouts <= 0:
            return
        for _ in range(self.warmup_rollouts):
            random_controls = torch.empty(self.timesteps, self.joints).uniform_(-self.max_action, self.max_action)
            reward, _ = self.rollout(random_controls)
            self.replay.add(random_controls, reward)
            self.autoencoder.online_update(random_controls.unsqueeze(0).to(self.device))

    def generate_nominal_gait(self) -> torch.Tensor:
        latent = self.autoencoder.sample_latent(1, device=self.device)
        decoded = self.autoencoder.decode_latent(latent)
        return decoded.squeeze(0).detach().cpu()

    def perturb(self, controls: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(controls) * self.perturb_std
        return self.clamp_controls(controls + noise)

    def online_step(self, iteration: int) -> Dict:
        nominal = self.generate_nominal_gait()
        candidates = [nominal] + [self.perturb(nominal) for _ in range(self.num_perturbations)]

        losses: List[float] = []
        rewards: List[float] = []

        for controls in candidates:
            reward, _ = self.rollout(controls)
            rewards.append(reward)
            self.replay.add(controls, reward)
            loss = self.autoencoder.online_update(controls.unsqueeze(0).to(self.device))
            losses.append(loss)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_gait = controls.detach().clone()

        for _ in range(self.updates_per_iter):
            if len(self.replay) < self.batch_size:
                break
            batch = self.replay.sample(self.batch_size).to(self.device)
            replay_loss = self.autoencoder.online_update(batch)
            losses.append(replay_loss)

        summary = {
            "iteration": iteration,
            "mean_reward": float(sum(rewards) / len(rewards)),
            "max_reward": max(rewards),
            "mean_loss": float(sum(losses) / len(losses)),
            "buffer_size": len(self.replay),
        }
        self.training_log.append(summary)
        return summary

    def save_results(self) -> None:
        model_path = os.path.join(self.results_dir, "autoencoder_online.pth")
        torch.save(self.autoencoder.state_dict(), model_path)

        if self.best_gait is not None:
            best_path = os.path.join(self.results_dir, "best_gait.pt")
            torch.save({"controls": self.best_gait, "reward": self.best_reward}, best_path)

        log_path = os.path.join(self.results_dir, "training_log.json")
        with open(log_path, "w") as handle:
            json.dump(self.training_log, handle, indent=2)

    def close(self) -> None:
        self.env.close()


def main() -> None:
    args = parse_args()
    trainer = OnlineGaitTrainer(args)
    try:
        trainer.warmup()
        for iteration in range(1, args.iterations + 1):
            summary = trainer.online_step(iteration)
            print(
                f"Iteration {iteration:03d} | Mean Reward: {summary['mean_reward']:.3f} | "
                f"Max Reward: {summary['max_reward']:.3f} | Mean Loss: {summary['mean_loss']:.6f} | "
                f"Buffer: {summary['buffer_size']}"
            )
    finally:
        trainer.save_results()
        trainer.close()


if __name__ == "__main__":
    main()

