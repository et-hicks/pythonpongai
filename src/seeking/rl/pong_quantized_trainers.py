"""Training utilities for quantized Pong controllers."""

from __future__ import annotations

from collections import deque
import random
from typing import Deque, Tuple

import torch
from torch.nn import functional as F

from .pong_quantized_models import ACTION_LABELS, PongActorCritic, PongDQN

NEUTRAL_ACTION_INDEX = ACTION_LABELS.index("neutral")


def create_dqn_controller() -> PongDQN:
    model = PongDQN()
    model.eval()
    return model


def create_actor_critic_controller() -> PongActorCritic:
    model = PongActorCritic()
    model.eval()
    return model


class DQNTrainer:
    """Lightweight replay-based trainer for the convolutional DQN controller."""

    def __init__(
        self,
        model: PongDQN,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        replay_size: int = 4096,
        epsilon: float = 0.05,
        idle_threshold: int = 20,
    ) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.idle_threshold = idle_threshold
        self.replay: Deque[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=replay_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.prev_state: torch.Tensor | None = None
        self.prev_action: int | None = None
        self.idle_steps = 0
        self.pending_idle_penalty = False

    def replace_model(self, model: PongDQN) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.replay.clear()
        self.reset_state()

    def reset_state(self) -> None:
        self.prev_state = None
        self.prev_action = None
        self.idle_steps = 0
        self.pending_idle_penalty = False

    def observe(self, next_state: torch.Tensor, reward: float, done: bool) -> None:
        next_state_detached = next_state.detach()
        if self.prev_state is not None and self.prev_action is not None:
            transition = (
                self.prev_state.clone(),
                self.prev_action,
                float(reward),
                next_state_detached.clone(),
                bool(done),
            )
            self.replay.append(transition)
            if len(self.replay) >= self.batch_size:
                self._train_batch()
        self.prev_state = next_state_detached

    def _train_batch(self) -> None:
        batch = random.sample(self.replay, self.batch_size)
        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([float(item[4]) for item in batch], dtype=torch.float32)

        self.model.train()
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.model(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        action, q_values = self.model.act(state, epsilon=self.epsilon)
        return action, q_values

    def register_action(self, action: int) -> None:
        self.prev_action = action
        if action == NEUTRAL_ACTION_INDEX:
            self.idle_steps += 1
            if self.idle_steps >= self.idle_threshold:
                self.pending_idle_penalty = True
        else:
            self.idle_steps = 0
            self.pending_idle_penalty = False


class ActorCriticTrainer:
    """Online actor-critic updater for the convolutional controller."""

    def __init__(
        self,
        model: PongActorCritic,
        lr: float = 1e-4,
        gamma: float = 0.99,
        idle_threshold: int = 20,
    ) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.idle_threshold = idle_threshold
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.prev_state: torch.Tensor | None = None
        self.prev_action: int | None = None
        self.idle_steps = 0
        self.pending_idle_penalty = False

    def replace_model(self, model: PongActorCritic) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.reset_state()

    def reset_state(self) -> None:
        self.prev_state = None
        self.prev_action = None
        self.idle_steps = 0
        self.pending_idle_penalty = False

    def observe(self, next_state: torch.Tensor, reward: float, done: bool) -> None:
        next_state_detached = next_state.detach()
        if self.prev_state is not None and self.prev_action is not None:
            self.model.train()
            logits, value = self.model(self.prev_state)
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = torch.tensor(self.prev_action, dtype=torch.long)
            log_prob = dist.log_prob(action_tensor)
            with torch.no_grad():
                target = torch.tensor(reward, dtype=torch.float32)
                if not done:
                    _, next_value = self.model(next_state_detached)
                    target = target + self.gamma * next_value
            advantage = target - value
            policy_loss = -(advantage.detach() * log_prob)
            value_loss = advantage.pow(2)
            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()
        self.prev_state = next_state_detached

    def act(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        action, logits, value = self.model.act(state)
        return action, logits, value

    def register_action(self, action: int) -> None:
        self.prev_action = action
        if action == NEUTRAL_ACTION_INDEX:
            self.idle_steps += 1
            if self.idle_steps >= self.idle_threshold:
                self.pending_idle_penalty = True
        else:
            self.idle_steps = 0
            self.pending_idle_penalty = False
