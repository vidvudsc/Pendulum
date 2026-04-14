#!/usr/bin/env python3
"""
PPO Training for Pendulum Balance - Continuous Action Space
With Disturbance System
"""

import asyncio
import concurrent.futures
import websockets
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import random
import os
from datetime import datetime

# ==================== CONFIGURATION ====================
START_POSITION = "down"

# Physics
GRAVITY = 9.81
CART_MASS = 1.0
BOB_MASS = 0.1
ARM_LENGTH = 1.5
FORCE_MAG = 10.0
DT = 0.02
MAX_CART_POS = 4.0
MAX_STEPS = 1500
DROP_RESET_ANGLE_UP = 0.8
DROP_RESET_PENALTY = 5.0

# PPO Hyperparameters
PPO_LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256
STEPS_PER_ROLLOUT = 2048
VALUE_COEF = 0.5
ENTROPY_COEF = 0.005   # enough to maintain exploration without explosion
NUM_ENVS = 16
HIDDEN_SIZE = 128

# Disturbance settings
DISTURBANCE_STRENGTH = 0.15  # 15% of max force per impulse
DISTURBANCE_MIN_COUNT = 2
DISTURBANCE_MAX_COUNT = 5
DISTURBANCE_MIN_TIME = 1.0  # seconds after episode start
DISTURBANCE_MAX_TIME = 30.0  # seconds after episode start

STATE_SIZE = 5


class ActorCritic(nn.Module):
    def __init__(self, state_size=STATE_SIZE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(HIDDEN_SIZE, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5]))  # initial std≈0.6; zeros caused entropy explosion
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = torch.tanh(self.actor_mean(h))
        value = self.critic(h)
        # clamp log_std: std stays in [exp(-2), exp(0.5)] ≈ [0.14, 1.65]
        # lower bound keeps exploration alive; upper bound prevents entropy explosion
        std = self.log_std.clamp(-2.0, 0.5).exp()
        return mean, std, value


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.infos = []

    def clear(self):
        self.__init__()


class PPOAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=PPO_LR)

    def act(self, state, training=True):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, value = self.model(s)

        dist = td.Normal(mean, std)
        if training:
            action = dist.sample()
        else:
            action = mean

        action_clipped = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action_clipped.item(), log_prob.item(), value.item()

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device).unsqueeze(-1)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, MINIBATCH_SIZE):
                mb_idx = indices[start : start + MINIBATCH_SIZE]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                mean, std, values = self.model(mb_states)
                std = std.unsqueeze(-1)

                dist = td.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (mb_returns - values.squeeze(-1)).pow(2).mean()
                entropy_loss = -ENTROPY_COEF * entropy

                loss = actor_loss + VALUE_COEF * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }


class PendulumPhysics:
    def __init__(self, start_position="down", disturbance_enabled=False):
        self.start_position = start_position
        self.disturbance_enabled = disturbance_enabled
        self.reset()

    def reset(self):
        self.cart_x = random.uniform(-0.5, 0.5)
        self.cart_v = 0.0
        self.time = 0.0
        self.steps = 0

        if self.start_position == "down":
            self.angle = np.pi + random.uniform(-0.3, 0.3)
        else:
            self.angle = random.uniform(-0.3, 0.3)

        self.angular_v = random.uniform(-0.2, 0.2)

        self.pending_disturbances = []
        self.active_disturbance = 0.0
        self.disturbance_timer = 0.0

        if self.disturbance_enabled:
            self._schedule_disturbances()

    def _schedule_disturbances(self):
        num_disturbances = random.randint(DISTURBANCE_MIN_COUNT, DISTURBANCE_MAX_COUNT)
        self.pending_disturbances = sorted(
            [
                random.uniform(DISTURBANCE_MIN_TIME, DISTURBANCE_MAX_TIME)
                for _ in range(num_disturbances)
            ]
        )

    def apply_disturbance(self):
        self.active_disturbance = (
            random.uniform(-DISTURBANCE_STRENGTH, DISTURBANCE_STRENGTH) * FORCE_MAG
        )
        self.disturbance_timer = 0.15

    def get_disturbance(self):
        if self.disturbance_timer > 0:
            self.disturbance_timer -= DT
            return self.active_disturbance
        return 0.0

    def get_state(self):
        return np.array(
            [
                self.cart_x / MAX_CART_POS,
                np.clip(self.cart_v / 8.0, -1, 1),
                np.sin(self.angle),
                np.cos(self.angle),
                np.clip(self.angular_v / 8.0, -1, 1),
            ],
            dtype=np.float32,
        )

    def step(self, action, apply_disturbance=True):
        base_force = float(np.clip(action, -1.0, 1.0)) * FORCE_MAG

        # Always tick and apply any active impulse (includes manual ones from the UI).
        # apply_disturbance only controls whether *new* scheduled impulses are triggered.
        disturbance = self.get_disturbance()  # decrements timer every step

        if apply_disturbance and disturbance == 0.0 and len(self.pending_disturbances) > 0:
            if self.time >= self.pending_disturbances[0]:
                self.pending_disturbances.pop(0)
                self.apply_disturbance()
                disturbance = self.get_disturbance()

        force = base_force + disturbance

        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)

        total_mass = CART_MASS + BOB_MASS
        angular_acc = (
            GRAVITY * sin_theta
            - cos_theta
            * (force + BOB_MASS * ARM_LENGTH * self.angular_v**2 * sin_theta)
            / total_mass
        ) / (ARM_LENGTH * (4.0 / 3.0 - BOB_MASS * cos_theta**2 / total_mass))

        cart_acc = (
            force
            + BOB_MASS
            * ARM_LENGTH
            * (self.angular_v**2 * sin_theta - angular_acc * cos_theta)
        ) / total_mass

        self.cart_v += cart_acc * DT
        self.cart_v *= 0.998
        self.cart_x += self.cart_v * DT

        self.angular_v += angular_acc * DT
        self.angular_v *= 0.998
        self.angle += self.angular_v * DT
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))

        self.time += DT
        self.steps += 1

        state = self.get_state()

        # Normalized upright: 0.0 when hanging straight down, 1.0 when perfectly upright.
        # Using cos(angle) directly gave -2/step while hanging, which made dying fast
        # (hitting the rail for -10) the optimal strategy — agent learned to rush to the wall.
        # Normalizing to [0,1] means survival is ALWAYS at least as good as dying.
        upright = (np.cos(self.angle) + 1.0) / 2.0  # [0, 1]

        # Linear centering: constant gradient everywhere so the agent always
        # has a clear signal to move toward x=0, not just near the edges.
        # Quadratic was nearly flat near center — agent settled anywhere within ~1m.
        centered = 1.0 - abs(self.cart_x) / MAX_CART_POS  # [0, 1], linear

        reward = upright + 0.5 * centered

        # Small action penalty (action is passed in as float in [-1,1])
        reward -= 0.001 * float(np.clip(action, -1.0, 1.0)) ** 2

        # Bonus for holding upright AND centered
        if abs(self.angle) < 0.20 and abs(self.cart_x) < 0.5:
            reward += 0.5

        done = False

        if self.start_position == "up" and abs(self.angle) > DROP_RESET_ANGLE_UP:
            done = True
            reward -= DROP_RESET_PENALTY

        elif abs(self.cart_x) >= MAX_CART_POS:
            done = True
            reward = -10.0

        elif self.steps >= MAX_STEPS:
            done = True
            reward += 50.0

        info = {
            "centered": abs(self.cart_x) < 0.5,
            "upright": abs(self.angle) < 0.26,
        }

        return state, reward, done, info


class RolloutCollector:
    def __init__(self, agent, num_envs, start_position, disturbance_enabled=False):
        self.agent = agent
        self.num_envs = num_envs
        self.start_position = start_position
        self.disturbance_enabled = disturbance_enabled
        self.envs = [
            PendulumPhysics(start_position, disturbance_enabled)
            for _ in range(num_envs)
        ]
        self.buffer = RolloutBuffer()
        # Per-env data kept separately so GAE can be computed correctly per env,
        # not across the interleaved boundary where env_i at step T would wrongly
        # bootstrap from env_{i+1} at step T.
        self.per_env = []

    def collect(self, steps):
        self.buffer.clear()
        states = [env.get_state() for env in self.envs]

        # Per-env temporary storage
        per_env = [
            dict(states=[], actions=[], log_probs=[], rewards=[], dones=[], values=[], infos=[])
            for _ in range(self.num_envs)
        ]

        for _ in range(steps):
            for i, env in enumerate(self.envs):
                action, log_prob, value = self.agent.act(states[i], training=True)
                next_state, reward, done, info = env.step(
                    action, apply_disturbance=True
                )

                d = per_env[i]
                d['states'].append(states[i])
                d['actions'].append(action)
                d['log_probs'].append(log_prob)
                d['rewards'].append(reward)
                d['dones'].append(done)
                d['values'].append(value)
                d['infos'].append(info)

                if done:
                    env.reset()
                    next_state = env.get_state()

                states[i] = next_state

        self.per_env = per_env

        # Flatten into shared buffer for metrics / PPO update
        for d in per_env:
            self.buffer.states.extend(d['states'])
            self.buffer.actions.extend(d['actions'])
            self.buffer.log_probs.extend(d['log_probs'])
            self.buffer.rewards.extend(d['rewards'])
            self.buffer.dones.extend(d['dones'])
            self.buffer.values.extend(d['values'])
            self.buffer.infos.extend(d['infos'])

        return self.buffer


def compute_gae(rewards, dones, values, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    values = values + [last_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


class TrainingServer:
    def __init__(self):
        self.agent = PPOAgent()
        self.generation = 0
        self.best_reward = -999.0
        self.training = False
        self.has_model = False        # True after load or first training gen completes
        self.model_was_loaded = False # True only when explicitly loaded from disk
        self.clients = set()
        self.start_position = START_POSITION
        self.disturbance_enabled = False

        # Separate agent for the live demo loop (synced from training agent)
        self.demo_agent = PPOAgent()
        self.demo_physics = PendulumPhysics(
            start_position=self.start_position, disturbance_enabled=False
        )

        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.entropy = 0.0
        self.avg_reward = 0.0
        self.avg_centered = 0.0
        self.avg_upright = 0.0
        self.avg_perfect = 0.0
        self.avg_episode_len = 0.0   # seconds, avg completed episode length this gen
        self.total_episodes = 0      # cumulative completed episodes across all gens
        self.total_steps = 0         # cumulative env steps across all gens

        # Training runs in a single-worker executor to keep the event loop free.
        self._train_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Separate executor for I/O (save/load) so saves don't queue behind training.
        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def register(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected. Total: {len(self.clients)}")
        # Restore full UI state for the reconnecting client
        if self.training:
            await websocket.send(json.dumps({
                "type": "server_mode", "idle": False, "trainingActive": True,
                "generation": self.generation
            }))
        elif self.has_model:
            await websocket.send(json.dumps({
                "type": "server_mode", "idle": False, "watchMode": True
            }))
        else:
            await websocket.send(json.dumps({"type": "server_mode", "idle": True}))

    async def unregister(self, websocket):
        self.clients.discard(websocket)
        print(f"Client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, message):
        msg_json = json.dumps(message)
        disconnected = []
        clients_snapshot = list(self.clients)
        for client in clients_snapshot:
            try:
                await client.send(msg_json)
            except Exception:
                disconnected.append(client)
        for d in disconnected:
            self.clients.discard(d)

    async def send_stats(self):
        stats = {
            "type": "stats",
            "generation": self.generation,
            "totalEpisodes": self.total_episodes,
            "totalSteps": self.total_steps,
            "actorLoss": round(self.actor_loss, 4),
            "criticLoss": round(self.critic_loss, 4),
            "entropy": round(self.entropy, 4),
            "avgReward": round(self.avg_reward, 1),
            "bestReward": round(self.best_reward, 1),
            "centeredTime": round(self.avg_centered, 1),
            "uprightTime": round(self.avg_upright, 1),
            "perfectRate": round(self.avg_perfect, 1),
            "avgEpisodeLen": round(self.avg_episode_len, 1),
            "disturbanceEnabled": self.disturbance_enabled,
        }
        await self.broadcast(stats)

    async def send_demo_state(self, reset=False):
        if not self.clients:
            return

        disturbance_force = 0.0
        if self.demo_physics.disturbance_timer > 0:
            disturbance_force = self.demo_physics.active_disturbance / FORCE_MAG

        await self.broadcast(
            {
                "type": "demo_state",
                "cartPos": self.demo_physics.cart_x,
                "cartVel": self.demo_physics.cart_v,
                "angle": self.demo_physics.angle,
                "angularVel": self.demo_physics.angular_v,
                "time": self.demo_physics.time,
                "generation": self.generation,
                "reset": reset,
                "disturbanceForce": round(disturbance_force, 3),
            }
        )

    def _reset_demo_episode(self):
        self.demo_physics = PendulumPhysics(
            start_position=self.start_position, disturbance_enabled=self.disturbance_enabled
        )

    async def run_demo_loop(self):
        """
        Runs the live AI demo at real-time speed.
        Only active when training is running OR a model has been loaded.
        When idle (no model, not training) it stays paused so the browser
        can run its own manual physics instead.
        """
        episode_reset = True

        while True:
            if self.training or self.has_model:
                state = self.demo_physics.get_state()
                action, _, _ = self.demo_agent.act(state, training=False)
                _, _, done, _ = self.demo_physics.step(action, apply_disturbance=self.disturbance_enabled)

                await self.send_demo_state(reset=episode_reset)
                episode_reset = False

                if done:
                    self._reset_demo_episode()
                    episode_reset = True
            else:
                # Idle — reset the flag so we send a proper reset when we wake up
                episode_reset = True

            await asyncio.sleep(DT)

    async def run_generation(self):
        if not self.training:
            return

        loop = asyncio.get_event_loop()

        collector = RolloutCollector(
            agent=self.agent,
            num_envs=NUM_ENVS,
            start_position=self.start_position,
            disturbance_enabled=self.disturbance_enabled,
        )

        # Run the CPU-heavy rollout collection in a thread so the event loop
        # stays free to send demo frames and handle WebSocket messages.
        buffer = await loop.run_in_executor(
            self._train_executor, lambda: collector.collect(STEPS_PER_ROLLOUT)
        )

        # Compute GAE per environment — NOT on the flattened interleaved buffer.
        # Flattened order is [env0_t0, env1_t0, ..., env0_t1, ...] so naively
        # iterating backwards would bootstrap env_i from env_{i+1}, not from the
        # correct next step of env_i.
        all_advantages = []
        all_returns = []
        for i, env in enumerate(collector.envs):
            _, _, last_value = self.agent.act(env.get_state(), training=False)
            d = collector.per_env[i]
            adv, ret = compute_gae(d['rewards'], d['dones'], d['values'], last_value)
            all_advantages.extend(adv)
            all_returns.extend(ret)

        advantages, returns = all_advantages, all_returns

        # PPO update is also CPU-heavy — run in executor
        loss_info = await loop.run_in_executor(
            self._train_executor,
            lambda: self.agent.ppo_update(
                buffer.states, buffer.actions, buffer.log_probs, returns, advantages
            ),
        )

        self.actor_loss = loss_info["actor_loss"]
        self.critic_loss = loss_info["critic_loss"]
        self.entropy = loss_info["entropy"]

        self.avg_reward = np.mean(buffer.rewards)
        if self.avg_reward > self.best_reward:
            self.best_reward = self.avg_reward

        # Count completed episodes this generation and accumulate totals
        episodes_this_gen = int(np.sum(buffer.dones))
        self.total_episodes += episodes_this_gen
        self.total_steps += len(buffer.rewards)

        # Average episode length from completed episodes (in seconds)
        if episodes_this_gen > 0:
            total_done_steps = int(np.sum(buffer.dones))
            self.avg_episode_len = (len(buffer.rewards) / max(total_done_steps, 1)) * DT
        else:
            self.avg_episode_len = (len(buffer.rewards) / NUM_ENVS) * DT

        centered_vals = [1.0 if info.get("centered", False) else 0.0 for info in buffer.infos]
        upright_vals = [1.0 if info.get("upright", False) else 0.0 for info in buffer.infos]
        self.avg_centered = np.mean(centered_vals) * 100
        self.avg_upright = np.mean(upright_vals) * 100
        self.avg_perfect = np.mean([c * u for c, u in zip(centered_vals, upright_vals)]) * 100

        # Sync demo agent with the latest trained weights
        self.demo_agent.model.load_state_dict(self.agent.model.state_dict())
        # Mark that we now have a real model to display
        self.has_model = True

        self.generation += 1

        if self.generation % 5 == 0:
            print(
                f"Gen {self.generation}: reward={self.avg_reward:.1f}, "
                f"actor={self.actor_loss:.4f}, critic={self.critic_loss:.4f}, "
                f"entropy={self.entropy:.4f}"
            )

        await self.send_stats()

        if self.training:
            asyncio.create_task(self.run_generation())

    async def handle_message(self, websocket, message):
        data = json.loads(message)
        msg_type = data.get("type")

        if msg_type == "command":
            action = data.get("action")

            if action == "start_training":
                if not self.training:
                    self.training = True
                    self.model_was_loaded = False
                    self.generation = 0
                    self.total_episodes = 0
                    self.total_steps = 0
                    self.best_reward = -999.0
                    self.actor_loss = 0.0
                    self.critic_loss = 0.0
                    self.entropy = 0.0

                    await self.send_stats()
                    print("\n*** Training Started ***")
                    asyncio.create_task(self.run_generation())
                    await self.broadcast(
                        {"type": "notification", "message": "Training started!"}
                    )

            elif action == "stop_training":
                self.training = False
                self.has_model = False  # reset so demo loop pauses
                print("\n*** Training Stopped ***")
                # Go back to manual control unless a model was explicitly loaded from disk.
                # Training-produced weights are discarded from the demo on stop.
                if not self.model_was_loaded:
                    await self.broadcast({"type": "server_mode", "idle": True})
                await self.broadcast(
                    {"type": "notification", "message": "Training stopped"}
                )

            elif action == "save_model":
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                os.makedirs(model_dir, exist_ok=True)
                fp = os.path.join(model_dir, f"gen{self.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                checkpoint = {
                    "model": self.agent.model.state_dict(),
                    "generation": self.generation,
                    "start_position": self.start_position,
                }
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._io_executor, lambda: torch.save(checkpoint, fp))
                print(f"Saved: {fp}")
                await self.broadcast(
                    {"type": "notification", "message": f"Saved gen {self.generation}"}
                )

            elif action == "list_models":
                # Send file list so browser can show a picker
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                if os.path.exists(model_dir):
                    files = sorted(
                        [f for f in os.listdir(model_dir) if f.endswith(".pth")],
                        reverse=True  # newest first
                    )
                else:
                    files = []
                await self.broadcast({"type": "model_list", "files": files})

            elif action == "load_model_file":
                filename = data.get("filename", "")
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                path = os.path.join(model_dir, filename)
                if not filename or not os.path.exists(path):
                    await self.broadcast({"type": "notification", "message": "File not found"})
                    return
                loop = asyncio.get_event_loop()
                checkpoint = await loop.run_in_executor(
                    self._io_executor,
                    lambda: torch.load(path, map_location="cpu", weights_only=False)
                )
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "policy" in checkpoint:
                    state_dict = checkpoint["policy"]
                else:
                    await self.broadcast({"type": "notification", "message": "Load failed: unknown format"})
                    return
                self.agent.model.load_state_dict(state_dict)
                self.demo_agent.model.load_state_dict(state_dict)
                self.generation = checkpoint.get("generation", 0)
                self.best_reward = -999.0
                self.has_model = True
                self.model_was_loaded = True
                if self.training:
                    self.training = False
                    print("*** Training stopped by load ***")
                print(f"Loaded: {filename} (Gen {self.generation})")
                self._reset_demo_episode()
                await self.broadcast({"type": "server_mode", "idle": False, "watchMode": True})
                await self.broadcast({"type": "notification", "message": f"Loaded: {filename}"})

            elif action == "apply_impulse":
                # User dragged on the canvas — apply a manual impulse to the demo physics.
                # force is in [-1, 1], scaled to actual force units.
                raw = float(data.get("force", 0.0))
                raw = max(-1.0, min(1.0, raw))
                self.demo_physics.active_disturbance = raw * FORCE_MAG
                self.demo_physics.disturbance_timer = 0.25  # hold for 0.25s

            elif action == "restart_episode":
                self._reset_demo_episode()
                await self.broadcast(
                    {"type": "notification", "message": "Episode restarted"}
                )

            elif action == "toggle_disturbance":
                self.disturbance_enabled = not self.disturbance_enabled
                # Reset demo so it picks up the new disturbance setting immediately
                self._reset_demo_episode()
                msg = (
                    "Disturbance ENABLED"
                    if self.disturbance_enabled
                    else "Disturbance DISABLED"
                )
                print(f"\n*** {msg} ***")
                await self.broadcast({"type": "notification", "message": msg})
                await self.send_stats()

        elif msg_type == "config":
            config = data.get("data", {})
            if "startPosition" in config:
                new_pos = config["startPosition"]
                if new_pos in ["down", "up"]:
                    self.start_position = new_pos
                    self._reset_demo_episode()
                    print(f"  Starting position: {new_pos}")

    async def handler(self, websocket, path=None):
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)


async def main():
    server = TrainingServer()
    print("=" * 60)
    print("PPO PENDULUM TRAINING - CONTINUOUS ACTION SPACE")
    print(f"Start: {server.start_position} | Device: {server.agent.device}")
    print(
        f"ARM_LENGTH: {ARM_LENGTH}m | Disturbance: {DISTURBANCE_STRENGTH * 100:.0f}%"
    )
    print(f"Envs: {NUM_ENVS} | Steps/rollout: {STEPS_PER_ROLLOUT}")
    print("=" * 60)
    print("Server idle — browser uses manual physics until training starts or model is loaded")
    asyncio.create_task(server.run_demo_loop())
    async with websockets.serve(server.handler, "localhost", 8765,
                                ping_interval=20, ping_timeout=120):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
