from time import time
from pathlib import Path
import os

import numpy as np
import torch
from tensordict import TensorDict
from trainer.base import Trainer

# Use non-interactive backend for servers (no DISPLAY)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

        # --- histories for plotting (EVAL) ---
        self.eval_steps = []
        self.eval_reward = []
        self.eval_success = []
        self.eval_length = []

        # directory to save plots
        self.plot_dir = Path(self.cfg.work_dir) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    # ===========================
    # 📊 COMMON METRICS
    # ===========================
    def common_metrics(self):
        elapsed_time = time() - self._start_time
        return dict(
            step=self._step,
            episode=self._ep_idx,
            elapsed_time=elapsed_time,
            steps_per_second=self._step / (elapsed_time + 1e-6),
        )

    # ===========================
    # 🔧 Helper: obs dict -> tensor
    # ===========================
    def _obs_to_tensor(self, obs):
        """
        Convert an observation (dict/np/torch) to a torch.Tensor expected by agent.act.
        Agent.act expects a raw torch tensor (encoder input) and will .to(device) internally.
        We return a torch.Tensor (not batched); the agent will unsqueeze(0) itself.
        """
        # If observation is a dict (preferred env interface), try to extract 'state'
        if isinstance(obs, dict):
            # Try common keys in order
            for key in ["state", "observation", "obs", "state_vector"]:
                if key in obs:
                    obs_val = obs[key]
                    break
            else:
                # fallback: take first value (useful if env returns {'state':..., 'image':...})
                # but prefer raising to be explicit
                raise KeyError(
                    "Observation dict passed to trainer._obs_to_tensor but no 'state' key found."
                )
        else:
            obs_val = obs

        # Convert numpy -> torch if necessary
        if isinstance(obs_val, np.ndarray):
            obs_tensor = torch.from_numpy(obs_val).float()
        elif isinstance(obs_val, torch.Tensor):
            obs_tensor = obs_val.float()
        else:
            raise TypeError("obs must be dict, numpy array, or torch tensor")

        return obs_tensor

    # ===========================
    # ✅ EVALUATION LOOP
    # ===========================
    def eval(self):
        ep_rewards, ep_successes, ep_lengths = [], [], []

        for i in range(self.cfg.eval_episodes):
            obs = self.env.reset()
            done, ep_reward, t = False, 0, 0

            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))

            while not done:
                # convert obs (dict) -> raw tensor for agent.act
                try:
                    act_input = self._obs_to_tensor(obs)
                except Exception as e:
                    # In case of unexpected format, re-raise with context
                    raise RuntimeError(f"Failed to convert obs to tensor in eval(): {e}") from e

                # agent.act will call .to(device) internally
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(act_input, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1

                if self.cfg.save_video:
                    self.logger.video.record(self.env)

            ep_rewards.append(ep_reward)
            ep_successes.append(float(info.get("success", 0.0)))
            ep_lengths.append(t)

            if self.cfg.save_video:
                self.logger.video.save(self._step)

        return dict(
            episode_reward=float(np.mean(ep_rewards)),
            episode_success=float(np.mean(ep_successes)),
            episode_length=float(np.mean(ep_lengths)),
        )

    # ===========================
    # 🧱 CONVERT TO TENSORDICT
    # ===========================
    def to_td(self, obs, action=None, reward=None, terminated=None):
        """
        Convert environment observation (possibly dict) into a TensorDict entry suitable
        for the replay buffer. This function now accepts dicts (it extracts 'state').
        """

        # If obs is a dict, extract 'state' (or similar)
        if isinstance(obs, dict):
            # Preferential keys
            for key in ["state", "observation", "obs", "state_vector"]:
                if key in obs:
                    obs_val = obs[key]
                    break
            else:
                raise KeyError(
                    "to_td received an observation dict but no 'state' key was found."
                )
        else:
            obs_val = obs

        # Observation -> torch tensor
        if isinstance(obs_val, np.ndarray):
            obs_tensor = torch.from_numpy(obs_val).float()
        elif isinstance(obs_val, torch.Tensor):
            obs_tensor = obs_val.float()
        else:
            raise TypeError("obs must be numpy array, torch tensor, or dict containing them")
        obs_tensor = obs_tensor.unsqueeze(0).cpu()

        # Action
        if action is None:
            action = self.env.rand_act()
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).float()
        elif isinstance(action, torch.Tensor):
            action_tensor = action.float()
        else:
            raise TypeError("action must be numpy array or torch tensor")
        action_tensor = action_tensor.unsqueeze(0).cpu()

        # Reward
        if reward is None:
            reward_tensor = torch.tensor(float("nan"))
        elif not isinstance(reward, torch.Tensor):
            reward_tensor = torch.tensor(float(reward))
        else:
            reward_tensor = reward
        reward_tensor = reward_tensor.unsqueeze(0)

        # Termination flag
        if terminated is None:
            terminated_tensor = torch.tensor(float("nan"))
        elif not isinstance(terminated, torch.Tensor):
            terminated_tensor = torch.tensor(float(terminated))
        else:
            terminated_tensor = terminated
        terminated_tensor = terminated_tensor.unsqueeze(0)

        return TensorDict(
            {
                "obs": obs_tensor,
                "action": action_tensor,
                "reward": reward_tensor,
                "terminated": terminated_tensor,
            },
            batch_size=(1,),
        )

    # ===========================
    # 📈 PLOTTING HELPER
    # ===========================
    def _plot_eval_curves(self):
        """Save eval curves as PNGs in self.plot_dir."""
        if len(self.eval_steps) == 0:
            return

        steps = np.array(self.eval_steps)
        rew = np.array(self.eval_reward)
        succ = np.array(self.eval_success)
        length = np.array(self.eval_length)

        # small helper
        def plot_and_save(x, y, ylabel, filename):
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("Training steps")
            plt.ylabel(ylabel)
            plt.title(ylabel)
            plt.grid(True)
            out_path = self.plot_dir / filename
            plt.savefig(out_path)
            plt.close()

        plot_and_save(steps, rew, "Eval episode reward", "Post_H5_eval_reward.png")
        plot_and_save(steps, succ, "Eval success rate", "Post_H5_eval_success.png")
        plot_and_save(steps, length, "Eval episode length", "Pos_eval_length.png")

    # ===========================
    # 🚂 MAIN TRAINING LOOP
    # ===========================
    def train(self):

        train_metrics, done, eval_next = {}, True, False

        while self._step <= self.cfg.steps:

            # ----------------------
            # EVALUATION SCHEDULING
            # ----------------------
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # ----------------------
            # EPISODE RESET
            # ----------------------
            if done:

                if eval_next:
                    # run eval
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")

                    # 👇 store eval history for plotting
                    self.eval_steps.append(self._step)
                    self.eval_reward.append(eval_metrics["episode_reward"])
                    self.eval_success.append(eval_metrics["episode_success"])
                    self.eval_length.append(eval_metrics["episode_length"])

                    # update plots
                    self._plot_eval_curves()

                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.stack(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=torch.tensor(float(info.get("success", 0))),
                        episode_length=len(self._tds),
                        episode_terminated=torch.tensor(
                            float(info.get("terminated", 0))
                        ),
                    )

                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")

                    # ✅ push episode into replay buffer
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                # reset env and rollout buffer
                obs = self.env.reset()
                self._tds = [self.to_td(obs)]

            # ----------------------
            # ACTION SELECTION
            # ----------------------
            if self._step > self.cfg.seed_steps:
                # convert obs -> tensor for agent.act
                try:
                    act_input = self._obs_to_tensor(obs)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert obs to tensor for agent.act(): {e}") from e

                action = self.agent.act(act_input, t0=(len(self._tds) == 1))
            else:
                action = self.env.rand_act()

            obs, reward, done, info = self.env.step(action)

            # ----------------------
            # STORE TRANSITION
            # ----------------------
            self._tds.append(
                self.to_td(obs, action, reward, info.get("terminated", False))
            )

            # ----------------------
            # UPDATE POLICY
            # ----------------------
            if self._step >= self.cfg.seed_steps and self.buffer.num_eps > 0:

                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1

                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)

                train_metrics.update(_train_metrics)

            # increment global step
            self._step += 1

        self.logger.finish(self.agent)
