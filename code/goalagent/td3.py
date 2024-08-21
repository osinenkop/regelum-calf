# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
import mlflow
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from goalagent.utils import (
    save_source_code,
    save_episodic_data,
    make_env,
    evaluate_policy,
)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


import srccode as rg
from goalagent import repo_root

config_path = repo_root / "presets" / "td3_sac_ours"
config_name = os.path.basename(__file__)[: -len(".py")]


@rg.main(
    config_path=config_path,
    config_name=config_name,
)
def launch(cfg):
    save_source_code()
    args = ~cfg
    seed = int(np.random.get_state()[1][0])
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = ~cfg

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    episodic_observations = []
    episodic_actions = []
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [
                    np.random.uniform(
                        low=envs.single_action_space.low,
                        high=envs.single_action_space.high,
                    )
                    for _ in range(envs.num_envs)
                ]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )
        episodic_observations.append(obs)
        episodic_actions.append(actions)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )

                mlflow.log_metric(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                mlflow.log_metric(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                save_episodic_data(
                    info,
                    global_step,
                    episodic_observations,
                    episodic_actions,
                    args.env_spec.observations_names,
                    args.env_spec.actions_names,
                )
                episodic_observations.clear()
                episodic_actions.clear()
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                mlflow.log_metric(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                mlflow.log_metric(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                mlflow.log_metric("losses/qf1_loss", qf1_loss.item(), global_step)
                mlflow.log_metric("losses/qf2_loss", qf2_loss.item(), global_step)
                mlflow.log_metric("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                mlflow.log_metric("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                mlflow.log_metric(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
            if args.env_spec.get("eval_env") is not None:
                if global_step % args.env_spec.eval_n_steps == 0:
                    evaluate_policy(
                        args.env_spec.eval_env,
                        actor,
                        global_step,
                        get_action=lambda obs: actor(obs),
                        device=device,
                        observations_names=args.env_spec.observations_names,
                        actions_names=args.env_spec.actions_names,
                    )

    envs.close()


if __name__ == "__main__":
    launch()
