from goalagent.calf.agent_calf import AgentCALF
from goalagent.env.pendulum import PendulumQuanser

# from goalagent.env.rg_env import RgEnv
# from goalagent.env.running_objective import GymPendulumRunningObjective
# from srccode.simulator import CasADi
import gymnasium as gym
from goalagent.utils import save_source_code, save_episodic_data, make_env
import numpy as np
from goalagent import repo_root
import os
import srccode as rg
import mlflow

config_path = repo_root / "presets" / "td3_sac_ours"
config_name = os.path.basename(__file__)[: -len(".py")]


@rg.main(
    config_path=config_path,
    config_name=config_name,
)
def launch(cfg):
    save_source_code()
    args = ~cfg
    calf_agent: AgentCALF = args.calf_agent

    envs = gym.vector.SyncVectorEnv([make_env(args.env) for i in range(1)])
    # envs = args.env
    obs, info = envs.reset()
    calf_agent.reset(global_step=0)  # args.total_timesteps)
    # mlflow.log_metric("charts/relax_probability", calfq_agent.relax_probability, step=0)

    episodic_observations = []
    episodic_actions = []

    for global_step in range(args.total_timesteps):
        action = calf_agent.get_action(obs)
        episodic_observations.append(obs)
        episodic_actions.append(action)
        obs, reward, terminations, truncations, infos = envs.step(action)

        if "final_info" in infos:
            mlflow.log_metric(
                "charts/relax_probability",
                calf_agent.relax_probability,
                step=global_step,
            )
            calf_agent.reset(global_step=global_step)
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
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


if __name__ == "__main__":
    launch()
