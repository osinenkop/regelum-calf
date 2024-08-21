import typer
import subprocess
from typing import Annotated
from enum import Enum
from exman import utils
from pathlib import Path
from typing import Optional
import mlflow

repo_root = Path(__file__).parent


class Agent(str, Enum):
    calf = "calf"
    nominal = "nominal"
    ppo = "ppo"
    vpg = "vpg"
    ddpg = "ddpg"
    reinforce = "reinforce"
    sac = "sac"
    td3 = "td3"


class Env(str, Enum):
    inv_pendulum = "pendulum"
    twotank = "2tank"
    robot = "3wrobot_kin"
    lunar_lander = "lunar_lander"
    kin_point = "omnibot"
    inverted_pendulum = "inverted_pendulum"


def main(
    agent: Agent = typer.Option(
        help="Agent name", show_default=False, show_choices=True
    ),
    env: Env = typer.Option(help="Environment name", show_default=False),
    plots_only: bool = typer.Option(
        default=False, help="Whether not to run experiment and just try to pop up plots"
    ),
    mlflow_tracking_uri: str = typer.Option(
        default="file://" + str(repo_root / "goalagent" / "srccode_data" / "mlruns"),
        help="MLflow tracking URI. Use the default value to ensure all experiment data is correctly saved.",
    ),
    experiment: Optional[str] = typer.Option(
        default=None,
        help="Which experiment from Mlflow to use for data extraction.",
    ),
):
    if not plots_only:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(f"{agent.value}_{env.value}")
        subprocess.run(["bash", f"bash/{agent.value}/{env.value}.sh"])
    metrics_pd, best_episode = utils.get_metrics_and_best_episode_data(
        mlflow_tracking_uri=mlflow_tracking_uri,
        env=env.value,
        agent=agent.value,
        experiment=experiment,
    )
    path = (
        repo_root
        / "goalagent"
        / "srccode_data"
        / "plots"
        / f"{agent.value}_{env.value}"
    )
    utils.save_plots(
        metrics_pd=metrics_pd if agent != Agent.nominal else None,
        best_episode=best_episode,
        agent=agent.value,
        env=env.value,
        path=path,
    )
    print(f"The plots were saved to {path}")


if __name__ == "__main__":
    typer.run(main)
