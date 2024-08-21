import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bezier
from pathlib import Path
import omegaconf
import scienceplots


plt.style.use("science")

mlflow_tracking_uri = (
    "file:///home/user/Repos/skoltech/goal-agent-cleanrl/goalagent/srccode_data/mlruns"
)

steps_per_episode = {
    "pendulum": 1001,
    "3wrobot_kin": 501,
    "inverted_pendulum": 1501,
    "2tank": 801,
    "lunar_lander": 401,
    "omnibot": 1001,
}

dim_action = {
    "omnibot": 2,
    "3wrobot_kin": 2,
    "pendulum": 1,
    "inverted_pendulum": 1,
    "2tank": 1,
    "lunar_lander": 2,
}

dim_state = {
    "omnibot": 2,
    "3wrobot_kin": 3,
    "pendulum": 2,
    "inverted_pendulum": 4,
    "2tank": 2,
    "lunar_lander": 6,
}

state_action_names = {
    "2tank": {
        "state_names": ["Intake Level [m]", "Sink Level [m]"],
        "action_names": ["Pressure [Pa]"],
    },
    "pendulum": {
        "state_names": ["angle [rad]", "angular vel. [rad/s]"],
        "action_names": ["momentum [kg*m/s]"],
    },
    "3wrobot_kin": {
        "state_names": ["x-coord. [m]", "y-coord. [m]", "angle [rad]"],
        "action_names": ["velocity [m/s]", "angular vel. [rad/s]"],
    },
    "inverted_pendulum": {
        "state_names": [
            "angle [rad]",
            "x-coord. [m]",
            "angular vel. [rad/s]",
            "x-velocity [m/s]",
        ],
        "action_names": ["Force [N]"],
    },
    "lunar_lander": {
        "state_names": [
            "x-coord. [m]",
            "y-coord. [m]",
            "angle [rad]",
            "x-vel. [m/s]",
            "y-vel. [m/s]",
            "ang. vel. [rad/s]",
        ],
        "action_names": ["vert. force [N]", "side force [N]"],
    },
    "omnibot": {
        "state_names": ["x-coord. [m]", "y-coord. [m]"],
        "action_names": ["x-velocity [m/s]", "y-velocity [m/s]"],
    },
}

mapping_agents_pretty_namings = {
    "vpg": "VPG",
    "calf": r"\textbf{our agent}",
    "ddpg": "DDPG",
    "ppo": "PPO",
    "sac": "SAC",
    "td3": "TD3",
    "reinforce": "REINFORCE",
    "nominal": r"$\pi_0$",
}

mapping_envs_pretty_namings = {
    "pendulum": "pendulum",
    "3wrobot_kin": "three-wheel robot",
    "inverted_pendulum": "inverted pendulum",
    "omnibot": "omnibot",
    "lunar_lander": "lunar lander",
    "2tank": "two-tank system",
}


def get_runs(
    mlflow_tracking_uri,
    experiment_name,
    runs_from=0,
    runs_to=None,
):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Unable to find the experiment data for {experiment_name}. "
            "It seems you are attempting to extract experiment data without running the experiment itself. "
            "Please try running reproduce.py without the --plots-only flag."
        )
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
    )
    runs = runs.sort_values("start_time", ascending=False)
    if runs_to is None:
        runs = runs.iloc[runs_from:]
    else:
        runs = runs.iloc[runs_from:runs_to]
    return runs


def extract_eval_episodic_returns_from_mlflow(runs, metrics_name, columns):
    metrics_data_list = []
    for i, artifact_uri in enumerate(runs.artifact_uri):
        metrics_path = (
            artifact_uri[len("file://") : -len("artifacts")] + "metrics/" + metrics_name
        )
        metrics_data = pd.read_csv(metrics_path, sep=" ", header=None).rename(
            columns={0: "timestamp", 1: "episodic_return", 2: "global_step"}
        )

        for col in columns:
            metrics_data[col] = runs.iloc[i][col]

        metrics_data_list.append(metrics_data)

    return pd.concat(metrics_data_list, axis=0)


def save_learning_curve(mlflow_tracking_uri, env, agent):
    runs = get_runs(mlflow_tracking_uri, f"{agent}_{env}", runs_to=10)

    if agent == "td3" or agent == "sac" or agent == "calf":
        if (agent == "td3" or agent == "sac") and env == "pendulum":
            metrics_name = "eval/total_reward"
        else:
            metrics_name = "episodic_return"

        metrics_pd = extract_eval_episodic_returns_from_mlflow(
            runs, metrics_name=metrics_name, columns=["runs"]
        )
    else:
        ...

    plot_learning_curve(
        metrics_pd,
        label=agent,
        column="episodic_return",
        quantile_below=0.4,
        quantile_above=0.6,
        max_loc=1000000,
        rolling_median_window=10,
        is_smooth_bezier=True,
    )


def smooth_bezier(x, y, num_points=200):
    """
    Smooth data via a Bézier curve.

    Parameters:
    - x: np.ndarray - Array of x-coordinates of the data points.
    - y: np.ndarray - Array of y-coordinates of the data points.
    - num_points: int - Number of points to evaluate the curve.

    Returns:
    - smooth_x: np.ndarray - Smoothed x-coordinates.
    - smooth_y: np.ndarray - Smoothed y-coordinates.
    """
    data_points = np.vstack((x, y)).T

    # Create a Bezier curve from the data points
    nodes = np.asfortranarray(data_points.T)
    curve = bezier.Curve(nodes, degree=len(data_points) - 1)

    # Evaluate the curve at many points to get a smooth line
    s_vals = np.linspace(0.0, 1.0, num_points)
    curve_points = curve.evaluate_multi(s_vals)

    smooth_x = curve_points[0, :]
    smooth_y = curve_points[1, :]

    return smooth_x, smooth_y


def plot_learning_curve(
    metrics_pd: pd.DataFrame,
    label: str,
    column: str,
    quantile_below: float,
    quantile_above: float,
    max_loc=1000000,
    rolling_median_window=3,
    from_nominal=None,
    is_with_pretrain_stats=False,
    is_smooth_bezier=False,
    smooth_bezier_num_points=200,
    ax=None,
    color=None,
):
    between = (
        metrics_pd.groupby("global_step")[column]
        .median()
        .rolling(rolling_median_window)
        .mean()
        .dropna()
    )
    moment_of_pi_0_perfomance = None

    if from_nominal is not None:
        if np.any(between < from_nominal) > 0:
            start_from = between[between < from_nominal].idxmax()
        else:
            start_from = between.index.min()
        if not is_with_pretrain_stats:
            between = between.loc[start_from:]
            between.index = between.index - start_from
            between -= from_nominal
        elif ax is None:
            plt.axhline(
                y=from_nominal,
                color="gray",
                linestyle="--",
                label="Cumulative reward of $\pi_0$",
            )

            if np.any(between >= from_nominal):
                plt.axvline(
                    x=start_from / 1000,
                    color="gray",
                    linestyle="dotted",
                    label="Moment of reaching of $\pi_0$ performance",
                )
                moment_of_pi_0_perfomance = start_from / 1000
        else:
            ax.axhline(
                y=from_nominal,
                color="gray",
                label="Cumulative reward of $\pi_0$",
                linestyle="--",
            )
            if np.any(between >= from_nominal):
                moment_of_pi_0_perfomance = start_from / 1000
                ax.axvline(
                    x=start_from / 1000,
                    color="gray",
                    linestyle="dotted",
                    label="Moment of reaching of $\pi_0$ performance",
                )
    between = between.loc[:max_loc]
    if is_smooth_bezier:
        x, between = smooth_bezier(
            between.index, between, num_points=smooth_bezier_num_points
        )
        between = pd.Series(data=between, index=x)

    below = (
        metrics_pd.groupby("global_step")[column]
        .quantile(quantile_below)
        .rolling(rolling_median_window)
        .mean()
        .dropna()
    )
    if from_nominal is not None and not is_with_pretrain_stats:
        below = below.loc[start_from:]
        below.index = below.index - start_from
        below -= from_nominal
    below = below.loc[:max_loc]
    if is_smooth_bezier:
        _, below = smooth_bezier(
            below.index, below, num_points=smooth_bezier_num_points
        )
        below = pd.Series(data=below, index=x)

    above = (
        metrics_pd.groupby("global_step")[column]
        .quantile(quantile_above)
        .rolling(rolling_median_window)
        .mean()
        .dropna()
    )

    if from_nominal is not None and not is_with_pretrain_stats:
        above = above.loc[start_from:]
        above.index = above.index - start_from
        above -= from_nominal
    above = above.loc[:max_loc]

    if is_smooth_bezier:
        _, above = smooth_bezier(
            above.index, above, num_points=smooth_bezier_num_points
        )
        above = pd.Series(data=above, index=x)
    if ax is None:
        if color is None:
            plt.fill_between(between.index / 1000, below, above, alpha=0.3)
            plt.plot(between.index / 1000, between, label=label)
        else:
            plt.fill_between(between.index / 1000, below, above, alpha=0.3, color=color)
            plt.plot(between.index / 1000, between, label=label, color=color)
    else:
        if color is None:
            ax.fill_between(between.index / 1000, below, above, alpha=0.3)
            ax.plot(between.index / 1000, between, label=label)
        else:
            ax.fill_between(between.index / 1000, below, above, alpha=0.3, color=color)
            ax.plot(between.index / 1000, between, label=label, color=color)

    return pd.DataFrame(
        index=between.index,
        data={
            "between": between,
            "below": below,
            "above": above,
            "vert": moment_of_pi_0_perfomance,
            "hor": from_nominal,
        },
    )


def read_metrics_from_srccode(runs, steps_per_episode):

    data = []
    for i, metrics_path in enumerate(
        runs["tags.run_path"] + "/.callbacks/ValueCallback/"
    ):
        run_data = []
        for h5_path in sorted(Path(metrics_path).glob("*.h5")):
            iteration_data = pd.read_hdf(h5_path).rename(
                columns={"episode": "episode_id", "objective": "episodic_return"}
            )
            iteration_data["iteration_id"] = int(h5_path.stem.split("_")[-1])
            run_data.append(iteration_data)

        run_data = pd.concat(run_data, axis=0)
        run_data["global_step"] = np.arange(1, len(run_data) + 1) * steps_per_episode
        run_data["run_id"] = runs.iloc[i].run_id
        data.append(run_data)

    data = pd.concat(data)

    return data


def get_best_episode(runs, metrics_pd, env, agent):

    best_episode = metrics_pd.iloc[metrics_pd["episodic_return"].argmax()]
    best_episode_run = runs[runs["run_id"] == best_episode.run_id].iloc[0]

    best_episode_raw = pd.read_hdf(
        Path(best_episode_run["tags.run_path"])
        / ".callbacks"
        / "HistoricalDataCallback"
        / f"observations_actions_it_{best_episode.iteration_id:0>5}_ep_{best_episode.episode_id:0>5}.h5"
    )

    best_episode = best_episode_raw.iloc[:, -dim_state[env] - dim_action[env] :].iloc[
        :,
        list(range(dim_action[env], dim_action[env] + dim_state[env]))
        + list(range(dim_action[env])),
    ]
    best_episode.columns = (
        state_action_names[env]["state_names"] + state_action_names[env]["action_names"]
    )

    # return best_episode_raw[best_episode_raw.columns[-dim_state[env] - dim_action[env]:]]
    return best_episode


def get_metrics_and_best_episode_data(mlflow_tracking_uri, env, agent, experiment=None):
    runs = get_runs(
        mlflow_tracking_uri,
        experiment_name=f"{agent}_{env}" if experiment is None else experiment,
        runs_to=1 if agent == "nominal" else 10,
    )

    if agent in ["vpg", "ddpg", "ppo", "reinforce"]:
        metrics_pd = read_metrics_from_srccode(runs, steps_per_episode[env])
        best_episode = get_best_episode(runs, metrics_pd, env, agent)

    elif agent in ["sac", "td3"]:
        metrics_pd = extract_eval_episodic_returns_from_mlflow(
            runs=runs,
            metrics_name=(
                "eval/total_reward" if env == "pendulum" else "episodic_return"
            ),
            columns=["run_id"],
        )
        best_episode_metrics = metrics_pd.iloc[metrics_pd["episodic_return"].argmax()]
        best_episode_run = runs[runs["run_id"] == best_episode_metrics.run_id].iloc[0]
        artifact_path = best_episode_run.artifact_uri[len("file://") :]
        best_observations = pd.read_hdf(
            artifact_path
            + ("/eval" if env == "pendulum" else "")
            + f"/raw/observations/observations_{best_episode_metrics.global_step:0>8}.h5"
        )
        best_actions = pd.read_hdf(
            artifact_path
            + ("/eval" if env == "pendulum" else "")
            + f"/raw/actions/actions_{best_episode_metrics.global_step:0>8}.h5"
        )
        best_actions.columns = state_action_names[env]["action_names"]
        if env == "pendulum":
            cos_sin_angles = best_observations.iloc[:, [0, 1]].values
            cos_angles = cos_sin_angles[:, 0]
            sin_angles = cos_sin_angles[:, 1]
            angles = np.arctan2(sin_angles, cos_angles)
            best_states = pd.DataFrame(
                data=np.hstack(
                    (
                        angles.reshape(-1, 1),
                        best_observations.iloc[:, 2].values.reshape(-1, 1),
                    )
                ),
                columns=state_action_names[env]["state_names"],
            )
        elif env == "inverted_pendulum":
            sin_angles = best_observations.iloc[:, 0].values
            cos_angles = 1 - best_observations.iloc[:, 1].values
            angles = np.arctan2(sin_angles, cos_angles).reshape(-1, 1)
            best_states = pd.DataFrame(
                data=np.hstack(
                    (
                        angles,
                        np.zeros_like(angles),
                        best_observations.iloc[:, 2:].values,
                    )
                ),
                columns=state_action_names[env]["state_names"],
            )
        else:
            best_states = best_observations
            best_states.columns = state_action_names[env]["state_names"]
        best_episode = pd.concat((best_states, best_actions), axis=1)
    elif agent == "calf" or agent == "nominal":
        metrics_pd = extract_eval_episodic_returns_from_mlflow(
            runs=runs,
            metrics_name="episodic_return",
            columns=["run_id"],
        )
        best_episode_metrics = metrics_pd.iloc[metrics_pd["episodic_return"].argmax()]
        best_episode_run = runs[runs["run_id"] == best_episode_metrics.run_id].iloc[0]
        artifact_path = best_episode_run.artifact_uri[len("file://") :]
        best_observations = pd.read_hdf(
            artifact_path
            + f"/raw/observations/observations_{best_episode_metrics.global_step:0>8}.h5"
        )
        best_actions = pd.read_hdf(
            artifact_path
            + f"/raw/actions/actions_{best_episode_metrics.global_step:0>8}.h5"
        )
        best_actions.columns = state_action_names[env]["action_names"]

        if env == "pendulum":
            best_states = best_observations
            best_states["angle"] = angle_normalize(best_states["angle"])
            best_states.columns = state_action_names[env]["state_names"]
        elif env == "lunar_lander":
            best_states = pd.DataFrame(
                data=best_observations.values
                + np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                columns=state_action_names[env]["state_names"],
            )
        elif env == "2tank":
            best_states = pd.DataFrame(
                data=best_observations.values + np.array([[0.4, 0.4]]),
                columns=state_action_names[env]["state_names"],
            )
        elif env == "inverted_pendulum":
            sin_angles = best_observations.iloc[:, 0].values
            cos_angles = 1 - best_observations.iloc[:, 1].values
            angles = np.arctan2(sin_angles, cos_angles).reshape(-1, 1)
            best_states = pd.DataFrame(
                data=np.hstack(
                    (
                        angles,
                        np.zeros_like(angles),
                        best_observations.iloc[:, 2:].values,
                    )
                ),
                columns=state_action_names[env]["state_names"],
            )
        else:
            best_states = pd.DataFrame(
                data=best_observations.values,
                columns=state_action_names[env]["state_names"],
            )
        best_episode = pd.concat((best_states, best_actions), axis=1)
    return metrics_pd, best_episode


def plot_episode(best_episode: pd.DataFrame, agent: str, env: str):
    axes = best_episode.plot(
        subplots=True,
        figsize=(4, 6),
        legend=False,
        grid=True,
        sharex=True,
    )

    for title, ax in zip(best_episode.columns, axes):
        ax.set_ylabel(title, rotation=70, fontsize=6, labelpad=10)
        ax.set_xlabel("Time step")
    axes[0].set_title(
        f"Selected state-action trajectory for\n{mapping_agents_pretty_namings[agent]} on {mapping_envs_pretty_namings[env]}"
    )
    target_set = omegaconf.OmegaConf.load("exman/target_sets.yaml")[env]

    for ax, center, radii in zip(axes, target_set["centers"], target_set["radii"]):
        if center is not None:

            t = ax.fill_between(
                best_episode.index,
                center - radii,
                center + radii,
                color="gray",
                alpha=0.3,
            )

            ax.legend([t], ["Goal set $\mathbb{G}$"])

    return axes[0].figure


def save_plots(metrics_pd, best_episode, agent, env, path: Path):
    path.mkdir(exist_ok=True, parents=True)
    figure = plot_episode(best_episode, agent=agent, env=env)
    figure.savefig(path / "best_state_action_trajectory.pdf")

    best_episode.astype(np.float32).to_hdf(
        path / "best_state_action_trajectory.h5", key="data"
    )

    with open(str(path / "target_set.yaml"), "w") as f:
        f.write(
            omegaconf.OmegaConf.to_yaml(
                omegaconf.OmegaConf.load("exman/target_sets.yaml")[env]
            )
        )

    if metrics_pd is not None:
        metrics_pd.to_hdf(path / "raw_step_data.h5", key="data")
        plt.figure(figsize=(4, 4))
        plot_data = plot_learning_curve(
            metrics_pd,
            label=None,
            column="episodic_return",
            **dict(
                omegaconf.OmegaConf.load("exman/plot_config.yaml")[f"{agent}_{env}"]
            ),
        )
        plt.legend(frameon=True)
        plt.title(
            f"{mapping_agents_pretty_namings[agent]} on {mapping_envs_pretty_namings[env]}"
        )
        plt.xlabel("Time step (in thousands)")
        plt.ylabel("Median episodic return")
        plt.grid()

        plt.savefig(
            path / "perfomance.pdf",
        )
        plot_data.astype(np.float32).to_hdf(path / "perfomance.h5", key="data")


def extract_metrics(runs, metrics_name, columns):

    metrics_data_list = []
    for i, artifact_uri in enumerate(runs.artifact_uri):
        metrics_path = (
            artifact_uri[len("file://") : -len("artifacts")] + "metrics/" + metrics_name
        )
        metrics_data = pd.read_csv(metrics_path, sep=" ", header=None).rename(
            columns={0: "timestamp", 1: metrics_name, 2: "global_step"}
        )

        for col in columns:
            metrics_data[col] = runs.iloc[i][col]

        metrics_data_list.append(metrics_data)

    return pd.concat(metrics_data_list, axis=0)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
