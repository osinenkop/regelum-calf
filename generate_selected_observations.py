from pathlib import Path
from omegaconf import OmegaConf
import shutil
import os

root_folder = Path(__file__).parent.parent
code_folder = root_folder / "code"
gfx_folder = root_folder / "gfx"
plots_folder = code_folder / "plots"
gfx_selected_observations = gfx_folder / "selected_observations"

shutil.rmtree(gfx_selected_observations)
gfx_selected_observations.mkdir(parents=True, exist_ok=True)

selected_observations_tex = ""
for agent_env_pair in OmegaConf.load(code_folder / "selected_observations.yaml"):
    source_file = plots_folder / agent_env_pair / "best_state_action_trajectory.pdf"
    destination_folder = gfx_selected_observations
    selected_observations_tex += (
        r"\includegraphics[width=0.5\textwidth]{gfx/selected_observations/"
        + agent_env_pair
        + ".pdf}\n"
    )
    shutil.copy(source_file, destination_folder)
    (destination_folder / "best_state_action_trajectory.pdf").rename(
        str(destination_folder / agent_env_pair) + ".pdf"
    )

with open(str(root_folder / "input" / "selected_observations.tex"), "w") as f:
    f.write(selected_observations_tex)
