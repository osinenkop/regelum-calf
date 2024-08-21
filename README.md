# Overview

This part of the repository is dedicated to plot generation and the corresponding code routines.

- All plot data is located in the [./plots](./plots/) folder, including extracted raw experiment data, the best observations, learning curves, and their sources.
- The data is natively sorted in the `{AGENT}_{ENV}` format. 
- Aggregated information plots are in the [./plots/aggregated](./plots/aggregated) folder.
- All plots used in the paper can be generated from the [plots.ipynb](./plots.ipynb) Jupyter notebook.

To generate selected observations, run [generate_selected_observations.py](generate_selected_observations.py). This script will place selected observations into the [gfx](../gfx/) folder. It parses the configuration file `selected_observations.yaml`, copies the corresponding observations from the `plots` folder, and updates the LaTeX file [input/selected_observations.tex](../input/selected_observations.tex).

The [goalagent](./goalagent/) folder contains the source code for the paper. The bash script [create_zip.sh](./create_zip.sh) sets up a Git repository inside the folder (necessary for regelum to work), makes an anonymous commit, copies the appendix PDF file inside, and zips everything into the file `supplementary.zip`. Just run 
```sh
bash create_zip.sh
```
to make it work.



## Setup Instructions

To prepare your Python environment (Python 3.10 preferably), run:

```sh
pip install -r requirements.txt
```