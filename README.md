# Overview

This is the code associated with the paper "Critic Monotone Envelope for Assisting Reinforcement Learning Agents".

## Installation

### Setup Python Virtual Environment

Note that this project requires **Python version 3.10**.

Before running the experiments from the source, it is recommended to setup a virtual environment to isolate the project dependencies. Here's how to do it:

1. **Install virtualenv if you haven't got it installed:**

   ```bash
   pip install virtualenv
   ```
   
2. **Create a virtual environment:**

   Navigate to the root of the repo and run:

   ```bash
   virtualenv venv
   ```
   
3. **Activate the virtual environment:**


   ```bash
   source venv/bin/activate
   ```

### Quick Start

To run the experiments, follow the steps below after activating the virtual environment:

1. Install the required packages:
   
   Update your system and install necessary dependencies:
   ```
   sudo apt update
   sudo apt install -y libgeos-dev libqt5x11extras5 default-jre texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
   ```
   
   Install the Python package in editable mode:
   ```
   pip install -e . --no-cache-dir
   ```
   
2. Execute the experiment script:
   
   Run the reproduce.py script with the required parameters:
   ```
   python reproduce.py --agent={NAME_OF_THE_AGENT} --env={NAME_OF_THE_ENVIRONMENT}
   ```

   Replace {NAME_OF_THE_AGENT} and {NAME_OF_THE_ENVIRONMENT} with the appropriate values from the lists below. For example:
   ```
   python reproduce.py --agent=calf --env=pendulum
   ```

   The script will run the specified agent-environment configuration by executing a corresponding bash file located in the `bash/` directory. We use MLflow for experiment management. The command performs the following steps:
   - Executes the specified run over 10 seeds by executing the corresponding bash file in the ./bash directory. Please note that this process can take a long time (several hours).
   - Extracts data from the MLflow experiment.
   - Generates plots representing the learning curve and the state-action trajectory with the greatest reward over all runs and stores the results into `goalagent/srccode_data/plots/` directory. 

   If you want to rebuild the plots with already run experiment then you can use `--plots-only` flag.

   ```
   python reproduce.py --agent=calf --env=pendulum --plots-only
   ```
3. To view the current progress of training, start the MLflow server:
   ```
   cd goalagent/srccode_data
   mlflow ui
   ```


## Abbreviations

### Environments

- `--env=inverted_pendulum` for the inverted pendulum 
- `--env=pendulum` for the pendulum 
- `--env=3wrobot_kin` for the three-wheel robot
- `--env=2tank` for the two-tank system
- `--env=lunar_lander` for the lunar lander
- `--env=omnibot` for the omnibot (kinematic point)

### Agents

- `--agent=calf` for **our agent**
- `--agent=nominal` for $\pi_0$ 
- `--agent=ppo` for Proximal Policy Optimization (PPO)
- `--agent=sdpg` for Vanilla Policy Gradient (VPG)
- `--agent=ddpg` for Deep Deterministic Policy Gradient (DDPG)
- `--agent=reinforce` for REINFORCE
- `--agent=sac` for Soft Actor Critic (SAC)
- `--agent=td3` for Twin-Delayed DDPG (TD3)

## Code structure

```
goal-agent-cleanrl
├─ bash  -  bash files launching 10 runs of specific pair agent-environment
├─ exman -  experiment manager. Contains utils for aggregation of experiments and drawing plots.
├─ reproduce.py   -  main script for reproducing experiments
├─ presets  -  storage of configuration .yaml files for all experiments
├─ goalagent   -  separate run scripts for experiments   
|  ├─calf   -  different implementations of our agent
|  |  ├─__init__.py
|  |  ├─agent_calf.py
|  |  ├─agent_calfq.py
|  |  ├─utilities.py
|  ├─env  -  environment specific functions and classes 
|  ├─utils  -  saving source code, observations, metrics evaluations 
|  ├─calf.py   -  run script for value function variant of our agent
|  ├─calfq.py  -  run script for action value function variant of our agent
|  ├─run_stable.py - entrypoint for PPO (our variant), VPG, DDGP, REINFORCE
|  ├─sac.py    - SAC run script
|  ├─td3.py    - TD3 run script
└─ srccode  -  source code for PPO (our variant), VPG, DDGP, REINFORCE
```

### `./bash`
Within this subdirectory, you will find all the Bash scripts intended for running a specific pair of agent-environment configurations. You may also override seeds or any hyperparameters here to obtain different sets of runs. 

To override the seeds, look for the line that iterates over seeds such as `+seed=0,1,2,3...` or `seed=1,2,3`, and modify it as you wish.



### Comments on internal logic

**Config files**: We utilize our own fork of [Hydra](https://hydra.cc/) to create reproducible pipelines, fully configured through a set of .yaml files that form a structured tree. To gain a deeper understanding of how our run scripts operate, we strongly recommend familiarizing yourself with [Hydra](https://hydra.cc/) and its core principles. All our run configurations are located in `presets/` folder. 

**Main script**: Every time you launch `reproduce.py` it will search and launch a respective bash script responsible for specific agent-environment pair, save learning curves and state-action trajectories. 

**Our agent**: Core logic of our agent is implemented in modules presented in `./goalagent/calf` folder. 

## Hardware and software requirements

### System Configuration

All experiments were conducted on the following machine
 * Operating system: Ubuntu 24.04 LTS
 * GPU: 2x GeForce RTX 3090
 * CPU: AMD Ryzen Threadripper 3990X 64-Core Processor
 * RAM: 128 GB

### Minimum System Requirements

- RAM: 32GB or higher
- CPU Cores: At least 16 logical cores
- Storage space: 250 GB for all experiment data
- GPU: Nvidia GPU with minimum of 10 GB of memory

### Software Requirements

Although the installation steps are detailed in \Cref{sec_installation}, we formally provide a summary of the dependencies here.

#### Python Dependencies
- All necessary Python dependencies are listed in the `pyproject.toml` file.

#### Non-Python Dependencies

Non-Python Dependencies can be installed using the following command:
```bash
sudo apt install -y libgeos-dev libqt5x11extras5 default-jre
```

