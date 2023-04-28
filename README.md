# DEX: Demonstration-Guided RL with Efficient Exploration for Task Automation of Surgical Robot
This is the official PyTorch implementation of the paper "**Demonstration-Guided Reinforcement Learning with
Efficient Exploration for Task Automation of Surgical Robot**" (ICRA 2023). 
<p align="left">
  <img width="98%" src="docs/resources/dex_teaser.png">
</p>

# Prerequisites
* Ubuntu 18.04
* Python 3.7+


# Installation Instructions

1. Clone this repository.
```bash
git clone --recursive https://github.com/med-air/DEX.git
cd DEX
```

2. Create a virtual environment
```bash
conda create -n dex python=3.8
conda activate dex
```

3. Install packages

```bash
pip3 install -e SurRoL/	# install surrol environments
pip3 install -r requirements.txt
pip3 install -e .
```

4. Then add one line of code at the top of `gym/gym/envs/__init__.py` to register SurRoL tasks:

```python
# directory: anaconda3/envs/dex/lib/python3.8/site-packages/
import surrol.gym
```

# Usage
Commands for DEX and all baselines. Results will be logged to WandB. Before running the commands below, please change the wandb entity in [```train.yaml```](dex/configs/train.yaml#L26) to match your account.

We collect demonstration data via the scripted controllers provided by SurRoL. Take the NeedlePick task as example:
```bash
mkdir SurRoL/surrol/data/demo
python SurRoL/surrol/data/data_generation.py --env NeedlePick-v0 
```
## Training Commands 

- Train **DEX**:
```bash
python3 train.py task=NeedlePick-v0 agent=dex use_wb=True
```

- Train **SAC**:
```bash
python3 train.py task=NeedlePick-v0 agent=sac use_wb=True
```

- Train **DDPG**:
```bash
python3 train.py task=NeedlePick-v0 agent=ddpg use_wb=True
```

- Train **DDPGBC**:
```bash
python3 train.py task=NeedlePick-v0 agent=ddpgbc use_wb=True
```

- Train **CoL**:
```bash
python3 train.py task=NeedlePick-v0 agent=col use_wb=True
```

- Train **AMP**:
```bash
python3 train.py task=NeedlePick-v0 agent=amp use_wb=True
```

- Train **AWAC**:
```bash
python3 train.py task=NeedlePick-v0 agent=awac use_wb=True
```

- Train **SQIL**:
```bash
python3 train.py task=NeedlePick-v0 agent=sqil use_wb=True
```

Again, all commands can be run on other surgical tasks by replacing NeedlePick with the respective environment in the commands (for both demo collection and RL training).

We also implement synchronous parallelization of RL training, e.g., launch 4 parallel training processes:
```
mpirun -np 4 python -m train agent=dex task=NeedlePick-v0 use_wb=True
```
It should be noted that parallel training will lead to inconsistent performance, which require hyperparameters tuning.

## Evaluation Commands
We also provide a script for evaluate the saved model. The directory of the to-be-evaluated model should be included in the configuration file [```eval.yaml```](dex/configs/eval.yaml), where the checkpoint is specified by `ckpt_episode`. For instance:
- Eval model trained by **DEX** in NeedlePick-v0:
```bash
python3 eval.py task=NeedlePick-v0 agent=dex ckpt_episode=latest
```

# Starting to Modify the Code
## Modifying the hyperparameters
The default hyperparameters are defined in `dex/configs`, where [```train.yaml```](dex/configs/train.yaml) defines the experiment settings and YAML file in the directory [```agent```](dex/configs/agent) defines the hyperparameters of each method. Modifications to these parameters can be directly defined in the experiment or agent config files, or passed through the terminal command. For example:
```bash
python3 train.py task=NeedleRegrasp-v0 agent=dex use_wb=True batch_size=256 agent.aux_weight=10
```
## Adding a new RL algorithm
The core RL algorithms are implemented within the `BaseAgent` class. For adding a new algorithm, a new file needs to be created in
`dex/agents` and [```BaseAgent```](dex/agents/base.py#L8) needs to be subclassed. In particular, any required
networks (actor, critic etc) need to be constructed and the `update(...)` function and `get_action(...)` needs to be overwritten. For an example, 
see the DDPGBC implementation in [```DDPGBC```](dex/agents/ddpgbc.py#L7). When implementation is done, a registration is needed in [```factory.py```](dex/agents/factory.py) and a config file should also be made in [```agent```](dex/configs/agent) to specify the model parameters. 

## Transfering to other simulation platform
Our code is designed for standard goal-conditioned gym-based environments and can be easily transfered to other platform if provide the same interfaces (e.g., OpenAI gym fetch). If no similar interface is provided, some modifications should be made to make it compatible, e.g., replay buffer and sampling utilities. We will make our code more generalizable in the future.

# Code Navigation

```
dex
  |- agents                # implements core algorithms in agent classes
  |- components            # reusable infrastructure for model training
  |    |- checkpointer.py  # handles saving + loading of model checkpoints
  |    |- normalizer.py    # normalizer for vectorized input
  |    |- logger.py        # implements core logging functionality using wandB
  |
  |- configs               # experiment configs 
  |    |- train.yaml       # configs for rl training
  |    |- eval.yaml        # configs for rl evaluation
  |    |- agent            # configs for each algorithm (dex, ddpg, ddpgbc, etc.)
  |
  |- modules               # reusable architecture components
  |    |- critic.py        # basic critic implementations (eg MLP-based critic)
  |    |- distributions.py # pytorch distribution utils for density model
  |    |- policy.py    	   # basic actor implementations
  |    |- replay_buffer.py # her replay buffer with future sampling strategy
  |    |- sampler.py       # rollout sampler for collecting experience
  |    |- subnetworks.py   # basic networks
  |
  |- trainers              # main model training script, builds all components + runs training loop and logging
  |
  |- utils                 # general and rl utilities, pytorch / visualization utilities etc
  |- train.py              # experiment launcher
  |- eval.py               # evaluation launcher
```

# Contact
For any questions, please feel free to email taou.cs13@gmail.com.
