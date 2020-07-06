# pytorch-a2c-ppo-acktr

## Please use hyper parameters from this readme. With other hyper parameters things might not work (it's RL after all)!

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)
* Generative Adversarial Imitation Learning [GAIL](https://arxiv.org/abs/1606.03476)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
    }

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) (via [dm_control2gym](https://github.com/martinseilair/dm_control2gym))

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.

To use the DeepMind Control Suite environments, set the flag `--env-name dm.<domain_name>.<task_name>`, where `domain_name` and `task_name` are the name of a domain (e.g. `hopper`) and a task within that domain (e.g. `stand`) from the DeepMind Control Suite. Refer to their repo and their [tech report](https://arxiv.org/abs/1801.00690) for a full list of available domains and tasks. Other than setting the task, the API for interacting with the environment is exactly the same as for all the Gym environments thanks to [dm_control2gym](https://github.com/martinseilair/dm_control2gym).

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

Also I'm searching for volunteers to run all experiments on Atari and MuJoCo (with multiple random seeds).

## Disclaimer

It's extremely difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information. I tried to reproduce OpenAI results as closely as possible. However, majors differences in performance can be caused even by minor differences in TensorFlow and PyTorch libraries.

### TODO

## Training
### Docker
If not installed yet, [set up](https://docs.docker.com/install/) docker on your machine.
Pull our docker container ``vioichigo/async`` from docker-hub:
```
docker pull vioichigo/async:latest
```
All the necessary dependencies are already installed inside the docker container.

### Setting up the doodad experiment launcher with EC2 support

Install AWS commandline interface

```
sudo apt-get install awscli
```

and configure the asw cli

```
aws configure
```

Clone the doodad repository 

```
git clone https://github.com/jonasrothfuss/doodad.git
```

Install the extra package requirements for doodad
```
cd doodad && pip install -r requirements.txt
```
Modifications: 
Modify ``cd doodad/scripts/run_experiment_lite_doodad.py``, add ``if __name__ == '__main__'`` before ``fn = doodad.get_args('run_method', failure)`` and ``fn()``:

Configure doodad for your ec2 account. First you have to specify the following environment variables in your ~/.bashrc: 
AWS_ACCESS_KEY, AWS_ACCESS_KEY, DOODAD_S3_BUCKET

Then run
```
python scripts/setup_ec2.py
```

Set S3_BUCKET_NAME in experiment_utils/config.py to your bucket name

## Experiments

### How to run experiments 
examples:

On your own machine:
```
python aws.py
```
You can change experiment name and hyperparameters in ``aws.py``. 
On docker:
```
python aws.py --mode local_docker
```
On aws:
```
python aws.py --mode ec2
```
To pull results from aws
```
python experiment_utils/sync_s3.py experiment_name
```
To check all the experiments running on aws
```
python experiment_utils/ec2ctl.py jobs
```
To kill experiments on aws
```
python experiment_utils/ec2ctl.py kill_f the_first_few_characters_of_your_experiment
```
OR
```
python experiment_utils/ec2ctl.py kill specific_full_name_of_an_experiment
```

