# pytorch-a2c-ppo-acktr
partially copied from meta-mb 
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

