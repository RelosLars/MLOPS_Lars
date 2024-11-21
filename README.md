# Introduction
Welcome to my repository for Project 2: Containerization! In this project, I focused on transforming a machine learning model training workflow into a fully containerized application using Docker. The goal was to ensure that the training process could be easily replicated across different environments, including local machines and cloud platforms, to streamline development and maintain consistency between environments.

# How to run
## Prerequisites:
- Docker installed and running
- Wandb account and its api key, if you want to track your experiments. You can get the key on this website: https://wandb.ai/authorize

## Configuration
- If you want to track your experiments you need to enter your wandb API key inside the provided ```config.yaml``` file you in the line ```api_key: ""```. I also recommend adjusting the project name in the config file so that the run gets properly logged in your wandb account. If you don't provide the api key in ```config.yaml``` the experiments won't be tracked but it will still run.
- The ```config.yaml``` contains the tunable hyperparameters. You can change them as you want.
- You can also configure the values by passing them as parameters at the end of the  ```docker run``` command, there is an example further down.

## Running as a docker container
- You first need to clone this repository. The Docker image is not available on the Docker hub.
- Open a commandline window in the root folder of the cloned repository
- Enter ```docker build -t mlops_lars_soler .``` to build the Docker image, feel free to replace ```mlops_lars_soler``` with any name
### Windows:
- Run the docker container with the following command, make sure to adjust the path
```
docker run -v C:/Users/larss/Documents/mlops/MLOPS_Lars/config.yaml:/app/config.yaml -v C:/Users/larss/Documents/mlops/MLOPS_Lars:/app/models mlops_lars_soler
```
Getting relative paths to work on Windows can be troublesome so this example uses absolte paths.

### Linux:
- Run the docker container with the following command,
```
docker run -v ./config.yaml:/app/config.yaml -v ./models:/app/models mlops_lars_soler
```
You should now see a run being executed. 

## Running with GPU support
In order to use CUDA to access your local NVIDIA GPU you have to add the ```--gpus all``` flag. The training should then run significantly faster.
Exampole: 
```
docker run --gpus all -v C:/Users/larss/Documents/mlops/MLOPS_Lars/config.yaml:/app/config.yaml -v C:/Users/larss/Documents/mlops/MLOPS_Lars:/app/models mlops_lars_soler
```

## Passing configuration as parameters
You can configure the hyperparameters/model/wandb with parameters in the ```docker run``` command aswell. If you leave out an argument then its value in ```config.yaml``` gets used as default. 
Example: 
```
docker run -v C:/Users/larss/Documents/mlops/MLOPS_Lars/config.yaml:/app/config.yaml -v C:/Users/larss/Documents/mlops/MLOPS_Lars:/app/models mlops_lars_soler --learning_rate 0.1 --epochs 1
```
The arguments must be placed at the end of the command.

### All configurable arguments
With ```--help``` you can get the list of all configurable values:
```options:
  -h, --help            show this help message and exit

Hyperparameters:
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps
  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimizer
  --optimizer_type OPTIMIZER_TYPE
                        Optimizer type (e.g., 'sgd', 'adam')
  --momentum MOMENTUM   Momentum factor (only used if optimizer_type is 'sgd')
  --beta1 BETA1         Beta1 parameter for the Adam optimizer
  --beta2 BETA2         Beta2 parameter for the Adam optimizer

WandB Configuration:
  --api_key API_KEY     api key for wandb
  --project PROJECT     project name for wandb
  --run_name RUN_NAME   run name for wandb

Model Configuration:
  --epochs EPOCHS       number of epochs for the training run
  --save_path SAVE_PATH
                        save path for the model ```
