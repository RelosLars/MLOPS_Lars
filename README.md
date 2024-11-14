# Introduction
This is a finetinuned distilbert-base-uncased model on paraphrase detection (MRPC). It is part of an MLOPS course at HSLU



# How to run
## Prerequisites:
- Docker installed and running
- Wandb account and its api key, if you want to track your experiments. You can get the key on this website: https://wandb.ai/authorize

## Configuration
- If you want to track your experiments you need to enter your wandb API key inside the provided ```config.yaml``` file you in the line ```api_key: ""```. I also recommend adjusting the project name in the config file so that the run gets properly logged in your wandb account. If you don't provide the api key in ```config.yaml``` the experiments won't be tracked but it will still run.
- The ```config.yaml``` contains the tunable hyperparameters. You can change them as you want.

## Running as a docker container
- You first need to clone the repository. The Docker image is not available on the Docker hub.
- Open a commandline window in the root folder of the cloned repository
- Enter ```docker build -t mlops_lars_soler .``` to build the Docker image, feel free to replace ```mlops_lars_soler``` with any name
- If you want to use your Nvidia GPU for the training run, you can use CUDA and build the docker image as follows: ```docker build -t mlops_lars_soler -f Dockerfile_GPU .```.
### Windows:
- Run the docker container with the following command, make sure to adjust the path ```docker run -v C:/Users/larss/Documents/mlops/config.yaml:/app/config.yaml -v C:/Users/larss/Documents/mlops:/app/models mlops_lars_soler```. Getting relative paths to work on Windows can be troublesome so this example uses absolte paths.

### Linux:
- Run the docker container with the following command,```docker run -v ./config.yaml:/app/config.yaml -v ./models:/app/models mlops_lars_soler```.
You should now see a run being executed. 
