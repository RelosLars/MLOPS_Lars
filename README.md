# Introduction
This is a finetinuned distilbert-base-uncased model on paraphrase detection (MRPC). It is part of an MLOPS course at HSLU



How to run
## Prerequisites:
- Docker installed and running
- You need a wandb account and its api key, you can get the key on this website: https://wandb.ai/authorize

## Configuration
- Inside the provided ```config.yaml``` file you first need to enter your wandb API key in the line ```api_key: ""```.
- I also recommend adjusting the project name in the config file so that the run gets properly logged in your wandb account.
- The ```config.yaml``` contains the tunable hyperparameters. You can change them as you want.

## Running as a docker container (Windows)
- You first need to clone the repository. The Docker image is not available on the Docker hub.
- Open a commandline window in the root folder of the cloned repository
- Enter ```docker build -t mlops_lars_soler .``` to build the Docker image, feel free to replace ```mlops_lars_soler``` with any name
- Run the docker container with the following command, make sure to adjust the path ```docker run -v C:/Users/larss/Documents/mlops/config.yaml:/app/config.yaml -v C:/Users/larss/Documents/mlops:/app/models mlops_lars_soler```. Getting relative paths to work on Windows can be troublesome so this example uses absolte paths.
- You should now see a run being executed. The training run runs on the CPU so it is pretty slow.
