# FL CODE BASE
Federated Averaging and Model Training 

## Docker Image and container instructions
```
docker pull docker.pkg.github.com/qtim-lab/rop_classification/rop:v1
```
- Open a container in bash mode using this image or use a docker image that satifies the requirements in requirements.txt




## To make use of the code 
- Provide the paths to the image files of different insitituions and complete the list in the relevant code block to create the dataloaders for different institutions.

# Training Flow
- Pass a copy of the global model to each insitutions and train on that instituions' dataset for one epoch
- Store the weieghts received from training on each instituions and average them
- load the global model with the averaged weights 
- Repeat the process for certain number of Federated rounds or monitor the global validation loss for stopping criteria

