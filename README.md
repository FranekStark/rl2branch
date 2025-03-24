# Learning to branch with Tree MDPs

Lara Scavuzzo, Feng Yang Chen, Didier Chételat, Maxime Gasse, Andrea Lodi, Neil Yorke-Smith, Karen Aardal

Official implementation of the paper *Learning to branch with Tree MDPs*.

## Installation

See installation instructions [here](INSTALL.md).

## Running the experiments

For a given TYPE in {setcover, cauctions, indset, ufacilities, mknapsack}, run the following to reproduce the experiments
```
# Generate MILP instances
python 01_generate_instances.py $TYPE

# Get train instance solutions
python 02_get_instance_solutions.py $TYPE -j 8    # number of parallel threads

# Generate supervised learning datasets
python 03_generate_il_samples.py $TYPE -j 8  # number of parallel threads

# Training supervised learning model
python 04_train_il.py $TYPE -g 0    # GPU id

# Training reinforcement learning learning models
python 05_train_rl.py $TYPE mdp -g 0    
python 05_train_rl.py $TYPE tmdp+DFS -g 0
python 05_train_rl.py $TYPE tmdp+ObjLim -g 0

# Evaluation
python evaluate.py $TYPE -g 0
```
Optional: run steps 4 and 5 with flag `--wandb` to log the training metrics using wandb. This requires a wandb installation, an account and the appropriate projects.

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.


# Train Franeks test set:
## Commands

On the PC
```bash
docker build rl2branch -t rl2branch:cuda  
docker run -it -v  ~/User/fstark/mip_rl/rl2branch/data/:/root/rl2branch/data/  --network="host" --gpus all rl2branch:cuda bash
python3.8 05_train_rl.py $TYPE mdp -g 0 --wandb
```


On the cluster

```bash
ssh fstark@hpc3-login.dfki.uni-bremen.de
docker run -it -v ~/SCRATCH/data/:/root/rl2branch/data/ ~/SCRATCH/actor/mimpc/0:/root/rl2branch/actor/mimpc/0  --network="host" --gpus all rl2branch:cuda bash
srun --account=deepl --nodelist=hpc-dnode04 --job-name=fstark_rlbranch1 --pty --mem-per-cpu=8g --cpus-per-task=16 --gres=gpu:1 --partition=gpu_ampere shifter -v --image=fstark/rl2branch:cuda bash
export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gluster/public/cudadrv/lib64
python3.8 05_train_rl.py $TYPE mdp -g 0 --wandb
```
