# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the reinforcement learning method. User must provide a      #
# mode in {mdp, tmdp+DFS, tmdp+ObjLim}. The training parameters are read from   #
# a file config.default.json which is overriden by command line inputs, if      #
# provided.                                                                     #
# Usage:                                                                        #
# python 04_train_il.py <type> -s <seed> -g <cudaId>                            #
# Optional: use flag --wandb to log metrics using wandb (requires account)      #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #


import os
import json
import time
import glob
import numpy as np
import argparse

import ecole
from pathlib import Path
from datetime import datetime
from scipy.stats.mstats import gmean



if __name__ == '__main__':
    # read default config file
    with open("config.default.json", 'r') as f:
        config = json.load(f)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack', 'mimpc'],
        nargs='?',
        default='mimpc',
    )
    parser.add_argument(
        'mode',
        help='Training mode.',
        choices=['mdp', 'tmdp+DFS', 'tmdp+ObjLim'],
        nargs='?',
        default='mdp',
    )
    parser.add_argument(
        '--wandb',
        help="Use wandb?",
        default=False,
        action="store_true",
    )
    # add all config parameters as optional command-line arguments
    for param, value in config.items():
        if param == 'gpu':
            parser.add_argument(
                '-g', '--gpu',
                type=type(value),
                help='CUDA GPU id (-1 for CPU).',
                default=argparse.SUPPRESS,
            )
        elif param == 'seed':
            parser.add_argument(
                '-s', '--seed',
                type=type(value),
                help = 'Random generator seed.',
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                f"--{param}",
                type=type(value),
                default=argparse.SUPPRESS,
            )
    args = parser.parse_args()

    # override config with the user config file if provided
    if os.path.isfile("config.json"):
        with open("config.json", 'r') as f:
            user_config = json.load(f)
        unknown_options = user_config.keys() - config.keys()
        if unknown_options:
            raise ValueError(f"Unknown options in config file: {unknown_options}")
        config.update(user_config)

    # override config with command-line arguments if provided
    args_config = {key: getattr(args, key) for key in config.keys() & vars(args).keys()}
    config.update(args_config)

    # configure gpu
    if config['gpu'] == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config['gpu']}"
        device = f"cuda:0"

    # import torch after gpu configuration
    import torch
    import torch.nn.functional as F
    import utilities
    from brain import Brain
    from agent import AgentPool

    if config['gpu'] > -1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Active CUDA Device: {torch.cuda.current_device()}")

    rng = np.random.RandomState(config['seed'])
    torch.manual_seed(config['seed'])

    logger = utilities.configure_logging()
    if args.wandb:
        import wandb
        wandb.init(project="rl2branch", config=config)


    # data
    if args.problem == "setcover":
        maximization = False
        valid_path = "data/instances/setcover/valid_400r_750c_0.05d"
        train_path = "data/instances/setcover/train_400r_750c_0.05d"

    elif args.problem == "cauctions":
        maximization = True
        valid_path = "data/instances/cauctions/valid_100_500"
        train_path = "data/instances/cauctions/train_100_500"

    elif args.problem == "indset":
        maximization = True
        valid_path = "data/instances/indset/valid_500_4"
        train_path = "data/instances/indset/train_500_4"

    elif args.problem == "ufacilities":
        maximization = False
        valid_path = "data/instances/ufacilities/valid_35_35_5"
        train_path = "data/instances/ufacilities/train_35_35_5"

    elif args.problem == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"

    elif args.problem == "mimpc":
        maximization = False
        valid_path = "data/instances/mimpc/valid"
        train_path = "data/instances/mimpc/train"

    # recover training / validation instances
    valid_instances = [f'{valid_path}/instance_{j+1}.lp' for j in range(config["num_valid_instances"])]
    train_instances = [f'{train_path}/instance_{j+1}.lp' for j in range(len(glob.glob(f'{train_path}/instance_*.lp')))]

    # collect the pre-computed optimal solutions for the training instances
    with open(f"{train_path}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)
    with open(f"{valid_path}/instance_solutions.json", "r") as f:
        valid_sols = json.load(f)
    
    # collect the tree sizes
    with open(f"{train_path}/instance_nnodes.json", "r") as f:
        train_nnodes = json.load(f)

    valid_batch = [{'path': instance, 'seed': seed, 'sol': valid_sols[instance]}
        for instance in valid_instances
        for seed in range(config['num_valid_seeds'])]

    def train_batch_generator():
        eps = -0.1 if maximization else 0.1
        while True:
            yield [{'path': instance, 'sol': train_sols[instance] + eps, 'seed': rng.randint(0, 2**32)}
                    for instance in rng.choice(train_instances, size=config['num_episodes_per_epoch'], replace=True)]
            
    def train_batch_generator_new():
        eps = -0.1 if maximization else 0.1

        while True:
            batch = []
            total_nodes = 0

            while total_nodes < config['num_nodes_per_epoch'] or len(batch) < config['num_episodes_per_epoch']:
                # Randomly sample an instance
                instance = rng.choice(train_instances)

                # Exclude instances with only 1 node
                if train_nnodes[instance] <= 1:
                    print(f"instance {instance} is exclued as it has only {train_nnodes[instance]} nodes")
                    continue

                # Add the instance to the batch
                batch.append({
                    'path': instance,
                    'sol': train_sols[instance] + eps,
                    'seed': rng.randint(0, 2**32)
                })

                # Accumulate the total number of nodes
                total_nodes += train_nnodes[instance]
            print(f"vield batch with {len(batch)} instances and total {total_nodes} nodes")
            yield batch

    train_batches = train_batch_generator_new()

    logger.info(f"Training on {len(train_instances)} training instances and {len(valid_instances)} validation instances")


    brain = Brain(config, device, args.problem, args.mode)
    agent_pool = AgentPool(brain, config['num_agents'], config['time_limit'], args.mode)
    agent_pool.start()
    is_validation_epoch = lambda epoch: (epoch % config['validate_every'] == 0) or (epoch == config['num_epochs'])
    is_training_epoch = lambda epoch: (epoch < config['num_epochs'])

    # Already start jobs
    if is_validation_epoch(0):
        _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True, heuristics_on=config['heuristics_during_validation_on'], heuristics_off=config['heuristics_during_validation_off'])

    if is_training_epoch(0):
        train_batch = next(train_batches)
        t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True,heuristics_on=config['heuristics_during_training'], heuristics_off= not config['heuristics_during_training'])

    # training loop
    start_time = datetime.now()
    best_tree_size = np.inf
    for epoch in range(config['num_epochs'] + 1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}

        # Allow preempted jobs to access policy
        if is_validation_epoch(epoch):
            v_stats, v_queue, v_access = v_stats_next, v_queue_next, v_access_next
            v_access.set()
            logger.info(f"  {len(valid_batch)} validation jobs running (preempted)")
            # do not do anything with the stats yet, we have to wait for the jobs to finish !
        else:
            logger.info(f"  validation skipped")
        if is_training_epoch(epoch):
            t_samples, t_stats, t_queue, t_access = t_samples_next, t_stats_next, t_queue_next, t_access_next
            t_access.set()
            logger.info(f"  {len(train_batch)} training jobs running (preempted)")
            # do not do anything with the samples or stats yet, we have to wait for the jobs to finish !
        else:
            logger.info(f"  training skipped")

        # Start next epoch's jobs
        if epoch + 1 <= config["num_epochs"]:
            if is_validation_epoch(epoch + 1):
                _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True, heuristics_on=config['heuristics_during_validation_on'], heuristics_off=config['heuristics_during_validation_off'])
            if is_training_epoch(epoch + 1):
                train_batch = next(train_batches)
                t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True,heuristics_on=config['heuristics_during_training'], heuristics_off= not config['heuristics_during_training'])

        # Validation
        if is_validation_epoch(epoch):
            v_queue.join()  # wait for all validation episodes to be processed
            logger.info('  validation jobs finished')

        for heur in [True, False]:
            v_nnodess = [s['info']['nnodes'] for s in v_stats if s['heuristics'] == heur]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats if s['heuristics'] == heur]
            v_num_lps = [s['info']['num_lps'] for s in v_stats if s['heuristics'] == heur]
            v_times = [s['info']['time'] for s in v_stats if s['heuristics'] == heur]
            v_subopt_gap = [s['info']['subopt_gap'] for s in v_stats if s['heuristics'] == heur]
            v_primal_obj = [s['info']['primal_obj'] for s in v_stats if s['heuristics'] == heur]
            v_gap = [s['info']['gap'] for s in v_stats if s['heuristics'] == heur]
            v_normed_gap_integral = [s['info']['normed_gap_integral'] for s in v_stats if s['heuristics'] == heur]
            v_normed_optimality_integral = [s['info']['normed_optimality_integral'] for s in v_stats if s['heuristics'] == heur]

            if(len(v_nnodess) == 0):
                continue

            heur_str = '_h' if heur else ''
            wandb_data.update({
                f'valid{heur_str}_nnodes_g': gmean(np.asarray(v_nnodess) + 1) - 1,
                f'valid{heur_str}_nnodes': np.mean(v_nnodess),
                f'valid{heur_str}_nnodes_max': np.amax(v_nnodess),
                f'valid{heur_str}_nnodes_min': np.amin(v_nnodess),
                f'valid{heur_str}_time': np.mean(v_times),
                f'valid{heur_str}_lpiters': np.mean(v_lpiterss),
                f'valid{heur_str}_num_lps': np.mean(v_num_lps),
                f'valid{heur_str}_num_lps_min': np.amin(v_num_lps),
                f'valid{heur_str}_num_lps_max': np.amax(v_num_lps),
                f'valid{heur_str}_subopt_gap' : np.mean(v_subopt_gap),
                f'valid{heur_str}_subopt_gap_median' : np.median(v_subopt_gap),
                f'valid{heur_str}_subopt_gap_ninf' : np.isinf(v_subopt_gap).sum(),
                f'valid{heur_str}_subopt_gap_max' : np.amax(v_subopt_gap),
                f'valid{heur_str}_subopt_gap_min' : np.amin(v_subopt_gap),
                f'valid{heur_str}_primal_obj' : np.mean(v_primal_obj),
                f'valid{heur_str}_primal_obj_max' : np.amax(v_primal_obj),
                f'valid{heur_str}_primal_obj_min' : np.amin(v_primal_obj),
                f'valid{heur_str}_gap' : np.mean(v_gap),
                f'valid{heur_str}_gap_median' : np.median(v_gap),
                f'valid{heur_str}_gap_ninf' : np.isinf(v_gap).sum(),
                f'valid{heur_str}_gap_max' : np.amax(v_gap),
                f'valid{heur_str}_gap_min' : np.amin(v_gap),
                f'valid{heur_str}_normed_gap_integral' : np.mean(v_normed_gap_integral),
                f'valid{heur_str}_normed_optimality_integral' : np.mean(v_normed_optimality_integral)
            })
            if epoch == 0:
                v_nnodes_0 = wandb_data[f'valid{heur_str}_nnodes'] if wandb_data[f'valid{heur_str}_nnodes'] != 0 else 1
                v_nnodes_g_0 = wandb_data[f'valid{heur_str}_nnodes_g'] if wandb_data[f'valid{heur_str}_nnodes_g']!= 0 else 1
            wandb_data.update({
                f'valid{heur_str}_nnodes_norm': wandb_data[f'valid{heur_str}_nnodes'] / v_nnodes_0,
                f'valid{heur_str}_nnodes_g_norm': wandb_data[f'valid{heur_str}_nnodes_g'] / v_nnodes_g_0,
            })

        if wandb_data['valid_h_nnodes_g'] < best_tree_size:
            best_tree_size = wandb_data['valid_h_nnodes_g']
            logger.info('Best parameters so far (1-shifted geometric mean), saving model.')
            brain.save()
        # saving every valid epoch:
        brain.save_epoch(epoch)

        # Training
        if is_training_epoch(epoch):
            t_queue.join()  # wait for all training episodes to be processed
            logger.info('  training jobs finished')
            logger.info(f"  {len(t_samples)} training samples collected")
            t_losses = brain.update(t_samples)
            logger.info('  model parameters were updated')

            t_nnodess = [s['info']['nnodes'] for s in t_stats]
            t_lpiterss = [s['info']['lpiters'] for s in t_stats]
            t_num_lps = [s['info']['num_lps'] for s in t_stats]
            t_times = [s['info']['time'] for s in t_stats]
            t_subopt_gap = [s['info']['subopt_gap'] for s in t_stats]
            t_primal_obj = [s['info']['primal_obj'] for s in t_stats]
            t_gap = [s['info']['gap'] for s in t_stats]
            t_normed_gap_integral = [s['info']['normed_gap_integral'] for s in t_stats]
            t_normed_optimality_integral = [s['info']['normed_optimality_integral'] for s in t_stats]


            wandb_data.update({
                'train_nnodes_g': gmean(t_nnodess),
                'train_nnodes': np.mean(t_nnodess),
                'train_time': np.mean(t_times),
                'train_lpiters': np.mean(t_lpiterss),
                'train_num_lps': np.mean(t_num_lps),
                'train_num_lps_min': np.amin(t_num_lps),
                'train_num_lps_max': np.amax(t_num_lps),
                'train_nsamples': len(t_samples),
                'train_loss': t_losses.get('loss', None),
                'train_reinforce_loss': t_losses.get('reinforce_loss', None),
                'train_entropy': t_losses.get('entropy', None),
                'train_subopt_gap' : np.mean(t_subopt_gap),
                'train_subopt_max' : np.amax(t_subopt_gap),
                'train_subopt_min' : np.amin(t_subopt_gap),
                'train_subopt_gap' : np.mean(t_subopt_gap),
                'train_subopt_max' : np.amax(t_subopt_gap),
                'train_subopt_min' : np.amin(t_subopt_gap),
                'train_primal_obj' : np.mean(t_primal_obj),
                'train_primal_obj_max' : np.amax(t_primal_obj),
                'train_primal_obj_min' : np.amin(t_primal_obj),
                'train_gap' : np.mean(t_gap),
                'train_gap_max' : np.amax(t_gap),
                'train_gap_min' : np.amin(t_gap),
                'train_normed_gap_integral' : np.mean(t_normed_gap_integral),
                'train_normed_optimality_integral' : np.mean(t_normed_optimality_integral),
                't_samples': len(t_samples)
            })

        # Send the stats to wandb
        if args.wandb:
            wandb.log(wandb_data, step = epoch)

        # If time limit is hit, stop process
        elapsed_time = datetime.now() - start_time
        if elapsed_time.days >= 6: break

    logger.info(f"Done. Elapset time: {elapsed_time}")
    if args.wandb:
        wandb.join()
        wandb.finish()

    v_access_next.set()
    t_access_next.set()
    agent_pool.close()
