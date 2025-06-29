import ecole
import threading
import queue

import ecole.observation
import ecole.reward
import utilities
import numpy as np
from collections import namedtuple
from pyscipopt import Model, SCIP_PARAMSETTING

from custom_rewards import *


class AgentPool():
    """
    Class holding the reference to the agents and the policy sampler.
    Puts jobs in the queue through job sponsors.
    """
    def __init__(self, brain, n_agents, time_limit, mode, optimal_vals):
        self.jobs_queue = queue.Queue()
        self.policy_queries_queue = queue.Queue()
        self.policy_sampler = PolicySampler("Policy Sampler", brain, self.policy_queries_queue)
        self.agents = [Agent(f"Agent {i}", time_limit, self.jobs_queue, self.policy_queries_queue, mode, optimal_vals) for i in range(n_agents)]

    def start(self):
        self.policy_sampler.start()
        for agent in self.agents:
            agent.start()

    def close(self):
        # order the episode sampling agents to stop
        for _ in self.agents:
            self.jobs_queue.put(None)
        self.jobs_queue.join()
        # order the policy sampler to stop
        self.policy_queries_queue.put(None)
        self.policy_queries_queue.join()

    def start_job(self, instances, sample_rate, greedy=False, block_policy=False, heuristics_on=True, heuristics_off=False):
        """
        Starts a job.
        A job is a set of tasks. A task consists of an instance that needs to be solved and instructions
        to do so (sample rate, greediness).
        The job queue is loaded with references to the job sponsor, which is in itself a queue specific
        to a job. It is the job sponsor who holds the lists of tasks. The role of the job sponsor is to
        keep track of which tasks have been completed.
        """
        job_sponsor = queue.Queue()
        samples = []
        stats = []

        policy_access = threading.Event()
        if not block_policy:
            policy_access.set()

        for instance in instances:
            if heuristics_on:
                task = {'instance': instance, 'sample_rate': sample_rate, 'greedy': greedy,
                        'samples': samples, 'stats': stats, 'policy_access': policy_access, 'heuristics': True}
                job_sponsor.put(task)
                self.jobs_queue.put(job_sponsor)
            if heuristics_off:
                task = {'instance': instance, 'sample_rate': sample_rate, 'greedy': greedy,
                        'samples': samples, 'stats': stats, 'policy_access': policy_access, 'heuristics': False}
                job_sponsor.put(task)
                self.jobs_queue.put(job_sponsor)

        ret = (samples, stats, job_sponsor)
        if block_policy:
            ret = (*ret, policy_access)

        return ret

    def wait_completion(self):
        # wait for all running episodes to finish
        self.jobs_queue.join()


class PolicySampler(threading.Thread):
    """
    Gathers policy sampling requests from the agents, and process them in a batch.
    """
    def __init__(self, name, brain, requests_queue):
        super().__init__(name=name)
        self.brain = brain
        self.requests_queue = requests_queue

    def run(self):
        stop_order_received = False
        while True:
            requests = []
            request = self.requests_queue.get()
            while True:
                # check for a stopping order
                if request is None:
                    self.requests_queue.task_done()
                    stop_order_received = True
                    break
                # add request to the batch
                requests.append(request)
                # keep collecting more requests if available, without waiting
                try:
                    request = self.requests_queue.get(block=False)
                except queue.Empty:
                    break

            states = [r['state'] for r in requests]
            greedys = [r['greedy'] for r in requests]
            receivers = [r['receiver'] for r in requests]

            # process all requests in a batch
            action_idxs = self.brain.sample_action_idx(states, greedys)
            for action_idx, receiver in zip(action_idxs, receivers):
                receiver.put(action_idx)
                self.requests_queue.task_done()

            if stop_order_received:
                break


class Agent(threading.Thread):
    """
    Agent class. Receives tasks from the job sponsor, runs them and samples transitions if
    requested.
    """
    def __init__(self, name, time_limit, jobs_queue, policy_queries_queue, mode, optimal_vals):
        super().__init__(name=name)
        self.jobs_queue = jobs_queue
        self.policy_queries_queue = policy_queries_queue
        self.policy_answers_queue = queue.Queue()
        self.mode = mode
        self.time_limit = time_limit
        self.optimal_val_lookup_fun  = lambda name : optimal_vals[name]
        # Setup Ecole environment
        scip_params={'separating/maxrounds': 0,
                     'presolving/maxrestarts': 0,
                     'timing/clocktype': 2,
                     'display/verblevel': 5
                     }
        observation_function=(
            ecole.observation.FocusNode(),
            ecole.observation.NodeBipartite(),
            ecole.observation.TreeRecorder()
            )
        reward_function=PrimalGapFunction(primal_bound_lookup_fun=self.optimal_val_lookup_fun)
        information_function= {
            'nnodes': ecole.reward.NNodes().cumsum(),
            'lpiters': ecole.reward.LpIterations().cumsum(),
            'num_lps': DeltaNumLPs().cumsum(),
            'time': ecole.reward.SolvingTime().cumsum(),
            'primal_obj': PrimalObj(),
            'primal_gap': PrimalGapFunction(primal_bound_lookup_fun=self.optimal_val_lookup_fun),
            'primal_integral_lpiters': (ecole.reward.LpIterations() * PrimalGapFunction(primal_bound_lookup_fun=self.optimal_val_lookup_fun)).cumsum(),
            'primal_integral_time': (ecole.reward.SolvingTime() * PrimalGapFunction(primal_bound_lookup_fun=self.optimal_val_lookup_fun)).cumsum(),
            'confined_primal_integral': ecole.reward.ConfinedPrimalGapIntegral(self.optimal_val_lookup_fun, self.time_limit, importance=0.0001, name='1'),
            'num_lps_for_first_feasible': BeforeFirstFesibleSol(DeltaNumLPs().cumsum()),
            'sub_optimality_0.01': ecole.reward.SubOptimality(0.01, self.optimal_val_lookup_fun),
            'sub_optimality_0.05': ecole.reward.SubOptimality(0.01, self.optimal_val_lookup_fun),
            'sub_optimality_0.1': ecole.reward.SubOptimality(0.1, self.optimal_val_lookup_fun),
            'sub_optimality_0.5': ecole.reward.SubOptimality(0.1, self.optimal_val_lookup_fun),
            'sub_optimality_0.2': ecole.reward.SubOptimality(0.2, self.optimal_val_lookup_fun),
            'sub_optimality_0.3': ecole.reward.SubOptimality(0.2, self.optimal_val_lookup_fun),
            'sub_optimality_0.4': ecole.reward.SubOptimality(0.2, self.optimal_val_lookup_fun),
            'sub_optimality_0.5': ecole.reward.SubOptimality(0.5, self.optimal_val_lookup_fun),
            'sub_optimality_1.0': ecole.reward.SubOptimality(1.0, self.optimal_val_lookup_fun),
            
        }

        

        if mode == 'tmdp+ObjLim':
            self.env = ObjLimBranchingEnv(scip_params=scip_params,
                                          pseudo_candidates=False,
                                          observation_function=observation_function,
                                          reward_function=reward_function,
                                          information_function=information_function)
        elif mode == 'tmdp+DFS':
            self.env = DFSBranchingEnv(scip_params=scip_params,
                                       pseudo_candidates=False,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        elif mode == 'mdp':
            self.env = MDPBranchingEnv(scip_params=scip_params,
                                       pseudo_candidates=False,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        else:
            raise NotImplementedError

    def run(self):
        while True:
            job_sponsor = self.jobs_queue.get()

            # check for a stopping order
            if job_sponsor is None:
                self.jobs_queue.task_done()
                break

            # Get task from job sponsor
            task = job_sponsor.get()
            instance = task['instance']
            sample_rate = task['sample_rate']
            greedy = task['greedy'] # should actions be chosen greedily w.r.t. the policy?
            training = not greedy
            samples = task['samples']
            stats = task['stats']
            policy_access = task['policy_access']
            seed = instance['seed']
            heuristics = task['heuristics']

            transitions = []
            self.env.seed(seed)
            rng = np.random.RandomState(seed)
            #if sample_rate > 0:
            #    tree_recorder = TreeRecorder()
            
            #print("before spiode reset")
            # Run episode
            observation, action_set, reward, done, info = self.env.reset(instance = instance['path'],
                                                                             primal_bound=instance.get('sol', None),
                                                                             training=training, heuristics=heuristics, time_limit=self.time_limit)
            #print(f"agent on {instance['path']} reset info {info}")

            policy_access.wait()
            iter_count = 0
            nan_found = False
            while not done:
                focus_node_obs, node_bipartite_obs, tree_record = observation
                                    
                state = utilities.extract_state(node_bipartite_obs, action_set, focus_node_obs.number)

                # send out policy queries
                self.policy_queries_queue.put({'state': state, 'greedy': greedy, 'receiver': self.policy_answers_queue})
                action_idx = self.policy_answers_queue.get()

                action = action_set[action_idx]

                # collect transition samples if requested
                if sample_rate > 0:
                    #tree_recorder.record_branching_decision(focus_node_obs, reward)
                    keep_sample = rng.rand() < sample_rate
                    if keep_sample:
                        transition = utilities.Transition(state, action_idx, reward, tree_record)
                        transitions.append(transition)

                observation, action_set, reward, done, info = self.env.step(action)
                #print(f"agent on {instance['path']} step reward {cum_nnodes}")
                iter_count += 1
                if (iter_count>50000) and training: done=True # avoid too large trees during training for stability

            print(f"agent on {instance['path']} finished after {iter_count} iters, primal_integral at {info['primal_integral_lpiters']}, primal_gap at {info['primal_gap']}. Num transitions recorded: {len(transitions)} - heuristics {heuristics}")

            if nan_found:
                job_sponsor.task_done()
                self.jobs_queue.task_done()
                continue

            if (iter_count>50000) and training: # avoid too large trees during training for stability
                job_sponsor.task_done()
                self.jobs_queue.task_done()
                continue

            # post-process the collected samples (credit assignment)
            if sample_rate > 0:
                if self.mode in ['tmdp+ObjLim', 'tmdp+DFS']:
                    for transition in transitions:
                        transition.returns = -transition.tree_record.get_sub_tree_confined_primal_gap_integral(instance.get('sol', None), self.time_limit, 0.8, True)
                else:
                    assert self.mode == 'mdp'
                    for transition in transitions:
                        gap_improvement = transition.cum_nnodes - reward  #"reward <= transition.cum_nnodes "

                        if gap_improvement <= np.finfo(float).eps:
                            transition.returns = -1
                        else:
                            transition.returns = gap_improvement


                

            
            # record episode samples and stats
            samples.extend(transitions)
            stats.append({'order': task, 'info': info, 'heuristics': heuristics})

            # tell both the agent pool and the original task sponsor that the task is done
            job_sponsor.task_done()
            self.jobs_queue.task_done()
            #input("press enter")


class TreeRecorder:
    """
    Records the branch-and-bound tree from a custom brancher.

    Every node in SCIP has a unique node ID. We identify nodes and their corresponding
    attributes through the same ID system.
    Depth groups keep track of groups of nodes at the same depth. This data structure
    is used to speed up the computation of the subtree size.
    """
    def __init__(self):
        self.tree = {}
        self.depth_groups = []

    def record_branching_decision(self, focus_node, loss, lp_cand=True):
        id = focus_node.number
        # Tree
        self.tree[id] = {'parent': focus_node.parent_number,
                         'lowerbound': focus_node.lowerbound,
                         'num_children': 2 if lp_cand else 3,
                         'loss': loss
                             }
        # Add to corresponding depth group
        if len(self.depth_groups) > focus_node.depth:
            self.depth_groups[focus_node.depth].append(id)
        else:
            self.depth_groups.append([id])

    def calculate_subtree_sizes(self):
        subtree_sizes = {id: 0 for id in self.tree.keys()}
        for group in self.depth_groups[::-1]:
            for id in group:
                parent_id = self.tree[id]['parent']
                subtree_sizes[id] += self.tree[id]['num_children']
                if parent_id >= 0: subtree_sizes[parent_id] += subtree_sizes[id]
        return subtree_sizes
    
    def calculate_subtree_lowest_child_loss(self, incubent_nodes, incubent_nodes_by_parent):
        sub_tree_lowest_losses = {}
        for group in self.depth_groups[::-1]:
            for id in group:
                parent_id = self.tree[id]['parent']
                if id in sub_tree_lowest_losses: # Has to take the loss from below me
                    loss = sub_tree_lowest_losses[id]
                else: # Is a child node and has to take is loss
                    loss = self.__get_lowest_leaf_loss(id, incubent_nodes, incubent_nodes_by_parent)
                if parent_id >= 0:
                    if parent_id not in sub_tree_lowest_losses:
                        sub_tree_lowest_losses[parent_id] = loss
                    elif sub_tree_lowest_losses[parent_id] > loss:
                        sub_tree_lowest_losses[parent_id] = loss

    def get_loss(self, id):
        return self.tree[id]['loss']
    
    def __get_lowest_leaf_loss(self, id, incubent_nodes, incubent_nodes_by_parent):
        if id in incubent_nodes_by_parent:
            if len(incubent_nodes_by_parent[id]) == 2:
                loss_a = incubent_nodes[incubent_nodes_by_parent[id]][0]['integral']
                loss_b = incubent_nodes[incubent_nodes_by_parent[id]][1]['integral']
                return min(loss_a, loss_b)
            else:
                assert len(incubent_nodes_by_parent[id]) == 1
                return incubent_nodes[incubent_nodes_by_parent[id]][0]['integral']
        else:
            return self.tree[id]['loss']



class DFSBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that changes the node strategy to DFS when training.
    """
    def reset_dynamics(self, model, primal_bound, training, heuristics, time_limit,*args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if training:
            # Set the dfs node selector as the least important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 666666)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 666666)
        else:
            # Set the dfs node selector as the most important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 0)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 0)
        
        if heuristics:
            pyscipopt_model.setHeuristics(SCIP_PARAMSETTING.DEFAULT)
        else:
            pyscipopt_model.setHeuristics(SCIP_PARAMSETTING.OFF)

        if training:
            pyscipopt_model.setParam(f"limits/time", time_limit)
        else:
            pyscipopt_model.setParam(f"limits/time", time_limit)

        return super().reset_dynamics(model, *args, **kwargs)

class DFSBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = DFSBranchingDynamics

class ObjLimBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that allows the user to set an initial primal bound.
    """
    def reset_dynamics(self, model, primal_bound, training, heuristics, time_limit, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if primal_bound is not None:
            pyscipopt_model.setObjlimit(primal_bound)
        
        if heuristics:
            pyscipopt_model.setHeuristics(SCIP_PARAMSETTING.DEFAULT)
        else:
            pyscipopt_model.setHeuristics(SCIP_PARAMSETTING.OFF)

        if training:
            pyscipopt_model.setParam(f"limits/time", time_limit)
        else:
            pyscipopt_model.setParam(f"limits/time", time_limit)

        return super().reset_dynamics(model, *args, **kwargs)

class ObjLimBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = ObjLimBranchingDynamics

class MDPBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Regular branching environment that allows extra input parameters, but does
    not use them.
    """
    def reset_dynamics(self, model, primal_bound, training, heuristics, time_limit, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if heuristics:
            pyscipopt_model.setHeuristics(SCIP_PARAMSETTING.DEFAULT)
        else:
            pyscipopt_model.setHeuristics(SCIP_PARAMSETTING.OFF)

        if training:
            pyscipopt_model.setParam(f"limits/time", time_limit)
        else:
            pyscipopt_model.setParam(f"limits/time", time_limit)

        return super().reset_dynamics(model, *args, **kwargs)
    

class MDPBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = MDPBranchingDynamics
