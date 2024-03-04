import ray
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
#test branche opti
from time import sleep
from gymnasium import spaces
from environments import Exemple_MultiAgentsSupervisorOperatorsEnv as exemple_env
from environments import MultiAgentsSupervisorOperatorsEnv as env
from environments import MyMonoAgent as MonoAgentEnv  
from ray import tune
from ray.air import CheckpointConfig

if __name__ == "__main__":
    type = 'mono'
    do = 'simple'
    
    
    if type =="multi" : 

        if do == 'train' : 
            print("train")
            ray.init()

            def select_policy(algorithm, framework):
                if algorithm == "PPO":
                    if framework == "torch":
                        return PPOTorchPolicy
                    elif framework == "tf":
                        return PPOTF1Policy
                    else:
                        return PPOTF2Policy
                else:
                    raise ValueError("Unknown algorithm: ", algorithm)

            taille_map_x = 6
            taille_map_y = 3
            subzones_size=3
            nbr_sup = 1
            nbr_op = 1
            nbr_of_subzones = taille_map_x/subzones_size + taille_map_y / subzones_size
            ppo_config = (
                PPOConfig()
                # or "corridor" if registered above
                .environment(env,
                            env_config={
                                
                                "num_boxes_grid_width":taille_map_x,
                                "num_boxes_grid_height":taille_map_y,
                                "subzones_width":subzones_size,
                                "num_supervisors" : nbr_sup,
                                "num_operators" : nbr_op,
                                "num_directions" : 4,
                                "step_limit": 1000,
                                "same_seed" : False


                            })
                .environment(disable_env_checking=True)

                .framework("torch")

                # disable filters, otherwise we would need to synchronize those
                # as well to the DQN agent
                .rollouts(observation_filter="MeanStdFilter")
                .training(
                    model={"vf_share_layers": True},
                    vf_loss_coeff=0.01,
                    num_sgd_iter=6,
                    _enable_learner_api=False,
                )
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                .resources(num_gpus=0)
                #.rollouts(num_rollout_workers=1)
                .rl_module(_enable_rl_module_api=False)

            )
            tail_obs_sup = 2 + nbr_op + nbr_op * 2
            tail_obs_op = subzones_size * subzones_size *2 + 2 
            print("trail_obs_sup",tail_obs_sup)
            obs_supervisor = spaces.Box(low=0, high=taille_map_x, shape=(tail_obs_sup,))
            obs_operator = spaces.Box(low=0, high=taille_map_x, shape=(tail_obs_op,))

            action_supervisor  = spaces.MultiDiscrete([4, nbr_of_subzones-1])
            action_operator  = spaces.Discrete(4)

            policies = {
                "supervisor_policy": (None,obs_supervisor,action_supervisor, {}),
                "operator_policy": (None,obs_operator,action_operator, {}),
                #"operator_1": (None,obs_operator,acti, {}),
                #"operator_2": (None,obs_operator,acti, {}),
            }

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                #print("#################",agent_id,"#####################################")
                agent_type = agent_id.split('_')[0]
                if agent_type == "supervisor" :
                    #print(agent_id,"supervisor_policy")
                    return "supervisor_policy"

                else :
                    #print(agent_id,"operator_policy")
                    return "operator_policy"
    
            ppo_config.multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            ppo = ppo_config.build()

          
           
            
            i=0 
            j=0
            intervalle = 20
            while True :

                i+=1
                j+=1
                print("== Iteration", i, "==")
                print("-- PPO --")
                result_ppo = ppo.train()
                print(pretty_print(result_ppo))
                if j == intervalle :
                    j=0
                    save_result = ppo.save()
            print(save_result)
            
            
            path_to_checkpoint = save_result
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )

            # Let's terminate the algo for demonstration purposes.
            ppo.stop()
            
        elif do =='load':

            #================LOAD================

            # Use the Algorithm's `from_checkpoint` utility to get a new algo instance
            # that has the exact same state as the old one, from which the checkpoint was
            # created in the first place:
            my_new_ppo =  Algorithm.from_checkpoint("/home/ia/ray_results/PPO_MultiAgentsSupervisorOperatorsEnv_2024-02-28_16-29-277ov4wwa4/checkpoint_000160")

            i=0 
            j=0
            intervalle = 20
            while True :

                i+=1
                j+=1
               
                my_new_result_ppo = my_new_ppo.train()
                
                if j == intervalle :
                    j=0 
                    print("== Iteration", i, "==")
                    print("-- PPO --")
                    save_result = my_new_ppo.save()
                    print(pretty_print(my_new_result_ppo))
                    # restored_algo = Algorithm.from_checkpoint(checkpoint)

            path_to_checkpoint='/home/ia/ray_results/PPO_MultiAgentsSupervisorOperatorsEnv_2024-02-21_13-44-53fywm4k7f/checkpoint_000002'
            algo = Algorithm.from_checkpoint(path_to_checkpoint)
    
        elif do =='simple' : 


            def inference_policy_mapping_fn(agent_id):
                agent_type = agent_id.split('_')[0]
                if agent_type == "supervisor" :

                    return "supervisor_policy"

                else :

                    return "operator_policy"
            
            taille_map_x = 6
            taille_map_y = 3
            subzones_size=3
            nbr_sup = 1
            nbr_op = 1
            env_config={
                                "implementation":"simple",
                                "num_boxes_grid_width":taille_map_x,
                                "num_boxes_grid_height":taille_map_y,
                                "subzones_width":subzones_size,
                                "num_supervisors" : nbr_sup,
                                "num_operators" : nbr_op,
                                "num_directions" : 4,
                                "step_limit": 100000,
                                "same_seed" : True}
            env = env(env_config)
            print("env")
            algo = Algorithm.from_checkpoint("/home/ia/ray_results/PPO_MultiAgentsSupervisorOperatorsEnv_2024-02-28_16-10-427f_33wzj/checkpoint_000040")

            obs = env.reset()
            print(obs)

            num_episodes = 0
            num_episodes_during_inference =100

        
            episode_reward = {}

            while num_episodes < num_episodes_during_inference:
                num_episodes +=1 
                action = {}
                print("next step : ",num_episodes)

                
                for agent_id, agent_obs in obs.items():
                    
                    policy_id = inference_policy_mapping_fn(agent_id)
                    action[agent_id] = algo.compute_single_action(observation=agent_obs, policy_id=policy_id)

                print(action)
                obs, reward, done, info = env.step(action)
                print("next step : ",num_episodes)

                for id, thing in obs.items() :
                    print("id",id,":",thing) 

                for id, thing in reward.items() :
                    print("id :",id,":",thing)
        
                for id, thing in done.items() :
                    print("id :",id,":",thing)

               

                env.render()

        elif do == 'real' : 
            print("real")


            def inference_policy_mapping_fn(agent_id):
                agent_type = agent_id.split('_')[0]
                if agent_type == "supervisor" :

                    return "supervisor_policy"

                else :

                    return "operator_policy"
            
            taille_map_x = 12
            taille_map_y = 9
            subzones_size=5

            env_config={    "implementation": "real",
                            "num_boxes_grid_width":taille_map_x,
                            "num_boxes_grid_height":taille_map_y,
                            "subzones_width":subzones_size,
                            "num_supervisors" : 1,
                            "num_operators" : 3,
                            "num_directions" : 4,
                            "step_limit": 6000 
                        }
            env = env(env_config)
            algo = Algorithm.from_checkpoint("/home/ia/ray_results/PPO_MultiAgentsSupervisorOperatorsEnv_2024-02-26_09-29-19m4g99996/checkpoint_000002")

            obs = env.reset()
            print(obs)

            num_episodes = 0
            num_episodes_during_inference =10

        
            episode_reward = {}

            while num_episodes < num_episodes_during_inference:
                num_episodes +=1 
                action = {}
                print("next step : ",num_episodes)
                for agent_id, agent_obs in obs.items():

                    print(agent_id)
                
                    
                    policy_id = inference_policy_mapping_fn(agent_id)
                    action[agent_id] = algo.compute_single_action( 
                        observation=agent_obs, policy_id=policy_id)
                print(action)
                obs, reward, done, info = env.step(action)

                env.render()

        if do == "test_solo_trio" : 
                       
            taille_map_x = 9
            taille_map_y = 6
            subzones_size=3
            nbr_sup = 1
            nbr_op = 3
            nbr_of_subzones = taille_map_x/subzones_size + taille_map_y / subzones_size
            env_config={
                "implementation" : "simple",      
                "num_boxes_grid_width":taille_map_x,
                "num_boxes_grid_height":taille_map_y,
                "subzones_width":subzones_size,
                "num_supervisors" : nbr_sup,
                "num_operators" : nbr_op,
                "num_directions" : 4,
                "step_limit": 1000,
                "same_seed" : True }
            my_env = exemple_env(env_config)
            obs = my_env.reset()
            my_env.render()

            action_dict = [
                        [{"operator_0" : 3, "operator_1" : 4, "operator_2": 3,"supervisor_0" : [3,1,1,1]}],

                        [{"operator_0" : 0, "operator_1" : 4, "operator_2": 0,"supervisor_0" : [0,1,1,1]}],
                        [{"operator_0" : 0, "operator_1" : 4, "operator_2": 0,"supervisor_0" : [4,1,1,1]}],

                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 3,"supervisor_0" : [4,1,1,1]}],

                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,1,1,1]}],

                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,1,1,1]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,1,1,1]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,1,1,1]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,1,1,1]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,3,4,5]}],

                    # [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [3,3,4,2]}],

                        [{"operator_0" : 0, "operator_1" : 3, "operator_2": 0,"supervisor_0" : [0,3,4,5]}],

                        [{"operator_0" : 0, "operator_1" : 0, "operator_2": 0,"supervisor_0" : [0,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 0, "operator_2": 4,"supervisor_0" : [0,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 0, "operator_2": 4,"supervisor_0" : [4,3,4,5]}],

                        [{"operator_0" : 4, "operator_1" : 0, "operator_2": 4,"supervisor_0" : [4,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [2,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [2,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [2,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [2,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [2,3,4,5]}],
                        [{"operator_0" : 4, "operator_1" : 4, "operator_2": 4,"supervisor_0" : [2,3,4,5]}],
                            ]
                            
                    
            for i in range(len(action_dict)):

                print("=============acton",str(i),"================")
                action = action_dict[i][0]
                print(action)
                obs,reward,term,info = my_env.step(action)
                print(" ")
                print(" ")

                for id, thing in obs.items() :
                    print("id",id,":",thing) 

                for id, thing in reward.items() :
                    print("id :",id,":",thing)
        
                for id, thing in term.items() :
                    print("id :",id,":",thing)

                #print("term",term)    
                my_env.render()   
                sleep(1)
            while True :
                my_env.render()
        

    elif type=="mono" : 

        if do == 'train' : 
                 
                    print("train")  
                    ray.init()

                    taille_map_x = 5
                    taille_map_y = 5
                    subzones_size = 3 
                    nbr_sup = 1
                    nbr_op = 1
                                    

                    tune_config={
                                    "env": MonoAgentEnv,
                                     "env_config":{
                                    "implementation":"simple",
                                    "subzones_width":subzones_size,
                                    "num_boxes_grid_width":taille_map_x,
                                    "num_boxes_grid_height":taille_map_y,
                                    "n_orders" : 4,
                                    "step_limit": 1000,
                                    "same_seed" : False,
                                    
                                    },

                                    "num_workers": 1,

                                    #"num_learner_workers" : 0,
                                    "num_gpus": 0,
                                    #"num_gpus_per_worker": 2,

                                    "num_cpus_per_worker": 20,

                                    "model":{
                                             "fcnet_hiddens": [64, 64],  # Architecture du réseau de neurones (couches cachées)
                                            },
                                    "optimizer": {
                                           "learning_rate": 0.001,  # Taux d'apprentissage
                                         }
                                }
                    analysis = tune.run('PPO',name="PPO", config=tune_config,stop={"timesteps_total": 10000000}, checkpoint_config=CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=50),storage_path='/home/eliott/Desktop/platform/IA_model')

        if do == 'simple' : 
            

            algo = Algorithm.from_checkpoint("/home/eliott/Desktop/platform/IA_model/PPO/PPO_MyMonoAgent_04583_00000_0_2024-03-04_08-58-41/checkpoint_000550")
            #po=algo.get_policy()
            #algo.export_policy_model("/home/ia/Desktop")
            #algo.evaluate()
            
            taille_map_x = 5
            taille_map_y = 5
            subzones_size = 3 
            nbr_sup = 1
            nbr_op = 1
            env_config={
                                    "implementation":"simple",
                                    "subzones_width":subzones_size,
                                    "num_boxes_grid_width":taille_map_x,
                                    "num_boxes_grid_height":taille_map_y,
                                    "n_orders" : 4,
                                    "step_limit": 1000,
                                    "same_seed" : False
                        }
            
            mono_env = MonoAgentEnv(env_config=env_config)
            agent_obs = mono_env.reset()
            print("obs",agent_obs)
            mono_env.render()


            while True : 

                action =  algo.compute_single_action( observation=agent_obs)
                print(action)
                agent_obs, reward, done, info = mono_env.step(action)
                print("obs",agent_obs)
                print("obs",reward)
            


                mono_env.render()
                if done :
                    mono_env = MonoAgentEnv(env_config=env_config)
                    agent_obs = mono_env.reset()
                    print("obs",agent_obs)
                    mono_env.render()