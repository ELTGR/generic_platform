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
from ray import tune    
from ray.air import CheckpointConfig



class DrlExperiments():

    def __init__(self,env_config,env) :

        self.env_config = env_config
        self.env_type= env
        
    def train(self,train_config) : 
        self.env_config['implementation'] = "simple"
        
        tune_config={
                                    "env": self.env_type,
                                    "env_config":self.env_config,
                                    "num_workers": train_config["num_workers"],
                                    #"num_learner_workers" : train_config["num_learner_workers"],
                                    "num_gpus": train_config["num_gpus"],
                                    #"num_gpus_per_worker": train_config["num_gpus_per_worker"],
                                    "num_cpus_per_worker": train_config["num_cpus_per_worker"],
                                    "model":train_config["model"],
                                    "optimizer": train_config["optimizer"],
                                }
        
        ray.init()
        algo = tune.run("PPO",name=train_config["name"],config = tune_config,stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path=train_config["path"]
                        )
                
      
                                                                  
    def train_from_checkpoint(self,train_config,path):
            
        self.env_config['implementation'] = "simple"
        ray.init()

        tune_config={
                                    "env": self.env_type,
                                    "env_config":self.env_config,
                                    "num_workers": train_config["num_workers"],
                                    #"num_learner_workers" : train_config["num_learner_workers"],
                                    "num_gpus": train_config["num_gpus"],
                                    #"num_gpus_per_worker": train_config["num_gpus_per_worker"],
                                    "num_cpus_per_worker": train_config["num_cpus_per_worker"],
                                    "model":train_config["model"],
                                    "optimizer": train_config["optimizer"],
                                }
        
        
        algo = tune.run("PPO",name=train_config["name"],config = tune_config,stop = {"timesteps_total": train_config["stop_step"]}, 
                        checkpoint_config = CheckpointConfig(checkpoint_at_end=True,checkpoint_frequency=train_config["checkpoint_freqency"] ),
                        storage_path=train_config["path"],restore=path
                        )
        
    
    def test(self,implementation, path) :
            
            self.env_config['implementation'] = implementation 
            print("config : ",self.env_config)

            env = self.env_type(env_config = self.env_config)
            loaded_model = Algorithm.from_checkpoint(path)
            agent_obs = env.reset()
            print("obs",agent_obs)
            env.render()

            while True : 

                action =  action = loaded_model.compute_single_action(agent_obs)
                print(action)
                agent_obs, reward, done, info = env.step(action)
                print("obs",agent_obs)
                print("obs",reward)
            
                env.render()
                if done :
                    env = self.env_type(env_config=self.env_config)
                    agent_obs = env.reset()
                    print("obs",agent_obs)
                    env.render()





if __name__ == '__main__':


    from Scenarios.UUV_Mono_Agent_TSP.env import UUVMonoAgentTSPEnv


    taille_map_x = 3
    taille_map_y = 3
    n_orders = 3
    step_limit = 100


    env_config={
                "implementation":"simple",
                
                "num_boxes_grid_width":taille_map_x,
                "num_boxes_grid_height":taille_map_y,
                "n_orders" : n_orders,
                "step_limit": step_limit,
                "same_seed" : False
                }

    train_config = {
                    "name" : str(taille_map_x)+"x"+str(taille_map_y)+"_"+str(n_orders)+"_"+str(step_limit),
                    "path" : "/home/ia/Desktop/generic_platform/Scenarios/UUV_Mono_Agent_TSP/models",
                    "checkpoint_freqency" : 10,
                    "stop_step" : 1000,
                    "num_workers": 1,
                    "num_learner_workers" : 0,
                    "num_gpus": 0,
                    "num_gpus_per_worker": 0,#
                    "num_cpus_per_worker": 5,
                    "model":{"fcnet_hiddens": [64, 64],},  # Architecture du réseau de neurones (couches cachées) 
                    "optimizer": {"learning_rate": 0.001,} # Taux d'apprentissage
    }

    my_platform = DrlExperiments(env_config=env_config,env = UUVMonoAgentTSPEnv)
    #my_platform.train(train_config=train_config) 

    #my_platform.test(implementation="simple",path="/home/ia/Desktop/generic_platform/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/PPO_UUVMonoAgentTSPEnv_8751c_00000_0_2024-03-07_11-07-39/checkpoint_000000")
    my_platform.train_from_checkpoint(train_config=train_config,path="/home/ia/Desktop/generic_platform/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/PPO_UUVMonoAgentTSPEnv_8751c_00000_0_2024-03-07_11-07-39/checkpoint_000000")