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
import torch 
import tensorflow as tf

class UtilsPlatform():

    def __init__(self,env_config,env) :

        self.env_config = env_config
        self.env_type= env
        
    def train(self,train_config) : 
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
                        storage_path=train_config["path"]
                        )
                
        # Retrieve the best trial
        best_trial = algo.get_best_trial("episode_reward_mean", mode="max")

        # Get the trained model from the best trial
        trained_model = best_trial.get_model()

        # Save the trained model (assuming it's a PyTorch model)
        torch.save(trained_model.state_dict(), "path/to/save/trained_model.pth")

                                                                    
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
            loaded_model = tf.keras.models.load_model(path)
            agent_obs = env.reset()
            print("obs",agent_obs)
            env.render()

            while True : 

                action =  action = loaded_model.predict(agent_obs)
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
 