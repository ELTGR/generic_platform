from utils_platform import UtilsPlatform
from environments import MonoAgentEnv


taille_map_x = 3
taille_map_y = 3
subzones_size = 3 
nbr_sup = 1
nbr_op = 1


env_config={
            "implementation":"simple",
            "subzones_width":subzones_size,
            "num_boxes_grid_width":taille_map_x,
            "num_boxes_grid_height":taille_map_y,
            "n_orders" : 3,
            "step_limit": 100,
            "same_seed" : False
            }


my_platfrom = UtilsPlatform(env_config=env_config,env = MonoAgentEnv)


#my_platfrom.train(checkpoint_freq=5)


my_platfrom.test(implementation = "simple", path="IA_model/PPO/PPO_MonoAgentEnv_0482c_00000_0_2024-03-04_14-20-49/checkpoint_000010")
