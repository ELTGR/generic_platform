
from environments.My_Custom_Env.implementations import SimpleImplementation, RealImplementation
#The bridge is where to create an instance of the implementation choose in the env_config
class Agent:

    def __init__(self, implementation="simple"):
        """
            implementation : choose if you use Bluerov2 or not ("bluerov2" or "simple")
        """
        if implementation == "simple":
            self.implementation = SimpleImplementation()

        elif implementation == "real":
            self.implementation = RealImplementation()

        else : 
            raise ValueError("Incorrect implementation value. Choose 'real' or 'simple'.")

   

    def get_info(self) : 
        return self.implementation.get_info()
