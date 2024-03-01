
from vehicle_implementations import UXVSimpleSimImplementation, Bluerov2RealImplementation

class UXV:
    def __init__(self, implementation="simple",ip_port=0):
        """
            implementation : choose if you use Bluerov2 or not ("bluerov2" or "simple")
        """
        if implementation == "simple":
            self.implementation = UXVSimpleSimImplementation()
        elif implementation == "real":
            print("in uxv real")
            self.implementation = Bluerov2RealImplementation(ip_port)
        else : 
            raise ValueError("Incorrect implementation value. Choose 'bluerov2' or 'simple'.")

    def get_pos(self):
        #print("UXV get_pos ")
        return self.implementation.get_pos()
     
    def set_pos(self,new_pose):
        #print("UXV set_pos : " , new_pose[0],new_pose[1])
        return self.implementation.set_pos(new_pose)
    def get_info(self) : 
        return self.implementation.get_info()