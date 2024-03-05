
# Generic Platform
-The goal is to create a generic platform to train planty IAs on environements.

## Organisation
-First you will found one folder "environments" and 2 files "start.py and utils_platfrom.py
![Screenshot from 2024-03-05 13-16-53](https://github.com/ELTGR/generic_platform/assets/122261448/50d39fbc-71fc-4645-af29-9f51f916a1c8)

### Lauchn
- Start.py is usefull to train, test your IA in a simple way.
- utils_platfrom.py is the file where you can found all the train, test fonction but less simple.
### Environments
- You will found 2 folder, one is the folder "UUV_Mono_Agent", it's an exemple of environment.
![image](https://github.com/ELTGR/generic_platform/assets/122261448/492a543b-f581-4563-b643-62b21f8b270e)
- The second "My_Custom_Env" is a esqueleton of your environement, copie and paste it to create your owne environment.
- 
![Screenshot from 2024-03-05 13-30-54](https://github.com/ELTGR/generic_platform/assets/122261448/3a438b89-a1ef-470f-8e8b-a158e8c8fa72)

- Ia_model  is just where u fill save your model.
- env is where the principal fonction are.
- utils is where your spécific fonction are.
- views2d provide a 2d visualisation of your env (usefull during testing).
- bridge allow you to swith between a simple vehicule (UUV,USV,UAV) to a real vehicle.
- Inside implementations.py you will found 2class SimpleImplementation and RealImplementation. Simple is for the trainning, it's where to fake a real vehicule during the trainning part. Real is for the testing part, but you can also test with simple implementation
