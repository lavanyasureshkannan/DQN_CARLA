# DQN_CARLA

### This repository includes the final project from the course ENPM690- ROBOT LEARNING
## 
## Authors
Akhilrajan Vethirajan
[v.akhilrajan@gmail.com](https://github.com/Akhilrajan-V) 

Lavanya Suresh Krishnan

## 
![output](https://github.com/lavanyasureshkannan/DQN_CARLA/blob/main/Outputs/simulation.gif)

# SYSTEM SETUP
## Requirements
---
- Carla Simulator Windows/Ubuntu (This Project was executed in Windows and it is more easier to boot Carla in Win than Ubuntu)
- A virtual environment loaded with:
   - Python 3.7
   - TensorFlow 1.15.0 
   - Keras 2.2.4
   - Pygame
   - Opencv
   - TQDM
   - Math
---

## Best Practice
**Use Anaconda as your virual environmet/similar**

While creating a new virtual environment using conda create

  **Create environment with tensorflow this will import and install required dependencies especially Numpy and Backend Estimators** 

  ```
  conda create -n your-env-name tensorflow-gpu
  
  conda activate your-env-name 
  ```
 
Refer Conda documentation [here](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)

---
**Note: Any other Tensorflow or Keras version does not work**

**TensorFlow 1.x has minimal support on Nvidia RTX GPU cards**

# PROCEDURE
## TRAINING THE MODEL
1. Download the carla simulator. 
 - Download [Carla](https://github.com/carla-simulator/carla/releases)
 - ***This project uses Carla 0.9.5 and was also tested with 0.9.13 (Latest version as of April 2022)***
2. For Windows:
    Run the Carla simulator executable file (.exe)
   For Ubuntu:
    Form terminal run the /.sh
3. Download the python DQN train and model deploy scripts from the Legacy Version directory
4. Paste the two python scripts in
 
    > carla_simulator_directory/PythonAPI/examples
    
5. Open the virtual environment command prompt (here Anaconda prompt)
6. Cd to the carla/PythonAPI/examples directory.
7. To Train the Model type
```
python train_model.py
```
8. The best trained model will be stored in ../PythonAPI/examples/models directory that gets created when the model starts training.
## DEPLOY TRAINED MODEL

1. To Run the trained model, launch the Carla simulator (if not already launched).
2. cd into the ../models directory and copy the name of the model file you want to deploy.
3. Open the Deploy_DQNmodel.py script that was copied into the ../examples directory earlier and replace the following,
```python
MODEL_PATH = 'MODEL_NAME.model'
```
3. In a new command prompt cd into the ../PythonAPI/examples directory and run  
```
python Deploy_DQNmodel.py
```
---
## NOTE
**A newer, better version using PyTorch and a reliable Deep Learning Architecture is in the works. Will keep you posted. Thanks!**

---
**NOTE**
This project was heavily inspired by [pythonprogramming.net](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/) tutorial.
---
[Read Carla Documentation](https://carla.readthedocs.io/en/latest/)
