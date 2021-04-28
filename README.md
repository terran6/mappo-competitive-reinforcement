# Competitive Reinforcement Learning through Multi-Agent Proximal Policy Optimization

## Results
In the `saved_files` directory, you may find the saved model weights and learning curve plots for the successful Actor-Critic networks.  The trained agents were able to solve the environment in 5,627 episodes utilizing the MAPPO training algorithm.  The graph below depicts the agents' performance over time in terms of relative score averaged over the past 100 episodes.

<br />

<div align="center">
  <img width="700" height="538" img src="saved_files/scores_5627.png">
</div>

<br />

## Dependencies
In order to run the above code, you will have to set up and activate a customized Python 3.6 environment.  Please follow the directions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) for setup instructions.

Next, please click the link corresponding to your operating system below which will download the respective UnityEnvironment.  You may then save the resulting file directly inside of your cloned repository in order to run the code.
* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Let's Play Tennis!
All of the relevant functionality and tools you will need in order to initialize and train the agents are available inside of this repository.  Please utilize the `run_tennis_main.py` file in order to run the training process.  If you would like to change any parameters to customize training, please update the relevant attributes in the function calls below the `if __name__ == '__main__':` block.
