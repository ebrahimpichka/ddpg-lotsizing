# ddpg-lotsizing
An implementation of the DDPG Reinforcement Learning algorithm for capacitated lot-sizing problem.

--------
**To train the agent:**

    > python train.py

**since collecting/generation training data (lot-sizing scenarios) is very costly and time- and resource-consuming, training the agent completely was not possible.**

--------
## `models.py`
In the models.py the Actor and Critic network classes are implemented using the PyTorch framework.Each network has two fully-connected feed-forward layers followed by a Layer Normalization layer and ReLu activation functions. The Actor has a 2-dimension layer with a Sigmoid activation as an output layer to approximate the deterministic action. The Critic has a 1-dimension layer without any activation as an output layer to approximate the state-action value (Q).

## `environment.py`
In the environment.py the environment class that simulates the lot-sizing problem called LotSizingEnv is implemented. The __init__ (construction) method takes the main problem-defining information as input, initializes the environment variables, and creates the environment object. The reset method re-initializes the environment state at the starting timestep and returns the starting state. The step method takes an action as input and transitions the environment to the next state and returns the corresponding reward, next state, and terminal flag. Other utility methods are created to be used inside of these three main methods.

## `ddpg_agent.py`
In the ddpg_agent.py the Ornstein Uhlenbeck Action Noise and the Replay Buffer classes are implemented, both of which are used in the DDPGAgent class. The DDPGAgent agent class is also implemented in this file. In the __init__ (construction) method, actor, target-actor, critic, target-critic, model optimizers, hyperparameters, replay buffer (memory), and random noise generator are initialized for the agent. The train method in this class, first samples a shuffled batch of (state, action, reward, next_state, done flag) data from the replay buffer (memory) object, computes the target value from this data using the target-actor and target-critic, calculates the critic and actor loss, and backpropagates through each network to update the weights with respect to the loss gradients. And at the end updates the target network weights.

The choose_action method takes an observation from the environment as input and outputs an action signal using the actor-network. And The remember method stores (state, action, reward, next_state, done_flag) sample into the replay buffer.

## `train.py`
In the train.py the main training loop is defined in a way that at each episode, the environment is initialized with a lot-sizing problem scenario (total_timesteps, demands, capacities, holding_cost, setup_cost, starting_inventory). The agent starts interacting with the environment until the last timestep and stores the transitions from this interaction in the replay buffer and simultaneously trains and updates its networks in an off-policy manner from the data stored in the memory to optimize its behaviour. After the training is finished the weights of the networks are saved in the models directory.
