import numpy as np
from ddpg_agent import DDPGAgent
from environment import LotSizingEnv

def train(num_iterations, agent, validate_steps, scenarios, save_model=True, save_often=25):

    score_history = []
    reward_log_history = []
    action_log_history = []
    for iteration in range(1, num_iterations+1):

        total_timesteps, demands, capacities,\
             holding_cost, setup_cost, starting_inventory,\
                  shortage_penalty = random.choice(scenarios)
        
        env = LotSizingEnv(total_timesteps, demands, capacities,
                           holding_cost, setup_cost, starting_inventory,
                           shortage_penalty)

        state = env.reset()
        done = False
        score = 0
        action_log = []
        reward_log = []
        while not done:
            act = agent.choose_action(state)
            new_state, reward, done = env.step(act)
            agent.remember(state, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            reward_log.append(reward)
            action_log.append(act)

        score_history.append(score)
        reward_log_history.append(reward_log)
        action_log_history.append(action_log)

        if save_model and  (iteration % save_often == 0):
           agent.save_models()

        print('episode ', iteration, 'score %.2f' % score,
            'last 10 episodes avg score %.3f' % np.mean(score_history[-10:]))

    return(agent, score_history, reward_log_history, action_log_history)

if __name__ == "__main__":


    LOAD_MODEL = False

    agent = DDPGAgent(num_states = 14, tau = 0.9 , gamma=0.99, num_actions=2, max_size=1000000,
                     hidden1_dims=400, hidden2_dims=300, batch_size=64, critic_lr=0.0003, actor_lr=0.0003)
    
    scenarios = [
        (
            # Tuple[total_timesteps, demands, capacities, holding_cost, setup_cost, starting_inventory]
            # (int, list, list, int, int, int)
            (10, [100,125,138,169,147,74,98,112,136,98], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100], 50, 10, 150)
            # ... real historical data or simulated data
        )
    ]
    
        
    if LOAD_MODEL:
        agent.load_models()

    ### TRAINING ###  
    agent, score_history, reward_log_history, action_log_history = \
        train(num_iterations, agent, validate_steps, scenarios)
    
    agent.save_models()
