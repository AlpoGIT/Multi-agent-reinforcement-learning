from unityagents import UnityEnvironment
import numpy as np
from collections import deque, defaultdict
import network, utils
import torch, torch.nn as nn, torch.nn.functional as F
import agent

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# define environment
env = UnityEnvironment(file_name="C:\\Users\AL\Documents\GitHub\deep-reinforcement-learning\p3_collab-compet\Tennis_Windows_x86_64\Tennis.exe", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# define target score
target_score = 0.5

# statistics
deque_scores = deque(maxlen=100)
scores = []
print_every = 1
keep_print_every = 50

# hyperparameters
M = 1000 # max number of episodes
start_steps = 1000  # explore action space uniformly for start_steps
params = {
    'action_size'       : brain.vector_action_space_size,
    'state_size'        : env_info.vector_observations.shape[1],
    'buffer_size'       : int(1e6),
    'batch_size'        : 128,
    'nb_agents'         : len(env_info.agents),
    'learning_rate_Q'   : 0.001,
    'learning_rate_mu'  : 0.001,
    'tau'               : 0.01,             # constant for soft update
    'gamma'             : 0.99,
    'lambda'            : 0.01,             # merge Q1 and Q2, mu1 and mu2
    'grad_clip'         : 1.0,              # gradient clip for mu
    'device'            : device
    }
agents = defaultdict(lambda : agent.agent(params))

# schedule for normal noise standard deviation
ln_sigma_i = 0.4
ln_sigma_f = -2.3   #-1.6   
x = np.array([(ln_sigma_f-ln_sigma_i)*t/M + ln_sigma_i for t in range(M)])
sigma = np.exp(x).clip(0.1)

# loop over episodes
for ep in range(M):                                         
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
    state = env_info.vector_observations                  # get the current state (for each agent)
    score = np.zeros(params['nb_agents'])                 # initialize the score (for each agent)
    
    d = torch.distributions.Normal(0, sigma[ep])          # initialize noise distribution

    # begin episode
    while True:
        # explore uniformly at the beginning
        if len(agents['0'].memory) < start_steps :
            action = np.random.uniform(-1,1,(params['nb_agents'], params['action_size']))
        else:
            with torch.no_grad():
                torch_states = torch.as_tensor(state, dtype=torch.float).to(device)
                list_actions = [agents[str(k)].mu(torch_states[k]) + d.sample() for k in range(params['nb_agents'])]
                action = torch.cat(list_actions, dim=0).view(params['nb_agents'], params['action_size'])
                action = np.clip(action.cpu().numpy(), -1, 1)


        env_info = env.step(action)[brain_name]           # send all actions to tne environment
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                         # get reward (for each agent)
        done = env_info.local_done                        # see if episode finished
        score += env_info.rewards                         # update the score (for each agent)

        # use the same memory buffer, keep general for now with different memories
        for i in range(params['nb_agents']):
            agents[str(i)].memory.add(state.flatten(), action.flatten(), reward, next_state.flatten(), done)
        
        
        # each agent learns
        if len(agents['0'].memory) > params['batch_size']: 
            for id in range(params['nb_agents']):
                agent.learn(agents, id)

            # update weights after agent.learn loop
            for i in range(params['nb_agents']):
                agents[str(i)].soft_update()
            
            # not stable if params['lambda'] too big
            # not necessary but if cooperation, average Q and mu. Stabilize convergence
            for target_param, local_param in zip(agents[str(0)].Q.parameters(), agents[str(1)].Q.parameters()):
                target_param.data.copy_(params['lambda']*local_param.data + (1.0-params['lambda'])*target_param.data)
                local_param.data.copy_((1.0-params['lambda'])*local_param.data + params['lambda']*target_param.data)
                
            for target_param, local_param in zip(agents[str(0)].mu.parameters(), agents[str(1)].mu.parameters()):
                target_param.data.copy_(params['lambda']*local_param.data + (1.0-params['lambda'])*target_param.data)
                local_param.data.copy_((1.0-params['lambda'])*local_param.data + params['lambda']*target_param.data)
            
        state = next_state

        if np.any(done):                                  # exit loop if episode finished
            break

    # statistics
    scores.append(np.max(score))
    deque_scores.append(np.max(score))
    average = np.mean(deque_scores)
    if ep%print_every == 0 and ep > 0:
        print("\r{}/{}\taverage: {:.4f}\tsigma: {:.2f}".format(ep, M, average, sigma[ep]), end='')
    if ep%keep_print_every == 0 and ep > 0:
        print("\r{}/{}\taverage: {:.4f}\tsigma: {:.2f}\t max: {:.2f}".format(ep, M, average, sigma[ep], np.max(deque_scores)))
        np.savetxt('scores.txt', scores, fmt='%f')
    if average > target_score:
            print("\nsolved in {} episodes.".format(ep), end ='')
            for i in range(params['nb_agents']):
                torch.save(agents[str(i)].Q.state_dict(), 'Q{}.pth'.format(i))
                torch.save(agents[str(i)].mu.state_dict(), 'mu{}.pth'.format(i))
            np.savetxt('scores.txt', scores, fmt='%f')
            break

env.close()

# print results
utils.print_results(scores, target_score)
