import utils
import network
import torch
import copy

class agent:
    def __init__(self, params):

        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.nb_agents = params['nb_agents']
        self.learning_rate_Q = params['learning_rate_Q']
        self.learning_rate_mu = params['learning_rate_mu']
        self.memory = utils.ReplayBuffer(self.buffer_size, self.batch_size)
        self.device = params['device']
        self.tau = params['tau']
        self.gamma = params['gamma']
        
        self.Q = network.Q_estimator(
                                    self.state_size*self.nb_agents,
                                    self.action_size*self.nb_agents
                                    ).to(self.device)
        self.Q_hat = network.Q_estimator(
                                    self.state_size*self.nb_agents,
                                    self.action_size*self.nb_agents
                                    ).to(self.device)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.optim_Q = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate_Q)

        self.mu = network.mu_estimator(self.state_size, self.action_size).to(self.device)
        self.mu_hat = network.mu_estimator(self.state_size, self.action_size).to(self.device)
        self.mu_hat.load_state_dict(self.mu.state_dict())
        self.optim_mu = torch.optim.Adam(self.mu.parameters(), lr=self.learning_rate_mu)
        
    def soft_update(self):
        for target_param, local_param in zip(self.Q_hat.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        for target_param, local_param in zip(self.mu_hat.parameters(), self.mu.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


def learn(agents, i):
        # generate sample from memory
        states, actions, rewards, next_states, dones = agents[str(i)].memory.sample()

        # learn Q
        with torch.no_grad():
            r = torch.split(rewards , 1 , dim=1)[i]
            dones = torch.split(dones , 1 , dim=1)[i]
            next_s = [torch.split(next_states , agents[str(i)].state_size , dim=1)[k] for k in range(agents[str(i)].nb_agents)]
            next_a = [agents[str(k)].mu_hat(next_s[k]) for k in range(agents[str(i)].nb_agents)]
            next_a = torch.cat(next_a , dim=1)
            y = r + agents[str(i)].gamma * agents[str(i)].Q_hat(next_states, next_a) * (1-dones)

        loss = (y-agents[str(i)].Q(states, actions)).pow(2).mean()
        agents[str(i)].optim_Q.zero_grad()
        loss.backward()
        agents[str(i)].optim_Q.step()

        # learn mu
        # generate sample from memory
       # states, actions, rewards, next_states, dones = agents[str(i)].memory.sample()

        #s = [torch.split(states , agents[str(i)].state_size , dim=1)[k] for k in range(agents[str(i)].nb_agents)]
        a = [torch.split(actions, agents[str(i)].action_size, dim=1)[k] for k  in range(agents[str(i)].nb_agents)]
        a[i] = agents[str(i)].mu( torch.split(states , agents[str(i)].state_size , dim=1)[i] )
        a = torch.cat(a, dim=1)

        loss = -agents[str(i)].Q(states, a).mean()
        agents[str(i)].optim_mu.zero_grad()
        loss.backward()
        agents[str(i)].optim_mu.step()