# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

# Implementing a function to make sure the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) # shifting the seed with rank to asynchronize each training agent
    env = create_atari_env(params.env_name) # creating an optimized environment thanks to the create_atari_env function
    env.seed(params.seed + rank) # aligning the seed of the environment on the seed of the agent
    model = ActorCritic(env.observation_space.shape[0], env.action_space) # creating the model from the ActorCritic class
    state = env.reset() # state is a numpy array of size 1*42*42, in black & white
    state = torch.from_numpy(state) # converting the numpy array into a torch tensor
    done = True # when the game is done
    episode_length = 0 # initializing the length of an episode to 0
    while True: # repeat
        episode_length += 1 # incrementing the episode length by one
        model.load_state_dict(shared_model.state_dict()) # synchronizing with the shared model - the agent gets the shared model to do an exploration on num_steps
        if done: # if it is the first iteration of the while loop or if the game was just done, then:
            cx = Variable(torch.zeros(1, 256)) # the cell states of the LSTM are reinitialized to zero
            hx = Variable(torch.zeros(1, 256)) # the hidden states of the LSTM are reinitialized to zero
        else: # else:
            cx = Variable(cx.data) # we keep the old cell states, making sure they are in a torch variable
            hx = Variable(hx.data) # we keep the old hidden states, making sure they are in a torch variable
        values = [] # initializing the list of values (V(S))
        log_probs = [] # initializing the list of log probabilities
        rewards = [] # initializing the list of rewards
        entropies = [] # initializing the list of entropies
        for step in range(params.num_steps): # going through the num_steps exploration steps
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) # getting from the model the output V(S) of the critic, the output Q(S,A) of the actor, and the new hidden & cell states
            prob = F.softmax(action_values) # generating a distribution of probabilities of the Q-values according to the softmax: prob(a) = exp(prob(a))/sum_b(exp(prob(b)))
            log_prob = F.log_softmax(action_values) # generating a distribution of log probabilities of the Q-values according to the log softmax: log_prob(a) = log(prob(a))
            entropy = -(log_prob * prob).sum(1) # H(p) = - sum_x p(x).log(p(x))
            entropies.append(entropy) # storing the computed entropy
            action = prob.multinomial().data # selecting an action by taking a random draw from the prob distribution
            log_prob = log_prob.gather(1, Variable(action)) # getting the log prob associated to this selected action
            values.append(value) # storing the value V(S) of the state
            log_probs.append(log_prob) # storing the log prob of the action
            state, reward, done, _ = env.step(action.numpy()) # playing the selected action, reaching the new state, and getting the new reward
            done = (done or episode_length >= params.max_episode_length) # if the episode lasts too long (the agent is stucked), then it is done
            reward = max(min(reward, 1), -1) # clamping the reward between -1 and +1
            if done: # if the episode is done:
                episode_length = 0 # we restart the environment
                state = env.reset() # we restart the environment
            state = torch.from_numpy(state) # tensorizing the new state
            rewards.append(reward) # storing the new observed reward
            if done: # if we are done
                break # we stop the exploration and we directly move on to the next step: the update of the shared model
        R = torch.zeros(1, 1) # intializing the cumulative reward
        if not done: # if we are not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx))) # we initialize the cumulative reward with the value of the last shared state
            R = value.data # we initialize the cumulative reward with the value of the last shared state
        values.append(Variable(R)) # storing the value V(S) of the last reached state S
        policy_loss = 0 # initializing the policy loss
        value_loss = 0 # initializing the value loss
        R = Variable(R) # making sure the cumulative reward R is a torch Variable
        gae = torch.zeros(1, 1) # initializing the Generalized Advantage Estimation to 0
        for i in reversed(range(len(rewards))): # starting from the last exploration step and going back in time
            R = params.gamma * R + rewards[i] # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
            advantage = R - values[i] # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # computing the value loss
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # computing the temporal difference
            gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] # computing the policy loss
        optimizer.zero_grad() # initializing the optimizer
        (policy_loss + 0.5 * value_loss).backward() # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        ensure_shared_grads(model, shared_model) # making sure the model of the agent and the shared model share the same gradient
        optimizer.step() # running the optimization step
