# Test Agent

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque

# Making the test agent (won't update the model but will just use the shared model to explore)
def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank) # asynchronizing the test agent
    env = create_atari_env(params.env_name, video=True) # running an environment with a video
    env.seed(params.seed + rank) # asynchronizing the environment
    model = ActorCritic(env.observation_space.shape[0], env.action_space) # creating one model
    model.eval() # putting the model in "eval" model because it won't be trained
    state = env.reset() # getting the input images as numpy arrays
    state = torch.from_numpy(state) # converting them into torch tensors
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) # cf https://pymotw.com/2/collections/deque.html
    episode_length = 0 # initializing the episode length to 0
    while True: # repeat
        episode_length += 1 # incrementing the episode length by one
        if done: # synchronizing with the shared model (same as train.py)
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].data.numpy() # the test agent does not explore, it directly plays the best action
        state, reward, done, _ = env.step(action[0, 0]) # done = done or episode_length >= params.max_episode_length
        reward_sum += reward
        if done: # printing the results at the end of each part
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            reward_sum = 0 # reinitializing the sum of rewards
            episode_length = 0 # reinitializing the episode length
            actions.clear() # reinitializing the actions
            state = env.reset() # reinitializing the environment
            time.sleep(60) # doing a one minute break to let the other agents practice (if the game is done)
        state = torch.from_numpy(state) # new state and we continue
