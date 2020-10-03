import torch
import time

def render_torch_environment(env, policy):
    done = False
    state = env.reset()

    while not done:
        action = policy.sample_action(torch.tensor(state).float())
        env.render()
        time.sleep(0.05)
        state, R, done, _ = env.step(action)


    env.close()