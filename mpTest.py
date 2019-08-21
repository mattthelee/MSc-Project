import multiprocessing as mp
import minerl
import gym
import pdb
env1 = gym.make('MineRLNavigateDense-v0')
env2 =  gym.make('MineRLNavigateDense-v0')

def envResetWrapper(env,obs):
    obs, _ = env.reset()
    return

done = False
for agents in range(10):
    print(f"Agent: {agents}")
    obs1,_ = env1.reset()
    obs2,_ = env2.reset()

    for step in range(20):
        if step % 2 == 0:
            if step != 0:
                env1Thread.join()
            env = env1
            env2Thread = mp.Process(name="env2Thread",target=envResetWrapper, args=(env2,obs2))
            env2Thread.daemon = True
            env2Thread.start()
        else:
            env2Thread.join()
            env = env2
            env1Thread = mp.Process(name="env1Thread",target=envResetWrapper, args=(env1,obs1))
            env1Thread.daemon = True
            env1Thread.start()
        action = env.action_space.sample()

        # One can also take a no_op action with
        # action =env.action_space.noop()


        obs, reward, done, info = env.step(
            action)
        print(f"Step: {step} reward: {reward}")
