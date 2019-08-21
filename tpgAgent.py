from __future__ import division
import gym
import matplotlib.pyplot as plt
import utils
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent
import time
import pickle
import csv
import datetime
import argparse
import minerl
import os
from dataEnv import DataEnv
import pdb
import numpy as np
import multiprocessing as mp
import vae

""" TPGAgent code was based on the example code available at: https://github.com/Ryan-Amaral/PyTPG/blob/master/tpg_examples.ipynb
"""

def main(args):
    """ Main function. Runs TPG against MineRL environment"""

    # Start timer
    tStart = time.time()
    print(f"Starting agent with args: {args}")
    if not os.path.isdir(args.output):
        print("WARNING: output directory does not exist or not set, therefore will not save TPG population at the end of each generation")

    if args.humanTraining != 0:
        # If doing human training, set arguments appropriately
        args.repetitions = 1
        if args.humanTraining < 110:
            print("must set human examples to at least 110, therfore defaulting to 110")
            args.humanTraining = 110
        env = DataEnv(args.env, args.humanTraining, data_dir= args.data_dir, always_attack= args.attack)
    elif args.multi:
        # If using multiprocessing, initialise environments
        print("Initialising multiprocessing environments")
        env1 = gym.make(args.env)
        env2 = gym.make(args.env)
        obs1,_ = env1.reset()
        obs2,_ = env2.reset()
        env = env1
        env1Thread = None
        env2Thread = None
    else:
        print("Initialising environment")
        env = gym.make(args.env)

    # Get list of actions for this task
    action_list = utils.setupActionSpace(env.action_space.noop(), args.env, args.attack)
    actions_space = range(len(action_list))

    # Create prefix to save results and TPG trainer object to
    trainerFilename = f"tpgTrainerObject-{args.gens}-{args.env}-ht-{args.humanTraining}-vae-{args.vae}-attack-{args.attack}"
    print(f"Action space size: {len(actions_space)}")

    if args.resume:
        print("Resuming from pretrained TPG trainer")
        if os.path.isdir(args.output):
            trainer = utils.pickleLoad(f"{args.output}/{trainerFilename}")
        else:
            trainer = utils.pickleLoad(trainerFilename)

        # This evolves the TPG population that has been loaded
        trainer.evolve()
        summaryScores = [] # record score summaries for each gen (min, max, avg)

    else:
        # parameters set to match the visdoom paper
        # ,with exception of number of teams which should be 450 but set to 50 due to time/memory limits
        trainer = TpgTrainer(actions=actions_space, teamPopSize=args.teams, rTeamPopSize=0,
            gap=0.5, pLearnerDelete=0.7, pLearnerAdd=0.7, pMutateAction=0.2,
            pActionIsTeam=0.5, maxTeamSize=9, maxProgramSize=1024,
            pProgramDelete=0.5, pProgramAdd=0.5, pProgramSwap=1.0,
            pProgramMutate=1.0)
        summaryScores = [('MinScore','MaxScore','AvgScore')] # record score summaries for each gen (min, max, avg)

    if args.vae:
        # load in the pretrained VAE and its structure, this will be used to preprocess images
        encoder = vae.train(True)


    envResetCount = 0

    # Iterate through the generations
    for gen in range(args.startGen, args.gens): # generation loop
        curScores = [] # new list per gen
        # Increase the step limit every X generations
        # This increases training speed for initial gens
        if gen < args.gens/5:
            steps = 100
        elif gen < 2*args.gens/5:
            steps = 200
        elif gen < 3*args.gens/5:
            steps = 300
        elif gen < 4*args.gens/5:
            steps = 500
        else:
            steps = 1000
        # Some envs require more time to complete
        if 'Tree' in args.env:
            steps *= 2
        elif 'Iron' in args.env:
            steps *= 10
        elif 'Diamond' in args.env:
            steps *= 20

        # If doing human training or if manually setting then overright step limit
        if args.humanTraining != 0:
            steps = args.humanTraining
        if args.full > 0:
            steps = args.full


        while True: # loop through agents
            teamNum = trainer.remainingAgents()
            agent = trainer.getNextAgent()
            if agent is None:
                break # no more agents, so proceed to next gen

            score = 0

            for repeat in range(args.repetitions): # For a number of repetitions
                #  Reset the environment using either the multiprocessing or standard methods
                if args.multi and not args.humanTraining:
                    env, obs, envResetCount, env1Thread, env2Thread = envReset(args, env,envResetCount, env1, obs1, env1Thread, env2, obs2, env2Thread)
                else:
                    obs = env.reset()
                    # Handle MineRL v0.2.2 and earlier versions
                    if type(obs) == tuple:
                        obs, _ = obs
                print(f"[{datetime.datetime.now().isoformat()[11:-7]}] 0, Gen #{str(gen)}, Team #{str(teamNum)}, Score: {str(score)}")
                for i in range(steps): # Run for pre-determined step limit

                    if args.vae:
                        # Get state with VAE preprocessed image
                        state = utils.getVaeState(obs, args.env, encoder)
                    else:
                        state = utils.getState(obs, args.env)

                    # Traverse the team's policy graph to find action
                    act = agent.act(state)
                    # Perform action and get new state from environment
                    obs, reward, isDone, debug = env.step(action_list[act])
                    if args.render:
                        # Render the environment
                        env.render()
                    if 'Navigate' in args.env:
                        print(f"[{datetime.datetime.now().isoformat()[11:-7]}] Action: {act}, reward: {reward}, compass: {obs['compassAngle']}")
                    else:
                        print(f"[{datetime.datetime.now().isoformat()[11:-7]}] Action: {act}, reward: {reward}")


                    score += reward # Sum rewards for score
                    print(f"[{datetime.datetime.now().isoformat()[11:-7]}] {i}, Gen #{str(gen)}, Team #{str(teamNum)}, Score: {str(score)}")
                    if isDone:
                        print("environment done, therefore finishing")
                        break # If environment signals done, finish
            agent.reward(score) # Assigns score to agent
            curScores.append(score) # Record scores

        # at end of generation, make summary of scores and save everything
        summaryScores.append((min(curScores), max(curScores),
                        sum(curScores)/len(curScores))) # min, max, avg
        with open(f"{trainerFilename}-summaryScores.csv","w") as f:
            wr = csv.writer(f)
            wr.writerows(summaryScores)

        if os.path.isdir(args.output):
            utils.pickleSave(trainer, f"{args.output}/{trainerFilename}")
        else:
            # don't save if no import location as too much mem for laptop and not enough quota on gpu machines
            #utils.pickleSave(trainer, trainerFilename)
            print("Not saving trainer")

        trainer.evolve() # Evolve the population

        if args.humanTraining and gen == args.gens -1:
            # Delete env at end of training to save memory
            del env

    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    for result in summaryScores:
        print(result[0],result[1],result[2])
    return summaryScores

def envResetWrapper(env,obs):
    # Wrapper function for env reset, required for multiprocessing
    obs = env.reset()
    # Handle MineRL v0.2.2 and earlier versions
    if type(obs) == tuple:
        obs, _ = obs
    return

def envReset(args,env,envResetCount, env1, obs1, env1Thread, env2, obs2, env2Thread):
    """ Performs the correct method of resetting the environment depending whether using multiprocessing or not.
    If not using multiprocessing then it simply does reset and counts number of times its called.
    If using multiprocessing it will maintain two environments at the same time and switch between them during reset. This should reduce time waiting for environemtn reset.
    """
    if args.multi:
        if envResetCount % 2 == 0:
            if envResetCount != 0:
                env1Thread.join()
            env = env1
            obs = obs1
            env2Thread = mp.Process(name="env2Thread",target=envResetWrapper, args=(env2,obs2))
            env2Thread.daemon = True
            env2Thread.start()
        else:
            env2Thread.join()
            env = env2
            obs = obs2
            env1Thread = mp.Process(name="env1Thread",target=envResetWrapper, args=(env1,obs1))
            env1Thread.daemon = True
            env1Thread.start()
        envResetCount += 1
        return env, obs, envResetCount, env1Thread, env2Thread
    else:
        obs = env.reset()
        # Handle MineRL v0.2.2 and earlier versions
        if type(obs) == tuple:
            obs, _ = obs
        envResetCount += 1
    return env, obs, envResetCount

if __name__ == "__main__":
    # Parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--env", help="The environment to run the agent in e.g. 'MineRLNavigateDense-v0'", default="MineRLNavigateDense-v0")
    parser.add_argument("-g", "--gens", help="How many generations to train for", type=int, default=200)
    parser.add_argument("-s", "--startGen", help="Starting generation if resuming", type=int, default=0)
    parser.add_argument("-r","--resume", help="Set to load the tpgTrainer object and continue training from that", action='store_true')
    parser.add_argument("-u","--humanTraining",help="If set will train on human data with number of example frames set by this. Min: 110. set to -1 to get all samples, recommended limit is 10000 to avoid memory issues.", type=int, default=0)
    parser.add_argument("-t","--teams",help="How many teams to use", type=int, default=50)
    parser.add_argument("-c","--render", help="Set to see the frames as the agent plays", action='store_true')
    parser.add_argument("-p","--repetitions",help="How many environments to measure the agent's score over", type=int, default=2)
    parser.add_argument("-m","--multi", help="Set to use multiprocessing", action='store_true')
    parser.add_argument("-v","--vae", help="Use VAE for dimensionality reduction. N.B. Requires a trained VAE model to be saved into local vae.h5 and vae.json files", action='store_true')
    parser.add_argument("-f","--full", help="Use full number of steps. This disables the graduated step limits so may greatly increase runtime.", type=int, default=0)
    parser.add_argument("-d","--data_dir", help="local path to data", default='/import/scratch/mdl31/data')
    parser.add_argument("-a","--attack",help="Sets agents to always attack", action='store_true')
    parser.add_argument("-o","--output",help="Directory where the TPG population will be saved at the end of each generation", default='/import/scratch/mdl31')

    print("arguments parsed")
    args = parser.parse_args()
    main(args)
