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
from PIL import Image
import random

def main(args):
    """
    *** Requires MineRL v0.2.3 not MineRL v0.2.2 which is used by other project python files. v0.2.3 is required for seeding which greatly reduces assessment time ***
    Loads in a TPG population from pickle file. Due to the randomness of MineRL it is difficult to choose the best agent off a single score.
    It take too long to assess every agetn for 100 episodes so this evaluates the agents against a small numbber of repetitions of the environment.
    Then takes the best agents from that and continues to evaluate them for X repetitions, recording their scores.
    Uses MineRL 0.2.3 ability to seed environment, thereby allowing number of preliminary repetitions to be reduced as there is less noise.
    Each agent is assessed on the same gameworlds as the others
    """

    if args.full == -1: # Check params set
        print("Must set -f argument")
        return 1

    # Generate seeds to be used for creating environments
    seeds = []
    for i in range(args.repetitions):
        seeds.append(random.randint(0,10000))

    # Generate environment and start the timer
    env = gym.make(args.env)
    tStart = time.time()

    # Setup action space
    action_list = utils.setupActionSpace(env.action_space.noop(), args.env)
    actions_space = range(len(action_list))

    # Construct trainer filename
    trainerFilename = f"tpgTrainerObject-{args.gens}-{args.env}-ht-{args.humanTraining}"
    print(f"Action space size: {len(actions_space)}")
    print(f"loading from {trainerFilename}")
    # Load TPG trainer object from saved pickleFile
    trainer = utils.pickleLoad(f"/import/scratch/mdl31/{trainerFilename}")

    assessmentScores = [["Agent","Repeat","Score","Steps"]]

    # Uses of MineRL 0.2.3 seeds currently only supported on Nav env
    if 'Navigate' in args.env:
        env.seed(seeds[0])

    # Initialise environment
    state = env.reset()

    # Set number or prelim reps
    # First assessment will do smaller number of runs in order to find the best agents
    # As navigate uses seeding, we need fewer prelim runs
    if 'Navigate' in args.env:
        prelimReps = 3
    else:
        prelimReps = 10

    scores = []
    steps = []
    prelimScores = []

    # Handle different versions of TPG trainer object and extract root teams
    try:
        teams = trainer.rootTeams
    except:
        teams = []
        trainer.evolve()
        for agent in trainer.getAllAgents():
            teams.append(agent.team)

    # do preliminary rep check to find best agents
    for teamNum, team in enumerate(teams):
        agent = TpgAgent(team, trainer) # Convert team to an agent
        prelimScore = 0
        scores.append([])
        steps.append([])

        # do prelimReps repetitions to work out which are the best agents
        for repeat in range(prelimReps):
            score, finalStep = runTest(args,env,agent, seeds, repeat, teamNum, action_list)
            prelimScore += score
            scores[teamNum].append(score)
            steps[teamNum].append(finalStep)
        prelimScores.append(prelimScore)
        print(f"Team: {teamNum}, scored: {prelimScore}")

    # Sort with prelimScores and teamnames to avoid errors when scores are the same
    teamNames = [ str(x) for x in teams]
    prelimSortedTeams = [x for _,x in sorted(zip(prelimScores,teamNames), reverse=True)]

    # Take only the best 3 agents and perform full assessment on them
    top3Agents = prelimSortedTeams[:3]
    for teamNum, team in enumerate(teams):
        # if not in the top three, skip
        if str(team) not in top3Agents:
            continue
        # Add the scores we've alread calculated to this agents assessments
        for i in range(len(scores[teamNum])):
            repeat = i
            score = scores[teamNum][i]
            print(f"A top 3 score: {score}, team: {teamNum}")
            lastStep = steps[teamNum][i]
            assessmentScores.append([teamNum,repeat,score,lastStep])

        agent = TpgAgent(team, trainer) # Convert team to agent

        # Perform remaining assessments and save scores
        for repeat in range(prelimReps,args.repetitions):
            score,lastStep = runTest(args,env,agent,seeds, repeat, teamNum, action_list)

            assessmentScores.append([teamNum,repeat,score,lastStep])
            with open(f"{trainerFilename}-assessmentScores.csv","w") as f:
                wr = csv.writer(f)
                wr.writerows(assessmentScores)
    return

def runTest(args,env,agent, seeds, repeat, teamNum, action_list):
    """ Runs a single episode against the given environment with a specified seed.
    """
    # get initial state and prep environment
    if 'Navigate' in args.env: # Seeding only supported for Nav environment
        env.seed(repeat)

    # Initialise env
    obs = env.reset()
    score = 0
    print(f"[{datetime.datetime.now().isoformat()[11:-7]}] Agent: {str(teamNum)}, Repeat:{str(repeat)}, Score: {str(score)}")
    for i in range(args.full): # run episodes that last X steps

        # Get state from observation
        state = utils.getState(obs, args.env)

        # Get action from agent
        act = agent.act(state)

        # Perform action and get updated state and reward
        obs, reward, isDone, debug = env.step(action_list[act])

        # If render is set, will save current frame to file
        if args.render:
            img = Image.fromarray(env.render())
            img.save(f'/import/scratch/mdl31/image{i}-action{act}.png')

        if 'Navigate' in args.env:
            print(f"[{datetime.datetime.now().isoformat()[11:-7]}] Action: {act}, reward: {reward}, compass: {obs['compassAngle']}")
        else:
            print(f"[{datetime.datetime.now().isoformat()[11:-7]}] Action: {act}, reward: {reward}")


        score += reward # accumulate reward in score
        print(f"[{datetime.datetime.now().isoformat()[11:-7]}] {i}, Agent: {str(teamNum)}, Repeat:{str(repeat)}, Score: {str(score)}")
        lastStep = i
        if isDone:
            print("environment done, therefore finishing")
            break # end early if env is done
    return score, lastStep

if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--env", help="The environment to run the agent in e.g. 'MineRLNavigateDense-v0'", default="MineRLNavigateDense-v0")
    parser.add_argument("-g", "--gens", help="How many generations to train for", type=int, default=200)
    parser.add_argument("-r","--resume", help="Set to load the tpgTrainer object and continue training from that", action='store_true')
    parser.add_argument("-u","--humanTraining",help="If set will do human training with number of example frames set by this. Min: 110. set to -1 to get all samples", type=int, default=0)
    parser.add_argument("-c","--render", help="Set to see the frames as the agent plays", action='store_true')
    parser.add_argument("-f","--full", help="Use full number of steps", type=int, default=-1)
    parser.add_argument("-p","--repetitions",help="How many environments to measure the agent's score over", type=int, default=100)
    args = parser.parse_args()
    main(args)
