import minerl
import numpy as np
import pdb

class Action_space:
    """ Subclass required to simulate gym env. Only exists to provide noop() method.
    """
    def noop(self):
        return {'attack': 0, 'back': 0, 'camera': [0., 0.], 'craft': 0, 'equip': 0, 'forward': 0, 'jump': 0, 'left': 0, 'nearbyCraft': 0, 'nearbySmelt': 0, 'place': 0, 'right': 0, 'sneak': 0, 'sprint': 0}

class DataEnv:
    """
    DataEnv is the class used to create the imitation environment.
    This class is designed to have the same step(), reset() and action_space.noop() methods as a gym environment to allow the TpgAgent to switch between the two easily.
    The class first retrieves data for the given environment, filters it to reduce the class imbalance problems and then stores it to its obsActionList.
    When the step() method is used, the action given (by the agent) is compared to the human action to determine what reward to grant it.
    """
    def __init__(self, mission = 'MineRLObtainDiamond-v0', limit = 0, filters=True, data_dir= '/import/scratch/mdl31/data', always_attack = False):
        """ Initialise the imitation environment. Mission sets the task to extract data for, limit sets maximum examples to prevent OOM errors.
        filters = True applies filtering to reduce the action imbalance and to also applies data quality filters such a minimum score and max steps used.
        data_dir should be a string giving the directory where the MineRL data has been downloaded.
        If always_attack=True then attacking is not rewarded or punished because the agent always attacks
        """
        self.data = minerl.data.make(mission, data_dir = data_dir)
        self.position = 0 # Records how far through the dataset the environment has gone through
        # Initialise variables
        self.action_space = Action_space()
        self.mission = mission
        self.meta = []
        # initialise self.countDict, which counts how many of each action have been seen, to allow balancing
        self.countDict = {}
        for key in self.action_space.noop().keys():
            if key == 'camera':
                continue
            self.countDict[key] = 0
        self.countDict['lookLeft'] = 0
        self.countDict['lookRight'] = 0
        self.countDict['lookUp'] = 0
        self.countDict['lookDown'] = 0

        # Extract data
        self.prepData(limit, filters)
        self.always_attack = always_attack


    def reset(self):
        """ Resets the environment and returns the first observation
        """
        self.position = 0
        obs, reward, isDone, debug = self.step(self.action_space.noop())
        return obs

    def prepData(self, limit = 100000, filters=True):
        """ Extracts and filters data. Limit set to prevent OOm errors.
        """
        self.obsActionList = []
        count = 0
        forwardCount = 0
        interestingActionCount = 0


        # Iterate through the human examples
        for obs, humanActions, reward, next_state, done, meta in self.data.sarsd_iter(num_epochs = 1, max_sequence_len =  1000, include_metadata=True):
            if limit != -1 and count >= limit:
                break
            # Whether to apply filters or not
            if filters:
                # For each task uses a score and step filter appropriate to that task
                if 'Navigate' in self.mission and (meta['total_reward'] < 100 or meta['duration_steps'] > 2500):
                    # if score is low then continue rather than learning from it
                    continue
                elif 'Tree' in self.mission and (meta['total_reward'] < 63 or meta['duration_steps'] > 3000):
                    continue
                elif 'Iron' in self.mission and (meta['total_reward'] < 550 or meta['duration_steps'] > 10000):
                    continue
                elif 'Diamond' in self.mission and (meta['total_reward'] < 1000 or meta['duration_steps'] > 35000):
                    continue

            # Observations and human actions require reformatting to match online env
            obs = self.formatObs(obs)
            self.meta.append(meta)
            humanActions = self.formatHumanActions(humanActions)
            residualAngle = 0

            # All observations in this trajectory are provided as a list within obs, therfore iterate through and reformat data to match online env
            for step in range(len(obs)):
                humanAction = humanActions[step]

                if limit != -1 and count >= limit:
                    break

                if 'Navigate' in self.mission:
                    # If the cameraAngle is greater than compass turn rate we want to add that to compass
                    # This means the compass angle in human demonstrations is accurate and does not require catching up
                    residualAngle += -humanAction['camera'][1]
                    if residualAngle > 3:
                        residualAngle += -3
                    elif residualAngle < -3:
                        residualAngle += 3
                    else:
                        residualAngle = 0
                    obs[step]['compassAngle'] += residualAngle

                # Balances the data by ignoring actions that are overrepresented until at least 1/2 of actions contain something of interest
                if self.interestingAction(humanAction):
                    interestingActionCount += 1
                elif interestingActionCount < count //2 and filters:
                    continue

                count += 1

                # Convert human action to discretised turns based on 3 degree turn minimum criterion
                self.obsActionList.append((obs[step],humanAction))
                for key in humanAction.keys():
                    if key == 'camera':
                        if humanAction[key][0] > 3:
                            self.countDict['lookRight'] += 1
                        if humanAction[key][0] < -3:
                            self.countDict['lookLeft'] += 1
                        if humanAction[key][1] > 3:
                            self.countDict['lookUp'] += 1
                        if humanAction[key][1] < -3:
                            self.countDict['lookDown'] += 1
                        continue
                    if humanAction[key] > 0:
                        self.countDict[key] += 1
        return

    def interestingAction(self, humanAction):
        """ Rulebased system for deciding if an action is 'interesting' i.e. a minority action.
        If not then it is a majority action and i may want to discard it
        """
        if self.nonAction(humanAction):
            return False
        # Rules are based on the task
        if 'Navigate' in self.mission:
            return abs(humanAction['camera'][1]) >= 3
        if 'Tree' in self.mission:
            return humanAction['attack'] == 0 and (abs(humanAction['camera'][1]) >= 3 or abs(humanAction['camera'][0]) >= 3)
        if 'Obtain' in self.mission:
            interestingActions = ['craft', 'nearbyCraft', 'nearbySmelt', 'place', 'equip']
            for interestingAct in interestingActions:
                if humanAction[interestingAct] > 0:
                    return True
            return humanAction['attack'] == 0 and (abs(humanAction['camera'][1]) >= 3 or abs(humanAction['camera'][0]) >= 3 > 0)

    def nonAction(self, humanAction):
        """ Checks if action is a noNaction. Human actions with minor (< 3 degree) camera changes a no other subactions are considered nonActions.
        NonActions are not useful to the agent so should be ignored.
        """
        noop = self.action_space.noop()
        action = humanAction.copy()

        # Sets human action camera angle to 0 if below 3 degree threshold
        if abs(humanAction['camera'][1]) < 3:
            humanAction['camera'][1] = 0
        if abs(humanAction['camera'][0]) < 3:
            humanAction['camera'][0] = 0
        return self.identicalActions(noop, action)

    def identicalActions(self,action1,action2):
        """ Checks if two actions are identical.
        """
        result = True
        for key in action2.keys():
            if key == 'camera':
                result = result and action1['camera'][0] == action2['camera'][0] and action1['camera'][1] == action2['camera'][1]
            else:
                result = result and action1[key] == action2[key]
        return result

    def formatHumanActions(self,humanActions):
        """ Formats a group of humanActions from the dataset into same format as the online env.
        This involves changing from a dictionary of lists to a list of dictionaries.
        """
        formattedHumanActions = []
        for step in range(len(humanActions['forward'])):
            formattedAct = {}
            for key in humanActions.keys():
                formattedAct[key] = humanActions[key][step]
            formattedHumanActions.append(formattedAct)
        return formattedHumanActions

    def formatObs(self,obs):
        """ Similar to formatHumanActions, changes observations from data set to the same format as the online env.
        To do this it converts from a dictionary of lists to a list of dictionaries
        """
        formattedObs = []
        for step in range(len(obs['pov'])):
            formattedOb = {'inventory' : {}, 'equipped_items' : {'mainhand' : {}}}
            if 'Navigate' in self.mission:
                formattedOb['compassAngle'] = obs['compassAngle'][step][0]
            formattedOb['pov'] = obs['pov'][step]
            if 'Obtain' in self.mission:
                for inventoryKey in  obs['inventory'].keys():
                    formattedOb['inventory'][inventoryKey] = obs['inventory'][inventoryKey][step]
                for key in obs['equipped_items']['mainhand'].keys():
                    formattedOb['equipped_items']['mainhand'][key] = obs['equipped_items']['mainhand'][key][step]
            formattedObs.append(formattedOb)
        return formattedObs

    def step(self,action):
        """ Takes a step in the imitation environment.
        This takes in action, finds its imitation score by comparing to human action and increments the position in the data set
        """
        isDone = False
        (obs, humanAction) = self.obsActionList[self.position]
        debug = humanAction

        # Get reward by comparing the two actions
        reward = self.compareActions(action, humanAction)
        self.position += 1
        if len(self.obsActionList) == self.position:
            isDone = True
        return obs, reward, isDone, debug

    def compareActions(self,action, humanAction):
        """ Compares the agents action to the total human action
        returns a reward signal based on them using similar actions
        weighting of reward is handled by dividing by the number of times that reward is seen in the data,
        therefore naively choosing any action gets the same reward as naively choosing any other, but correclty choosing can greatly benefit the agent
        """

        # As attack, craft, nearbyCraft,nearbySmelth, equip and place are mutaully exclusive their order in the priority doesn't matter
        # They all have greater priority than looking or movement as they are the human's 'intent'
        if 'Obtain' in self.mission:
            for key in ['craft','nearbyCraft', 'nearbySmelt', 'equip','place']:
                if humanAction[key] != 0:
                    if humanAction[key] == action[key]:
                        # return reward inversely proportional to the frequency in the dataset
                        return 1./self.countDict[key]
                    else:
                        return 0

        if 'Navigate' not in self.mission and not self.always_attack:
            for key in ['attack']:
                if humanAction[key] != 0:
                    if humanAction[key] == action[key]:
                        # return reward inversely proportional to the frequency in the dataset
                        return 1./self.countDict[key]
                    else:
                        return 0



        # Next, grant a reward if the agent is turning in correct direction,
        # punish if it turns opposite way, do not punish if it turns but human doesn't
        yawWeight = 2./(self.countDict['lookLeft'] + self.countDict['lookRight']) # harmonic mean
        pitchWeight =  2./(self.countDict['lookDown'] + self.countDict['lookUp'])
        reward = 0
        humanRight = humanAction['camera'][1] >= 3
        humanLeft = humanAction['camera'][1] <= -3
        humanUp = humanAction['camera'][0] >= 3
        humanDown = humanAction['camera'][0] <= -3

        agentRight = action['camera'][1] >= 3
        agentLeft = action['camera'][1] <= -3
        agentUp = action['camera'][0] >= 3
        agentDown = action['camera'][0] <= -3


        if humanRight and agentRight:
            reward += yawWeight
        elif humanRight and agentLeft:
            reward -= yawWeight

        if humanLeft and agentLeft:
            reward += yawWeight
        elif humanLeft and agentRight:
            reward -= yawWeight

        # In obtain environments looking up and down as included in action space
        if 'Obtain' in self.mission:
            if humanUp and agentUp:
                reward += pitchWeight
            elif humanUp and agentDown:
                reward -= pitchWeight

            if humanDown and agentDown:
                reward += pitchWeight
            elif humanDown and agentUp:
                reward -= pitchWeight

        if agentRight or agentLeft or agentUp or agentDown:
            return reward

        # Jumping and forwards are the lowest priority as the human is often doing them while doing something more important for the agent to learn.
        for key in ['jump','forward']:
            if humanAction[key] != 0:
                if humanAction[key] == action[key]:
                    return 1./self.countDict[key]
                else:
                    return 0

        return 0
