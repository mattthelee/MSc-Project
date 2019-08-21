import json
from numpy import genfromtxt
from matplotlib import pyplot as plt
import pdb
from time import sleep
from PIL import Image
import subprocess
import os
import collections
import pickle
import numpy as np

""" Utils provides a number of useful functions for use by agents on MineRL. Having a separate file ensured changes applied across MineRL and the imitation environment
"""


def getState(state, env):
    """ Performs preprocessing on the observations returned from MineRL.
    In particular performs image flattening and adds compass angle to state multiple times
    """
    # Flatten image and divide by max pixel value to bound between 0 and 1
    state2 = state['pov'].flatten()/255.

    # If navigate in environment name, then compass angle must be used
    if 'Navigate' in env:
        # append the compass angle so that 50% of data is compass angle
        state2 = np.append(state2, np.ones(len(state2))*state['compassAngle']/180.)

    # If obtain environment, then pre-process inventory state and equipped state
    if 'Obtain' in env:
        # Append the inventory and equipped state
        state2 = np.append(state2, list(state['inventory'].values()))

        state2 = np.append(state2, equippedEncode(state))
    return state2

def getVaeState(state, env, encoder):
    """ Uses the encoder to process image and perform dimensionality reduction
    """
    # Get the reduced dimension from the encoder network
    # Some indexing and reshaping required as expecting a list of images to predict
    state2 = encoder.predict(state['pov'].reshape(-1,64,64,3))[2][0]
    #state2 = np.array([1])
    #state2 = np.ones(4000)
    if 'Navigate' in env:
        state2 = np.append(state2, np.ones(len(state2))*state['compassAngle']/180.)

    # If obtain environment, then pre-process inventory state and equipped state
    if 'Obtain' in env:
        # Append the inventory and equipped state
        state2 = np.append(state2, list(state['inventory'].values()))

        state2 = np.append(state2, equippedEncode(state))
    return state2

def equippedEncode(state):
    """ Convert equipped state information to onehot encoded vector
    """
    equippable = [0,'air','wooden_axe','wooden_pickaxe','stone_axe','stone_pickaxe','iron_axe','iron_pickaxe']
    equippedItem = state['equipped_items']['mainhand']['type']
    if type(equippedItem) == np.int32:
        # Handle incorrect state cases
        equippedItem = 0

    # Handle cases where the item is equipped but not equippable,
    # e.g. if player mines a block with bare hand, Minecraft auto-equips the block, but that block might not be in equippable list
    if equippedItem in equippable:
        objectIndex = equippable.index(equippedItem)
    else:
        objectIndex = 0
    equippedState = []

    # Convert equipped item to onehot encoded vector
    for i in range(len(equippable)):
        if i == objectIndex:
            equippedState.append(1)
        else:
            equippedState.append(0)
    return equippedState


def setupActionSpace(noopAction, env = 'MineRLNavigateDense-v0', always_attack= False):
    """ Sets up action space. If always attack is set then every action will include attack subaction and attack will be removed from action space.
        Actions available are task dependent. Also navigate task has 3 degree turns, others use 10 degree turns. Jump action is a forward jump.
    """
    # Set angle change for task
    if 'Navigate' in env:
        angleChange = 3.
    else:
        angleChange = 10.

    # Gets basic action structure from noopAction supplied by env
    baseAction = noopAction.copy()
    if always_attack:
        baseAction['attack'] = 1

    # For each action, make change required from the base action
    forward = baseAction.copy()
    forward['forward'] = 1
    # Jump is in fact a forward jump
    jump = forward.copy()
    jump['jump'] = 1
    back = baseAction.copy() # not used
    back['back'] = 1
    right = baseAction.copy() # not used
    right['right'] = 1
    left = baseAction.copy() # not used
    left['left'] = 1

    attack = baseAction.copy()
    attack['attack'] = 1
    sprint = baseAction.copy() # not used
    sprint['sprint'] = 1
    sneak = baseAction.copy() # not used
    sneak['sneak'] = 1
    lookLeft = baseAction.copy()
    lookLeft['camera'] = [0.,-angleChange]
    lookRight = baseAction.copy()
    lookRight['camera'] = [0.,angleChange]
    lookUp = baseAction.copy()
    lookUp['camera'] = [-angleChange,0.]
    lookDown = baseAction.copy()
    lookDown['camera'] = [angleChange,0.]

    craftTorch = baseAction.copy()
    craftTorch['craft'] = 1
    craftSticks = baseAction.copy()
    craftSticks['craft'] = 2
    craftPlanks = baseAction.copy()
    craftPlanks['craft'] = 3
    craftTable = baseAction.copy()
    craftTable['craft'] = 4

    craftWoodenAxe = baseAction.copy()
    craftWoodenAxe['nearbyCraft'] = 1
    craftWoodenPickaxe = baseAction.copy()
    craftWoodenPickaxe['nearbyCraft'] = 2
    craftStoneAxe = baseAction.copy()
    craftStoneAxe['nearbyCraft'] = 3
    craftStonePickaxe = baseAction.copy()
    craftStonePickaxe['nearbyCraft'] = 4
    craftIronAxe = baseAction.copy()
    craftIronAxe['nearbyCraft'] = 5
    craftIronPickaxe = baseAction.copy()
    craftIronPickaxe['nearbyCraft'] = 6
    craftFurnace = baseAction.copy()
    craftFurnace['nearbyCraft'] = 7

    smeltIron = baseAction.copy()
    smeltIron['nearbySmelt'] = 1
    smeltCoal = baseAction.copy()
    smeltCoal['nearbySmelt'] = 2

    placeCraftingTable = baseAction.copy()
    placeCraftingTable['place'] = 4
    placeFurnace = baseAction.copy()
    placeFurnace['place'] = 5
    placeTorch = baseAction.copy()
    placeTorch['place'] = 6

    equipWoodenAxe = baseAction.copy()
    equipWoodenAxe['equip'] = 1
    equipWoodenPickaxe = baseAction.copy()
    equipWoodenPickaxe['equip'] = 2
    equipStoneAxe = baseAction.copy()
    equipStoneAxe['equip'] = 3
    equipStonePickaxe = baseAction.copy()
    equipStonePickaxe['equip'] = 4
    equipIronAxe = baseAction.copy()
    equipIronAxe['equip'] = 5
    equipIronPickaxe = baseAction.copy()
    equipIronPickaxe['equip'] = 6

    # Return action space depending on the task
    if 'Navigate' in env:
        return [forward, lookLeft, lookRight, jump]
    elif 'Tree' in env and always_attack:
        return [forward, lookLeft, lookRight, jump]
    elif 'Tree' in env:
        return [forward, lookLeft, lookRight, jump, attack]
    elif always_attack:
        return [forward, lookLeft, lookRight, jump, lookUp, lookDown, craftTorch, craftSticks, craftPlanks,
        craftTable, craftWoodenAxe, craftWoodenPickaxe, craftStoneAxe, craftStonePickaxe,
        craftIronAxe, craftIronPickaxe, craftFurnace, smeltIron, smeltCoal, placeCraftingTable, placeFurnace, placeTorch,
        equipWoodenAxe, equipWoodenPickaxe, equipStoneAxe, equipStonePickaxe, equipIronAxe, equipIronPickaxe]
    else:
        return [forward, lookLeft, lookRight, jump, attack, lookUp, lookDown, craftTorch, craftSticks, craftPlanks,
        craftTable, craftWoodenAxe, craftWoodenPickaxe, craftStoneAxe, craftStonePickaxe,
        craftIronAxe, craftIronPickaxe, craftFurnace, smeltIron, smeltCoal, placeCraftingTable, placeFurnace, placeTorch,
        equipWoodenAxe, equipWoodenPickaxe, equipStoneAxe, equipStonePickaxe, equipIronAxe, equipIronPickaxe]

def pickleSave(obj,filename):
    """ Save the object to the file with pickle
    """
    with open(filename,'wb') as file:
        pickle.dump(obj,file)
    return

def pickleLoad(filename):
    """ Load the object from the file with pickle
    """
    with open(filename,'rb') as file:
        return pickle.load(file)
