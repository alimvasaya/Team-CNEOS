from itertools import count
from time import *
from vpython import *
from state import * 
import numpy as np
import random
import seaborn as sb
import matplotlib.pyplot as plt

class Qlearning:
    def __init__(self,experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False,seed=None):
        self.seed = seed
        random.seed(seed)  # for reproducibility
        np.random.seed(seed)
        self.experiment = experiment
        self.subExperiment = subExperiment
        self.experimentName = f"Experiment {str(experiment)}"
        if subExperiment != None:
            self.experimentName += f"{subExperiment}"
        self.world = world
        self.cube_size = self.world.cube_size
        self.cube_spacing = self.world.cube_spacing
        self.qtable = qtable
        self.qTable = qtable.qtable
        self.qTable_ = qtable.qtable_
        self.dropOffCells = self.world.Dropoff_cell
        self.dropOffCellCapacity = 5
        self.pickUpCells = self.world.Pickup_cell
        self.stepCounter = 0
        self.pickUpCellsStartWith = self.world.pickup_items
        F_x, F_y, F_z, F_i, M_x, M_y, M_z, M_i = self.world.state.get()
        self.malePos = np.array([M_x,M_y,M_z]) # initial location of male agent
        self.femalePos = np.array([F_x, F_y, F_z])  # initial location of female agent
        self.MaleState = False if M_i == 0 else True      # represent whether the male player is holding
        self.FemaleState = False if F_i == 0 else True
        self.RewardsMalePerStep = []  # rewards per step for male in current episode
        self.RewardsFemalePerStep = []  # rewards per step for female in current pisode
        self.maleRewardsPerEpisode = []  # stores total rewards for male in each episode
        self.femaleRewardsPerEpisode = []  # stores total rewards for female in each episode
        self.totalRewardsPerEpisode = [] # stores the total rewards for both male and female earned in each episode
        # these next two are list of lists where each inner list is a nupy array representing position, a string representing the movement-action taken in that position, and the reward for that movement - just had to change the NextStep function to implement updates
        # these are used for SARSA
        self.MaleRewards = []
        self.Femalerewards = []
        self.ACTIONS = {"East": np.array([1, 0, 0]),"West": np.array([-1, 0, 0]), "North": np.array([0, 1, 0]), "South": np.array([0, -1, 0]), "Up": np.array([0, 0, 1]), "Down":np.array([0, 0, -1]) }
        self.learning_rate = learning_rate
        self.discount_factor = Discount_rate
        self.SARSA = None
        self.terminalStatesReached = 0
        self.stepsPerTerminalState = []
        self.manhattanDistance = [] # stores manhatten distance for current episode and get reset after reaching terminal state
        self.manhattanDistancePerState = []  # stores manhatten distance for all terminal states
        self.blockCounts = 0  # counts how many times agent block each other
        self.closeToEachOther = []  # when manhattan distance is 1 they are closest
        self.resetWorld()
        self.SARSA = SARSA
            
    def Turn(self,step):  # based on step count, determine whose turn it is, their position on the gameboard, and whether they're holding
        if step % 2 == 1:
            male = True
            x = self.malePos[0]
            y = self.malePos[1]
            z = self.malePos[2]
            holding = self.MaleState
        else:
            male = False
            x = self.femalePos[0]
            y = self.femalePos[1]
            z = self.femalePos[2]
            holding = self.FemaleState
        return male, x, y, z, holding

    def pickUp(self, male, x, y, z, holding):
        if male:
            self.MaleState = True
            self.RewardsMalePerStep.append(self.getRewards((x,y,z), holding))  # adding rewards for male
            indx = self.pickUpCells.index((x,y,z))
            self.world.pickup_items[indx] -=1
            pickup_idx = self.world.pickup_items[indx] # Updated index for accessing pickups
            if indx == 1:
                self.world.pickups2[pickup_idx].visible = False
            else: self.world.pickups[pickup_idx].visible = False
            self.world.sphere2.color = color.red
            
        else:
            self.FemaleState = True
            self.RewardsFemalePerStep.append(self.getRewards((x,y,z), holding))  # adding rewards female 
            indx2 = self.pickUpCells.index((x,y,z))
            self.world.pickup_items[indx2] -=1
            pickup_idx_2 = self.world.pickup_items[indx2] # Updated index for accessing pickups
            if indx2 == 1:
                self.world.pickups2[pickup_idx_2].visible = False
            else: self.world.pickups[pickup_idx_2].visible = False
            self.world.sphere1.color = color.red       

    def dropOff(self, male, x,y,z, holding):
        if male:
            if self.MaleState:
                self.MaleState = False                
                indx = self.dropOffCells.index((x,y,z))
                self.world.dropoff_capacity[indx] += 1

                dropoffindx = self.world.dropoff_capacity[indx]
                if indx == 0:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points.append(self.world.dropoff)
                    
                elif indx == 1:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points2.append(self.world.dropoff) 
                elif indx == 2:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points3.append(self.world.dropoff)
                else:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points4.append(self.world.dropoff)         
                
                self.world.sphere2.color = color.black
                self.RewardsMalePerStep.append(self.getRewards((x,y,z), holding))  # adding rewards for male
        else:  # must be female
            if self.FemaleState:
                self.FemaleState = False
                indx2 = self.dropOffCells.index((x,y,z))
                self.world.dropoff_capacity[indx2] += 1

                dropoffindx2 = self.world.dropoff_capacity[indx2]
                if indx2 == 0:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx2 * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points.append(self.world.dropoff)
                elif indx2 == 1:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx2 * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points2.append(self.world.dropoff)
                elif indx2 == 2:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx2 * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points3.append(self.world.dropoff)
                else:
                    dropoff_x = x * (self.cube_size + self.cube_spacing) + dropoffindx2 * self.cube_spacing
                    dropoff_y = y * (self.cube_size + self.cube_spacing)
                    dropoff_z = z * (self.cube_size + self.cube_spacing)
                    self.world.dropoff = sphere(pos=vector(dropoff_x-1,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                    self.world.dropoff_points4.append(self.world.dropoff)
                
                self.world.sphere1.color = color.orange
                self.RewardsFemalePerStep.append(self.getRewards((x,y,z), holding))  # adding rewards female

    def getRewards(self, pos, holding):
            x,y,z= pos
            if holding:
                if (x,y,z) in self.dropOffCells and self.world.dropoff_capacity[self.world.Dropoff_cell.index((x,y,z))] < 5:
                    return 14
                elif (x,y,z) in self.world.Risky_cell:
                    return -2    
            elif (x,y,z) in self.pickUpCells and self.world.pickup_items[self.pickUpCells.index((x,y,z))] > 0:
                return 14
            elif (x,y,z) in self.world.Risky_cell:
                return -2
            return -1

    def getPickUpIndex(self):  # returns an index to access the array in the notHolding values
        pickUp1, pickUp2 = self.pickUpCells  # positions
        indx1 , indx2 = self.pickUpCells.index(pickUp1) ,self.pickUpCells.index(pickUp2)
        pickUp1, pickUp2 = self.world.pickup_items[indx1], self.world.pickup_items[indx2]  # tokens left in each (used as bools)
        if pickUp1 and pickUp2:
            return 3 # both have tokens
        else:  # not both
            if pickUp2:
                return 2 
            elif pickUp1:
                return 1 
            else:
                return 0

    def getDropOffIndex(self): 
        dropOff1, dropOff2, dropOff3, dropOff4 = self.dropOffCells  # positions
        dropOff1, dropOff2, dropOff3, dropOff4 = self.world.dropoff_capacity[self.world.Dropoff_cell.index((dropOff1))] < 5, self.world.dropoff_capacity[self.world.Dropoff_cell.index((dropOff2))] < 5, self.world.dropoff_capacity[self.world.Dropoff_cell.index((dropOff3))] < 5, self.world.dropoff_capacity[self.world.Dropoff_cell.index((dropOff4))] < 5  # bools represent whether they can take more
        if not (dropOff1 or dropOff2 or dropOff3 or dropOff4):  # if all are full
            return 0
        else:  # at least one must not be full
            # this layer represents where only one is not full
            if dropOff1 and not (dropOff2 or dropOff3 or dropOff4):
                return 1
            elif dropOff2 and not (dropOff1 or dropOff3 or dropOff4):  # these indices aren't being returned 2,5,8
                return 2
            elif dropOff3 and not (dropOff1 or dropOff2 or dropOff4):
                return 3
            elif dropOff4 and not (dropOff1 or dropOff2 or dropOff3):
                return 4
            else:  # at least two must not be full
                if dropOff1 and dropOff2 and not (dropOff3 or dropOff4):
                    return 5
                elif dropOff1 and dropOff3 and not (dropOff2 or dropOff4):
                    return 6
                elif dropOff1 and dropOff4 and not (dropOff2 or dropOff3):
                    return 7
                elif dropOff2 and dropOff3 and not (dropOff1 or dropOff4):
                    return 8
                elif dropOff2 and dropOff4 and not (dropOff1 or dropOff3):
                    return 9
                elif dropOff3 and dropOff4 and not (dropOff1 or dropOff2):
                    return 10
                else:  # at least 3 must not be full
                    if dropOff1 and dropOff2 and dropOff3 and not dropOff4:
                        return 11
                    elif dropOff1 and dropOff2 and dropOff4 and not dropOff3:
                        return 12
                    elif dropOff1 and dropOff3 and dropOff4 and not dropOff2:
                        return 13
                    elif dropOff2 and dropOff3 and dropOff4 and not dropOff1:
                        return 14
                    else:  # all must not be full
                        return 15
    def terminalState(self):  
        if not self.MaleState and not self.FemaleState:  # both agents must not be holding in order to terminate
            # all dropOffCells should be full
            for cell in self.dropOffCells:  # for every cell in dropOffCells
                if self.world.dropoff_capacity[self.dropOffCells.index((cell))] != self.dropOffCellCapacity:  # if it's not full then "return "False
                    return False
            # all pickUPCells should be empty
            for cell in self.pickUpCells:  # for every cellin pickUpCells
                if self.world.pickup_items[self.pickUpCells.index((cell))] != 0:  # if it's not empty then return False
                    return False
            return True  # both agents were empty handed, all dropOffCells were full and all pickUpCells were empty
        return False
    def resetWorld(self):
            self.world.reset()
            # Delete old dropoff points
            for dropoff in self.world.dropoff_points:
                dropoff.visible = False
            del self.world.dropoff_points[:]
            for dropoff in self.world.dropoff_points2:
                dropoff.visible = False
            del self.world.dropoff_points2[:]
            for dropoff in self.world.dropoff_points3:
                dropoff.visible = False
            del self.world.dropoff_points3[:]
            for dropoff in self.world.dropoff_points4:
                dropoff.visible = False
            del self.world.dropoff_points4[:]
            self.malePos = np.array([2,1,2]) # initial location of male agent
            self.femalePos = np.array([0,0,0])  # initial location of female agent
            self.MaleState = False
            self.FemaleState = False

    def manhattan(self, a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def NextStep(self, male, NextAction, holding, qTableIndex):  # this is used to verify that the agents aren't going to occupy the same position
            i = 0
            #NextAction = [NextAction[0]] + random.sample(NextAction[1:], len(NextAction[1:])) # better than shuffling outside of the function
            if male:
                reward = self.getRewards(self.malePos, holding)
                self.RewardsMalePerStep.append(reward)  # adding rewards for male
                oldPos = self.malePos
                if NextAction[i] == 'Dropoff' or NextAction[i] == 'Pickup':
                    i += 1
                newPos = oldPos + self.ACTIONS[NextAction[i]]
                # male's new position can't be the same as the female's position, while it is, recalculate
                while ((newPos == self.femalePos).all() or
                        newPos[0] < 0 or newPos[0] > 2 or
                        newPos[1] < 0 or newPos[1] > 2 or
                        newPos[2] < 0 or newPos[2] > 2):
                    i += 1
                    self.blockCounts += 1
                    if NextAction[i] == 'Dropoff' or NextAction[i] == 'Pickup':
                        i += 1
                    newPos = oldPos + self.ACTIONS[NextAction[i]]
                self.MaleRewards.append([oldPos, NextAction[i], reward, holding, qTableIndex])
                self.malePos = newPos
                self.world.sphere2.pos = vector(self.malePos[0]*(self.world.cube_size + self.world.cube_spacing),self.malePos[1]*(self.world.cube_size + self.world.cube_spacing),self.malePos[2]*(self.world.cube_size + self.world.cube_spacing))
                sleep(1)
            else:
                reward = self.getRewards(self.femalePos, holding)
                self.RewardsFemalePerStep.append(reward)  # adding rewards female
                oldPos = self.femalePos
                if NextAction[i] == 'Dropoff' or NextAction[i] == 'Pickup':
                    i += 1
                newPos = oldPos + self.ACTIONS[NextAction[i]]
                while((newPos == self.malePos).all() or
                        newPos[0] < 0 or newPos[0] > 2 or
                        newPos[1] < 0 or newPos[1] > 2 or
                        newPos[2] < 0 or newPos[2] > 2):
                    # female's new position can't be the same as the male's position, while it is, recalculate
                    i += 1
                    self.blockCounts += 1
                    if NextAction[i] == 'Dropoff' or NextAction[i] == 'Pickup':
                        i += 1
                    newPos = oldPos + self.ACTIONS[NextAction[i]]
                self.Femalerewards.append([oldPos, NextAction[i], reward, holding, qTableIndex])
                self.femalePos = newPos
                self.world.sphere1.pos = vector(self.femalePos[0]*(self.world.cube_size + self.world.cube_spacing),self.femalePos[1]* (self.world.cube_size + self.world.cube_spacing),self.femalePos[2]* (self.world.cube_size + self.world.cube_spacing))
                sleep(1)
            return NextAction[i]

    def POLICY(self, step: int, policy: str):
        if self.terminalState():
            self.terminalStatesReached += 1
            self.maleRewardsPerEpisode.append(sum(self.RewardsMalePerStep))  # adding sum of male reward from current terminal state
            self.RewardsMalePerStep = []  # initializing male reward to 0
            self.femaleRewardsPerEpisode.append(sum(self.RewardsFemalePerStep))  # adding sum of female reward from current terminal state
            self.RewardsFemalePerStep = []  # initializing female reward to 0
            self.totalRewardsPerEpisode.append(self.maleRewardsPerEpisode[-1] + self.femaleRewardsPerEpisode[-1])
            self.stepsPerTerminalState.append(self.stepCounter)
            self.stepCounter = 0
            self.manhattanDistancePerState.append(self.manhattanDistance)
            self.closeToEachOther.append(self.manhattanDistance.count(1))
            self.manhattanDistance = []
            self.resetWorld()
        self.stepCounter += 1
        male, x,y,z, holding = self.Turn(step)  # uses step count to determine which players turn it is and the position of that player
        self.manhattanDistance.append(self.manhattan(self.malePos, self.femalePos) if male else self.manhattan(self.femalePos, self.malePos))
        if not holding and ((x,y,z)) in self.pickUpCells and self.world.pickup_items[self.pickUpCells.index((x,y,z))] > 0:  # if we can pick up then do so
            self.pickUp(male, x,y,z, holding)
            nextDirection = 'Pickup'
            if not male:
                self.Femalerewards.append([(x,y,z), 'Pickup', self.getRewards((x,y,z), holding), holding, self.getDropOffIndex() if holding else self.getPickUpIndex()])
            else:
                self.MaleRewards.append([(x,y,z), 'Pickup', self.getRewards((x,y,z), holding), holding, self.getDropOffIndex() if holding else self.getPickUpIndex()])
        elif ((x,y,z)) in self.dropOffCells and self.world.dropoff_capacity[self.dropOffCells.index((x,y,z))] < 5  and ((male and self.MaleState) or (not male and self.FemaleState)):  # else if we can drop off then do so else if we can drop off then do so
            self.dropOff(male, x,y,z, holding)
            nextDirection = 'Dropoff'
            if not male:
                self.Femalerewards.append([(x,y,z), 'Dropoff', self.getRewards((x,y,z), holding), holding, self.getDropOffIndex() if holding else self.getPickUpIndex()])
            else:
                self.MaleRewards.append([(x,y,z), 'Dropoff', self.getRewards((x,y,z), holding), holding, self.getDropOffIndex() if holding else self.getPickUpIndex()])
            
        else:  # else make a move
            if policy == "PRANDOM":
                NextAction = random.sample(list(self.qTable[(x,y,z)].keys()) if male else list(self.qTable_[(x,y,z)].keys()),
                    len(self.qTable[(x,y,z)]) if male else len(self.qTable_[(x,y,z)]))  # directions ordered randomly
                nextDirection= self.NextStep(male, NextAction, holding, self.getDropOffIndex() if holding else self.getPickUpIndex())
            elif policy == "PGREEDY":
                NextAction = sorted(self.qTable[(x,y,z)] if male else self.qTable_[(x,y,z)] , key=lambda i: self.qTable[(x,y,z)][i][holding][
                    self.getDropOffIndex() if holding else self.getPickUpIndex()] if male else self.qTable_[(x,y,z)][i][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()],
                        reverse=True)  # directions ordered best to worst
                nextDirection= self.NextStep(male, NextAction, holding, self.getDropOffIndex() if holding else self.getPickUpIndex())

            elif policy == "PEXPLOIT":
                decideWhich = np.random.uniform()
                if decideWhich < 0.8:
                    NextAction = sorted(self.qTable[(x,y,z)] if male else self.qTable_[(x,y,z)],
                        key=lambda i: self.qTable[(x,y,z)][i][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()] if male else
                        self.qTable_[(x,y,z)][i][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()], reverse=True)  # directions ordered best to worst
                    nextDirection = self.NextStep(male, NextAction, holding, self.getDropOffIndex() if holding else self.getPickUpIndex())
                else:
                    NextAction = random.sample(list(self.qTable[(x,y,z)].keys()) if male else list(self.qTable_[(x,y,z)].keys()),
                        len(self.qTable[(x,y,z)]) if male else len(self.qTable_[(x,y,z)]))  # directions ordered randomly
                    nextDirection = self.NextStep(male, NextAction, holding, self.getDropOffIndex() if holding else self.getPickUpIndex())
            else:
                print("Incorrect specification of policy name. Should be 'PRANDOM', 'PGREEDY', or 'PEXPLOIT'")
        
        oldPos = (x,y,z)  # curent position coordinates
        curPos = tuple(self.malePos) if male else tuple(self.femalePos)  # next position coordinates
        state(self.femalePos[0],self.femalePos[1],self.femalePos[2],1 if self.FemaleState else 0, self.malePos[0],self.malePos[1],self.malePos[2],1 if self.MaleState else 0)
        Qtable = self.qTable if male else self.qTable_
        self.updateQtable(curPos, oldPos, nextDirection, male, holding, step, Qtable)  # updating qTable
       
    def updateQtable(self, nextpos, currPos, direction, male, holding, step, Qtable):  # update q table for every move regardless of agent FOR CURRENT POSITION
        if step >= 2 and self.SARSA:
            S, oldMove, oldReward, oldHolding, old_qTableIndex = self.MaleRewards[-1] if male else self.Femalerewards[-1]
            S = tuple(S)
            Qtable[S][oldMove][oldHolding][old_qTableIndex] += \
                self.learning_rate * \
                    (oldReward + \
                    self.discount_factor * Qtable[currPos][direction][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()] - \
                    Qtable[S][oldMove][oldHolding][old_qTableIndex])
        else:
            # getting max q-value from next position
            Qtable[currPos][direction][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()] += \
                1-self.learning_rate * Qtable[currPos][direction][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()]+\
                self.learning_rate * \
                    (self.getRewards(currPos, holding) + \
                    self.discount_factor * max([val[holding][self.getDropOffIndex() if holding else self.getPickUpIndex()] for val in Qtable[nextpos].values()]) -\
                    Qtable[currPos][direction][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()])
        print(currPos,direction,holding,self.getDropOffIndex() if holding else self.getPickUpIndex(),Qtable[currPos][direction][holding][self.getDropOffIndex() if holding else self.getPickUpIndex()])

    def visualize_steps_per_terminal_state(self):
        y = self.stepsPerTerminalState
        x = range(1,len(y)+1)
        title = f"{self.experimentName}\nSteps per Terminal State"
        plot = sb.lineplot(x=x, y=y, marker = 'o').set(title=title, xlabel="Terminal State", ylabel="Steps")
        plt.show()
    
    def visualize_rewards_per_terminal_state(self):
        maleRewardsPerTerminalState = self.maleRewardsPerEpisode
        femaleRewardsPerTerminalState = self.femaleRewardsPerEpisode
        y = np.array(maleRewardsPerTerminalState) + np.array(femaleRewardsPerTerminalState)
        x = range(1,len(y)+1)
        title = f"{self.experimentName}\nRewards per Terminal State"
        sb.lineplot(x=x, y=y, marker = 'o').set(title=title, xlabel="Terminal State", ylabel="Rewards for both Agents")
        plt.show()
    
    def output_qTable(self, fileName='qTable.txt'):
        with open(fileName, 'w') as f:
            # make all states
            states = [(x,y,z,h,p_s,d_s) for x in range(3) for y in range(3) for z in range(3) for h in [False,True] for p_s in range(4) for d_s in range(16)]
            actions = ['North','East', 'South','West','Up', 'Down']
            qTable = {state:{action:self.qTable[state[:3]][action][state[3]][state[5] if state[3] else state[4]] if action in self.qTable[state[:3]] else 0 
                        for action in actions} 
                            for state in states}
            qTable_ = {state:{action:self.qTable_[state[:3]][action][state[3]][state[5] if state[3] else state[4]] if action in self.qTable_[state[:3]] else 0 
                        for action in actions} 
                            for state in states}
            f.write(" male Agent \n")
            f.write("x = X, y = Y, z = Z, h = holdingStatus, pi = PickupIndex, di = DropOffIndex\n")
            f.write(f"{'states':^20s}{'actions':^40s}\n")  
            f.write(f"{' x, y, z,  h,   pi, di':20s}{'North':>10s}{'East':>10s}{'South':>10s}{'west':>10s}{'Up':>10s}{'Down':>10s}\n")
            for state in states:
                north,east,south,west,up,down = f"{qTable[state]['North']:.3f}",f"{qTable[state]['East']:.3f}",f"{qTable[state]['South']:.3f}",f"{qTable[state]['West']:.3f}",f"{qTable[state]['Up']:.3f}",f"{qTable[state]['Down']:.3f}"
                f.write(f"{str(state):20s}{north:>10s}{east:>10s}{south:>10s}{west:>10s}{up:>10s}{down:>10s}\n")
            f.write(" Female Agent \n")
            f.write("x = X, y = Y, z = Z, h = holdingStatus, pi = PickupIndex, di = DropOffIndex\n")
            f.write(f"{'states':^20s}{'actions':^40s}\n")  
            f.write(f"{' x, y, z,  h,   pi, di':20s}{'North':>10s}{'East':>10s}{'South':>10s}{'west':>10s}{'Up':>10s}{'Down':>10s}\n")
            for state in states:
                north,east,south,west,up,down = f"{qTable_[state]['North']:.3f}",f"{qTable_[state]['East']:.3f}",f"{qTable_[state]['South']:.3f}",f"{qTable_[state]['West']:.3f}",f"{qTable_[state]['Up']:.3f}",f"{qTable_[state]['Down']:.3f}"
                f.write(f"{str(state):20s}{north:>10s}{east:>10s}{south:>10s}{west:>10s}{up:>10s}{down:>10s}\n")