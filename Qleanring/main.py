from Qtable import qtable
from world import PDWorld
from time import *
from state import state
from vpython import *
from RL import *
import sys


def experiment1(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
    RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
    for step in range(500):
        male, x, y,z, i = RL.Turn(step)
        RL.POLICY(step, "PRANDOM")
        
    if subExperiment == 'a':
        experiment1_a(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate)
    elif subExperiment == 'b':
        experiment1_b(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate)
    elif subExperiment == 'c':
        experiment1_c(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate)
    else:
        print("You're trying to run experiment 1, but you need to specify which subExperiment.")
    
def experiment1_a(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
    RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
    for step in range(500,9500):
        male, _, _,_, _ = RL.Turn(step)
        RL.POLICY(step, "PRANDOM")
        

    with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w") as file:
        sys.stdout = file
        print(f"Terminal States reached: {RL.terminalStatesReached}")
        print(f"Steps   per terminal state: {RL.stepsPerTerminalState}")
        print(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}")
        RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_final_qTable.txt")
        RL.visualize_steps_per_terminal_state()
        return RL
    
def experiment1_b(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
    RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
    for step in range(500,9500):
        male, _, _,_, holding = RL.Turn(step)
        RL.POLICY(step, "PGREEDY")
        
        
    with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w") as file:
        sys.stdout = file
        print(f"Terminal States reached: {RL.terminalStatesReached}")
        print(f"Steps   per terminal state: {RL.stepsPerTerminalState}")
        print(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}")
        RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_final_qTable.txt")
        RL.visualize_steps_per_terminal_state()
        return RL
    

def experiment1_c(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
    RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
    for step in range(500,9500):
        male, _, _,_, _ = RL.Turn(step)
        RL.POLICY(step, "PEXPLOIT")
        
        
    with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w") as file:
        sys.stdout = file
        print(f"Terminal States reached: {RL.terminalStatesReached}")
        print(f"Steps   per terminal state: {RL.stepsPerTerminalState}")
        print(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}")
        RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_final_qTable.txt")
        RL.visualize_steps_per_terminal_state()
        return RL

def experiment2(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
    RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
    for step in range(500):
        male, _, _,_, _ = RL.Turn(step)
    for step in range(500,9500):
        male, _, _,_, _ = RL.Turn(step)
        RL.POLICY(step, "PEXPLOIT")
        
    with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w") as file:
        sys.stdout = file
        print(f"Terminal States reached: {RL.terminalStatesReached}")
        print(f"Steps   per terminal state: {RL.stepsPerTerminalState}")
        print(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}")
        RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_final_qTable.txt")
        RL.visualize_steps_per_terminal_state()
        return RL
    
    
def experiment3(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
        print(f"Experiment 3 with alpha={learning_rate} ", end="")
        RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
        for step in range(500):
            male, _, _,_, _ = RL.Turn(step)
            RL.POLICY(step, "PRANDOM")
            
        for step in range(500,9500):
            male, _, _,_, _ = RL.Turn(step)
            RL.POLICY(step, "PEXPLOIT")
            
        with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w") as file:
            sys.stdout = file
            print(f"Terminal States reached: {RL.terminalStatesReached}")
            print(f"Steps   per terminal state: {RL.stepsPerTerminalState}")
            print(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}")
            RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_final_qTable.txt")
            RL.visualize_steps_per_terminal_state()
            return RL
        
def experiment4(experiment,subExperiment, algorithm, world, qtable, seed, learning_rate, Discount_rate):
    RL = Qlearning(experiment,subExperiment,world,qtable,learning_rate,Discount_rate,SARSA=False if algorithm == 0 else True,seed=seed)
    for step in range(500):
        male, _, _,_, _ = RL.Turn(step)
        RL.POLICY(step, "PRANDOM")
        
    for step in count():
        male, _, _,_, _ = RL.Turn(step)
        RL.POLICY(step, "PEXPLOIT")
        
        if RL.terminalStatesReached == 3:
            break
    output_file1 = open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w")

    with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_output.txt", "w") as file:
        print_to_file_and_console(f"Terminal States reached: {RL.terminalStatesReached}", output_file1)
        print_to_file_and_console(f"Steps   per terminal state: {RL.stepsPerTerminalState}", output_file1)
        print_to_file_and_console(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}", output_file1)
        RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_final_qTable.txt")
    output_file1.close()      
    RL.qTable[(1, 1, 0)].pop('Pickup')
    RL.qTable[(2, 2, 1)].pop('Pickup')
    RL.qTable[(1, 2, 2)]['Pickup'] = [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    RL.qTable[(0, 2, 0)]['Pickup'] = [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    RL.qTable_[(1, 1, 0)].pop('Pickup')
    RL.qTable_[(2, 2, 1)].pop('Pickup')
    RL.qTable_[(1, 2, 2)]['Pickup'] = [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    RL.qTable_[(0, 2, 0)]['Pickup'] = [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    RL.pickUpCells[0] = (1, 2, 2)
    # Update the second pickup cell
    RL.pickUpCells[1] = (0, 2, 0)
    sleep(1)
    for pickup in RL.world.pickups:
        pickup.visible = False
    del RL.world.pickups
    for pickup in RL.world.pickups2:
        pickup.visible = False
    del RL.world.pickups2
    sleep(1)
    RL.world.create_grid()
    RL.world.create_pickup_points()
    sleep(5)
    RL.resetWorld()
    for step in count():
        RL.POLICY(step, "PEXPLOIT")
        
        if RL.terminalStatesReached == 6:
            break
    output_file2 = open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_2_output.txt", "w")
    with open(f"experiment_{experiment}_{subExperiment}_seed_{seed}_2_output.txt", "w") as file:
        print_to_file_and_console(f"Terminal States reached: {RL.terminalStatesReached}", output_file2)
        print_to_file_and_console(f"Steps   per terminal state: {RL.stepsPerTerminalState}", output_file2)
        print_to_file_and_console(f"Rewards per terminal state: {RL.totalRewardsPerEpisode}", output_file2)
        RL.output_qTable(f"_{experiment}_{subExperiment}_seed_{seed}_changed_final_qTable.txt")
        RL.visualize_steps_per_terminal_state()
    output_file2.close()
    return RL  
    
def print_to_file_and_console(text, file):
    print(text)
    file.write(text + "\n")
def main():
    # Define grid parameters
    num_cubes1 = [3, 3, 3]
    cube_size1 = 5
    cube_spacing1 = 0.3
    Pickup_cell1 = [(1, 1, 0), (2, 2, 1)]
    Dropoff_cell1 = [(0, 0, 1), (0, 0, 2), (2, 0, 0), (2, 1, 2)]
    pickup_items1 = [10,10]
    dropoff_capacity1 = [0,0,0,0]
    Risky_cell1 = [(1, 1, 1), (2, 1, 0)]
    Starting_state = state(0,0,0,0,2,1,2,0)
    learning_rate = 0.3
    Discount_rate = 0.5
    seedC = int(input("Enter Seed: "))
    experiment = int(input("Enter experiment number (1-4): "))
    algorithm = int(input("Enter 0 for Q learning and 1 for SARSA: "))
   
    
    if experiment == 1 or experiment == 3:
        subExperiment = input("Enter sub-experiment (a, b, c): ")
    elif experiment == 2 or experiment == 4:
        subExperiment = ''

    one = PDWorld(num_cubes1, cube_size1, cube_spacing1, Pickup_cell1, Dropoff_cell1, pickup_items1, dropoff_capacity1, Risky_cell1, Starting_state)
    qtable1 = qtable()
    qtable2 = qtable()
    qtable3 = qtable()
    qtable4 = qtable()
    if experiment == 1 or experiment == 2 or experiment == 4:
        if experiment == 1:
            experiment1(experiment,subExperiment,algorithm, one, qtable1, seedC, learning_rate, Discount_rate)
        elif experiment == 2:
            experiment2(experiment,subExperiment,algorithm, one, qtable2, seedC, learning_rate, Discount_rate)
        elif experiment == 4:
            experiment4(experiment,subExperiment,algorithm, one, qtable3, seedC, learning_rate, Discount_rate)
    elif experiment == 3:
        if subExperiment == 'c':
            learning_rate = 0.3
            Discount_rate = 0.5
        elif subExperiment == 'a':
            learning_rate = 0.1
            Discount_rate = 0.5
        elif subExperiment == 'b':
            learning_rate = 0.5
            Discount_rate = 0.5
        experiment3(experiment,subExperiment,algorithm, one, qtable4, seedC, learning_rate, Discount_rate)
    else:
        print("Invalid experiment number.")
    exit()
if __name__ == "__main__":
    main()