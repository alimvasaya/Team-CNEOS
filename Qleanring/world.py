from vpython import *
from state import state
from time import *

class PDWorld:
    def __init__(self, num_cubes, cube_size, cube_spacing, Pickup_cell, Dropoff_cell, pickup_items, dropoff_capacity, Risky_cell,state):
        self.num_cubes = num_cubes
        self.cube_size = cube_size
        self.cube_spacing = cube_spacing
        self.Pickup_cell = Pickup_cell
        self.Dropoff_cell = Dropoff_cell
        self.pickup_items = pickup_items
        self.dropoff_capacity = dropoff_capacity
        self.Risky_cell = Risky_cell
        self.state = state
        self.agent_female_current_X = 0
        self.agent_female_current_Y = 0 
        self.agent_female_current_Z = 0
        self.agent_male_current_X = 0 
        self.agent_male_current_Y = 0 
        self.agent_male_current_Z = 0
        

        self.create_grid()
        self.create_agents()
        self.create_pickup_points()
        self.create_dropoff_points()


    def stateToCoordinate(self, state, cube_size, cube_spacing):
        F_x, F_y, F_z, F_i, M_x, M_y, M_z, M_i = state.get()
        return [
            F_x * (cube_size + cube_spacing),
            F_y * (cube_size + cube_spacing),
            F_z * (cube_size + cube_spacing),
            M_x * (cube_size + cube_spacing),
            M_y * (cube_size + cube_spacing),
            M_z * (cube_size + cube_spacing)
        ]
    

    def create_grid(self):
        self.cube_list = []
        for i in range(self.num_cubes[0]):
            for j in range(self.num_cubes[1]):
                for k in range(self.num_cubes[2]):
                    x = i * (self.cube_size + self.cube_spacing)
                    y = j * (self.cube_size + self.cube_spacing)
                    z = k * (self.cube_size + self.cube_spacing)
                    if (i, j, k) in self.Risky_cell:
                        cube_risky = box(pos=vector(x, y, z), size=vector(self.cube_size, self.cube_size, self.cube_size), color=color.red, opacity=0.5)
                    elif (i, j, k) in self.Pickup_cell:
                        cube_pickup = box(pos=vector(x, y, z), size=vector(self.cube_size, self.cube_size, self.cube_size), color=color.yellow, opacity=0.5)
                    elif (i, j, k) in self.Dropoff_cell:
                        cube_dropoff = box(pos=vector(x, y, z), size=vector(self.cube_size, self.cube_size, self.cube_size), color=color.green, opacity=0.5)
                    else:
                        cube = box(pos=vector(x, y, z), size=vector(self.cube_size, self.cube_size, self.cube_size), color=color.white, opacity=0.5)
                        self.cube_list.append(cube)

    def create_agents(self):
        self.agent_female_current_X, self.agent_female_current_Y,self.agent_female_current_Z,self.agent_male_current_X,self.agent_male_current_Y,self.agent_male_current_Z = self.stateToCoordinate(self.state, self.cube_size, self.cube_spacing)
        self.sphere1 = sphere(pos=vector(self.agent_female_current_X, self.agent_female_current_Y, self.agent_female_current_Z),radius=1, color=color.orange)
        self.sphere2 = sphere(pos=vector(self.agent_male_current_X, self.agent_male_current_Y, self.agent_male_current_Z),radius=1, color=color.black)
        
        
    def create_pickup_points(self):  
        # Create pickup points
        self.pickups = []
        self.pickups2 = []
        offset_points = 2
        for cell in self.Pickup_cell:
            x,y,z = cell
            indx = self.Pickup_cell.index(cell)
            totalP = self.pickup_items[indx]
            for i in range(totalP):
                pickup_x = x * (self.cube_size + self.cube_spacing) + i * self.cube_spacing
                pickup_y = y * (self.cube_size + self.cube_spacing)
                pickup_z = z * (self.cube_size + self.cube_spacing)
                self.pickup_points = sphere(pos=vector(pickup_x-offset_points, pickup_y, pickup_z), radius=0.1, color=color.red)
                if indx == 0:
                    self.pickups.append(self.pickup_points)
                else:
                    self.pickups2.append(self.pickup_points)
            

    def create_dropoff_points(self):     
        # Create dropoff points
        self.dropoff_points = []
        self.dropoff_points2 = []
        self.dropoff_points3 = []
        self.dropoff_points4 = []
        offset_droppoints =  1      
        for cell in self.Dropoff_cell:
            x,y,z = cell
            indx = self.Dropoff_cell.index(cell)
            self.totalD = self.dropoff_capacity[indx]
            for d in range(self.totalD):
                dropoff_x = x * (self.cube_size + self.cube_spacing) + d * self.cube_spacing
                dropoff_y = y * (self.cube_size + self.cube_spacing)
                dropoff_z = z * (self.cube_size + self.cube_spacing)
                self.dropoff = sphere(pos=vector(dropoff_x-offset_droppoints,dropoff_y,dropoff_z),radius=0.2, color=color.blue)
                if indx == 0:
                    self.dropoff_points.append(self.dropoff)
                elif indx == 1:
                    self.dropoff_points2.append(self.dropoff)
                elif indx == 2:
                    self.dropoff_points3.append(self.dropoff)
                else:
                    self.dropoff_points4.append(self.dropoff)


    def reset(self):
        self.pickup_items = [10,10]
        self.dropoff_capacity = [0,0,0,0]
        # Delete old pickup points
        for pickup in self.pickups:
            pickup.visible = True

        for pickup in self.pickups2:
            pickup.visible = True

        return
