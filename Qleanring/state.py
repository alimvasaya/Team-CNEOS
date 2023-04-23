class state:
    def __init__(self,F_x, F_y, F_z, F_i, M_x, M_y, M_z, M_i):
        self.F_x = F_x
        self.F_y = F_y
        self.F_z = F_z
        self.F_i = F_i
        self.M_x = M_x
        self.M_y = M_y
        self.M_z = M_z
        self.M_i = M_i
        self.qvalues = 0
    def get(self):
        return[self.F_x,self.F_y,self.F_z,self.F_i,self.M_x,self.M_y,self.M_z,self.M_i]
    