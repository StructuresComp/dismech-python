import numpy as np

class StretchSpring:

    def __init__(self, length, node_indices, robot, EA=0):
        if EA == 0:
            self.__EA = robot.EA
        else:
            self.__EA = EA

        self.__ref_len = length
        self.__node_indices = node_indices
        self.__indices = np.concat([robot.map_node_to_dof(self.node_indices[0]), robot.map_node_to_dof(self.node_indices[1])])

    @property
    def EA(self):
        return self.__EA

    @property
    def ref_len(self):
        return self.__ref_len

    @property
    def node_indices(self):
        return self.__node_indices
    
    @property
    def indices(self):
        return self.__indices