import abc

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display

class Visualizer(metaclass=abc.ABCMeta):
    """ Abstract class to visualize a robot throughout a simulation. """

    @abc.abstractmethod
    def update(self, robot, t):
        """ Called periodically by timestepper to update the visualization. """
        pass

class MatplotlibLogger:
    """Logs the (x,y,z) coordinates of provided nodes on a position vs time line graph."""
    
    def __init__(self, nodes: np.ndarray):
        self._nodes = nodes
        self.time_history = []
        self.q_history = []

        # Initialize the figure and axis
        self.fig, self.ax = plt.subplots()
        self.lines = {}  # Dictionary to store line objects for each node
        
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("q (Position)")
        self.ax.set_title("Position vs Time")

    def update(self, robot, t):
        inds = robot.map_node_to_dof(self._nodes)
        q = robot.state.q[inds]

        self.time_history.append(t)
        self.q_history.append(q)

        time_array = np.array(self.time_history)
        q_array = np.array(self.q_history)

        # Update or create lines for each node
        for i, ind in enumerate(inds):
            if ind not in self.lines:
                (line,) = self.ax.plot([], [], label=f"{ind}")
                self.lines[ind] = line
            
            self.lines[ind].set_data(time_array, q_array[:, i])

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        
        display.clear_output(wait=True)
        display.display(self.fig)