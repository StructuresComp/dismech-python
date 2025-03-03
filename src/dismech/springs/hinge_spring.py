import numpy as np


class HingeSpring:
    def __init__(self, nodes_ind: np.ndarray, robot, kb: float = None):
        self.kb = kb or robot.kb
        self.nodes_ind = nodes_ind
        self.ind = np.concatenate([robot.map_node_to_dof(i)
                                  for i in nodes_ind], axis=0)
        self.theta_bar = self.get_theta(*np.split(robot.state.q[self.ind], 4))

    # TODO: migrate this to hinge_energy
    @staticmethod
    def get_theta(n0, n1, n2, n3):
        m_e0 = n1 - n0
        m_e1 = n2 - n0
        m_e2 = n3 - n0

        c0 = np.cross(m_e0, m_e1)
        c1 = np.cross(m_e2, m_e0)

        w = np.cross(c0, c1)
        angle = np.atan2(np.linalg.norm(w), np.dot(c0, c1))
        return -angle if np.dot(m_e0, w) < 0 else angle
