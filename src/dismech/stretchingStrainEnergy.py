import numpy as np
from typing import Tuple

class StretchingStrainEnergy:
    def __init__(self):
        self.eps: float = 0.0
        self.grad_eps: np.ndarray = np.zeros((1, 6))
        self.hess_eps: np.ndarray = np.zeros((6, 6))
        self.E: float = 0.0
        self.gradE: np.ndarray = np.zeros((1, 6))
        self.hessE: np.ndarray = np.zeros((6, 6))
        self.F: np.ndarray = np.zeros((6, 1))
        self.J: np.ndarray = np.zeros((6, 6))

    def get_strain_stretch_edge(self, node0: np.ndarray, node1: np.ndarray, l_k: float) -> float:
        edge = node1 - node0
        edgeLen = np.linalg.norm(edge)
        self.eps = edgeLen / l_k - 1
        return self.eps

    def grad_and_hess_strain_stretch_edge(self, node0: np.ndarray, node1: np.ndarray, l_k: float) -> Tuple[np.ndarray, np.ndarray]:
        edge = node1 - node0
        edgeLen = np.linalg.norm(edge)
        tangent = edge / edgeLen
        self.eps = self.get_strain_stretch_edge2D(node0, node1, l_k)

        dF_unit = tangent / l_k
        dF = np.zeros((4,))
        dF[0:2] = -dF_unit
        dF[2:4] = dF_unit
        self.grad_eps = dF

        Id3 = np.eye(2)
        M = 2.0 / l_k * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * (np.outer(edge, edge)) / edgeLen ** 2)

        if self.eps == 0:
            M2 = np.zeros_like(M)
        else:
            M2 = 1.0 / (2.0 * self.eps) * (M - 2.0 * np.outer(dF_unit, dF_unit))

        dJ = np.zeros((4, 4))
        dJ[0:2, 0:2] = M2
        dJ[2:4, 2:4] = M2
        dJ[0:2, 2:4] = -M2
        dJ[2:4, 0:2] = -M2
        self.hess_eps = dJ

        return self.grad_eps, self.hess_eps

    def get_energy_stretch_linear_elastic(self, node0: np.ndarray, node1: np.ndarray, l_eff: float, EA: float) -> float:
        self.E = 0.5 * EA * self.get_strain_stretch_edge2D(node0, node1, l_eff)**2.0 * l_eff
        return self.E

    def grad_and_hess_energy_stretch_linear_elastic(self, node0: np.ndarray, node1: np.ndarray, l_eff: float, EA: float) -> Tuple[np.ndarray, np.ndarray]:
        self.eps = self.get_strain_stretch_edge2D(node0, node1, l_eff)
        self.grad_eps, self.hess_eps = self.grad_and_hess_strain_stretch_edge2D(node0, node1, l_eff)

        gradE_strain = EA * self.eps * l_eff
        hessE_strain = EA * l_eff

        self.gradE = gradE_strain * self.grad_eps
        self.hessE = gradE_strain * self.hess_eps + (self.grad_eps.T * hessE_strain * self.grad_eps)
        self.F = -self.gradE.T
        self.J = -self.hessE

        return self.F, self.J
