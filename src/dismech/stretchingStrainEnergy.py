import numpy as np
from typing import Tuple, Dict
from .elastic_energy import ElasticEnergy  # Import the base class

class stretchingStrainEnergy(ElasticEnergy):
    def __init__(self, material_properties):
        super().__init__(material_properties)
        self.eps: float = 0.0
        self.grad_eps: np.ndarray = np.zeros((1, 6))
        self.hess_eps: np.ndarray = np.zeros((6, 6))
        self.E: float = 0.0
        self.gradE: np.ndarray = np.zeros((1, 6))
        self.hessE: np.ndarray = np.zeros((6, 6))
        self.F: np.ndarray = np.zeros((6, 1))
        self.J: np.ndarray = np.zeros((6, 6))

    def get_strain(self, deformation: Dict[str, np.ndarray]) -> float:
        node0, node1, l_k = deformation["node0"], deformation["node1"], deformation["nat_strain"]
        edge = node1 - node0
        edgeLen = np.linalg.norm(edge)
        self.eps = edgeLen / l_k - 1
        return self.eps

    def grad_hess_strain(self, deformation: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        node0, node1, l_k = deformation["node0"], deformation["node1"], deformation["nat_strain"]
        edge = node1 - node0
        edgeLen = np.linalg.norm(edge)
        tangent = edge / edgeLen
        self.eps = self.get_strain(deformation)

        dF_unit = tangent / l_k
        dF = np.zeros((6,))
        dF[0:3] = -dF_unit
        dF[3:6] = dF_unit
        self.grad_eps = dF

        Id3 = np.eye(3)
        M = 2.0 / l_k * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * (np.outer(edge, edge)) / edgeLen ** 2)

        if self.eps == 0:
            M2 = np.zeros_like(M)
        else:
            M2 = 1.0 / (2.0 * self.eps) * (M - 2.0 * np.outer(dF_unit, dF_unit))

        dJ = np.zeros((6, 6))
        dJ[0:3, 0:3] = M2
        dJ[3:6, 3:6] = M2
        dJ[0:3, 3:6] = -M2
        dJ[3:6, 0:3] = -M2
        self.hess_eps = dJ

        return self.grad_eps, self.hess_eps