import numpy as np
from typing import Tuple, Dict
from .elastic_energy import ElasticEnergy  # Import the base class

class twistingStrainEnergy(ElasticEnergy):
    def __init__(self, material_properties):
        super().__init__(material_properties)
        self.twist = np.zeros((1, 2))
        self.gradTwist = np.zeros((2, 11))
        self.hessTwist = np.zeros((2, 11, 11))
        self.E = 0.0
        self.gradE = np.zeros((1, 11))
        self.hessE = np.zeros((11, 11))
        self.F = np.zeros((11, 1))
        self.J = np.zeros((11, 11))

    def get_strain(self, deformation: Dict[str, np.ndarray]) -> np.ndarray:
        theta_f, theta_e, refTwist = deformation["theta_f"], deformation["theta_e"], deformation["refTwist"]
        self.twist = theta_f - theta_e + refTwist
        return self.twist
    
    def grad_hess_strain(self, deformation: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        node0, node1, node2, theta_f, theta_e, refTwist = deformation["node0"], deformation["node1"], deformation["node2"], deformation["theta_f"], deformation["theta_e"], deformation["refTwist"]
        
        ee = node1 - node0
        ef = node2 - node1
        norm_e = np.linalg.norm(ee)
        norm_f = np.linalg.norm(ef)
        te = ee / norm_e
        tf = ef / norm_f

        # Curvature binormal
        kb = self.get_strain_curvature(self, node0, node1, node2)

        gradTwist = np.zeros(11)
        gradTwist[0:3] = -0.5 / norm_e * kb
        gradTwist[6:9] = 0.5 / norm_f * kb
        gradTwist[3:6] = -(gradTwist[0:3] + gradTwist[6:9])
        gradTwist[9] = -1
        gradTwist[10] = 1