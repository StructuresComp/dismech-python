import numpy as np
from typing import Tuple, Dict
from .elastic_energy import ElasticEnergy  # Import the base class

class twistingStrainEnergy(ElasticEnergy):
    def __init__(self, material_properties):
        super().__init__(material_properties)
        self.twist = 0.0
        self.gradTwist = np.zeros((11))
        self.hessTwist = np.zeros((11, 11))
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
        norm2_e = norm_e ** 2
        norm2_f = norm_f ** 2
        te = ee / norm_e
        tf = ef / norm_f

        # Curvature binormal
        kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

        # Gradient of twist
        gradTwist = np.zeros(11)
        gradTwist[0:3] = -0.5 / norm_e * kb
        gradTwist[6:9] = 0.5 / norm_f * kb
        gradTwist[3:6] = -(gradTwist[0:3] + gradTwist[6:9])
        gradTwist[9] = -1
        gradTwist[10] = 1

        chi = 1.0 + np.dot(te,tf)
        tilde_t = (te + tf) / chi
        te_plus_tilde_t = te + tilde_t
        kb_o_te = np.outer(kb, te_plus_tilde_t)
        te_o_kb = np.outer(te_plus_tilde_t, kb)
        tf_plus_tilde_t = tf + tilde_t
        kb_o_tf = np.outer(kb, tf_plus_tilde_t)
        tf_o_kb = np.outer(tf_plus_tilde_t, kb)
        kb_o_tilde_t = np.outer(kb, tilde_t)

        ## Hessian of twist wrt DOFs
        hessTwist  = np.zeros((11,11))
        # Bergou 2010 Formulation is below.
        # D2mDe2 = - 0.25 / norm2_e * (kb_o_te + te_o_kb)
        # D2mDf2 = - 0.25 / norm2_f * (kb_o_tf + tf_o_kb)
        # D2mDeDf = 0.5 / (norm_e * norm_f) * (2.0 / chi * self.crossMat(te) - kb_o_tilde_t)
        # D2mDfDe = np.transpose(D2mDeDf)
        # Panetta 2019 formulation
        D2mDe2 = -0.5 / norm2_e * (np.outer(kb, (te + tilde_t)) + 2.0 / chi * self.crossMat(tf))
        D2mDf2 = -0.5 / norm2_f * (np.outer(kb, (tf + tilde_t)) - 2.0 / chi * self.crossMat(te))
        D2mDfDe = 0.5 / (norm_e * norm_f) * (2.0 / chi * self.crossMat(te) - np.outer(kb, tilde_t)) # CAREFUL: D2mDfDe means \partial^2 m/\partial e^i \partial e^{i-1}
        D2mDeDf = 0.5 / (norm_e * norm_f) * (-2.0 / chi * self.crossMat(tf) - np.outer(kb, tilde_t))

        # See Line 1145 of https://github.com/jpanetta/ElasticRods/blob/master/ElasticRod.cc
        hessTwist [0:3,0:3] = D2mDe2
        hessTwist [0:3,3:6] = - D2mDe2 + D2mDfDe
        hessTwist [3:6,0:3] = - D2mDe2 + D2mDeDf
        hessTwist [3:6,3:6] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
        hessTwist [0:3,6:9] = - D2mDfDe
        hessTwist [6:9,0:3] = - D2mDeDf
        hessTwist [6:9,3:6] = D2mDeDf - D2mDf2
        hessTwist [3:6,6:9] = D2mDfDe - D2mDf2
        hessTwist [6:9,6:9] = D2mDf2

        return gradTwist, hessTwist


    @staticmethod
    def crossMat(a):
        """
        Returns the cross product matrix of vector 'a'.

        Parameters:
        a : np.ndarray
            A 3-element array representing a vector.

        Returns:
        A : np.ndarray
            The cross product matrix corresponding to vector 'a'.
        """
        A = np.array([[0, -a[2], a[1]],
                    [a[2], 0, -a[0]],
                    [-a[1], a[0], 0]])

        return A
