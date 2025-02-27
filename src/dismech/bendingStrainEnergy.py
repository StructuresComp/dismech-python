import numpy as np
from typing import Tuple, Dict
from .elastic_energy import ElasticEnergy  # Import the base class

class bendingStrainEnergy(ElasticEnergy):
    def __init__(self, material_properties):
        super().__init__(material_properties)
        self.kappa = np.zeros((1, 2))
        self.gradKappa = np.zeros((2, 11))
        self.hessKappa = np.zeros((2, 11, 11))
        self.E = 0.0
        self.gradE = np.zeros((1, 11))
        self.hessE = np.zeros((11, 11))
        self.F = np.zeros((11, 1))
        self.J = np.zeros((11, 11))

    def get_strain(self, deformation: Dict[str, np.ndarray]) -> np.ndarray:
        node0, node1, node2 = deformation["node0"], deformation["node1"], deformation["node2"]
        ee = node1 - node0
        ef = node2 - node1
        
        te = ee / np.linalg.norm(ee)
        tf = ef / np.linalg.norm(ef)
        
        kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))
        
        return kb

    def grad_hess_strain(self, deformation: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        node0, node1, node2, m1e, m2e, m1f, m2f = deformation["node0"], deformation["node1"], deformation["node2"], deformation["m1e"], deformation["m2e"], deformation["m1f"], deformation["m2f"]

        ee = node1 - node0
        ef = node2 - node1
        norm_e = np.linalg.norm(ee)
        norm_f = np.linalg.norm(ef)
        te = ee / norm_e
        tf = ef / norm_f

        # Curvature binormal
        kb = self.get_strain_curvature(self, node0, node1, node2)
        chi = 1.0 + np.dot(te, tf)
        tilde_t = (te + tf) / chi
        tilde_d1 = (m1e + m1f) / chi
        tilde_d2 = (m2e + m2f) / chi

        # Curvatures
        kappa1 = 0.5 * np.dot(kb, m2e + m2f)
        kappa2 = - 0.5 * np.dot(kb, m1e + m1f)

        # Gradient
        Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
        Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))
        Dkappa2De = 1.0 / norm_e * (-kappa2 * tilde_t - np.cross(tf, tilde_d1))
        Dkappa2Df = 1.0 / norm_f * (-kappa2 * tilde_t + np.cross(te, tilde_d1))
        
        self.gradKappa[0, 0:3] = -Dkappa1De
        self.gradKappa[0, 3:6] = Dkappa1De - Dkappa1Df
        self.gradKappa[0, 6:9] = Dkappa1Df
        self.gradKappa[1, 0:3] = -Dkappa2De
        self.gradKappa[1, 3:6] = Dkappa2De - Dkappa2Df
        self.gradKappa[1, 6:9] = Dkappa2Df

        self.gradKappa[0, 9] = -0.5 * np.dot(kb, m1e)
        self.gradKappa[0, 10] = -0.5 * np.dot(kb, m1f)
        self.gradKappa[1, 9] = -0.5 * np.dot(kb, m2e)
        self.gradKappa[1, 10] = -0.5 * np.dot(kb, m2f)

        # Hessian
        # Initialize Hessians
        DDkappa1 = np.zeros((11, 11)) # Hessian of kappa1
        DDkappa2 = np.zeros((11, 11)) # Hessian of kappa2

        norm2_e = norm_e ** 2
        norm2_f = norm_f ** 2

        tt_o_tt = np.outer(tilde_t, tilde_t)
        tmp = np.cross(tf,tilde_d2)
        tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
        tt_o_tf_c_d2t = np.transpose(tf_c_d2t_o_tt)
        kb_o_d2e = np.outer(kb, m2e)
        d2e_o_kb = np.transpose(kb_o_d2e) # Not used in Panetta 2019
        te_o_te = np.outer(te, te)
        Id3 = np.eye(3)

        # Bergou 2010
        # D2kappa1De2 = 1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) - kappa1 / (chi * norm2_e) * (Id3 - te_o_te ) + 1.0 / (4.0 * norm2_e) * (kb_o_d2e + d2e_o_kb)
        # Panetta 2019
        D2kappa1De2 = 1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) - kappa1 / (chi * norm2_e) * (Id3 - te_o_te ) + 1.0 / (2.0 * norm2_e) * (kb_o_d2e)


        tmp = np.cross(te,tilde_d2)
        te_c_d2t_o_tt = np.outer(tmp, tilde_t)
        tt_o_te_c_d2t = np.transpose(te_c_d2t_o_tt)
        kb_o_d2f = np.outer(kb, m2f)
        d2f_o_kb = np.transpose(kb_o_d2f) # Not used in Panetta 2019
        tf_o_tf = np.outer( tf, tf )

        # Bergou 2010
        # D2kappa1Df2 = 1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) - kappa1 / (chi * norm2_f) * (Id3 - tf_o_tf) + 1.0 / (4.0 * norm2_f) * (kb_o_d2f + d2f_o_kb)
        # Panetta 2019
        D2kappa1Df2 = 1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) - kappa1 / (chi * norm2_f) * (Id3 - tf_o_tf) + 1.0 / (2.0 * norm2_f) * (kb_o_d2f)


        te_o_tf = np.outer(te, tf)
        D2kappa1DfDe = - kappa1 / (chi * norm_e * norm_f) * (Id3 + te_o_tf) + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - self.crossMat(tilde_d2))
        D2kappa1DeDf = np.transpose(D2kappa1DfDe)

        tmp = np.cross(tf,tilde_d1)
        tf_c_d1t_o_tt = np.outer(tmp, tilde_t)
        tt_o_tf_c_d1t = np.transpose(tf_c_d1t_o_tt)
        kb_o_d1e = np.outer(kb, m1e)
        d1e_o_kb = np.transpose(kb_o_d1e) # Not used in Panetta 2019

        # Bergou 2010
        # D2kappa2De2 = 1.0 / norm2_e * (2.0 * kappa2 * tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t) - kappa2 / (chi * norm2_e) * (Id3 - te_o_te) - 1.0 / (4.0 * norm2_e) * (kb_o_d1e + d1e_o_kb)
        # Panetta 2019
        D2kappa2De2 = 1.0 / norm2_e * (2.0 * kappa2 * tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t) - kappa2 / (chi * norm2_e) * (Id3 - te_o_te) - 1.0 / (2.0 * norm2_e) * (kb_o_d1e)

        tmp = np.cross(te,tilde_d1)
        te_c_d1t_o_tt = np.outer(tmp, tilde_t)
        tt_o_te_c_d1t = np.transpose(te_c_d1t_o_tt)
        kb_o_d1f = np.outer(kb, m1f)
        d1f_o_kb = np.transpose(kb_o_d1f) # Not used in Panetta 2019

        # Bergou 2010
        # D2kappa2Df2 = 1.0 / norm2_f * (2 * kappa2 * tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t) - kappa2 / (chi * norm2_f) * (Id3 - tf_o_tf) - 1.0 / (4.0 * norm2_f) * (kb_o_d1f + d1f_o_kb)
        # Panetta 2019
        D2kappa2Df2 = 1.0 / norm2_f * (2 * kappa2 * tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t) - kappa2 / (chi * norm2_f) * (Id3 - tf_o_tf) - 1.0 / (2.0 * norm2_f) * (kb_o_d1f)

        D2kappa2DfDe = - kappa2 / (chi * norm_e * norm_f) * (Id3 + te_o_tf) + 1.0 / (norm_e * norm_f) * (2 * kappa2 * tt_o_tt + tf_c_d1t_o_tt - tt_o_te_c_d1t + self.crossMat(tilde_d1))
        D2kappa2DeDf = np.transpose(D2kappa2DfDe)

        D2kappa1Dthetae2 = - 0.5 * np.dot(kb,m2e)
        D2kappa1Dthetaf2 = - 0.5 * np.dot(kb,m2f)
        D2kappa2Dthetae2 = 0.5 * np.dot(kb,m1e)
        D2kappa2Dthetaf2 = 0.5 * np.dot(kb,m1f)

        D2kappa1DeDthetae = 1.0 / norm_e * (0.5 * np.dot(kb,m1e) * tilde_t - 1.0 / chi * np.cross(tf,m1e))
        D2kappa1DeDthetaf = 1.0 / norm_e * (0.5 * np.dot(kb,m1f) * tilde_t - 1.0 / chi * np.cross(tf,m1f))
        D2kappa1DfDthetae = 1.0 / norm_f * (0.5 * np.dot(kb,m1e) * tilde_t + 1.0 / chi * np.cross(te,m1e))
        D2kappa1DfDthetaf = 1.0 / norm_f * (0.5 * np.dot(kb,m1f) * tilde_t + 1.0 / chi * np.cross(te,m1f))
        D2kappa2DeDthetae = 1.0 / norm_e * (0.5 * np.dot(kb,m2e) * tilde_t - 1.0 / chi * np.cross(tf,m2e))
        D2kappa2DeDthetaf = 1.0 / norm_e * (0.5 * np.dot(kb,m2f) * tilde_t - 1.0 / chi * np.cross(tf,m2f))
        D2kappa2DfDthetae = 1.0 / norm_f * (0.5 * np.dot(kb,m2e) * tilde_t + 1.0 / chi * np.cross(te,m2e))
        D2kappa2DfDthetaf = 1.0 / norm_f * (0.5 * np.dot(kb,m2f) * tilde_t + 1.0 / chi * np.cross(te,m2f))

        # Curvature terms
        DDkappa1[0:3,0:3] = D2kappa1De2
        DDkappa1[0:3,3:6] = - D2kappa1De2 + D2kappa1DfDe
        DDkappa1[0:3,6:9] = - D2kappa1DfDe
        DDkappa1[3:6,0:3] = - D2kappa1De2 + D2kappa1DeDf
        DDkappa1[3:6,3:6] = D2kappa1De2 - D2kappa1DeDf - D2kappa1DfDe + D2kappa1Df2
        DDkappa1[3:6,6:9] = D2kappa1DfDe - D2kappa1Df2
        DDkappa1[6:9,0:3] = - D2kappa1DeDf
        DDkappa1[6:9,3:6] = D2kappa1DeDf - D2kappa1Df2
        DDkappa1[6:9,6:9] = D2kappa1Df2

        # Twist terms
        DDkappa1[9,9] = D2kappa1Dthetae2
        DDkappa1[10,10] = D2kappa1Dthetaf2

        # Curvature-twist coupled terms
        DDkappa1[0:3,9] = - D2kappa1DeDthetae
        DDkappa1[3:6,9] = D2kappa1DeDthetae - D2kappa1DfDthetae
        DDkappa1[6:9,9] = D2kappa1DfDthetae
        DDkappa1[9,0:3] = np.transpose(DDkappa1[0:3,9])
        DDkappa1[9,3:6] = np.transpose(DDkappa1[3:6,9])
        DDkappa1[9,6:9] = np.transpose(DDkappa1[6:9,9])

        # Curvature-twist coupled terms
        DDkappa1[0:3,10] = - D2kappa1DeDthetaf
        DDkappa1[3:6,10] = D2kappa1DeDthetaf - D2kappa1DfDthetaf
        DDkappa1[6:9,10] = D2kappa1DfDthetaf
        DDkappa1[10,0:3] = np.transpose(DDkappa1[0:3,10])
        DDkappa1[10,3:6] = np.transpose(DDkappa1[3:6,10])
        DDkappa1[10,6:9] = np.transpose(DDkappa1[6:9,10])

        # Curvature terms
        DDkappa2[0:3,0:3] = D2kappa2De2
        DDkappa2[0:3,3:6] = - D2kappa2De2 + D2kappa2DfDe
        DDkappa2[0:3,6:9] = - D2kappa2DfDe
        DDkappa2[3:6,0:3] = - D2kappa2De2 + D2kappa2DeDf
        DDkappa2[3:6,3:6] = D2kappa2De2 - D2kappa2DeDf - D2kappa2DfDe + D2kappa2Df2
        DDkappa2[3:6,6:9] = D2kappa2DfDe - D2kappa2Df2
        DDkappa2[6:9,0:3] = - D2kappa2DeDf
        DDkappa2[6:9,3:6] = D2kappa2DeDf - D2kappa2Df2
        DDkappa2[6:9,6:9] = D2kappa2Df2

        # Twist terms
        DDkappa2[9,9] = D2kappa2Dthetae2
        DDkappa2[10,10] = D2kappa2Dthetaf2

        # Curvature-twist coupled terms
        DDkappa2[0:3,9] = - D2kappa2DeDthetae
        DDkappa2[3:6,9] = D2kappa2DeDthetae - D2kappa2DfDthetae
        DDkappa2[6:9,9] = D2kappa2DfDthetae
        DDkappa2[9,0:3] = np.transpose(DDkappa2[0:3,9])
        DDkappa2[9,3:6] = np.transpose(DDkappa2[3:6,9])
        DDkappa2[9,6:9] = np.transpose(DDkappa2[6:9,9])

        # Curvature-twist coupled terms
        DDkappa2[0:3,10] = - D2kappa2DeDthetaf
        DDkappa2[3:6,10] = D2kappa2DeDthetaf - D2kappa2DfDthetaf
        DDkappa2[6:9,10] = D2kappa2DfDthetaf
        DDkappa2[10,0:3] = np.transpose(DDkappa2[0:3,10])
        DDkappa2[10,3:6] = np.transpose(DDkappa2[3:6,10])
        DDkappa2[10,6:9] = np.transpose(DDkappa2[6:9,10])

        self.hessKappa = np.concatenate((DDkappa1[np.newaxis, :, :], DDkappa2[np.newaxis, :, :]), axis=0)
        
        return self.gradKappa, self.hessKappa

    # def get_energy_bending_linear_elastic(self, node0: np.ndarray, node1: np.ndarray, node2: np.ndarray, l_eff: float, EI1: float, EI2: float, kappaBar: np.ndarray) -> float:
    #     self.kappa = self.get_strain_curvature(node0, node1, node2)
    #     dKappa = self.kappa - kappaBar
    #     self.E = 0.5 * dKappa * np.array([[0, EI1],[EI2,0]]) * dKappa.T / l_eff # check this
    #     return self.E

    # def grad_and_hess_energy_bending_linear_elastic(self, node0: np.ndarray, node1: np.ndarray, node2: np.ndarray, l_eff: float, EI1: float, EI2: float, kappaBar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     self.kappa = self.get_strain_curvature(node0, node1, node2)
    #     self.gradKappa, self.hessKappa = self.grad_and_hess_strain_curvature(node0, node1, node2)
    #     dKappa = self.kappa - kappaBar

    #     gradE_strain = dKappa* np.array([[0, EI1],[EI2,0]]) / l_eff
    #     hessE_strain = np.array([[0, EI1],[EI2,0]]) / l_eff

    #     self.gradE = gradE_strain * self.gradKappa
    #     self.hessE = gradE_strain * self.hessKappa + (self.gradKappa.T * hessE_strain *  self.gradKappa)

    #     self.F = -self.gradE.T
    #     self.J = -self.hessE

    #     return self.F, self.J
    
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
