import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class ElasticEnergy(ABC):
    def __init__(self, material_properties):
        self.material_properties = material_properties
    
    def get_energy_linear_elastic(self, nat_strain, strain):
        K = self.material_properties["K"]  # stiffness (with discrete geometry considerations) : unit Nm (same unit as energy)
        del_strain = strain - nat_strain
        return 0.5 * del_strain.T * K * del_strain
    
    def grad_hess_energy_linear_elastic(self, nat_strain, strain, grad_strain, hess_strain):
        K = self.material_properties["K"] # K unit: Nm (same unit as energy)
        del_strain = strain - nat_strain
        gradE_strain = K * del_strain
        hessE_strain = K  
        grad_energy = gradE_strain * grad_strain
        hess_energy = gradE_strain * hess_strain + grad_strain.T * hessE_strain * grad_strain
        return grad_energy, hess_energy
    
    @abstractmethod
    def get_strain(self, deformation: Dict[str, np.ndarray]): # will be uniquely defined for each individual elastic energy type
        pass
    
    @abstractmethod
    def grad_hess_strain(self, deformation: Dict[str, np.ndarray]): # will be uniquely defined for each individual elastic energy type
        pass
