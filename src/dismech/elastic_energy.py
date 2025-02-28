import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

class ElasticEnergy(ABC):
    def __init__(self, material_properties):
        self.material_properties = material_properties
    
    def get_energy_linear_elastic(self, dictionary):
        K = self.material_properties["K"] # stiffness (with discrete geometry considerations) : unit Nm (same unit as energy)
        nat_strain = dictionary["nat_strain"]
        strain = self.get_strain(dictionary)
        del_strain = strain - nat_strain
        
        if isinstance(K, np.ndarray): # rod bending
            del_strain = del_strain.reshape(2, 1)
            Energy = 0.5 * del_strain.T @ K @ del_strain
        else:
            Energy = 0.5 * del_strain**2
        return Energy
    
    def grad_hess_energy_linear_elastic(self, dictionary):
        K = self.material_properties["K"] # stiffness (with discrete geometry considerations) : unit Nm (same unit as energy)
        nat_strain = dictionary["nat_strain"]
        
        strain = self.get_strain(dictionary)
        grad_strain, hess_strain = self.grad_hess_strain(dictionary)
        
        del_strain = strain - nat_strain

        if isinstance(K, np.ndarray): # rod bending
            grad_strain = grad_strain.reshape(11, 2)
            gradE_strain = (del_strain.reshape(1,2) @ K).flatten()
            hessE_strain = K  
            grad_energy = grad_strain @ gradE_strain
            hess_energy = (gradE_strain[0] * hess_strain[0,:,:] + gradE_strain[1] * hess_strain[1,:,:]) + grad_strain @ K @ grad_strain.T
            grad_energy = grad_energy.flatten()
        else:
            gradE_strain = K * del_strain
            hessE_strain = K  
            grad_energy = gradE_strain * grad_strain
            hess_energy = gradE_strain * hess_strain + hessE_strain * np.outer(grad_strain, grad_strain)


        return grad_energy, hess_energy
    
    @abstractmethod
    def get_strain(self, deformation: Dict[str, np.ndarray]): # will be uniquely defined for each individual elastic energy type
        pass
    
    @abstractmethod
    def grad_hess_strain(self, deformation: Dict[str, np.ndarray]): # will be uniquely defined for each individual elastic energy type
        pass
