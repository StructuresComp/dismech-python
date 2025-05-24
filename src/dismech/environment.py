class Environment:

    def __init__(self):
        self.__ext_force_list = []

    def add_force(self, key, **kwargs) -> None:
        if key == 'gravity':
            self.g = kwargs['g']
        elif key == 'buoyancy':
            self.rho = kwargs['rho']
        elif key == 'rft':
            self.ct = kwargs['ct']
            self.cn = kwargs['cn']
        elif key == 'damping':
            self.eta = kwargs['eta']
        elif key == 'hydrodynamics':
            self.rho = kwargs['rho']    # REUSING
            self.cd = kwargs['cd']
        elif key == 'aerodynamics':
            self.rho = kwargs['rho']    # REUSING
            self.cd = kwargs['cd']
        elif key == 'thrust':
            self.thrust_coeff = kwargs['thrust_coeff']
        elif key == 'pointForce':
            self.pt_force = kwargs['pt_force']
            self.pt_force_node = kwargs['pt_force_node']
        # TODO: Translate the following forces
        elif key == 'selfContact':
            self.delta = kwargs['delta']
            self.h = kwargs['h']
            self.imc_stiffness = kwargs['kc']
        elif key == 'selfFriction':
            pass
        elif key == 'floorContact':
            self.ground_z = kwargs['ground_z']
            self.ground_stiffness = kwargs['stiffness']
            self.ground_delta = kwargs['delta']
            self.ground_h = kwargs['h']
        elif key == 'floorFriction':
            self.ground_mu = kwargs['mu']
            self.ground_vel_tol = kwargs['vel_tol']
        else:
            raise KeyError
        
        self.__ext_force_list.append(key)

    @property
    def ext_force_list(self):
        return self.__ext_force_list

    def set_static(self):
        if hasattr(self, 'g'):
            self.static_g = self.g
