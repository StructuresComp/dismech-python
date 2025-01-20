class Environment:

    def __init__(self):
        self.__ext_force_list = []

    def add_force(self, key, **kwargs) -> None:
        if key == 'gravity':
            self.g = kwargs['g']
        elif key == 'buoyancy':
            self.rho = kwargs['rho']
        elif key == 'viscous':
            self.eta = kwargs['eta']
        elif key == 'aerodynamics':
            self.rho = kwargs['rho']    # REUSING
            self.cd = kwargs['cd']
        elif key == 'pointForce':
            self.pt_force = kwargs['pt_force']
            self.pt_force_node = kwargs['pt_force_node']
        # TODO: Translate the following forces
        elif key == 'selfContact':
            pass
        elif key == 'selfFriction':
            pass
        elif key == 'floorContact':
            pass
        elif key == 'floorFriction':
            pass
        else:
            raise KeyError
        
        self.__ext_force_list.append(key)

    @property
    def ext_force_list(self):
        return self.__ext_force_list

    def set_static(self):
        if hasattr(self, 'g'):
            self.static_g = self.g
