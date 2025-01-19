class Environment:

    def __init__(self):
        pass

    def add_force(self, key, **kwargs) -> None:
        match(key):
            case 'gravity':
                self.g = kwargs['g']
            case 'buoyancy':
                self.rho = kwargs['rho']
            case 'viscous':
                self.eta = kwargs['eta']
            case'aerodynamics':
                self.rho = kwargs['rho']    # REUSING
                self.cd = kwargs['cd']
            case 'pointForce':
                self.pt_force = kwargs['pt_force']
                self.pt_force_node = kwargs['pt_force_node']
            # TODO: Translate the following forces
            case 'selfContact':
                pass
            case 'selfFriction':
                pass
            case 'floorContact':
                pass
            case 'floorFriction':
                pass
            case '_':
                raise KeyError

    def set_static(self):
        if hasattr(self, 'g'):
            self.static_g = self.g
