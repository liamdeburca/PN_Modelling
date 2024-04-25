from functools import lru_cache

def pCN_change(x0:float, step_size:float, pCN_beta:float, bounds:tuple=(None, None)):
    """
    Function which randomly alters a parameters using the "preconditioned Crank-Nicolson" algorithm. 

    - x0: original parameter value. 
    - step_size: standard deviation used in the Gaussian distributed randomly generated number. 
    - beta: beta-value used in the pCN algorithm. 
    - bounds: any physical limitations on the parameter value. 

    """
    from numpy.random import normal

    rng = normal(loc=0, scale=step_size)
    x1 = (1 - pCN_beta**2)**0.5 * x0 + pCN_beta * rng

    x_min, x_max = bounds

    if x_min is not None:
        x1 = max(x1, x_min)
    if x_max is not None:
        x1 = min(x1, x_max)

    return x1

class Distance:
    """
    Distance object containing the distance to the object in pc.

    - getElementSize():
        Calculates the "size" of a pixel, given a distance to the object. 

    - change():
        Randomly changes the distance parameter. 

    """
    def __init__(self, d, amplitude, bounds, dtheta=0.2, change_weight=1):
        self.d = d
        self.dtheta = dtheta
        self.amplitude = amplitude
        self.bounds = bounds

        self.parsec = 3.086e18
        self.au = 1.496e13

        self.object_type = 'Distance'
        self.change_weight = change_weight

    @lru_cache(maxsize=2)
    def getElementSize(self):
        return self.dtheta * (self.d * self.parsec) * self.au

    def change(self, pCN_beta):

        lb, ub = self.bounds
        amp = self.amplitude

        new = Distance(self.d, self.amplitude, self.bounds)
        new.d = pCN_change(self.d, amp, pCN_beta, bounds=(lb, ub))

        return new
    

############################################Coordinates#############################################
    
class Coordinates:
    """
    Coordinates object containing information on the centre of the PN.
    
    - base_matrices():
        Calculates the two-dimensional x and z matrices in units of index. 
        Should therefore be multiplied by a cell size given by the distance to the object.

    - matrices():
        Calculates the two-dimensional x and z matrices in units of index, centred on the centre of the PN. 

    - change():
        Method to randomly change the centre-parameter.

    """
    
    def __init__(self, centre, amplitudes, bounds, just_started=True, base=None, fname=''):
        from astropy.io.fits import getheader

        header0 = getheader(fname, ext = 0)
        header1 = getheader(fname, ext = 1)

        self.N1 = header1['NAXIS1']
        self.N2 = header1['NAXIS2']

        self.centre = centre
        self.i0 = self.centre[0]
        self.j0 = self.centre[1]

        self.amplitudes = amplitudes
        self.bounds = bounds

        self.change_weight = 2

        if header0['HIERARCH ESO INS AO FOCU1 CONFIG'] == 'WFM':
            self.spatial_sampling = 0.2
        else:
            self.spatial_sampling = 0.025

    @lru_cache(maxsize=2)
    def base_matrices(self):
        from numpy import arange, meshgrid

        x, z = arange(self.N1), arange(self.N2)
        xx, zz = meshgrid(x, z, indexing = 'ij')
        return xx, zz

    @lru_cache(maxsize=2)
    def matrices(self):
        xx, zz = self.base_matrices()
        i0, j0 = self.i0, self.j0
        return xx - i0, zz - j0
    
    def change(self, pCN_beta):
        from numpy.random import choice

        new = Coordinates(self.centre, self.amplitude, self.bounds, matrices=self.matrices)
        
        param_names = ['i0', 'j0']
        decision = choice(param_names)
        
        amp = self.amplitudes[decision]
        lb, ub = self.bounds[decision]
        
        if decision == 'i0':
            new_i0 = pCN_change(self.i0, amp, pCN_beta, bounds=(lb, amp))
            new_j0 = self.j0
        else:
            new_i0 = self.i0
            new_j0 = pCN_change(self.j0, amp, pCN_beta, bounds=(lb, ub))
        
        new.centre = (new_i0, new_j0)
            
        return new
    
#############################################Dimensions#############################################

class Dimensions:
    """
    Object containing the shapes of three ellipsoids, the ellipsoidal voids and the torus.

    - everything():
        Returns a list of every parameter in the Dimensions object.

    - produce_overlaps():
        Calculates the overlaps given the parameters, and the coordinate matrices. 
        Results are caches to improve efficiency. 

    - change():
        Method to randomly pick and change a parameter. 

    """
    
    def __init__(self, offset, radius, a_shells, a_void, a_torus, b_shells, b_void, b_torus, pitch_yaw, amplitudes, bounds):
        self.offset = offset
        self.radius = radius
        self.a_shells = a_shells
        self.a_void = a_void
        self.a_torus = a_torus
        self.b_shells = b_shells
        self.b_void = b_void
        self.b_torus = b_torus
        self.pitch_yaw = pitch_yaw
        
        self.amplitudes = amplitudes
        self.bounds = bounds
        
        self.object_type = 'dimensions'
        
        self.change_weight = int(2*len(a_shells) + 6)
        
    def everything(self):
        return [self.offset, self.radius] + self.a_shells + [self.a_void, self.a_torus] + self.b_shells + [self.b_void, self.b_torus] + self.pitch_yaw
        
    @lru_cache(maxsize=2)
    def produce_overlaps(self, coords):
        from polynomials import ellipsoid_intercepts, void_intercepts, torus_intercepts, produce_overlaps

        alpha, beta = self.pitch_yaw

        a1, a2, a3 = self.a_shells
        b1, b2, b3 = self.b_shells

        ellipsoid1 = ellipsoid_intercepts(coords, a1, b1, alpha=alpha, beta=beta)
        ellipsoid2 = ellipsoid_intercepts(coords, a2, b2, alpha=alpha, beta=beta)
        ellipsoid3 = ellipsoid_intercepts(coords, a3, b3, alpha=alpha, beta=beta)

        voids = void_intercepts(coords, self.a_void, self.b_void, self.offset, alpha=alpha, beta=beta)

        torus = torus_intercepts(coords, self.a_torus, self.b_torus, self.radius, alpha=alpha, beta=beta)

        result = produce_overlaps(ellipsoid1, ellipsoid2, ellipsoid3, voids, torus)

        return [result[0], result[1], result[2]], result[3], result[4]
    
    def change(self, pCN_beta):
        from numpy.random import choice
        from numpy import array, sum, copy
        
        new = Dimensions(self.offset, self.radius, list(copy(self.a_shells)), self.a_void, self.a_torus, list(copy(self.b_shells)), self.b_void, self.b_torus, self.pitch_yaw, self.amplitudes, self.bounds)
        
        param_names = ['offset', 'radius', 'a_shells', 'a_void', 'a_torus', 'b_shells', 'b_void', 'b_torus', 'pitch_yaw']
        p = array([1, 1, len(self.a_shells), 1, 1, len(self.b_shells), 1, 1, len(self.pitch_yaw)])
        p = p / sum(p)
        decision = choice(param_names, p = p)
        
        amp = self.amplitudes[decision]
        lb, ub = self.bounds[decision]
        
        if decision == 'offset':
            a_max = new.a_shells[-1]
            bounds = (lb, a_max - self.a_void)
            new.offset = pCN_change(self.offset, amp, pCN_beta, bounds=bounds)
        elif decision == 'radius':
            b_max = new.b_shells[-1]
            bounds = (self.b_torus, b_max - self.b_torus)
            new.radius = pCN_change(self.radius, amp, pCN_beta, bounds=bounds)
        elif decision == 'a_void':
            a_max = new.a_shells[-1]
            bounds = (max([lb, self.b_void]), a_max - self.offset)
            new.a_void = pCN_change(self.a_void, amp, pCN_beta, bounds=bounds)
        elif decision == 'a_torus':
            new.a_torus = pCN_change(self.a_torus, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'b_void':
            b_max = self.b_shells[-1]
            bounds = (lb, min([b_max, self.a_void]))
            new.b_void = pCN_change(self.b_void, amp, pCN_beta, bounds=bounds)
        elif decision == 'b_torus':
            b_max = new.b_shells[-1]
            bounds = (lb, min([ub, b_max - self.radius]))
            new.b_torus = pCN_change(self.b_torus, amp, pCN_beta, bounds=bounds)
        elif decision == 'pitch_yaw':
            index = choice(len(new.pitch_yaw))
            new_angle = new.pitch_yaw
            new_angle[index] = pCN_change(new_angle[index], amp, pCN_beta) # No bounds !!!
            new.pitch_yaw = new_angle
            
        elif decision == 'a_shells':
            index = choice(len(new.a_shells))
            
            if index == 0:
                bounds = (max([lb, self.b_shells[index]]), self.a_shells[index+1])
            elif index == int(len(new.a_shells)-1):
                bounds = (max([self.b_shells[index], self.a_shells[index-1], self.offset+self.a_void]), ub)
            else:
                bounds = (max([self.a_shells[index-1], self.b_shells[index]]), self.a_shells[index+1])
                
            new.a_shells[index] = pCN_change(self.a_shells[index], amp, pCN_beta, bounds=bounds)
                
        elif decision == 'b_shells':
            index = choice(len(new.b_shells))
        
            if index == 0:
                bounds = (lb, min([self.b_shells[index+1], self.a_shells[index]]))
            elif index == int(len(new.b_shells)-1):
                bounds = (max([self.b_shells[index-1], self.radius+self.b_torus]), self.a_shells[index])
            else:
                bounds = (self.b_shells[index-1], min([self.b_shells[index+1], self.a_shells[index]]))
                
            new.b_shells[index] = pCN_change(self.b_shells[index], amp, pCN_beta, bounds=bounds)
            
        return new
    
###########################################Ion Densities############################################

class IonDensities:
    """
    Object containing information (density in each substructure) a particular elemental species (ionisation). 

    - all_rho():
        Returns all ion densities.

    - everything():
        Returns a list of all ion densities. 
    
    - change():
        Returns IonDensities object with one slightly changed parameter. 
        The change amplitude and bounds are governed by the bounds dictionary.
    """
    
    def __init__(self, rho_shells, rho_void, rho_torus, amplitudes, bounds, element = '', ionisation = ''):
        self.rho_shells = rho_shells
        self.rho_void = rho_void
        self.rho_torus = rho_torus
        
        self.amplitudes = amplitudes
        self.bounds = bounds
        
        self.element = element
        self.ionisation = ionisation
        self.name = element + ionisation
        
        self.object_type = 'ion_densities'
        
        self.change_weight = int(len(rho_shells) + 2)
        
    def all_rho(self):
        return self.rho_shells, self.rho_void, self.rho_torus
        
    def everything(self):
        return self.rho_shells + [self.rho_void, self.rho_torus]
        
    def change(self, pCN_beta):
        from numpy.random import choice
        from numpy import array, sum, copy
        
        new = IonDensities(list(copy(self.rho_shells)), self.rho_void, self.rho_torus, self.amplitudes, self.bounds, element = self.element, ionisation = self.ionisation)
        
        param_names = ['rho_shells', 'rho_void', 'rho_torus']
        p = array([len(self.rho_shells), 1, 1])
        p = p / sum(p)
        decision = choice(param_names, p = p)
        
        amp = self.amplitudes[decision]
        lb, ub = self.bounds[decision]
        
        if decision == 'rho_void':
            new.rho_void = pCN_change(self.rho_void, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'rho_torus':
            new.rho_torus = pCN_change(self.rho_torus, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'rho_shells':
            index = choice(len(new.rho_shells))
            new.rho_shells[index] = pCN_change(self.rho_shells[index], amp, pCN_beta, bounds=(lb, ub))
            
        return new, 'ion_change'
    
#########################################Electron Densities#########################################

class ElectronParams:
    """
    Object containing general properties of the object, e.g. electron density and temperature, and filling values.

    - all_rho():
        Returns all electron densities.

    - all_T():
        Returns all electron temperatures. 

    - all_filling():
        Returns all filling values. 
    
    - change():
        Randomly picks and changes a parameter. 
    """
    def __init__(self, rho_shells, rho_void, rho_torus, T_shells, T_void, T_torus, filling_shells, filling_void, filling_torus, amplitudes, bounds, change_filling = False):
        self.rho_shells = rho_shells
        self.rho_void = rho_void
        self.rho_torus = rho_torus
        
        self.T_shells = T_shells
        self.T_void = T_void
        self.T_torus = T_torus
        
        self.filling_shells = filling_shells
        self.filling_void = filling_void
        self.filling_torus = filling_torus
        
        self.amplitudes = amplitudes
        self.bounds = bounds
        self.change_filling = change_filling
        
        self.object_type = 'electron_params'
        
        self.change_weight = int(3*len(rho_shells) + 6)
        
    def all_rho(self):
        return self.rho_shells, self.rho_void, self.rho_torus
    
    def all_T(self):
        return self.T_shells, self.T_void, self.T_torus
    
    def all_filling(self):
        return self.filling_shells, self.filling_void, self.filling_torus
        
    def everything(self):
        return self.rho_shells + [self.rho_void, self.rho_torus] + self.T_shells + [self.T_void, self.T_torus] + self.filling_shells + [self.filling_void, self.filling_torus]
        
    def change(self, pCN_beta):
        from numpy.random import choice
        from numpy import array, sum, copy
        
        new = ElectronParams(list(copy(self.rho_shells)), self.rho_void, self.rho_torus, list(copy(self.T_shells)), self.T_void, self.T_torus, list(copy(self.filling_shells)), self.filling_void, self.filling_torus, self.amplitudes, self.bounds, change_filling = self.change_filling)
        
        if self.change_filling:
            param_names = ['rho_shells', 'rho_void', 'rho_torus', 'T_shells', 'T_void', 'T_torus', 'filling_shells', 'filling_void', 'filling_torus']
            p = array([len(self.rho_shells), 1, 1, len(self.T_shells), 1, 1, len(self.filling_shells), 1, 1])
            p = p / sum(p)
            decision = choice(param_names, p = p)
            
        else:
            param_names = ['rho_shells', 'rho_void', 'rho_torus', 'T_shells', 'T_void', 'T_torus']
            p = array([len(self.rho_shells), 1, 1, len(self.T_shells), 1, 1])
            p = p / sum(p)
            decision = choice(param_names, p = p)
            
        amp = self.amplitudes[decision]
        lb, ub = self.bounds[decision]
            
        if decision == 'rho_void':
            new.rho_void = pCN_change(self.rho_void, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'rho_torus':
            new.rho_torus = pCN_change(self.rho_torus, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'T_void':
            new.T_void = pCN_change(self.T_void, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'T_torus':
            new.T_torus = pCN_change(self.T_torus, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'filling_void':
            new.filling_void = pCN_change(self.filling_void, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'filling_torus':
            new.filling_torus = pCN_change(self.filling_torus, amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'rho_shells':
            index = choice(len(self.rho_shells))
            new.rho_shells[index] = pCN_change(self.rho_shells[index], amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'T_shells':
            index = choice(len(self.T_shells))
            new.T_shells[index] = pCN_change(self.T_shells[index], amp, pCN_beta, bounds=(lb, ub))
        elif decision == 'filling_shells':
            index = choice(len(self.filling_shells))
            new.filling_shells[index] = pCN_change(self.filling_shells, amp, pCN_beta, bounds=(lb, ub))
            
        return new, 'electron_change'