from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import warnings


@dataclass
class MaterialProperties:
    E: float        # Young modulus
    G: float        # Shear modulus
    tau: float      # Shear strength
    f_m: float       # Masonry compressive strength
    gamma: float   # Density [N/m3] 
    nu: Optional[float] = None  # Poisson ratio
    mu: Optional[float] = None  # Friction coefficients
    f_u: Optional[float] = None  # tensile strength
      

###############################################################################
###############################################################################
# Geometrical properties
###############################################################################
###############################################################################
@dataclass
class GeometricProperties:
    L: float                       # Length
    w: float                       # Thickness
    h: float                       # Height
    Cx: float                      # Centroid x
    Cy: float                      # Centroid y
    alpha: float                   # Orientation angle
    points: Optional[np.ndarray]=None  # expected shape (8,3) - Vertext point coordinates


###############################################################################
###############################################################################
# Stress state
###############################################################################
###############################################################################
@dataclass
class StressState:
    top: Dict[str, float] = field(default_factory=dict)   #str: expected options "ULS" or "SLS"
    midheight: Dict[str, float] = field(default_factory=dict)  #str: expected options "ULS" or "SLS"
    #bottom: float


###############################################################################
###############################################################################
# Linkage
###############################################################################
###############################################################################
@dataclass
class Linkage:
    element: "FrameElement"   # linked element
    contact_length: float     # length of contact

###############################################################################
###############################################################################
# Linkages
###############################################################################
###############################################################################
@dataclass
class Linkages:
    connections: List[Linkage] = field(default_factory=list)
#------------------------------------------------------------------------------
    def add(self, link):
        self.connections.append(link) 

    def __add__(self, other):
        if not isinstance(other, Linkages):
            return NotImplemented
        return Linkages(self.union(other))

    def __getitem__(self, index):
        return self.connections[index]

    def __len__(self):
        return len(self.connections)

    def __iter__(self):
        return iter(self.connections)


###############################################################################
###############################################################################
# Frame element (wall)
###############################################################################
###############################################################################
@dataclass
class FrameElement:
    id: int
    global_id: str
    floor: int 
    material: MaterialProperties
    geometry: GeometricProperties
    stress: StressState
    supported_by: Linkages = field(default_factory=Linkages)
    loaded_by: Linkages = field(default_factory=Linkages)
    delta_u_duct: np.ndarray = None  # max diplacement based on ductility 2 values (X,Y)
    delta_u_drift: np.ndarray = None # max diplacement based on drift 2 values (X,Y)
    k: np.ndarray = None # lateral stiffness of the wall in direction x and y 
    H_Rd: np.ndarray = None # Shear resistace of the wall in direction x and y 

###############################################################################
# Methods for linear static analysis
###############################################################################

    #--------------------------------------------------------------------------
    #  Compute resistance
    #--------------------------------------------------------------------------

    def compute_resistances(self, gamma_m: float=2):
        """Compute shear and bending resistances for this wall."""
        # gamma_m is the safety factor (resistance is computed resistance divided by gamma_m)
        # stress in the midsection of the wall
        sigma = self.stress.midheight['SLS']        #????SLE?
        f_m = self.material.f_m
        tau = self.material.tau
        L, w = self.geometry.L, self.geometry.w
        A = L * w

        # Shear resistance
        V_Rd = A * tau / gamma_m * np.sqrt(1 + sigma / (1.5 * tau / gamma_m))

        # In-plane bending
        M_Rd = (L**2 * w * sigma / 2) * (1 - sigma / (0.85 * f_m / gamma_m))
        M_Rd = max(M_Rd, 0)

        # Out-of-plane bending
        M_Rd_ort = (L * w**2 * sigma / 2) * (1 - sigma / (0.85 * f_m / gamma_m))
        M_Rd_ort = max(M_Rd_ort, 0)

        return V_Rd, M_Rd, M_Rd_ort

    #--------------------------------------------------------------------------
    #  Compute safety factors
    #--------------------------------------------------------------------------

    def compute_safety_factors(self, acc_rel: float, acc_rel_ort:float):
        """Compute safety indexes for this wall using its own properties."""
        sigma = self.stress.midheight['SLS']
        gamma = self.material.gamma

        # Retrieve gometric properties
        L, w, h = self.geometry.L, self.geometry.w, self.geometry.h          #Do I need the floor height or is it fine to use height of the wall?
        alpha = self.geometry.alpha
       

        # Resistances
        V_Rd, M_Rd, M_Rd_ort = self.compute_resistances()

        # Demands
        T_Ed = acc_rel * sigma * L * w
        M = ((sigma - gamma * h) * L * w * h / 2 + gamma * h * L * w * h / 3)
        M_Ed = acc_rel * M

        # Acceleration to be used for out of plane
        M_Ed_ort = acc_rel_ort * M

        # Assign to X or Y based on orientation
        if alpha < 45 or alpha >= 135:
            return {
                "Wall_ID": self.id,        #????? which Id should we use here
                "SF_VX": V_Rd / T_Ed,
                "SF_MX": M_Rd / M_Ed,
                "SF_MOX": M_Rd_ort / M_Ed_ort
            }
        else:
            return {
                "Wall_ID": self.id,         #????? which Id should we use here
                "SF_VY": V_Rd / T_Ed,
                "SF_MY": M_Rd / M_Ed,
                "SF_MOY": M_Rd_ort / M_Ed_ort
            }

###############################################################################
# Methods for pushover analysis
###############################################################################

#--------------------------------------------------------------------------
#  Set-get ultimate displacement of the wall
#--------------------------------------------------------------------------
    def set_ultimate_displacement_based_on_ductility_limit_alt(self, mu=1.5):
        k, H_Rd = self.get_stiffness_and_resistance()
        d_e = H_Rd/k
        return mu*d_e
       


    def set_ultimate_displacement_based_on_ductility_limit(self, mu=1.5):          # THIS could be directly computed from kx/H_rdx and k_y/H_rdy, isn't it?
        """
        Set hoirzontal displacement limit of the wall based on  
        ductility (limiting delta_ult/delta_elastic) limit
        Sets ultimate displacement arrays of size (2,) - dir x and y.
        mu is the ductility factor delta_u = mu*delta_e
        """

        L = self.geometry.L                                                    
        w = self.geometry.w
        h = self.geometry.h
        alpha = self.geometry.alpha

        sigma = self.stress.midheight["SLS"]

        tau = self.material.tau
        E = self.material.E
        G = self.material.G

        a = G / tau
        b = E / G
       
       # Initiate vector for displacement limit
        d_u = np.zeros(2)
        
        for j in range(2):         # for x an y direction
            if alpha < 45 or alpha >= 135:
                l = L
            else:
                l = w if j == 0 else l
            d_e = np.sqrt(1 + sigma / (1.5*tau) )* 1.2/a *h * (1 + 1/(1.2*b)*(h/l)**2)
            d_u[j] = mu * d_e

        self.delta_u_duct = d_u
        return d_u

    def get_ultimate_diplacement_based_on_ductility_limit(self):
        d_u = self.delta_u_duct
        if d_u is None:
          d_u = self.set_ultimate_displacement_based_on_ductility_limit()
        return d_u

    def get_ultimate_displacement(self, method="ductility"):
        if method == "ductility":
          return self.get_ultimate_diplacement_based_on_ductility_limit()
        elif method == "drift":
          return self.get_ultimate_diplacement_based_on_drift_limit()
        else:
          warnings.warn(
            f"Unknown method '{method}'. Defaulting to 'ductility'.",
            UserWarning
            )
          return self.get_ultimate_diplacement_based_on_ductility_limit()
          
    def set_ultimate_displacement_based_on_drift_limit(self, drift_lim=0.004):
        """
        Compute hoirzontal displacement limit of the wall based on  
        drift (limiting delta_ult/height) limits.
        Sets ultimate displacement arrays of size (2,) - dir x and y.
        mu is the ductility factor delta_u = mu*delta_e
        """
        h = self.geometry.h
        d_u = drift_lim * np.ones(2)
        self.delta_u_drift = d_u
        return d_u

    def get_ultimate_diplacement_based_on_drift_limit(self):
        d_u = self.delta_u_ddrift
        if d_u is None:
          d_u = self.set_ultimate_displacement_based_on_drift_limit()
        return d_u

#--------------------------------------------------------------------------
#  Set-get stiffness values
#--------------------------------------------------------------------------

    def set_stiffness_and_resistance(self):
        """
        Compute wall stiffness in 'X' or 'Y' direction for pushover analysis.
        Returns k (float) and effective horizontal force H0 (float).
        """
        # Wall's geometric properties
        L = self.geometry.L 
        w = self.geometry.w 
        h = self.geometry.h  
        theta = np.radians(self.geometry.alpha)

        # Wall's material properties
        tau = self.material.tau
        G = self.material.G 
        E = self.material.E

        sigma = self.stress.midheight["SLS"]

        K0 = G * L * w / (1.2 * h * (1 + G/(1.2*E)*(h/L)**2))

        # Stiffness and resistance in direction X
        k_x = K0 * np.cos(theta)**2 
        H_Rd_x = L * w * tau * np.sqrt(1 + sigma/(1.5*tau))
      # Stiffness of the wall
        k_y = K0 * np.sin(theta)**2
        # Resistance of the wall
        H_Rd_y = L * w * tau * np.sqrt(1 + sigma/(1.5*tau))
        self.k = np.array([k_x, k_y])
        self.H_Rd = np.array([H_Rd_x, H_Rd_y])
        return self.k, self.H_Rd

    def get_stiffness_and_resistance(self):
        k = self.k
        H_Rd = self.H_Rd
        if k is None or H_Rd is None:
          k, H_Rd = self.set_stiffness_and_resistance()
        return k, H_Rd
