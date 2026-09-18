from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import warnings


@dataclass
class MaterialProperties:
    """ Class representing material properties of a frame element. 
    
        Attributes
        ----------
        E : float
            Young modulus
        G : float
            Shear modulus
        tau : float
            Shear strength
        f_m : float
            Masonry compressive strength
        gamma : float
            Density [N/m3]
        nu : float, optional
            Poisson ratio
        mu : float, optional
            Friction coefficients
        f_u : float, optional
            Tensile strength """
    
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
    """ Class representing geometrical properties of a frame element.

        Attributes
        ----------
        L : float
            Length
        w : float
            Thickness
        h : float
            Height
        Cx : float
            Centroid x
        Cy : float
            Centroid y
        alpha : float
            Orientation angle
        points : np.ndarray, optional
            Expected shape (8,3) - Vertex point coordinates """
    
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
    """ Class representing the stress state of a frame element.

        Attributes
        ----------
        top : dict
            Dictionary containing stress values at the top of the element. Expected keys are "ULS" or "SLS".
        midheight : dict
            Dictionary containing stress values at the midheight of the element. Expected keys are "ULS" or "SLS". """
        
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
    """ Class representing a linkage between frame elements.

        Attributes
        ----------
        element : FrameElement
            The linked frame element.
        contact_length : float
            The length of the contact between the elements. """
    
    element: "FrameElement"   # linked element
    contact_length: float     # length of contact

###############################################################################
###############################################################################
# Linkages
###############################################################################
###############################################################################
@dataclass
class Linkages:
    """ Class representing a collection of linkages between frame elements.

        Attributes
        ----------
        connections : list of Linkage
            List containing all the linkages. """
    
    connections: List[Linkage] = field(default_factory=list)
#------------------------------------------------------------------------------
    def add(self, link):
        """ Add a new linkage to the collection.

            Parameters
            ----------
            link : Linkage
                The linkage to be added. """
        
        self.connections.append(link) 

    def __add__(self, other):
        """ Add two Linkages collections together.

            Parameters
            ----------
            other : Linkages
                The other Linkages collection to be added.

            Returns
            -------
            Linkages
                A new Linkages collection containing all linkages from both collections. """
        
        if not isinstance(other, Linkages):
            return NotImplemented
        return Linkages(self.union(other))

    def __getitem__(self, index):
        """ Retrieve a linkage by its index.

            Parameters
            ----------
            index : int
                The index of the linkage to retrieve.

            Returns
            -------
            Linkage
                The linkage at the specified index. """
        
        return self.connections[index]

    def __len__(self):
        """ Return the number of linkages in the collection.

            Returns
            -------
            int
                The number of linkages. """
        
        return len(self.connections)

    def __iter__(self):
        """ Return an iterator over the linkages in the collection.

            Returns
            -------
            iterator
                An iterator over the linkages. """
        
        return iter(self.connections)


###############################################################################
###############################################################################
# Frame element (wall)
###############################################################################
###############################################################################
@dataclass
class FrameElement:
    """ Class representing a frame element (wall) in a building model.
    
        Attributes
        ----------
        id: int
            Unique identifier for the frame element.
        global_id: str
            Global identifier for the frame element.
        floor: int
            Floor number where the frame element is located.
        material: MaterialProperties
            Material properties of the frame element.
        geometry: GeometricProperties
            Geometric properties of the frame element.
        stress: StressState
            Stress state of the frame element.
        supported_by: Linkages
            Linkages representing the supports of the frame element.
        loaded_by: Linkages
            Linkages representing the loads applied to the frame element.
        delta_u_duct: np.ndarray
            Maximum displacement based on ductility (X, Y).
        delta_u_drift: np.ndarray
            Maximum displacement based on drift (X, Y).
        k: np.ndarray
            Lateral stiffness of the wall in direction x and y.
        H_Rd: np.ndarray
            Shear resistance of the wall in direction x and y. """
    
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
        """ Compute shear and bending resistances for this wall.
        
            Parameter
            ---------
            gamma_m : float
                Partial safety factor for material resistance.
                
            Returns
            -------
            V_Rd : float
                Shear resistance of the wall.
            M_Rd : float
                In-plane bending resistance of the wall.
            M_Rd_ort : float
                Out-of-plane bending resistance of the wall. """
        
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
        """ Compute safety indexes for this wall using its own properties.
        
            Parameters
            ----------
            acc_rel : float
                Relative acceleration in the in-plane direction.
            acc_rel_ort : float
                Relative acceleration in the out-of-plane direction.

            Returns
            -------
            results : dict
                Dictionary containing the safety factors for the wall in both in-plane and out-of-plane directions. """
        
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
            results = {
                "Wall_ID": self.id,        #????? which Id should we use here
                "SF_VX": V_Rd / T_Ed,
                "SF_MX": M_Rd / M_Ed,
                "SF_MOX": M_Rd_ort / M_Ed_ort
            }
        else:
            results = {
                "Wall_ID": self.id,         #????? which Id should we use here
                "SF_VY": V_Rd / T_Ed,
                "SF_MY": M_Rd / M_Ed,
                "SF_MOY": M_Rd_ort / M_Ed_ort
            }

        return results

###############################################################################
# Methods for pushover analysis
###############################################################################

#--------------------------------------------------------------------------
#  Set-get ultimate displacement of the wall
#--------------------------------------------------------------------------
    def set_ultimate_displacement_based_on_ductility_limit_alt(self, mu=1.5):
        """ Set the ultimate displacement of the wall based on the alternative ductility limit. 
        
            Parameters
            ----------
            mu : float, optional
                Ductility factor (default is 1.5).

            Returns
            -------
            ult_displacement : float
                Ultimate displacement of the wall based on the alternative ductility limit. """
        
        k, H_Rd = self.get_stiffness_and_resistance()
        d_e = H_Rd/k
        ult_displacement = mu*d_e
        return ult_displacement
       


    def set_ultimate_displacement_based_on_ductility_limit(self, mu=1.5):          # THIS could be directly computed from kx/H_rdx and k_y/H_rdy, isn't it?
        """ Set hoirzontal displacement limit of the wall based on ductility (limiting delta_ult/delta_elastic) limit.
        
            Parameters
            ----------
            mu : float, optional
                Ductility factor (default is 1.5).

            Returns
            -------
            d_u : ndarray
                Ultimate displacement of the wall in x and y directions based on the ductility limit. """

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
        """ Get ultimate displacement of the wall in x and y directions based on the ductility limit.
        
            Returns
            -------
            d_u : ndarray
                Ultimate displacement of the wall in x and y directions based on the ductility limit. """
        
        d_u = self.delta_u_duct
        if d_u is None:
          d_u = self.set_ultimate_displacement_based_on_ductility_limit()
        return d_u

    def get_ultimate_displacement(self, method="ductility"):
        """ Get ultimate displacement of the wall in x and y directions based on the specified method.

            Parameters
            ----------
            method : str, optional
                Method to compute ultimate displacement. Options are 'ductility' or 'drift' (default is 'ductility').

            Returns
            -------
            d_u : ndarray
                Ultimate displacement of the wall in x and y directions based on the specified method. """
        
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
        """ Compute hoirzontal displacement limit of the wall based on drift (limiting delta_ult/height) limits.
        
            Parameters
            ----------
            drift_lim : float, optional
                Drift limit for the wall (default is 0.004).

            Returns
            -------
            d_u : ndarray
                Ultimate displacement of the wall in x and y directions based on the drift limit. """
        
        h = self.geometry.h
        d_u = drift_lim * np.ones(2)
        self.delta_u_drift = d_u
        return d_u

    def get_ultimate_diplacement_based_on_drift_limit(self):
        """ Get ultimate displacement of the wall in x and y directions based on the drift limit.

            Returns
            -------
            d_u : ndarray
                Ultimate displacement of the wall in x and y directions based on the drift limit. """
        
        d_u = self.delta_u_ddrift
        if d_u is None:
          d_u = self.set_ultimate_displacement_based_on_drift_limit()
        return d_u

#--------------------------------------------------------------------------
#  Set-get stiffness values
#--------------------------------------------------------------------------

    def set_stiffness_and_resistance(self):
        """ Compute wall stiffness in 'X' or 'Y' direction for pushover analysis.
            
            Returns
            -------
            k : ndarray
                Stiffness of the wall in x and y directions.
            H_Rd : ndarray
                Ultimate horizontal resistance of the wall in x and y directions. """
        
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
        """ Get wall stiffness and ultimate horizontal resistance in x and y directions.

            Returns
            -------
            k : ndarray
                Stiffness of the wall in x and y directions.
            H_Rd : ndarray
                Ultimate horizontal resistance of the wall in x and y directions. """
        
        k = self.k
        H_Rd = self.H_Rd
        if k is None or H_Rd is None:
          k, H_Rd = self.set_stiffness_and_resistance()
        return k, H_Rd
