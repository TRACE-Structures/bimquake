from .floor import Floor
from .frame_element import FrameElement, MaterialProperties, GeometricProperties, StressState
from typing import List
import numpy as np
import pandas as pd
import re
from datetime import datetime, timezone
from os import error
from .seismic_loading import  get_seismic_loading
from .location_to_seismic_loading import get_Parameters
import scipy.linalg as la


from .pushover_utils import (
    get_force_displacement_diagram,
    get_global_vulnerability_metrics,
    compute_seismic_performance_assesment,
    get_bilinear_points_coord,
    get_current_data
)

from .plot_utils import(
    plot_capacity_curve,
    plot_ADRS_demand_and_capacity
)


def create_equivalent_frame_model_from_excel(sheets):
    """ Create an EquivalentFrameModel instance from an Excel file represented as a dictionary of sheets. 
    
        Parameters
        ----------
        sheets : dict
            Dictionary of Excel sheets, where keys are sheet names and values are pandas DataFrames.

        Returns
        -------
        EFM : EquivalentFrameModel
            The created equivalent frame model. """
    
    # sheets = pd.read_excel(file_name, sheet_name=None)
    # Read floor properties sheet
    floor_props = sheets.parse("data")
    wall_props = {}
    #Floor heights
    h = floor_props["H [m]"]
    #Floor weights
    w = floor_props["W [kN]"]

    for i in range(len(sheets.sheet_names[1:])):
        df = sheets.parse(sheets.sheet_names[i+1])
        wall_props[i] = df
    
    # for name, df in sheets.items():
    #     match = re.match(r"Floor(\d+)", name)
    #     if match:
    #         idx = int(match.group(1)) - 1
    #         wall_props[idx] = df
    
    EFM = create_equivalent_frame_model_from_dataframes(wall_props, h, w)
    return EFM

def create_equivalent_frame_model_from_dataframes(
  props_dict,  # Dictionary with keys 1,2,3... (floor numbers) and values of dataframes
  floor_heights,  #numpy or list with heights in m
  floor_weights   #numpy or list with weights in kN
  ) -> 'EquivalentFrameModel':
    """ Create an EquivalentFrame instance from a list of pandas DataFrames (one per floor).
        Assumes the following columns in each DataFrame:
            ["Wall", "L [m]", "w [m]", "H [m]", "Cx [m]", "Cy [m]", "α",
            "σ [N/mm²]", "τ [N/mm²]", "fₘ [N/mm²]", "γ [kN/m³]", "E [N/mm²]", "G [N/mm²]"]
        Floor heights are given in [m], weights in [kN]
        All stresses will be converted to N/m² (Pa), densities to N/m³.
        
        Parameters
        ----------
        props_dict : dict
            Dictionary with keys 1,2,3... (floor numbers) and values of dataframes
        floor_heights : list or numpy array
            List or array with heights in m
        floor_weights : list or numpy array
            List or array with weights in kN

        Returns
        -------
        eq_frame : EquivalentFrameModel
            The created equivalent frame model. """
    
    eq_frame = EquivalentFrameModel()

    for i, df in props_dict.items():   #loop over floor
        floor = Floor(id=i, height=floor_heights[i], weight=floor_weights[i]*1000)

        for _, row in df.iterrows():
            # Convert units to SI
            sigma = row["σ [N/mm²]"] * 1e6  # N/mm² → N/m²
            tau = row["τ [N/mm²]"] * 1e6    # N/mm² → N/m²
            f_m = row["fₘ [N/mm²]"] * 1e6   # N/mm² → N/m²
            gamma = row["γ [kN/m³]"] * 1000 # kN/m³ → N/m³
            E = row["E [N/mm²]"] * 1e6      # N/mm² → N/m²
            G = row["G [N/mm²]"] * 1e6      # N/mm² → N/m²

            # Create material properties
            material = MaterialProperties(tau=tau, f_m=f_m, gamma=gamma, E=E, G=G)

            # Geometry
            geometry = GeometricProperties(
              L= row["L [m]"], 
              w = row["w [m]"], 
              h = row["H [m]"], 
              Cx = row["Cx [m]"], 
              Cy = row["Cy [m]"], 
              alpha = row["α"]
              )

            # Stress state: 
            stress = StressState(
                top={"ULS": None, "SLS": None},
                midheight={"ULS": None, "SLS": sigma}
            )

            # Create the frame element
            frame_elem = FrameElement(
                id=int(row["Wall"]),
                global_id=f"F{i}_W{int(row['Wall'])}",
                floor=i,
                material=material,
                geometry=geometry,
                stress=stress,
                k=None,     # To be computed later
                H_Rd=None   # To be computed later
            )

            floor.add_wall(frame_elem)

        eq_frame.floors.append(floor)

    return eq_frame




def create_jsonld_from_equivalent_frame_model(efm: 'EquivalentFrameModel', title: str = "EFM model") -> dict:
    """ Serialize an EquivalentFrameModel into a JSON-LD document, capturing the same information as create_equivalent_frame_model_from_dataframes (floors,
        walls, material/geometry/stress properties) plus the model's location. 
        
        Parameters
        ----------
        efm : EquivalentFrameModel
            The equivalent frame model to serialize.
        title : str, optional
            The title of the JSON-LD document (default is "EFM model").

        Returns
        -------
        jsonld : dict
            The JSON-LD representation of the equivalent frame model. """
    
    floors_jsonld = []
    for floor in efm.floors:
        walls_jsonld = []
        for wall in floor.walls:
            walls_jsonld.append({
                "@type": "dkg:FrameElement",
                "@id": wall.global_id,
                "dkg:wallId": wall.id,
                "dkg:floor": wall.floor,
                "dkg:material": {
                    "dkg:E": wall.material.E,
                    "dkg:G": wall.material.G,
                    "dkg:tau": wall.material.tau,
                    "dkg:f_m": wall.material.f_m,
                    "dkg:gamma": wall.material.gamma,
                    "dkg:nu": wall.material.nu,
                    "dkg:mu": wall.material.mu,
                    "dkg:f_u": wall.material.f_u,
                },
                "dkg:geometry": {
                    "dkg:L": wall.geometry.L,
                    "dkg:w": wall.geometry.w,
                    "dkg:h": wall.geometry.h,
                    "dkg:Cx": wall.geometry.Cx,
                    "dkg:Cy": wall.geometry.Cy,
                    "dkg:alpha": wall.geometry.alpha,
                },
                "dkg:stress": {
                    "dkg:top": wall.stress.top,
                    "dkg:midheight": wall.stress.midheight,
                },
            })

        floors_jsonld.append({
            "@type": "dkg:Floor",
            "@id": f"floor:{floor.id}",
            "dkg:floorId": floor.id,
            "dkg:height": floor.height,
            "dkg:weight": floor.weight,
            "dkg:hasWall": walls_jsonld,
        })

    jsonld = {
        "@context": {
            "schema": "https://schema.org/",
            "dcterms": "http://purl.org/dc/terms/",
            "dkg": "https://ontology.origintrail.io/dkg/1.0#",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "schema:Dataset",
        "schema:name": "EFM model",
        "dcterms:title": title,
        "dcterms:created": datetime.now(timezone.utc).isoformat(),
        "dkg:latitude": efm.latitude,
        "dkg:longitude": efm.longitude,
        "dkg:hasFloor": floors_jsonld,
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "schema:name": "Trace Structures API",
            "schema:url": "https://buildchain.ilab.sztaki.hu/",
        },
    }
    return jsonld


def create_equivalent_frame_model_from_jsonld(jsonld: dict) -> 'EquivalentFrameModel':
    """ Reconstruct an EquivalentFrameModel from a JSON-LD document produced by create_jsonld_from_equivalent_frame_model.

        Parameters
        ----------
        jsonld : dict
            The JSON-LD document representing the equivalent frame model.

        Returns
        -------
        efm : EquivalentFrameModel
            The reconstructed equivalent frame model. """
    
    efm = EquivalentFrameModel()
    efm.latitude = jsonld.get("dkg:latitude")
    efm.longitude = jsonld.get("dkg:longitude")

    for floor_data in jsonld.get("dkg:hasFloor", []):
        floor = Floor(
            id=floor_data["dkg:floorId"],
            height=floor_data["dkg:height"],
            weight=floor_data["dkg:weight"],
        )

        for wall_data in floor_data.get("dkg:hasWall", []):
            mat = wall_data["dkg:material"]
            geo = wall_data["dkg:geometry"]
            stress_data = wall_data["dkg:stress"]

            material = MaterialProperties(
                E=mat["dkg:E"],
                G=mat["dkg:G"],
                tau=mat["dkg:tau"],
                f_m=mat["dkg:f_m"],
                gamma=mat["dkg:gamma"],
                nu=mat.get("dkg:nu"),
                mu=mat.get("dkg:mu"),
                f_u=mat.get("dkg:f_u"),
            )
            geometry = GeometricProperties(
                L=geo["dkg:L"],
                w=geo["dkg:w"],
                h=geo["dkg:h"],
                Cx=geo["dkg:Cx"],
                Cy=geo["dkg:Cy"],
                alpha=geo["dkg:alpha"],
            )
            stress = StressState(
                top=stress_data.get("dkg:top", {}),
                midheight=stress_data.get("dkg:midheight", {}),
            )

            wall = FrameElement(
                id=wall_data["dkg:wallId"],
                global_id=wall_data["@id"],
                floor=wall_data["dkg:floor"],
                material=material,
                geometry=geometry,
                stress=stress,
            )
            floor.add_wall(wall)

        efm.floors.append(floor)

    return efm


class EquivalentFrameModel:
      """ Equivalent frame model representing a building with multiple floors and their properties.
      
        Attributes
        ----------
        floors : List[Floor]
            The list of floors in the building.
        latitude : float
            The latitude of the building's location.
        longitude : float
            The longitude of the building's location.
        soil_category : str
            The soil category for seismic design.
        service_life : float
            The service life of the building.
        topographic_category : str
            The topographic category for seismic design.
        importance_class : str
            The importance class of the building.
        behaviour_factor : float
            The behaviour factor for seismic design.
        paraTR : float
            Seismic parameter paraTR.
        gamma_m : float
            Seismic parameter gamma_m.
        derived_design_params : dict
            Derived design parameters. """
    
      def __init__(self):
        """ Initialize an empty equivalent frame model."""

        self.floors: List[Floor] = []
         # --- location ---
        self.latitude = None
        self.longitude = None

        # --- seismic / design parameters ---
        self.soil_category = None
        self.service_life = None
        self.topographic_category = None
        self.importance_class = None
        self.behaviour_factor = None

        # seismic loading
        self.paraTR = None
        self.gamma_m = None
        self.derived_design_params = None
        # self.q_ort = None

#------------------------------------------------------------------------------        
      @property
      def num_floors(self) -> float:
          """ Get the number of floors in the building. """
          return len(self.floors)

      @property
      def total_height(self)-> float:
          """ Get the total height of the building. """
          return sum(f_i.height for f_i in self.floors)

      @property
      def floor_heights(self)-> List:
          """ Get the heights of all floors in the building. """
          return [f_i.height for f_i in self.floors]

      @property 
      def floor_heights_from_bottom(self) -> np.ndarray:   # alt_s
       """ Get the cumulative heights of all floors from the bottom. """
       hs = self.floor_heights
       return np.cumsum(hs)       

      @property
      def masses(self)-> np.ndarray:
        """ Get the masses of all floors in the building. """
        weights_r = [f_i.weight for f_i in self.floors]
        masses = np.array(weights_r)/9.81
        return masses

      @property
      def red_F(self) -> np.ndarray:    #distribution factor from top to bottom to distribute base shear force, sums to 1 
        """ Get the distribution factor for each floor from top to bottom, used to distribute the base shear force. """
        alt_s = self.floor_heights_from_bottom
        Ms = self.masses
        red_F = Ms*alt_s/np.dot(alt_s, Ms)  #Ms*alt /\sum (Ms*alt)  #is this reversed?????
        return red_F

      @property
      def _plot_bounds(self):
        """ Calculate the global boundaries (x_min, x_max, y_min, y_max) for the entire building."""

        all_centers = []
        all_sizes = []
        for floor in self.floors:
            wall_props = floor.walls_geometric_properties
            all_centers.append(wall_props[["Cx", "Cy"]].values)
            all_sizes.append(wall_props[["L", "w"]].values)
        
        all_centers = np.vstack(all_centers)
        all_sizes = np.vstack(all_sizes)

        x_min = np.min(all_centers[:, 0] - all_sizes[:, 0] / 2)
        x_max = np.max(all_centers[:, 0] + all_sizes[:, 0] / 2)
        y_min = np.min(all_centers[:, 1] - all_sizes[:, 1] / 2)
        y_max = np.max(all_centers[:, 1] + all_sizes[:, 1] / 2)
      
        return (x_min, x_max, y_min, y_max)

#--------------------------------------------------------------------------
#  Set properties
#--------------------------------------------------------------------------

      def set_location(self, latitude: float, longitude: float):
          """ Set the geographical location of the building. 
          
            Parameters
            ----------
            latitude : float
                The latitude of the building's location.
            longitude : float
                The longitude of the building's location. """
          
          self.latitude = latitude
          self.longitude = longitude

      def set_seismic_design_parameters(
          self,
          soil_category: str,
          nominal_life: float,
          topographic_category: str,
          importance_class: str):
          """Set seismic design parameters.
          
            Parameters
            ----------
            soil_category : str
                The soil category of the building's location.
            nominal_life : float
                The nominal life of the building.
            topographic_category : str
                The topographic category of the building's location.
            importance_class : str
                The importance class of the building. """
          
          self.soil_category = soil_category
          self.service_life = nominal_life                      #???????
          self.topographic_category = topographic_category
          self.importance_class = importance_class
        #   self.behaviour_factor = behaviour_factor

          self.derive_and_set_seismic_loading_params()

          # Set altitude of floors for computation of relative accelerations
          alt_s = self.floor_heights_from_bottom
          seismic_params = self.get_derived_seismic_design_params()
          H = self.total_height

          for i, F_i in enumerate(self.floors):
            F_i.set_altitude(alt_s[i])
            F_i.set_relative_accelerations(seismic_params, H)

#--------------------------------------------------------------------------
#  Get-Set Seismic Loading (dependent on given sismic design params)
#--------------------------------------------------------------------------
      def is_seismic_loading_input_valid(self, ParaTR):
          """ Check if the seismic loading input is valid. 
          
            Parameters
            ----------
            ParaTR : dict
                The seismic loading parameters table.

            Returns
            -------
            bool
                True if the seismic loading input is valid, False otherwise. """
          
        #TODO check whether necessary data is available
          return True

      
      def set_seismic_loading(self, ParaTR=None, q_ort=3, gamma_m=2):
          """ Set the seismic loading parameters.
          
            Parameters
            ----------
            ParaTR : dict, optional
                The seismic loading parameters table. If None, it will be obtained based on the building's latitude and longitude.
            q_ort : float, optional
                Factor for out-of-plane seismic loading. Default is 3.
            gamma_m : float, optional
                Partial safety factor for material strength. Default is 2. """
          
          if not None in (self.latitude, self.longitude) and ParaTR is None:
            ParaTR, _, _ = get_Parameters(self.latitude, self.longitude)
          else:
            if ParaTR is None:
              error("Either first set latitude and longitude, or give valid paraTR table")
            elif not self.is_seismic_loading_input_valid(ParaTR):
              error("ParaTR seismic loading table in is not valid")
          self.ParaTR = ParaTR
          self.gamma_m =gamma_m
          

      def get_derived_seismic_design_params(self):
          """ Get the derived seismic design parameters.
          
            Returns
            -------
            derived_design_params : dict
                The derived seismic design parameters.
          """
          if self.derived_design_params is None:
            self.derive_and_set_seismic_loading_params()
          derived_design_params = self.derived_design_params
          return derived_design_params

      
      def derive_and_set_seismic_loading_params(self):
          """ Derive and set the seismic loading parameters.
          
            This method calculates the derived seismic design parameters based on the seismic loading table (ParaTR)
            and the building's characteristics, and stores them in the instance variable `self.derived_design_params`. """
          
          ParaTR = self.ParaTR
          # print(ParaTR)
          Se_SLV_T, ag_SLV, S_geo, ADRS, tstep, func, TC = get_seismic_loading(
                      ParaTR,
                      self.service_life,
                      self.importance_class,
                      self.soil_category,
                      self.topographic_category,
                      self.total_height)
          self.derived_design_params = {
            "Se_SLV_T": Se_SLV_T,  #relative acceleration
            # "q_ort": q_ort,    # factor for out of plane ...
            # "q": q,  #behavior factor
            "ag_SLV": ag_SLV,
            "S_geo": S_geo,
            "ADRS": ADRS,   # for pushover analysis
            "tstep": tstep,  # for pushover analyiss
            "spectral_func": func,
            "TC": TC,
            "ParaTR": ParaTR
          }


#--------------------------------------------------------------------------
#  Add floor, floor walls
#--------------------------------------------------------------------------

      def add_floor(self, floor: Floor):
          """ Add a floor to the building.

            Parameters
            ----------
            floor : Floor
                The floor object to be added. """
          
          self.floors.append(floor)

      def add_wall_to_floor(self, floor_index: int, wall: 'FrameElement'):
          """ Add a wall to a specific floor.

            Parameters
            ----------
            floor_index : int
                The index of the floor to which the wall will be added.
            wall : FrameElement
                The wall object to be added. """
          
          if floor_index < 0 or floor_index >= len(self.floors):
              raise IndexError("Floor index out of range")
          self.floors[floor_index].add_wall(wall)

#--------------------------------------------------------------------------
#  Run Linear Static Analysis
#--------------------------------------------------------------------------

      def run_linear_static_analysis(self, q=1.5, q_ort=3, gamma_m=2):
          """ Run a linear static analysis for the building.

            Parameters
            ----------
            q : float, optional
                Behavior factor for in-plane walls (default is 1.5).
            q_ort : float, optional
                Behavior factor for out-of-plane walls (default is 3).
            gamma_m : float, optional
                Safety factor on the resistance side (default is 2).

            Returns
            -------
            SFs : list of DataFrame
                List of dataframes containing the safety factors for each floor. """
          
          #-------
          # safety factor on the resistance side
          # gamma_m = 2
                    
          # Initiate list of dataframes with safety factors 
          SFs = []
          for j, floor in enumerate(self.floors):
            SFs_j = floor.compute_wall_safety_factors(q, q_ort, gamma_m)
            SFs.append(SFs_j)
          return SFs


#--------------------------------------------------------------------------
#  Run Pushover Analysis
#--------------------------------------------------------------------------

      def get_force_displacement_capacity_curve(self, method="Ductility", algorithm="incr", mu=1.5):
          """ Get the force-displacement capacity curve for the building.

            Parameters
            ----------
            method : str, optional
                Method to compute the ultimate displacement ("Ductility" or "Drift", default is "Ductility").
            algorithm : str, optional
                Incremental algorithm to use ("incr" or "add", default is "incr").
            mu : float, optional
                Ductility factor (default is 1.5).

            Returns
            -------
            K : array
                Stiffness values.
            H : array
                Force values.
            v : array
                Displacement values.
            ind_state : array
                State indicators for each step. """
          
          n_floors, floor_id, _, alt_s, _, Ms, red_F = self.get_floor_data()
          D, X, S, V, alpha, G, NZ = self.get_wall_data()
          u_max = get_current_data(n_floors, D, mu, S, G, V, NZ, alpha, method)  # to be changed to my ultimate displacement function
          passo = 5
          incr = 1.001
          delta_red = 0.2
          #design_vals = self.get_seismic_design_loading
          #ADRS = design_vals["ADRS"]
          K, H, v, ind_state =get_force_displacement_diagram(
                                D, X, S, V, G,
                                u_max,
                                n_floors, NZ,
                                Ms,
                                red_F,
                                incr,
                                delta_red,
                                alpha,
                                passo,
                                algorithm)
          return K, H, v, ind_state

      def get_bilinear_capacity_curve_coords(self, v, H):
          """ Get the coordinates of the bilinear approximation of the capacity curve.

            Parameters
            ----------
            v : array
                Displacement values.
            H : array
                Force values.

            Returns
            -------
            v_bl : array
                Displacement values of the bilinear approximation.
            H_bl : array
                Force values of the bilinear approximation. """
          
          v_bl, H_bl = get_bilinear_points_coord(v, H)
          return v_bl, H_bl

      def assess_seismic_performance_get_ADRS_params(self, K, H, v, H_bl, v_bl):
          """ Assess the seismic performance and get ADRS parameters.

            Parameters
            ----------
            K : array
                Stiffness values.
            H : array
                Force values.
            v : array
                Displacement values.
            H_bl : array
                Force values of the bilinear approximation.
            v_bl : array
                Displacement values of the bilinear approximation.

            Returns
            -------
            pushover_results : dict
                Results of the pushover analysis including ADRS parameters. """
          
          Masses = np.array([floor.weight for floor in reversed(self.floors)]) / 10000
          design_params = self.get_derived_seismic_design_params()
          pushover_results = compute_seismic_performance_assesment(
                              design_params, 
                              Masses, 
                              v, H, K, 
                              v_bl,
                              H_bl,
                              self.soil_category)
          return pushover_results

      def get_global_vulnerability_metrics(self, v_bl, pushover_results):
          """ Get the global vulnerability metrics.

            Parameters
            ----------
            v_bl : array
                Displacement values of the bilinear approximation.
            pushover_results : dict
                Results of the pushover analysis including ADRS parameters.

            Returns
            -------
            df : DataFrame
                DataFrame containing the global vulnerability metrics. """
          
          # Put main metrics in dataframe
          df = get_global_vulnerability_metrics(v_bl, pushover_results)
          return df
      
      def plot_capacity_curve(self, v, H, v_bl, H_bl):
          """ Plot the capacity curve.

            Parameters
            ----------
            v : array
                Displacement values.
            H : array
                Force values.
            v_bl : array
                Displacement values of the bilinear approximation.
            H_bl : array
                Force values of the bilinear approximation.

            Returns
            -------
            fig : Figure
                Figure object of the capacity curve. """
          
          fig = plot_capacity_curve(v, H, v_bl, H_bl)
          return fig
      
      def plot_ADRS_demand_and_capacity(self, v_bl, pushover_results):
           """ Plot the ADRS demand and capacity.

            Parameters
            ----------
            v_bl : array
                Displacement values of the bilinear approximation.
            pushover_results : dict
                Results of the pushover analysis including ADRS parameters.

            Returns
            -------
            fig : Figure
                Figure object of the ADRS demand and capacity plot. """
           
           fig = plot_ADRS_demand_and_capacity(v_bl, pushover_results)
           return fig

      def get_floor_data(self):
        """ Get the floor data including heights, masses, and reduction factors.

            Returns
            -------
            N : int
                Number of floors.
            floorID : list
                List of floor identifiers.
            alt : array
                Array of floor heights.
            alt_s : array
                Array of cumulative floor heights from top to bottom.
            hTOT : float
                Total height of the building.
            Masses : array
                Array of floor masses.
            red_F : array
                Array of reduction factors for each floor. """
        
        N = len(self.floors)
        floorID = ["Floor{}".format(floor.id+1) for floor in self.floors]
        alt = np.array([floor.height for floor in reversed(self.floors)])
        alt_s = np.zeros(N)

        for i in range(N):
            alt_s[i] = np.sum(alt[i:])

        hTOT = np.sum(alt)
        Masses = np.array([floor.weight for floor in reversed(self.floors)]) / 10000
        # Masses = np.array(list(reversed(xlsx.parse(0)['W [kN]'].values))) / 10

        denominator = np.dot(alt_s, Masses)
        red_F=Masses*alt_s/denominator

        return N, floorID, alt, alt_s, hTOT, Masses, red_F
    

      def get_wall_data(self):
        """ Get the wall data including dimensions, positions, material properties, and other relevant parameters.

            Returns
            -------
            D : list
                List of arrays containing wall dimensions for each floor.
            X : list
                List of arrays containing wall positions for each floor.
            S : list
                List of wall stress values for each floor.
            V : list
                List of wall heights for each floor.
            alpha : list
                List of wall orientation angles for each floor.
            G : list
                List of arrays containing wall material properties for each floor.
            NZ : list
                List of the number of walls for each floor. """
        
        N = len(self.floors)
        D = []
        X = []
        S = []
        V = []
        alpha = []
        G = []
        NZ = []

        df = self.get_wall_initial_properties()
        for k in range(N):
            D.append(np.array([df[N-k-1]['L [m]'], df[N-k-1]['w [m]']]).T)
            X.append(np.array([df[N-k-1]['Cx [m]'], df[N-k-1]['Cy [m]']]).T)
            S.append(df[N-k-1]['σ [N/mm²]'] * 100)
            V.append(df[N-k-1]['H [m]'])
            alpha.append(df[N-k-1]['α'])
            # mud.append(df[N-k-1]['μ'])
            G.append(np.array([df[N-k-1]['G [N/mm²]'] *100, df[N-k-1]['τ [N/mm²]'] *100, df[N-k-1]['E [N/mm²]'] *100, df[N-k-1]['fₘ [N/mm²]'] *100, df[N-k-1]['γ [kN/m³]'] / 10]).T)
            NZ.append(len(D[k]))
        return D, X, S, V, alpha, G, NZ

      def get_wall_initial_properties(self):
          """ Get the initial properties of walls for each floor.

            Returns
            -------
            prop_dfs : dict
                Dictionary of DataFrames containing wall properties for each floor. """
          
          # initiate list of dataframes for the different floors
          prop_dfs = {}
          # Loop over floors from top to bottom
          for i, floor in enumerate(self.floors):
            walls = floor.walls
            records = []
            for W_j in walls:              
              record = {}
              record['Wall'] = W_j.id
              record["GlobalId"] = W_j.global_id
              record["L [m]"] = W_j.geometry.L
              record["w [m]"] = W_j.geometry.w
              record["H [m]"] = W_j.geometry.h
              record["Cx [m]"] = W_j.geometry.Cx
              record["Cy [m]"] = W_j.geometry.Cy
              record["α"] = W_j.geometry.alpha
              record["γ [kN/m³]"] = W_j.material.gamma /1e3
              sigma_SLS = W_j.stress.midheight["SLS"]
              record["σ [N/mm²]"] = sigma_SLS/1e6 if sigma_SLS is not None else None
              # Wall elastic moduli
              record["E [N/mm²]"] = W_j.material.E / 1e6 if W_j.material.E is not None else None
              # wall's ShearModulus
              record["G [N/mm²]"] = W_j.material.G / 1e6 if W_j.material.G is not None else None
              # wall's TensileStrength
              # record["f_u [N/mm²]"] = W_j.material.f_u/1e6 if W_j.material.f_u is not None else None
              # wall's ShearStrength
              record["τ [N/mm²]"] = W_j.material.tau/1e6 if W_j.material.tau is not None else None
              # wall's compressive strength
              record["fₘ [N/mm²]"] = W_j.material.f_m/1e6 if W_j.material.f_m is not None else None
              # wall's PoissonRatio
              record["nu"] = W_j.material.nu
              # Add record to list
              records.append(record)
            df = pd.DataFrame(records).sort_values("Wall")
            df = df[["Wall", "L [m]", "w [m]", "H [m]", "Cx [m]", "Cy [m]", 
              "α", "σ [N/mm²]", "τ [N/mm²]", "fₘ [N/mm²]",
              "γ [kN/m³]", "E [N/mm²]", "G [N/mm²]"]]
            prop_dfs[i] = df

          return prop_dfs

#--------------------------------------------------------------------------
#  Plotting functionalities
#--------------------------------------------------------------------------

      def plot_wall_safety_factors(self, SFs, direction, floor_num, type: str):
        """ Plot safety factors for each wall on a specific floor using Plotly.
            
            Parameters
            ----------
            SFs : dict
                Dictionary containing safety factors for each floor and wall.
            direction : str
                Direction of the safety factors ('X' or 'Y').
            floor_num : int
                Floor number for which to plot the safety factors.
            type : str
                Type of safety factor ('Shear', 'Bending in plane', 'Bending out of plane').

            Returns
            -------
            fig : Figure
                Figure object of the safety factors plot. """
        
        mapping_var = {
            'Shear': 'SF_V',
            'Bending in plane': 'SF_M',
            'Bending out of plane': 'SF_MO'
        }
        var = f"{mapping_var[type]}{direction}"
        bounds = self._plot_bounds
        # floor_num is 1-indexed, so floor_idx = floor_num - 1
        fig = self.floors[floor_num].plot_safety_factors(
            bounds, 
            SFs[floor_num][var],
            direction=direction,
            type=type
        )
        return fig
      
      def plot_failing_walls(self, failing_walls, direction, floor_num):
        """ Plot failing walls for each wall on a specific floor using Plotly.
            
            Parameters
            ----------
            failing_walls : dict
                Dictionary containing failing walls for each floor and direction.
            direction : str
                Direction of the failing walls ('X' or 'Y').
            floor_num : int
                Floor number for which to plot the failing walls.

            Returns
            -------
            fig : Figure
                Figure object of the failing walls plot. """
        
        if direction == 'X':
            failing_indices = failing_walls[0]
        else:
            failing_indices = failing_walls[1]
            
        bounds = self._plot_bounds

        idx = self.num_floors - 1 - floor_num   
        
        fig = self.floors[floor_num].plot_failing_walls(
            bounds,
            failing_indices[idx],
            direction=direction
        )
        return fig

      def plot_floor_layout(self, floor_num):
        """ Plot the basic structural wall layout for a specific floor using Plotly.

            Parameters
            ----------
            floor_num : int
                Floor number for which to plot the layout.

            Returns
            -------
            fig : Figure
                Figure object of the floor layout plot. """
        
        bounds = self._plot_bounds
        # floor_num is 1-indexed, so floor_idx = floor_num - 1
        fig = self.floors[floor_num].plot_2d_layout(
            bounds
        )
        return fig

      def plot_wall_type_layout(self, floor_num, indices):
        """ Plot the basic structural wall layout for a specific floor using Plotly.

            Parameters
            ----------
            floor_num : int
                Floor number for which to plot the layout.
            indices : list of int
                List of wall indices to highlight or plot.

            Returns
            -------
            fig : Figure
                Figure object of the wall type layout plot. """
        
        bounds = self._plot_bounds
        # floor_num is 1-indexed, so floor_idx = floor_num - 1
        fig = self.floors[floor_num].plot_wall_type_layout(
            bounds, indices
        )
        return fig
      

      def calculate_eigenfrequencies(self):
        """ Calculate eigenfrequencies and eigenvectors for the building model.
                
            Returns
            -------
            frequencies_X_damp : np.ndarray
                Eigenfrequencies for X direction with damping.
                
            frequencies_Y_damp : np.ndarray
                Eigenfrequencies for Y direction with damping. """
                    
        D, _, _, V, alpha, G, NZ = self.get_wall_data()
        N, _, _, _, _, Masses, _ = self.get_floor_data()

            
        # Initialize main floor(storey) properties (center of mass, center of rigidity) 
        KXel=np.zeros(N) #Rigidezza Totale Elastica X
        KYel=np.zeros(N) #Rigidezza Totale Elastica Y
        K = []

        for j in range(N):
            D0=D[j] #;%Dimensioni Pareti di Piano
            alpha0=alpha[j]
            Gvar1=G[j]
            V0=V[j]

            #Stifness of the walls
            K0=np.zeros((NZ[j],2))
            for i in range(NZ[j]):
                Gvar0 = Gvar1[i,2]/Gvar1[i,0]
                theta=alpha0[i]*np.pi/180
                K0[i,0]=Gvar1[i,0]*D0[i,0]*D0[i,1]/(1.2*V0[i]*(1+1/(1.2*Gvar0)*(V0[i]/D0[i,0])**2))*np.cos(theta)**2
                K0[i,1]=Gvar1[i,0]*D0[i,0]*D0[i,1]/(1.2*V0[i]*(1+1/(1.2*Gvar0)*(V0[i]/D0[i,0])**2))*np.sin(theta)**2 

            K.append(K0)
            KXel[j]=np.sum(K0[:,0])
            KYel[j]=np.sum(K0[:,1])
        
        M = 10**3*np.diag(np.flip(Masses))
        damp=0.05

        if N == 1:
            frequencies_X_damp = ((KXel * 10**4 / M)**0.5) * (1 - damp**2)**0.5 / (2 * np.pi)
            frequencies_Y_damp = ((KYel * 10**4 / M)**0.5) * (1 - damp**2)**0.5 / (2 * np.pi)

        elif N > 1:
            KX_mat = np.zeros((N, N))
            KY_mat = np.zeros((N, N))
            for i in range(N):
                if i == 0:
                    KX_mat[i, i] = KXel[i] + KXel[i+1]
                    KY_mat[i, i] = KYel[i] + KYel[i+1]
                elif i == N-1:
                    KX_mat[i, i] = KXel[i]
                    KY_mat[i, i] = KYel[i]
                    KX_mat[i, i-1] = -KXel[i]
                    KY_mat[i, i-1] = -KYel[i]
                else:
                    KX_mat[i, i] = KXel[i] + KXel[i+1]
                    KY_mat[i, i] = KYel[i] + KYel[i+1]
                    KX_mat[i, i-1] = -KXel[i]
                    KY_mat[i, i-1] = -KYel[i]
                if i < N-1:
                    KX_mat[i, i+1] = -KXel[i+1]
                    KY_mat[i, i+1] = -KYel[i+1]

            KX_mat = 10**4 * KX_mat
            KY_mat = 10**4 * KY_mat

            eigenvalues_X, _ = la.eig(KX_mat, M)
            eigenvalues_Y, _ = la.eig(KY_mat, M)

            eigenvalues_X = np.diag(eigenvalues_X.real)
            eigenvalues_Y = np.diag(eigenvalues_Y.real)

            eigenvalues_X = np.diag(eigenvalues_X)
            eigenvalues_Y = np.diag(eigenvalues_Y)

            frequencies_Y = np.sqrt(eigenvalues_Y)/(2*np.pi)
            frequencies_X = np.sqrt(eigenvalues_X)/(2*np.pi)

            frequencies_Y_damp = frequencies_Y*(1-damp**2)**0.5
            frequencies_X_damp = frequencies_X*(1-damp**2)**0.5

        return [frequencies_X_damp, frequencies_Y_damp]

 
#--------------------------------------------------------------------------
#  Display
#--------------------------------------------------------------------------

      def __repr__(self):
          """ String representation of the EquivalentFrame object. """
          return f"EquivalentFrame(floors={self.num_floors})"