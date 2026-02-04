
from .simple_objects import SolidObjectSet
from .slab import Slab
from .wall import Wall
import numpy as np
from collections import defaultdict
import pandas as pd
from .IFC_objects import IfcObject
import warnings

class IfcObjectSet(SolidObjectSet):
      """ Class to handle a set of IfcObjects (walls, slabs, etc.) 
      
          Attributes
          ----------
          objects : list
              List of IfcObject instances contained in the set.
              
          Methods
          -------
          add_object(obj, only_geometry=False, verbose=False)
              Adds a single IfcObject to the set.
              
          add_objects(obj_list, only_geometry=False, verbose=False)
              Adds multiple IfcObjects to the set.
              
          filter_objects_by_type(ifc_type, objs=None)
              Filters objects by their IFC type.
              
          filter_walls_by_floor(floor, objs=None)
              Filters wall objects by floor number.
              
          filter_objects_by_floor(floor, objs=None)
              Filters objects by floor number.
              
          find_object_by_id(id, objs=None)
              Finds an object by its IFC ID.
              
          filter_objects(floor=None, type=None, objs=None)
              Filters objects by floor and/or type.
              
          get_all_meshes_highlighting_elements_with_id(elem_id=[], objs=None)
              Generates meshes highlighting specified elements by their IDs.
              
          get_mesh_by_type(obj_type, opacity=0.8, objs=None)
              Generates meshes for objects of a specific type.
              
          get_mesh_of_wall_partitions(mode="with_loading_elements", opacity=0.8, objs=None)
              Generates meshes for wall partitions.
              
          get_mesh_by_floor(floor, color=None, opacity=0.8, objs=None)
              Generates meshes for objects on a specific floor.
              
          get_meshes(objs=None, color=None, opacity=0.8)
              Generates meshes for a list of objects.
              
          get_mesh_of_slab_partitions(color=None, opacity = 0.8, objs=None)
              Generates meshes for slab partitions."""
      
      def __init__(self):
        """ Initialize an IfcObjectSet instance."""

        super().__init__()
        self._seen_ids = set()

      def add_objects(self, obj_list,  only_geometry=False, verbose=False):
        """ Adds multiple IfcObjects to the set.

            Parameters
            ----------
            obj_list : list
                List of IfcObject instances to be added.

            only_geometry : bool, optional
                If True, only geometry is considered (default is False).

            verbose : bool, optional
                If True, prints information about added objects (default is False).

            Returns
            -------
            objects : list
                Updated list of IfcObject instances in the set. """
        
        if not isinstance(obj_list, list):
          obj_list = [obj_list]
        for obj in obj_list:
            self.add_object(obj,  only_geometry=False, verbose=False)
        objects = self.objects
        return objects

      def add_object(self, obj, only_geometry=False, verbose=False):
        """ Adds a single IfcObject to the set.

            Parameters
            ----------
            obj : IfcObject
                The IfcObject instance to be added.

            only_geometry : bool, optional
                If True, only geometry is considered (default is False).

            verbose : bool, optional
                If True, prints information about the added object (default is False). """
        
        gid = obj.GlobalId
        if gid in self._seen_ids:
            print(f"Duplicate ID: {gid}")
            return
        self._seen_ids.add(gid)
        if obj.is_a("IfcBuildingElement"):
              if obj.is_a("IfcWall"):
                self.objects.append(Wall(obj))
              elif obj.is_a("IfcSlab"):
                self.objects.append(Slab(obj, create_partitions= not only_geometry))
                if verbose:
                  print(f"added {obj}")
              else:
                self.objects.append(IfcObject(obj))

################################################################################
#            Basic methods to add/filter objects
################################################################################

      def filter_objects_by_type(self, ifc_type, objs=None):
          """ Filters objects by their IFC type.

              Parameters
              ----------
              ifc_type : str
                  The IFC type to filter by (e.g., "IfcWall", "IfcSlab").

              objs : list, optional
                  List of IfcObject instances to filter. If None, uses all objects in the set.

              Returns
              -------
              filtered_objects : list
                  List of IfcObject instances matching the specified IFC type. """
          
          if objs is None:
            objs = self.objects
          filtered_objects = [obj for obj in objs if obj.ifc_obj.is_a(ifc_type)]
          return filtered_objects

      def filter_walls_by_floor(self, floor, objs=None):
          """ Filters wall objects by floor number.

              Parameters
              ----------
              floor : int
                  The floor number to filter walls by.

              objs : list, optional
                  List of IfcObject instances to filter. If None, uses all objects in the set.

              Returns
              -------
              filtered_objects : list
                  List of Wall instances on the specified floor. """
          
          if objs is None:
            objs = self.objects
          walls = self.filter_objects_by_type("IfcWall", objs=objs)
          filtered_objects = [W_i for W_i in walls if W_i.floor == floor]
          return filtered_objects

      def filter_objects_by_floor(self, floor, objs=None):
          """ Filters objects by floor number.

              Parameters
              ----------
              floor : int
                  The floor number to filter objects by.

              objs : list, optional
                  List of IfcObject instances to filter. If None, uses all objects in the set.

              Returns
              -------
              filtered_objects : list
                  List of IfcObject instances on the specified floor. """
          
          if objs is None:
            objs = self.objects
          filtered_objects = [obj for obj in objs if obj.floor == floor]
          return filtered_objects

      def find_object_by_id(self, id, objs=None):
          """ Finds an object by its IFC ID.

              Parameters
              ----------
              id : int
                  The IFC ID of the object to find.

              objs : list, optional
                  List of IfcObject instances to search. If None, uses all objects in the set.

              Returns
              -------
              objs : IfcObject or None
                  The IfcObject instance with the specified ID, or None if not found. """
          
          if objs is None:
              objs = self.objects
          for obj in objs:
              if obj.ifc_obj.id() == id:
                  return obj
          obj = None
          return obj

      def filter_objects(self, floor=None, type=None, objs=None):
          """ Filters objects by floor and/or type.

              Parameters
              ----------
              floor : int, optional
                  The floor number to filter objects by.

              type : str, optional
                  The IFC type to filter by (e.g., "IfcWall", "IfcSlab").

              objs : list, optional
                  List of IfcObject instances to filter. If None, uses all objects in the set.

              Returns
              -------
              objs : list
                  List of IfcObject instances matching the specified criteria. """
          
          if objs is None:
            objs = self.objects
          if floor is not None:
            objs = self.filter_objects_by_floor(floor)
          if type is not None:
            objs = self.filter_objects_by_type(type, objs=objs)
          return objs


  ################################################################################
  #            methods to generate mesh 
  ################################################################################

      def get_all_meshes_highlighting_elements_with_id(self, elem_id=[], objs=None):
          """ Generates meshes highlighting specified elements by their IDs.

              Parameters
              ----------
              elem_id : int or list, optional
                  The IFC ID(s) of the elements to highlight.

              objs : list, optional
                  List of IfcObject instances to generate meshes from. If None, uses all objects in the set.

              Returns
              -------
              meshes : list
                  List of mesh representations of the objects, with specified elements highlighted. """
          
          meshes = []
          if not isinstance(elem_id, list):
              elem_id = [elem_id]

          if objs is None:
            objs = self.objects

          for id_i in elem_id:
            main_obj = self.find_object_by_id(id_i, objs=objs)
            meshes.append(main_obj.get_mesh(color = "orange", opacity=1))
          # Other (not highlighted) elements
          
          for obj in objs:
              if obj.id not in elem_id:
                  meshes.append(obj.get_mesh(color = "grey", opacity=0.3))
          return meshes

      def get_mesh_by_type(self, obj_type, opacity=0.8, objs=None):
          """ Generates meshes for objects of a specific type.

              Parameters
              ----------
              obj_type : str
                  The IFC type to filter by (e.g., "IfcWall", "IfcSlab").

              opacity : float, optional
                  Opacity of the generated meshes (default is 0.8).

              objs : list, optional
                  List of IfcObject instances to generate meshes from. If None, uses all objects in the set.

              Returns
              -------
              meshes : list
                  List of mesh representations of the filtered objects. """
          
          filtered_objects = self.filter_objects_by_type(obj_type, objs=objs)
          meshes = []
          for obj in filtered_objects:
              meshes.append(obj.get_mesh(color=self.random_pastel(), opacity=opacity))
          return meshes

      def get_mesh_of_wall_partitions(self, mode="with_loading_elements", opacity=0.8, objs=None):
          """ Generates meshes for wall partitions.

              Parameters
              ----------
              mode : str, optional
                  Mode for generating wall partition meshes (default is "with_loading_elements").

              opacity : float, optional
                  Opacity of the generated meshes (default is 0.8).

              objs : list, optional
                  List of IfcObject instances to generate meshes from. If None, uses all objects in the set.

              Returns
              -------
              WP_meshes : list
                  List of mesh representations of the wall partitions. """
          
          WP_meshes = []
          if objs is None:
            objs = self.objects
          walls =self.filter_objects_by_type("IfcWall", objs=objs)
          for W in walls:
            WPs = W.set_wall_partitions()
            WP_meshes.extend(WPs.get_meshes(mode=mode, opacity=opacity))
          return WP_meshes

      def get_mesh_by_floor(self, floor, color=None, opacity=0.8, objs=None):
          """ Generates meshes for objects on a specific floor.

              Parameters
              ----------
              floor : int
                  The floor number to filter objects by.

              color : str or tuple, optional
                  Color of the generated meshes (default is None).

              opacity : float, optional
                  Opacity of the generated meshes (default is 0.8).

              objs : list, optional
                  List of IfcObject instances to generate meshes from. If None, uses all objects in the set.

              Returns
              -------
              meshes : list
                  List of mesh representations of the filtered objects on the specified floor. """
          
          filtered_objects = self.filter_objects_by_floor(floor, objs=objs)
          meshes = self.get_meshes(objs=filtered_objects, color=color, opacity=opacity)
          return meshes

      def get_meshes(self, objs=None, color=None, opacity=0.8):
          """ Generates meshes for a list of objects.

              Parameters
              ----------
              objs : list, optional
                  List of IfcObject instances to generate meshes from. If None, uses all objects in the set.

              color : str or tuple, optional
                  Color of the generated meshes (default is None).

              opacity : float, optional
                  Opacity of the generated meshes (default is 0.8).

              Returns
              -------
              meshes : list
                  List of mesh representations of the specified objects. """
          
          if objs is None:
              objs = self.objects
          meshes = []
          for obj in objs:
              meshes.append(obj.get_mesh(color=color, opacity=0.8))
          return meshes

      def get_mesh_of_slab_partitions(self, color=None, opacity = 0.8, objs=None):
          """ Generates meshes for slab partitions.

              Parameters
              ----------
              color : str or tuple, optional
                  Color of the generated meshes (default is None).

              opacity : float, optional
                  Opacity of the generated meshes (default is 0.8).

              objs : list, optional
                  List of IfcObject instances to generate meshes from. If None, uses all objects in the set.

              Returns
              -------
              meshes : list
                  List of mesh representations of the slab partitions. """
          
          slabs = self.filter_objects_by_type("IfcSlab", objs=objs)
          
          meshes = []
          for S_i in slabs:
              meshes.extend(S_i.slab_partitions.get_meshes())
          return meshes

################################################################################
################################################################################
#        BUILDING
################################################################################
################################################################################

class Building(IfcObjectSet):
    """ Class to handle a building represented in an IFC file.

        Attributes
        ----------
        ifc_file : ifcopenshell.file
            The IFC file representing the building.

        floor_wall_dict : dict
            A dictionary mapping floor numbers to lists of wall objects on that floor.

        floor_heights : dict
            A dictionary mapping floor numbers to their bottom and top heights.

        slab_wall_links_df : pandas.DataFrame
            A DataFrame containing links between slabs and supporting walls.

        wall_wall_links_df : pandas.DataFrame
            A DataFrame containing links between walls and supporting walls.

        Methods
        -------
        collect_ifc_objects(ifc_type="IfcProduct", only_geometry=False, verbose=False)
            Collects IFC objects of a specified type from the IFC file.

        set_floor_ids_of_wall_partitions()
            Sets floor IDs for wall partitions.

        set_transmitting_distributed_loads()
            Sets transmitting distributed loads for walls.

        get_slab_wall_links()
            Links slabs with supporting walls.

        get_wall_slab_links()
            Links walls with loaded slabs.    

        get_wall_below_wall_links()
            Links walls with supporting walls.

        get_WP_WP_links()
            Links wall partitions with supporting wall partitions.
            
        create_wall_partitions()
            Creates wall partitions for walls.
            
        get_wall_partition_properties(detailed=False)
            Gets properties of wall partitions.
            
        group_walls_by_floor_get_floor_heights()
            Groups walls by floor and gets floor heights.

        set_floor_of_slabs()
            Sets floor numbers for slabs based on their bottom heights.

        get_floor_weights()
            Calculates the total weight of slabs on each floor.

        get_floor_properties()
            Gets properties of floors in the building. """

    def __init__(self, ifc_file, only_geometry=False, verbose=False):
        """ Initialize a Building instance.

            Parameters
            ----------
            ifc_file : ifcopenshell.file
                The IFC file representing the building.

            only_geometry : bool, optional
                If True, only geometry is considered (default is False).

            verbose : bool, optional
                If True, prints information during processing (default is False). """
        
        super().__init__()
        self.ifc_file = ifc_file
        self._seen_ids = set()
        if ifc_file is not None:
          self.collect_ifc_objects(only_geometry=only_geometry, verbose=verbose)
        if not only_geometry:
          if verbose:
            print("Grouping wall by floor, getting floor heights")
          self.floor_wall_dict, self.floor_heights = self.group_walls_by_floor_get_floor_heights()
          if verbose:
            print("Setting floor levels of slabs")
          self.set_floor_of_slabs()
          
          self.slab_wall_links_df = None
          self.wall_wall_links_df = None
          if verbose:
            print("Linking slabs with supporting walls")
          self.slab_wall_links_df = self.get_slab_wall_links()
          if verbose:
            print("Linking walls with supporting walls")
          self.wall_wall_links_df = self.get_wall_below_wall_links()
          if verbose:
            print("Setting transmitting distributed loads")
          self.set_transmitting_distributed_loads()
          if verbose:
            print("Creating wall partitions")
          self.create_wall_partitions()
          if verbose:
            print("Setting floor ids of wall partitions")
          self.set_floor_ids_of_wall_partitions()


    def collect_ifc_objects(self, ifc_type="IfcProduct", only_geometry=False, verbose=False):
        """ Collects IFC objects of a specified type from the IFC file.

            Parameters
            ----------
            ifc_type : str, optional
                The IFC type to collect (default is "IfcProduct").

            only_geometry : bool, optional
                If True, only geometry is considered (default is False).

            verbose : bool, optional
                If True, prints information during processing (default is False).

            Returns
            -------
            objects : list
                List of IfcObject instances collected from the IFC file. """
        
        for obj in self.ifc_file.by_type(ifc_type):
            self.add_object(obj, only_geometry=only_geometry, verbose=verbose)

        objects = self.objects
        return objects

################################################################################
#        Methods related to interconnection between elements
################################################################################

    def set_floor_ids_of_wall_partitions(self):
        """ Assigns floor IDs to wall partitions based on their supporting elements."""

        WP_counter = 0
        for floor in self.floor_wall_dict:
          walls = self.filter_walls_by_floor(floor)
          for W in walls:
            WPs = W.get_wall_partitions()
            for i, WP_i in enumerate(WPs.objects):
              if floor == 0:
                WP_counter += 1
                WP_i.set_floor_id(WP_counter)
              else:
                WP_i.set_floor_id(WP_i.supporting_elements[0].floor_id)
              # with this numbering it is not resolved when WPs are supported 
              # by several WPs. If this is the case, it can happen that two WPs get the same numbering

    def set_transmitting_distributed_loads(self):
        """ Sets transmitting distributed loads for walls based on their supporting elements."""

        # Loop over floors from top to bottom
        for floor in reversed([floor for floor in self.floor_wall_dict]):
          walls = self.filter_walls_by_floor(floor)
          for W_i in walls:
            W_i.set_transmitting_distributed_loads()


    def get_slab_wall_links(self):
        """ Links slabs with supporting walls.

            Returns
            -------
            slab_wall_links_df : pandas.DataFrame
                A DataFrame containing links between slabs and supporting walls. """
        
        if self.slab_wall_links_df is not None:
            slab_wall_links_df = self.slab_wall_links_df
            return slab_wall_links_df

        slabs = self.filter_objects_by_type("IfcSlab")
        walls = self.filter_objects_by_type("IfcWall")
        S_W_links = []
        for S_k in slabs:
          for SP_i in S_k.slab_partitions.objects:
            e_points_i = SP_i.e_points
            for W_j in walls:
                # Check if any edge point of the partition is close to the wall's top face
                  tol_z = S_k.get_dimensions()["Height"] + 0.02
                  is_linked, contact_info = W_j.is_edge_on_wall_top(e_points_i, tol_z=tol_z)

                  if is_linked:
                      entry = {}
                      entry["SlabId"] = S_k.id
                      entry["SlabPartitionId"] = SP_i.id
                      entry["ContactLength"] = contact_info["ContactLength"]
                      entry["ContactStart"] = contact_info["ContactStart"]
                      entry["ContactEnd"] = contact_info["ContactEnd"]
                      entry["WallLength"] = W_j.get_dimensions()["Length"],
                      entry["SlabEdgeLength"] = np.linalg.norm(e_points_i[0] - e_points_i[1])
                      entry["SlabHeight"] = S_k.get_dimensions()["Height"]
                      entry[ "LoadedWallId"] = W_j.id
                      W_j.add_loading_slab_partition({
                          "Object": SP_i,
                          "ContactLength":contact_info["ContactLength"]} )
                      SP_i.add_supporting_elements([W_j])
                      S_W_links.append(entry)
            # Send a warning if edge is not supported at all
            if SP_i.supporting_elements == []:
              warnings.warn("The slab partition with id {} of slab {} is not supported by any wall.".format(SP_i.id, S_k.id), UserWarning)

        slab_wall_links_df = pd.DataFrame(S_W_links)
        return slab_wall_links_df


    def get_wall_slab_links(self):
        """ Links walls with loaded slabs.

            Returns
            -------
            wall_slab_links : pandas.DataFrame
                A DataFrame containing links between walls and loaded slabs. """
        
        df = self.slab_wall_links_df
        wall_slab_links = df.groupby("LoadedWallId").agg({
            "SlabId": lambda x: sorted(list(set(x))),
            "SlabPartitionId": lambda x: sorted(list(set(x)))
        })
        return wall_slab_links


    def get_wall_below_wall_links(self):
        """ Links walls with supporting walls.

            Returns
            -------
            wall_wall_links_df : pandas.DataFrame
                A DataFrame containing links between walls and supporting walls. """
        
        if self.wall_wall_links_df is not None:
            wall_wall_links_df = self.wall_wall_links_df
            return wall_wall_links_df
        W_W_links = []
        # loop over floors from the top to the bottom
        for floor in reversed([floor for floor in self.floor_wall_dict if floor != 0]):
          walls_floor = self.filter_walls_by_floor(floor)
          walls_bellow_floor = self.filter_walls_by_floor(floor-1)
          for W_i in walls_floor:  #loop over walls within floor
            axis_points = W_i.get_lower_face_axis_points()
            supported = False
            for W_b_j in walls_bellow_floor:
              is_linked, contact_info = W_b_j.is_edge_on_wall_top(axis_points, tol_z=0.3)
              if is_linked:
                supported = True
                entry = {}
                entry["LoadingWallId"] = W_i.id
                entry["SupportingWallId"] = W_b_j.id
                entry["ContactLength"] = contact_info["ContactLength"]
                entry["WallLength"] = W_i.get_dimensions()["Length"]
     
                W_i.add_supporting_wall({
                    "Object": W_b_j,
                     "ContactLength": contact_info["ContactLength"],
                     "ContactStart": contact_info["ContactStart"],
                     "ContactEnd" : contact_info["ContactEnd"]} )
                entry["ContactStart"] = contact_info["ContactStart"]
                entry["ContactEnd"] = contact_info["ContactEnd"]
                W_b_j.add_loading_wall({
                    "Object": W_i,
                    "ContactLength": contact_info["ContactLength"],
                    "ContactStart": contact_info["ContactStart"],
                     "ContactEnd" : contact_info["ContactEnd"]} )
                W_W_links.append(entry)
          if not supported:
              W_W_links.append({
              "loading_wall_id": W_i.id,
              "supporting_wall_id": None
              })

        wall_wall_links_df = pd.DataFrame(W_W_links)
        return wall_wall_links_df

    def get_WP_WP_links(self):
        """ Links wall partitions with supporting wall partitions.

            Returns
            -------
            df_WP_WP : pandas.DataFrame
                A DataFrame containing links between wall partitions and supporting wall partitions. """
        
        records = []
        for floor in reversed([floor for floor in self.floor_wall_dict if floor != 0]):
          walls = self.filter_walls_by_floor(floor)
          for W_i in walls:
            for WP_j in W_i.wall_partitions.objects:
              record = {}
              record["WallIfcId"] = W_i.id
              record["PartitionNumber"] = WP_j.id
              record["FloorId"] = WP_j.floor_id
              record["SupportingElement"] = [wp.id for wp in WP_j.supporting_elements]
              record["SupportingElementFloorId"] = [wp.floor_id for wp in WP_j.supporting_elements]
              # Add record to list
              records.append(record)
        df_WP_WP = pd.DataFrame(records)
        df_WP_WP.set_index("FloorId")
        return df_WP_WP
    

    def export_building_data_to_excel(self, file_path):
        """ Exports building data to an Excel file in the format required for BIMQuake earthquake vulnerability calculations.

            Parameters
            ----------
            file_path : str
                The path to the Excel file where the data will be exported. """

        df = self.get_floor_properties()
        props = self.get_wall_partition_properties(detailed=False)

        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name='Data')
            for floor, prop_df in props.items():
                prop_df.to_excel(writer, sheet_name=f'Floor{floor+1}')

################################################################################
#        Generate wall partition, get its properties
################################################################################
    
    def create_wall_partitions(self):
        """ Creates wall partitions for walls."""
        
        # loop over floors from the bottom
        for floor, walls_floor in self.floor_wall_dict.items():
          for W_i in walls_floor:
            # Create wall partitions
            W_i.set_wall_partitions()

            
    def get_wall_partition_properties(self, detailed=False):
        """ Gets properties of wall partitions.

            Parameters
            ----------
            detailed : bool, optional
                If True, includes detailed properties (default is False).

            Returns
            -------
            prop_dfs : dict
                A dictionary mapping floor numbers to DataFrames containing wall partition properties. """
        
        # initiate list of dataframes for the different floors
        prop_dfs = {}
        # Loop over floors from top to bottom
        for floor in self.floor_wall_dict:
          walls = self.filter_walls_by_floor(floor)
          records = []
          for W_j in walls:
            WPs = W_j.get_wall_partitions()
            if WPs is None or WPs.objects==[]:
              continue
            for i, WP_i in enumerate(WPs.objects):   # loop over wall partitions
                # set floor id, to have the same id for walls that have the same x,y position
                record = {}
                points = WP_i.points
                record["WallIfcId"] = W_j.id
                record["PartitionNumber"] = WP_i.id
                record['Wall'] = WP_i.floor_id
                dim_ji = W_j.get_dimensions(points=points)
                L_ji = dim_ji["Length"]
                w_ji = dim_ji["Width"]
                h_ji = dim_ji["Height"]
                record["L [m]"] = L_ji
                record["w [m]"] = w_ji
                record["H [m]"] = h_ji
                record["Cx [m]"] = dim_ji["CenterPointX"]
                record["Cy [m]"] = dim_ji["CenterPointY"]
                # Angle of wall (degree between global x axis and local wall axis)
                record["α"] = W_j.get_angle_of_axis()
                # wall density
                record["γ [kN/m³]"] = W_j.properties.density *9.81/1e3
                # normal stresses on the wall bottom
                sigma_SLU = W_j.get_limit_state_stress(limit_state="SLU")
                sigma_SLE = W_j.get_limit_state_stress(limit_state="SLE")
                record["σ {SLU} [N/mm²]"] = sigma_SLU/1e6 if sigma_SLU is not None else None
                record["σ [N/mm²]"] = sigma_SLE/1e6 if sigma_SLE is not None else None
                # Wall elastic moduli
                record["E [N/mm²]"] = W_j.properties.E / 1e6 if W_j.properties.E is not None else None
                # wall's ShearModulus
                record["G [N/mm²]"] = W_j.properties.G / 1e6 if W_j.properties.G is not None else None
                # wall's TensileStrength
                record["f_u [N/mm²]"] = W_j.properties.f_u/1e6 if W_j.properties.f_u is not None else None
                # wall's ShearStrength
                record["τ [N/mm²]"] = W_j.properties.f_u/1.5e6 if W_j.properties.f_u is not None else None
                # wall's compressive strength
                record["fₘ [N/mm²]"] = W_j.properties.f_c/1e6 if W_j.properties.f_c is not None else None
                # wall's PoissonRatio
                record["μ"] = W_j.properties.nu
                # Add record to list
                records.append(record)
          df = pd.DataFrame(records).set_index("Wall")
          if not detailed:
            df = df[["L [m]", "w [m]", "H [m]", "Cx [m]", "Cy [m]", 
            "α", "σ [N/mm²]", "τ [N/mm²]", "fₘ [N/mm²]",
            "γ [kN/m³]", "E [N/mm²]", "G [N/mm²]", "μ"]]
          prop_dfs[floor] = df

        return prop_dfs

################################################################################
#  Methods to identify floors, define its heights, categorize elements by floor
################################################################################

    def group_walls_by_floor_get_floor_heights(self):
        """ Groups walls by floor and gets floor heights.

            Returns
            -------
            floor_wall_dict : dict
                A dictionary mapping floor numbers to lists of wall objects on that floor.

            floor_hs_dict : dict
                A dictionary mapping floor numbers to their bottom and top heights. """
        
        walls = super().filter_objects_by_type("IfcWall")
        # Create array of [bottom_height, top_height]
        hs = []
        for W in walls:
            # coordinate of wall bottom and top
            z_b = np.round(W.get_bottom_height(), decimals=1)
            z_t = np.round(W.get_top_height(), decimals=1)
            hs.append([z_b, z_t])
        hs = np.array(hs)

        # Find unique height pairs and inverse indices
        floor_hs, inv_ind = np.unique(hs, axis=0, return_inverse=True)

        # Sort by bottom height and remap to floor numbers
        sort_ind = np.argsort(floor_hs[:, 0])  # sort by bottom height
        floor_hs_dict = {}
        for i, ind_i in enumerate(sort_ind):
            floor_hs_dict[i] = {
                "BottomHeight": floor_hs[i,0],
                "TopHeight": floor_hs[i,1]}

        floor_no_map = {tuple(floor_hs[i]): idx for idx, i in enumerate(sort_ind)}

        # Group walls by those unique indices
        floor_wall_dict = defaultdict(list)
        for idx, W in zip(inv_ind, walls):
            key = tuple(floor_hs[idx])
            floor_no = floor_no_map[key]
            W.set_floor(floor_no)
            floor_wall_dict[floor_no].append(W)

        return floor_wall_dict, floor_hs_dict


    def set_floor_of_slabs(self):
        """ Sets floor numbers for slabs based on their bottom heights."""

        slabs = super().filter_objects_by_type("IfcSlab")
        h_ground = self.floor_heights[0]["BottomHeight"]
        for S in slabs:
            h_top = S.get_top_height()
            h_bottom = S.get_bottom_height()
            if np.isclose(h_top, h_ground) or np.isclose(h_bottom, h_ground):
                S.set_floor(0)
                continue
            for floor, props in self.floor_heights.items():
                h_floor = props["TopHeight"]
                if np.isclose(h_top, h_floor, atol=0.2) or np.isclose(h_bottom, h_floor, atol=0.2):
                    S.set_floor(floor+1)
                    break

################################################################################
#            compute floor weights
################################################################################
    def get_floor_weights(self):
      """ Calculates the total weight of slabs on each floor.

          Returns
          -------
          floor_weight_dict : dict
              A dictionary mapping floor numbers to their total slab weights. """
      
      slabs = self.filter_objects_by_type("IfcSlab")
      floor_weight_dict ={}
      for floor in self.floor_wall_dict:
        w = 0
        # get slab weight
        slabs_i = super().filter_objects_by_floor(floor+1, objs=slabs)
        for S_ji in slabs_i:
          w += S_ji.get_weight()
        # get lower wall top half weight
        walls_b_i = super().filter_walls_by_floor(floor)
        for W_b_ji in walls_b_i:
            w += W_b_ji.get_upper_half_weight()
        # get upper wall top half weight
        if floor != max(self.floor_wall_dict):
          walls_t_i = super().filter_walls_by_floor(floor+1)
          for W_t_ji in walls_t_i:
            w += W_t_ji.get_lower_half_weight()
        # For the last top wall
        floor_weight_dict[floor] = w

      return floor_weight_dict

    def get_floor_properties(self):
        """ Gets properties of floors in the building.

            Returns
            -------
            floor_props_df : pandas.DataFrame
                A DataFrame containing properties of each floor. """
      
        floor_heights = self.floor_heights
        floor_weights = self.get_floor_weights()

        floor_names =  []
        floor_values = np.zeros((len(floor_heights), 2))

        for floor in floor_heights:
            floor_values[floor, 0] = floor_heights[floor]["TopHeight"]-floor_heights[floor]["BottomHeight"]
            floor_values[floor, 1] = floor_weights[floor]/1000
            floor_names.append(f'Floor{floor+1}')

        floor_names = np.array(floor_names).reshape(-1, 1)
        floor_values = np.concatenate((floor_names, floor_values), axis=1)

        columns = ['Floor', 'H [m]', 'W [kN]']
        floor_props_df = pd.DataFrame(floor_values, columns=columns)
        return floor_props_df
        
    def __repr__(self):
        """ String representation of the Building instance. """
        
        return f"<IfcObjectSet with {len(self.objects)} objects>"
