from .simple_objects import SolidObject, SolidObjectSet
import numpy as np
from .frame_element import GeometricProperties, StressState, Linkage, Linkages, FrameElement

################################################################################
################################################################################
#        StructuralUnit
################################################################################
################################################################################

# Class to define structural partitions (base class for slab and wall partition classes)
class StructuralUnit(SolidObject):
    """ Base class for structural units such as slabs and walls. 

        Attributes
        ----------
        id : int
            The ID of the structural unit.
        name : str
            The name of the structural unit.
        floor_id : int or None
            The ID of the floor the structural unit belongs to.
        supported_by : Linkages
            The elements supporting this structural unit.
        loaded_by : Linkages
            The elements loading this structural unit. """
    
    def __init__(self, coords, id, supporting_elements=None):
        """ Initializes a structural unit.

            Parameters
            ----------
            coords : array-like
                The coordinates defining the structural unit.
            id : int
                The ID of the structural unit.
            supporting_elements : list, optional
                The elements supporting this structural unit. """
        
        super().__init__(coords)
        self.id = id
        self.name = self.id
        self.floor_id = None
        self.supported_by = Linkages()
        if supporting_elements is not None:
          self.add_supporting_element(supporting_elements)
        self.loaded_by = Linkages()

    @property
    def supporting_elements(self):
      """ Returns the elements supporting this structural unit. """
      return [link.element for link in self.supported_by]

    @property
    def loading_elements(self):
      """ Returns the elements loading this structural unit. """
      return [link.element for link in self.loaded_by]

    def set_floor_id(self, floor_id):
        """ Sets the floor ID for this structural unit.
        
            Parameter
            ---------
            floor_id : int
                The ID of the floor the structural unit belongs to. """
        self.floor_id = floor_id

    def add_supporting_element(self, obj):
        """ Adds a supporting element to this structural unit.

            Parameters
            ----------
            obj : Linkage, Linkages, or list
                The supporting element(s) to add. """
        
        if isinstance(obj, (Linkages, list)):
          self.add_supporting_elements(obj)
        elif isinstance(obj, Linkage):
          self.supported_by.add(obj)
        else:
          self.supported_by.add(Linkage(element=obj, contact_length=None))

    def add_supporting_elements(self, objs):
          """ Adds multiple supporting elements to this structural unit.

            Parameters
            ----------
            objs : list
                The supporting elements to add. """
          
          [self.add_supporting_element(obj) for obj in objs]

    def add_loading_element(self, obj):
        """ Adds a loading element to this structural unit.

            Parameters
            ----------
            obj : Linkage, Linkages, or list
                The loading element(s) to add. """
        
        if isinstance(obj, (Linkages, list)):
          self.add_loading_elements(obj)
        elif isinstance(obj, Linkage):
          self.loaded_by.add(obj)
        else:
          self.loaded_by.add(Linkage(element=obj, contact_length=None))
        

    def add_loading_elements(self, objs):
        """ Adds multiple loading elements to this structural unit.

            Parameters
            ----------
            objs : list
                The loading elements to add. """
        
        [self.add_loading_element(obj) for obj in objs]

    
################################################################################
################################################################################
#        WallPartition
################################################################################
################################################################################
# Class to define wall partitions and the
# belonging  structural segments (loading cuboids) that load this part
class WallPartition(StructuralUnit):
    """ Class to define wall partitions and the belonging structural segments (loading cuboids) that load this part.
    
        Attributes
        ----------
        wall : Wall
            The wall this partition belongs to.
        struct_type : str
            The type of the structural unit (e.g., 'LoadBearing').
        stress : dict or None
            The stress values for this wall partition.
        linked_struct : list
            The structural segments (loading cuboids) linked to this wall partition. """
    
    def __init__(self, coords, id, wall, linked_struct_points=None, struct_type='LoadBearing', supporting_elements=None):
        """ Initializes a WallPartition instance.

            Parameters
            ----------
            coords : list
                The coordinates defining the wall partition.
            id : str
                The unique identifier for this wall partition.
            wall : Wall
                The wall this partition belongs to.
            linked_struct_points : list or None, optional
                The points defining the linked structural segments (loading cuboids), by default None.
            struct_type : str, optional
                The type of the structural unit (e.g., 'LoadBearing'), by default 'LoadBearing'.
            supporting_elements : list or None, optional
                The supporting elements for this wall partition, by default None. """
        
        super().__init__(coords, id, supporting_elements=supporting_elements)
        self.wall = wall
        self.struct_type = struct_type
        self.stress = None
        if linked_struct_points is None:
            self.linked_struct = []   
        else:
            self.set_linked_struct(linked_struct_points)


    def set_linked_struct(self, linked_struct_points):
        """ Sets the linked structural segments (loading cuboids) for this wall partition.

            Parameters
            ----------
            linked_struct_points : list or array-like
                The points defining the linked structural segments (loading cuboids). """
        
        if isinstance(linked_struct_points, list):
            self.linked_struct = [WallPartition(points_i, f"{self.id}_{i}", self.wall, struct_type='Loading', supporting_elements = [self.id])
            for i, points_i in enumerate(linked_struct_points)]
        else:
            self.linked_struct = [WallPartition(linked_struct_points, f"{self.id}_1", self.wall, struct_type='Loading', supporting_elements = [self.id])]


    def set_stress_values(self, sigma_dict):
        """ Sets the stress values for this wall partition.

            Parameters
            ----------
            sigma_dict : dict
                A dictionary containing the stress values for this wall partition. """
        
        self.stress = sigma_dict

    def get_stress_state(self):
        """ Gets the stress state for this wall partition.

            Returns
            -------
            stress_state : StressState
                The stress state of this wall partition. """
        stress_state = StressState(top = self.stress["top"], midheight=self.stress["half-height"])
        return stress_state

    def add_linked_struct(self, linked_struct_points):
        """ Adds a linked structural segment (loading cuboid) to this wall partition.

            Parameters
            ----------
            linked_struct_points : list or array-like
                The points defining the linked structural segment (loading cuboid). """
        
        l = len(self.linked_struct)
        self.linked_struct.append(StructuralUnit(linked_struct_points, f"{self.id}_{l+1}", struct_type='Loading'))


    def get_total_volume_of_linked_parts(self):
        """ Computes the total volume of all linked structural parts (loading cuboids) for this wall partition.

            Returns
            -------
            V_LP : float
                The total volume of all linked structural parts. """
        # Initiate total volume of wall parts linked to this wall partition (from above openings)
        V_LP = 0.
        # computed total volume and length of linked parts
        for LP_k in self.linked_struct:
            V_LP += self.wall.get_volume(points=LP_k.points)
        return (V_LP)
    

    def get_total_length_of_linked_parts(self):
        """ Computes the total length of all linked structural parts (loading cuboids) for this wall partition.

            Returns
            -------
            L_LP : float
                The total length of all linked structural parts. """
        
        # Initiate total length of wall parts linked to this wall partition (from above openings)
        L_LP = 0
        # computed total volume and length of linked parts
        for LP_k in self.linked_struct:
            L_LP += self.wall.get_dimensions(points=LP_k.points)["Length"]
        return (L_LP)
    
    
    def get_mesh_with_linked_structs(self, base_name="HierarchicalStructure", color='blue'):
        """ Generates a mesh representation of the wall partition along with its linked structural parts.

            Parameters
            ----------
            base_name : str, optional
                The base name for the mesh objects (default is "HierarchicalStructure").
            color : str, optional
                The color of the mesh objects (default is 'blue').

            Returns
            -------
            meshes : list
                A list of mesh objects representing the wall partition and its linked structural parts. """
        
        meshes = [self.get_mesh(name=f'{base_name} - Main',
                  color=color)]
        if hasattr(self, 'linked_struct'):
            for i, part in enumerate(self.linked_struct):
                meshes.append(part.get_mesh(name=part.id,
                                            color=color, opacity=0.6,
                                            ).update(
                            hovertext=part.id,
                            hoverinfo='text'))
        return meshes

    @property
    def geometric_properties(self):
        """ Computes and returns the geometric properties of the wall partition.

              Returns
              -------
              geom_props : GeometricProperties
                  The geometric properties of the wall partition. """
          
          # angle of axis
        alpha = self.wall.get_angle_of_axis()
        dim = self.wall.get_dimensions(points=self.points)
        geom_props = GeometricProperties(
          L =  dim["Length"],         # Length
          w = dim["Width"],           # Thickness
          h =  dim["Height"],         # Height
          Cx =  dim["CenterPointX"],  # Centroid x
          Cy =  dim["CenterPointY"],  # Centroid y
          alpha = alpha,              # Orientation angle
          points =  self.points       # expected shape (8,3) - Vertext point coordinates
        )
        return geom_props

    @property
    def material_properties(self):
      """ Returns the material properties of the wall partition. """
      return self.wall.get_material_properties()
      
    
    def get_frame_element(self, elem_id):
      """ Generates and returns a frame element representation of the wall partition.

          Parameters
          ----------
          elem_id : str
              The identifier for the frame element.

          Returns
          -------
          F : FrameElement
              The frame element representing the wall partition. """
      
      F = FrameElement(
        id = elem_id,
        global_id = self.id,
        floor = self.wall.floor,
        material = self.material_properties,
        geometry = self.geometric_properties,
        stress = self.get_stress_state(),
        supported_by = self.supported_by,
        loaded_by = self.loaded_by
      )
      return F

    def __repr__(self):
        """ Returns a string representation of the wall partition. """

        return (
            f"<WallPartition id={self.id!r}>"
        )


        
################################################################################
################################################################################
#        StructuralUnitSet
################################################################################
################################################################################

class StructuralUnitSet(SolidObjectSet):
    """ Represents a set of structural units (slabs and walls). 
        
        Attributes
        ----------
        objects : list
            The list of structural unit objects contained in the set. """
    
    def __init__(self):
        """ Initializes an empty set of structural units. """

        super().__init__()

    def add_linked_structure(self, h_structure):
        """ Adds a linked structural unit to the set.

            Parameters
            ----------
            h_structure : StructuralUnit
                The structural unit to be added.

            Raises
            ------
            TypeError
                If the provided object is not a StructuralUnit (SlabPartition or WallPartition). """
        if isinstance(h_structure, StructuralUnit):
            self.objects.append(h_structure)
        else:
            raise TypeError("Only StructuralUnit (SlabPartition or WallPartition) objects can be added.")

    def get_meshes(self, mode="only_main", opacity=0.8):
      """ Returns the meshes of the structural units in the set.

          Parameters
          ----------
          mode : str, optional
              The mode of mesh retrieval. Options are "only_main" (default) or "with_linked".
          opacity : float, optional
              The opacity of the meshes (default is 0.8).

          Returns
          -------
          meshes : list
              A list of meshes representing the structural units. """
      
      meshes = []
      if mode == "only_main":
        for i, obj in enumerate(self.objects):
          meshes.append(obj.get_mesh(name=f"Part {i}",
                                      color= self.random_pastel(), opacity=opacity))
      else:
        for i, obj in enumerate(self.objects):
          color = self.random_pastel()
          meshes.extend(obj.get_mesh_with_linked_structs(base_name=obj.id, color=color))
      return meshes

    def __repr__(self):
      """ Returns a string representation of the hierarchical structures set. """

      return f"<HierarchicalStructuresSet with {len(self.objects)} objects>"

################################################################################
################################################################################
#        SlabPartition
################################################################################
################################################################################

class SlabPartition(StructuralUnit):
    """ Represents a slab partition within the structural system. 
    
        Attributes
        ----------
        coords : array-like
            The coordinates of the slab partition.
        id : int
            The unique identifier of the slab partition.
        slab : Slab
            The slab to which the partition belongs.
        e_points : list
            The edge points transmitting the load.
        A : float
            The area of the partition.
        g1 : float, optional
            The structural uniform permanent surface load on the slab (default is 0).
        g2 : float, optional
            The non-structural uniform permanent surface load on the slab (default is 0).
        q : dict, optional
            A dictionary with keys corresponding to different categories (A, B, ..H)
            and values of the vertical uniform surface load on the slab partition (default is 0). """
    
    def __init__(self, coords, id, slab, e_points, A, g1=0, g2=0, q=0):
      """ Initializes a SlabPartition instance.

          Parameters
          ----------
          coords : array-like
              The coordinates of the slab partition.
          id : int
              The unique identifier of the slab partition.
          slab : Slab
              The slab to which the partition belongs.
          e_points : list
              The edge points transmitting the load.
          A : float
              The area of the partition.
          g1 : float, optional
              The structural uniform permanent surface load on the slab (default is 0).
          g2 : float, optional
              The non-structural uniform permanent surface load on the slab (default is 0).
          q : dict, optional
              A dictionary with keys corresponding to different categories (A, B, ..H)
              and values of the vertical uniform surface load on the slab partition (default is 0). """
      
      super().__init__(coords, id)
      # Load transmitting edge points
      self.e_points = e_points
      self.name = "SP"+str(id)
      self.slab = slab
      # Area of partitions
      self.A = A
      # Total vertical load at the transmitting edge [N] and distributed force [N/m]
      self.G1, self.p_g1 = self.get_edge_load(g1)
      self.G2, self.p_g2 = self.get_edge_load(g2)
      self.Q, self.p_q = self.get_edge_load_from_q(q)

    def get_edge_load(self, p):
      """ Calculates the edge load based on the given surface load.

          Parameters
          ----------
          p : float
              The uniform surface load on the slab partition.

          Returns
          -------
          F_p : float
              The total vertical load at the transmitting edge [N].
          f_p : float
              The distributed force along the edge [N/m]. """
      
      if p is None:
        return None, None
      F_p = p * self.A
      L = np.linalg.norm(self.e_points[1] - self.e_points[0])
      f_p  = F_p/L
      return F_p, f_p

    def get_edge_load_from_q(self, q):
      """ Calculates the edge loads based on the given dictionary of surface loads.

          Parameters
          ----------
          q : dict
              A dictionary with keys corresponding to different categories (A, B, ..H)
              and values of the vertical uniform surface load on the slab partition.

          Returns
          -------
          F_q : dict
              A dictionary with keys corresponding to different categories (A, B, ..H)
              and values of the total vertical load at the transmitting edge [N].
          f_q : dict
              A dictionary with keys corresponding to different categories (A, B, ..H)
              and values of the distributed force along the edge [N/m]. """
      
      F_q = {cat_i: self.get_edge_load(val_i)[0] for cat_i, val_i in q.items()}
      f_q = {cat_i: self.get_edge_load(val_i)[1] for cat_i, val_i in q.items()}
      return F_q, f_q

    def __repr__(self):
        """ Returns a string representation of the SlabPartition object, including its ID and edge loads. """
        
        string = f"""<SlabPartition {self.id}
         with edge loads 
          G1: {np.round(self.G1,2)}N
          G2: {np.round(self.G2,2)}N,
          QA: {np.round(self.Q["A"],2)}N
          QB: {np.round(self.Q["A"],2)}N
          QC: {np.round(self.Q["A"],2)}N
          QD: {np.round(self.Q["A"],2)}N
          QE: {np.round(self.Q["A"],2)}N
          QF: {np.round(self.Q["A"],2)}N
          QG: {np.round(self.Q["A"],2)}N
          QH: {np.round(self.Q["A"],2)}N >"""
        return string
        