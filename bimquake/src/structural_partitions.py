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
    def __init__(self, coords, id, supporting_elements=None):
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
      return [link.element for link in self.supported_by]

    @property
    def loading_elements(self):
      return [link.element for link in self.loaded_by]

    def set_floor_id(self, floor_id):
        self.floor_id = floor_id

    def add_supporting_element(self, obj):
        if isinstance(obj, (Linkages, list)):
          self.add_supporting_elements(obj)
        elif isinstance(obj, Linkage):
          self.supported_by.add(obj)
        else:
          self.supported_by.add(Linkage(element=obj, contact_length=None))

    def add_supporting_elements(self, objs):
          [self.add_supporting_element(obj) for obj in objs]

    def add_loading_element(self, obj):
        if isinstance(obj, (Linkages, list)):
          self.add_loading_elements(obj)
        elif isinstance(obj, Linkage):
          self.loaded_by.add(obj)
        else:
          self.loaded_by.add(Linkage(element=obj, contact_length=None))
        

    def add_loading_elements(self, objs):
        [self.add_loading_element(obj) for obj in objs]

    
################################################################################
################################################################################
#        WallPartition
################################################################################
################################################################################
# Class to define wall partitions and the
# belonging  structural segments (loading cuboids) that load this part
class WallPartition(StructuralUnit):
    def __init__(self, coords, id, wall, linked_struct_points=None, struct_type='LoadBearing', supporting_elements=None):
        super().__init__(coords, id, supporting_elements=supporting_elements)
        self.wall = wall
        self.struct_type = struct_type
        self.stress = None
        if linked_struct_points is None:
            self.linked_struct = []   
        else:
            self.set_linked_struct(linked_struct_points)


    def set_linked_struct(self, linked_struct_points):
        if isinstance(linked_struct_points, list):
            self.linked_struct = [WallPartition(points_i, f"{self.id}_{i}", self.wall, struct_type='Loading', supporting_elements = [self.id])
            for i, points_i in enumerate(linked_struct_points)]
        else:
            self.linked_struct = [WallPartition(linked_struct_points, f"{self.id}_1", self.wall, struct_type='Loading', supporting_elements = [self.id])]


    def set_stress_values(self, sigma_dict):
        self.stress = sigma_dict

    def get_stress_state(self):
        return StressState(top = self.stress["top"], midheight=self.stress["half-height"])

    def add_linked_struct(self, linked_struct_points):
        l = len(self.linked_struct)
        self.linked_struct.append(StructuralUnit(linked_struct_points, f"{self.id}_{l+1}", struct_type='Loading'))


    def get_total_volume_of_linked_parts(self):
         # Initiate total volume of wall parts linked to this wall partition (from above openings)
          V_LP = 0.
          # computed total volume and length of linked parts
          for LP_k in self.linked_struct:
              V_LP += self.wall.get_volume(points=LP_k.points)
          return(V_LP)
    

    def get_total_length_of_linked_parts(self):
         # Initiate total length of wall parts linked to this wall partition (from above openings)
          L_LP = 0
          # computed total volume and length of linked parts
          for LP_k in self.linked_struct:
              L_LP += self.wall.get_dimensions(points=LP_k.points)["Length"]
          return(L_LP)
    
    
    def get_mesh_with_linked_structs(self, base_name="HierarchicalStructure", color='blue'):
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
        return self.wall.get_material_properties()
      
    
    def get_frame_element(self, elem_id):
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
        return (
            f"<WallPartition id={self.id!r}>"
        )


        
################################################################################
################################################################################
#        StructuralUnitSet
################################################################################
################################################################################

class StructuralUnitSet(SolidObjectSet):
    def __init__(self):
        super().__init__()

    def add_linked_structure(self, h_structure):
        if isinstance(h_structure, StructuralUnit):
            self.objects.append(h_structure)
        else:
            raise TypeError("Only StructuralUnit (SlabPartition or WallPartition) objects can be added.")

    def get_meshes(self, mode="only_main", opacity=0.8):
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
        return f"<HierarchicalStructuresSet with {len(self.objects)} objects>"

################################################################################
################################################################################
#        SlabPartition
################################################################################
################################################################################

class SlabPartition(StructuralUnit):
    def __init__(self, coords, id, slab, e_points, A, g1=0, g2=0, q=0):
      """
      rho is the density
      g1, g2 is the structural and non-structural uniform pernament surface load on the slab
      q is a dictionary with keys corresponding to different categories (A, B, ..H)
      and values of the vertical unifrom surface load on the slab partition
      """
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
      if p is None:
        return None, None
      F_p = p * self.A
      L = distance = np.linalg.norm(self.e_points[1] - self.e_points[0])
      f_p  = F_p/L
      return F_p, f_p

    def get_edge_load_from_q(self, q):
      F_q = {cat_i: self.get_edge_load(val_i)[0] for cat_i, val_i in q.items()}
      f_q = {cat_i: self.get_edge_load(val_i)[1] for cat_i, val_i in q.items()}
      return F_q, f_q

    def __repr__(self):
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
        