
from .simple_objects import SolidObject, SolidObjectSet
import numpy as np

# Class to define wall partitions and the
# belonging  structural segments (loading cuboids) that load this part
class StructuralUnit(SolidObject):
    """ Class to define wall partitions and the belonging  structural segments (loading cuboids) that load this part
    
        Attributes
        ----------
        coords : np.ndarray
            The coordinates defining the solid object.
            
        id : str
            Unique identifier for the structural unit.
            
        name : str
            Name of the structural unit, initialized to the id.
            
        floor_id : str or None
            Identifier for the floor associated with the structural unit.
            
        supporting_elements : list
            List of elements that support this structural unit.
            
        struct_type : str
            Type of the structural unit, default is 'LoadBearing'.
        
        linked_struct : list
            List of linked structural units (loading cuboids) associated with this structural unit.
            
        Methods
        -------
        set_floor_id(floor_id)
            Sets the floor identifier for the structural unit.
            
        set_linked_struct(linked_struct_points)
            Sets the linked structural units based on provided points.

        add_linked_struct(linked_struct_points)
            Adds a new linked structural unit based on provided points.

        add_supporting_elements(supporting_elements)
            Adds supporting elements to the structural unit.

        get_mesh_with_linked_structs(base_name="HierarchicalStructure", color='blue')
            Generates mesh representations for the structural unit and its linked structures. """
        
    def __init__(self, coords, id, linked_struct_points=None, struct_type='LoadBearing', supporting_elements=None):
        """ Initializes a StructuralUnit object.
        
            Parameters
            ----------
            coords : np.ndarray
                The coordinates defining the solid object.
                
            id : str
                Unique identifier for the structural unit.
                
            linked_struct_points : list or np.ndarray, optional
                Points defining linked structural units (loading cuboids). Default is None.
                
            struct_type : str, optional
                Type of the structural unit. Default is 'LoadBearing'.
                
            supporting_elements : list, optional
                List of elements that support this structural unit. Default is None. """
        
        super().__init__(coords)
        self.id = id
        self.name = self.id
        self.floor_id = None
        self.supporting_elements = []
        if supporting_elements is not None:
          self.add_supporting_elements(supporting_elements)
        self.struct_type = struct_type
        if linked_struct_points is None:
            self.linked_struct = []
        else:
            self.set_linked_struct(linked_struct_points)

    def set_floor_id(self, floor_id):
        """ Sets the floor identifier for the structural unit.
        
            Parameters
            ----------
            floor_id : str
                Identifier for the floor associated with the structural unit. """
        
        self.floor_id = floor_id

    def set_linked_struct(self, linked_struct_points):
        """ Sets the linked structural units based on provided points.
        
            Parameters
            ----------
            linked_struct_points : list or np.ndarray
                Points defining linked structural units (loading cuboids). """
        
        if isinstance(linked_struct_points, list):
            self.linked_struct = [StructuralUnit(points_i, f"{self.id}_{i}", struct_type='Loading')
            for i, points_i in enumerate(linked_struct_points)]
        else:
            self.linked_struct = [StructuralUnit(linked_struct_points, f"{self.id}_1", struct_type='Loading')]

    def add_linked_struct(self, linked_struct_points):
        """ Adds a new linked structural unit based on provided points.
        
            Parameters
            ----------
            linked_struct_points : np.ndarray
                Points defining the new linked structural unit (loading cuboid). """
        
        l = len(self.linked_struct)
        self.linked_struct.append(StructuralUnit(linked_struct_points, f"{self.id}_{l+1}", struct_type='Loading'))

    def add_supporting_elements(self, supporting_elements):
        """ Adds supporting elements to the structural unit.
        
            Parameters
            ----------
            supporting_elements : list
                List of elements that support this structural unit. """
        
        self.supporting_elements += supporting_elements

    def get_mesh_with_linked_structs(self, base_name="HierarchicalStructure", color='blue'):
        """ Generates mesh representations for the structural unit and its linked structures.
        
            Parameters
            ----------
            base_name : str, optional
                Base name for the main structural unit mesh. Default is "HierarchicalStructure".
                
            color : str, optional
                Color for the meshes. Default is 'blue'.
                
            Returns
            -------
            meshes : list
                List of mesh representations for the structural unit and its linked structures. """
        
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
        
        
class StructuralUnitSet(SolidObjectSet):
    """ Class to manage a collection of StructuralUnit objects.
    
        Inherits from SolidObjectSet.
        
        Methods
        -------
        add_linked_structure(h_structure)
            Adds a StructuralUnit object to the collection.
            
        get_meshes(mode="only_main", opacity=0.8)
            Generates mesh representations for the structural units in the collection. """
    
    def __init__(self):
        """ Initializes a StructuralUnitSet object. """

        super().__init__()

    def add_linked_structure(self, h_structure):
        """ Adds a StructuralUnit object to the collection.
        
            Parameters
            ----------
            h_structure : StructuralUnit
                The StructuralUnit object to be added to the collection.
                
            Raises
            ------
            TypeError
                If the provided object is not an instance of StructuralUnit. """
        
        if isinstance(h_structure, StructuralUnit):
            self.objects.append(h_structure)
        else:
            raise TypeError("Only HierarchicalStructures objects can be added.")

    def get_meshes(self, mode="only_main", opacity=0.8):
        """ Generates mesh representations for the structural units in the collection.

            Parameters
            ----------
            mode : str, optional
                Mode for mesh generation. "only_main" generates meshes for main structures only,
                while other modes include linked structures. Default is "only_main".

            opacity : float, optional
                Opacity for the meshes. Default is 0.8.

            Returns
            -------
            meshes : list
                List of mesh representations for the structural units in the collection. """
        
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
        """ Returns a string representation of the StructuralUnitSet object. """

        return f"<HierarchicalStructuresSet with {len(self.objects)} objects>"


class SlabPartition(StructuralUnit):
    """ Class to define slab partitions with load properties.
    
        Inherits from StructuralUnit.
        
        Attributes
        ----------
        e_points : numpy.ndarray
            Points defining the load transmitting edge of the slab partition.

        name : str
            Name of the slab partition, initialized to "SP" followed by the id.
        
        A : float
            Area of the slab partition.

        G1 : float
            Total vertical load at the transmitting edge due to structural permanent surface load [N].

        p_g1 : float
            Distributed force at the transmitting edge due to structural permanent surface load [N/m].

        G2 : float
            Total vertical load at the transmitting edge due to non-structural permanent surface load [N].

        p_g2 : float
            Distributed force at the transmitting edge due to non-structural permanent surface load [N/m].

        Q : dict
            Total vertical loads at the transmitting edge for different categories (A, B, ..H) [N].

        p_q : dict
            Distributed forces at the transmitting edge for different categories (A, B, ..H) [N/m].

        Methods
        -------
        get_edge_load(p)
            Calculates total and distributed edge load based on surface load.

        get_edge_load_from_q(q)
            Calculates total and distributed edge loads for different categories based on surface loads. """
    
    def __init__(self, coords, id, e_points, A, g1=0, g2=0, q=0, linked_struct_points=None):
        """ Initializes a SlabPartition object.
            
            Parameters
            ----------
            coords : np.ndarray
                The coordinates defining the slab partition.

            id : str
                Unique identifier for the slab partition.

            e_points : np.ndarray
                Points defining the load transmitting edge of the slab partition.

            A : float
                Area of the slab partition.

            g1 : float, optional
                Structural uniform permanent surface load on the slab [N/m²]. Default is 0.

            g2 : float, optional
                Non-structural uniform permanent surface load on the slab [N/m²]. Default is 0.

            q : dict, optional
                Dictionary with keys corresponding to different categories (A, B, ..H)
                and values of the vertical uniform surface load on the slab partition [N/m²]. Default is 0.

            linked_struct_points : list or np.ndarray, optional
                Points defining linked structural units (loading cuboids). Default is None."""
            
      
        super().__init__(coords, id, linked_struct_points)
        # Load transmitting edge points
        self.e_points = e_points
        self.name = "SP"+str(id)
        # Area of partitions
        self.A = A
        # Total vertical load at the transmitting edge [N] and distributed force [N/m]
        self.G1, self.p_g1 = self.get_edge_load(g1)
        self.G2, self.p_g2 = self.get_edge_load(g2)
        self.Q, self.p_q = self.get_edge_load_from_q(q)

    def get_edge_load(self, p):
        """ Calculates total and distributed edge load based on surface load.
        
            Parameters
            ----------
            p : float
                Uniform surface load on the slab partition [N/m²].
                
            Returns
            -------
            F_p : float or None
                Total edge load [N]. Returns None if p is None.
                
            f_p : float or None
                Distributed edge load [N/m]. Returns None if p is None. """
        
        if p is None:
            return None, None
        F_p = p * self.A
        L = distance = np.linalg.norm(self.e_points[1] - self.e_points[0])
        f_p  = F_p/L
        return F_p, f_p

    def get_edge_load_from_q(self, q):
        """ Calculates total and distributed edge loads for different categories based on surface loads.

            Parameters
            ----------
            q : dict
                Dictionary with keys corresponding to different categories (A, B, ..H)
                and values of the vertical uniform surface load on the slab partition [N/m²].

            Returns
            -------
            F_q : dict
                Dictionary with total edge loads [N] for each category.

            f_q : dict
                Dictionary with distributed edge loads [N/m] for each category. """
        
        F_q = {cat_i: self.get_edge_load(val_i)[0] for cat_i, val_i in q.items()}
        f_q = {cat_i: self.get_edge_load(val_i)[1] for cat_i, val_i in q.items()}
        return F_q, f_q
        