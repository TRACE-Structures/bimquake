# Define class of wall created from the IFC product
from .IFC_objects import IfcObject
from .structural_partitions import StructuralUnit, StructuralUnitSet
import numpy as np
from .simple_objects import SolidObject, SolidObjectSet
from scipy.spatial import QhullError

from .utils import get_line_polygon_intersection_and_gaps, segments_overlap

class Wall(IfcObject):
    """ Class representing a wall element derived from an IFC object.

        Inherits from IfcObject and includes methods for geometrical and mechanical
        properties specific to wall elements.
        
        Attributes
        ----------
        wall_partitions : StructuralUnitSet
            A set of structural wall partitions associated with the wall.
            
        supported_by_walls : list
            A list of walls that provide support to this wall.
            
        loaded_by_slab_partition : list
            A list of slab partitions that load this wall.
            
        loaded_by_walls : list
            A list of walls that load this wall.
            
        openings : SolidObjectSet
            A set of solid objects representing openings in the wall.
            
        Methods
        -------
        add_supporting_wall(contact_dict)
            Adds a wall that supports this wall.
            
        add_loading_slab_partition(contact_dict)
            Adds a slab partition that loads this wall.
            
        add_loading_wall(contact_dict)
            Adds a wall that loads this wall.
            
        get_average_x_coords(coords)
            Returns coordinates with averaged x-values.
            
        get_index_of_corner_points(points=None, warn_flag=True)
            Returns indices of corner points of the wall.
            
        get_index_of_upper_corner_points(points=None)
            Returns indices of upper corner points of the wall.
            
        get_upper_face_axis_points(points=None)
            Returns axis points of the upper face of the wall.
            
        get_lower_face_axis_points(points=None)
            Returns axis points of the lower face of the wall.
            
        get_index_of_opening_points()
            Returns indices of points corresponding to openings in the wall.
            
        filter_opening_points(ind_openings, x_tol=0.01, y_tol=0.01, z_tol=0.01)
            Filters opening points to retain only valid rectangular openings.
            
        orthogonalize_end_faces()
            Adjusts the end faces of the wall to be orthogonal.
            
        set_points_without_inner_wall_points()
            Sets wall points excluding inner wall points.
            
        set_openings()
            Identifies and sets openings in the wall.
            
        get_openings()
            Returns the set of openings in the wall.
            
        get_net_volume()
            Returns the net volume of the wall, accounting for openings.
            
        get_half_net_volumes(z_cut=None)
            Returns the net volumes of the lower and upper halves of the wall.
            
        project_opening_points_to_wall_bottom_and_top()
            Projects opening points to the wall's bottom and top surfaces.
            
        is_edge_on_wall_top(e_points, face_points=None, tol_z=0.02, tol_y=None)
            Checks if an edge is on the top face of the wall.
            
        get_upper_half_weight()
            Returns the weight of the upper half of the wall.
            
        get_lower_half_weight()
            Returns the weight of the lower half of the wall.
            
        get_distributed_loads_from_slabs()
            Returns distributed loads on the wall from slab partitions.
            
        set_transmitting_distributed_loads()
            Sets the distributed loads transmitted by the wall.
            
        get_distributed_loads_from_above_walls()
            Returns distributed loads on the wall from above walls.
            
        get_limit_state_stress(limit_state="SLE")
            Returns the stress in the wall for a given limit state.
            
        get_WP_support_info(points_start, points_end)
            Returns support information for a wall partition.
            
        register_wall_partition(WP_object)
            Registers a wall partition with the wall.
            
        merge_sections_and_ignore_small_ones(sections, tol_merge=0.05, tol_ignore=0.2)
            Merges overlapping sections and ignores small ones.
            
        set_wall_partitions()
            Identifies and sets wall partitions for the wall.
            
        get_wall_partitions()
            Returns the set of wall partitions associated with the wall."""
    
    def __init__(self, ifc_object):
        """ Initialize Wall object from an IFC object.
        
          Parameters
          ----------
          ifc_object : IfcObject
              An instance of IfcObject representing the wall from IFC data."""
        
        super().__init__(ifc_object)
        self.set_points_without_inner_wall_points()
        self.orthogonalize_end_faces()
        self.wall_partitions = StructuralUnitSet()
        self.supported_by_walls = []
        self.loaded_by_slab_partition = []
        self.loaded_by_walls = []
        self.openings = SolidObjectSet()

  ##################################################################
  #                   Methods to Link Elements
  ##################################################################

    def add_supporting_wall(self, contact_dict):
        """ Adds a wall that supports this wall.
        
            Parameters
            ----------
            contact_dict : dict
                A dictionary containing information about the supporting wall and contact details."""
        
        self.supported_by_walls.append(contact_dict)

    def add_loading_slab_partition(self, contact_dict):
        """ Adds a slab partition that loads this wall.
        
            Parameters
            ----------
            contact_dict : dict
                A dictionary containing information about the loading slab partition and contact details."""
        
        self.loaded_by_slab_partition.append(contact_dict)

    def add_loading_wall(self, contact_dict):
        """ Adds a wall that loads this wall.
        
            Parameters
            ----------
            contact_dict : dict
                A dictionary containing information about the loading wall and contact details."""
        
        self.loaded_by_walls.append(contact_dict)

  ##################################################################
  #                Methods to get geometrical properties
  ##################################################################
    def get_average_x_coords(self, coords):
        """ Returns coordinates with averaged x-values.
        
            Parameters
            ----------
            coords : numpy.ndarray, shape (N, 3)
                An array of coordinates.

            Returns
            -------
            coords : numpy.ndarray, shape (N, 3)
                The input coordinates with the x-values averaged."""
        if not np.allclose(coords[:,0], coords[0,0]):
          coords[:, 0] = np.mean(coords[:, 0])
        return coords

    def get_index_of_corner_points(self, points=None, warn_flag=True):
        """ Returns indices of corner points of the wall.
        
            Parameters
            ----------
            points : numpy.ndarray, shape (N, 3), optional
                An array of wall points. If None, uses the wall's own points. Default is None.

            warn_flag : bool, optional
                Whether to print a warning if bounding box corner indices cannot be found. Default is True.

            Returns
            -------
            ind_start_face : numpy.ndarray, shape (4,)
                Indices of the corner points of the starting face of the wall.

            ind_end_face : numpy.ndarray, shape (4,)
                Indices of the corner points of the ending face of the wall."""
        
        if points is None:
          points = self.points
        try:
          ind_corner = self.get_bounding_box_corner_indices(points=points, tol=0.1)
          ind_sort = self.sort_points_by_local_coordinates(points=points[ind_corner], sort_by_axis="xzy")
          ind_start_face = ind_corner[ind_sort[:4]]
          ind_end_face = ind_corner[ind_sort[-4:]]
        except ValueError:
          if warn_flag:
            print(f"For wall element {self.id}, finding bounding box corner indices failed with tol {0.1}, probably the wall is not a rectangular prism, or it starts with an opening")
          # Get index of four corner points of the wall (order first by x, then z, then y)
          ind_sort = self.sort_points_by_local_coordinates(points=points, sort_by_axis="xzy")
          ind_start_face = ind_sort[:4]
          ind_end_face = ind_sort[-4:]
        return ind_start_face, ind_end_face

    def get_index_of_upper_corner_points(self, points=None):
        """ Returns indices of upper corner points of the wall.

          Parameters
          ----------
          points : numpy.ndarray, shape (N, 3), optional
              An array of wall points. If None, uses the wall's own points. Default is None.

          Returns
          -------
          ind_upper_face : numpy.ndarray, shape (4,)
              Indices of the corner points of the upper face of the wall."""
        
        if points is None:
          points = self.points
        ind_corner = self.get_bounding_box_corner_indices(points=points, tol=0.02) 
        ind_sort = self.sort_points_by_coordinates(points=points[ind_corner], sort_by_axis="zxy")
        ind_upper_face = ind_corner[ind_sort[-4:]]
        return ind_upper_face

    def get_upper_face_axis_points(self, points=None):
        """ Returns axis points of the upper face of the wall.

          Parameters
          ----------
          points : numpy.ndarray, shape (N, 3), optional
              An array of wall points. If None, uses the wall's own points. Default is None.

          Returns
          -------
          p : numpy.ndarray, shape (2, 3)
              The start and end points of the upper face axis of the wall."""
        
        if points is None:
          points = self.points
        ind_u = self.get_index_of_upper_corner_points(points=points)
        point_l_u = self.transform_to_local(points[ind_u])
        ind_sort = self.sort_points_by_coordinates(points=point_l_u, sort_by_axis="xyz")

        p1 = np.mean(point_l_u[ind_sort[:2]], axis=0)
        p2 = np.mean(point_l_u[ind_sort[2:4]], axis=0)
        p = np.array([p1, p2])
        p = self.transform_to_global(p)
        return p

    def get_lower_face_axis_points(self, points=None):
        """ Returns axis points of the lower face of the wall.

          Parameters
          ----------
          points : numpy.ndarray, shape (N, 3), optional
              An array of wall points. If None, uses the wall's own points. Default is None.

          Returns
          -------
          p : numpy.ndarray, shape (2, 3)
              The start and end points of the lower face axis of the wall."""
        
        if points is None:
          points = self.points
        ind_start, ind_end = self.get_index_of_corner_points(points=points)
        # Local coords of bottom points of start and end face
        point_b_start = self.transform_to_local(points[ind_start[:2]])
        point_b_end = self.transform_to_local(points[ind_end[:2]])
        p1 = np.mean(point_b_start, axis=0)
        p2 = np.mean(point_b_end, axis=0)

        p = np.array([p1, p2])
        p = self.transform_to_global(p)
        return p

    def get_index_of_opening_points(self):
        """ Returns indices of points corresponding to openings in the wall.

            Returns
            -------
            ind_openings : numpy.ndarray, shape (M,)
                Indices of points corresponding to openings in the wall, or None if no openings are found."""
         
        n = self.points.shape[0]
        if n > 8:
          all_indices = np.arange(n)
          ind_corner = self.get_bounding_box_corner_indices(tol=0.02)
          ind_openings = np.setdiff1d(all_indices, ind_corner)
          ind_openings = self.filter_opening_points(ind_openings)

        else:
          ind_openings=None
        return ind_openings

    def filter_opening_points(self, ind_openings, x_tol=0.01, y_tol=0.01, z_tol=0.01):
        """ Filters opening points to retain only valid rectangular openings.

            Parameters
            ----------
            ind_openings : numpy.ndarray, shape (M,)
                Indices of points corresponding to potential openings in the wall.

            x_tol : float, optional
                Tolerance for grouping x-coordinates. Default is 0.01.

            y_tol : float, optional
                Tolerance for grouping y-coordinates. Default is 0.01.

            z_tol : float, optional
                Tolerance for grouping z-coordinates. Default is 0.01.

            Returns
            -------
            filtered_indices : numpy.ndarray, shape (K,)
                Filtered indices of points corresponding to valid rectangular openings."""
        
        o_points = self.points[ind_openings]
        local_points = self.transform_to_local(o_points)
        ind_sort = self.sort_points_by_coordinates(local_points)
        points = local_points[ind_sort]
        # Round x-values to group similar values
        x_rounded = np.round(points[:, 0] / x_tol) * x_tol
        # Get unique rounded x values and their inverse indices
        x_vals, inv_ind = np.unique(x_rounded, return_inverse=True)
        filtered_ind = []
        for i, x_val in enumerate(x_vals):
          ind_same_x = np.where(inv_ind == i)[0]
          if len(ind_same_x) == 4:
              # Let's check the four points form a rectangular shape
              z = np.round(points[ind_same_x, 2]/z_tol)*z_tol
              y = np.round(points[ind_same_x, 1]/y_tol)*y_tol
              # Count occurrences
              y_unique, y_counts = np.unique(y, return_counts=True)
              z_unique, z_counts = np.unique(z, return_counts=True)
              is_paired = (len(y_unique) == 2 and np.all(y_counts == 2) and
                        len(z_unique) == 2 and np.all(z_counts == 2))
              if is_paired:
                filtered_ind += [ind_sort[int(i)] for i in ind_same_x]
              #TODO: it should be also checked if the groups of two subsequent four points shape a rectangular shape
        
        filtered_indicies = ind_openings[filtered_ind]
        return filtered_indicies

    def orthogonalize_end_faces(self):
        """ Adjusts the end faces of the wall to be orthogonal."""

        # Transform point coordinates to local coordinates
        local_points = self.transform_to_local(self.points)
        # Get index of corner points
        ind_start_face, ind_end_face = self.get_index_of_corner_points(warn_flag=False)
        # Get coordinates of four corner points of the wall
        points_start = local_points[ind_start_face]
        points_end = local_points[ind_end_face]
        # change first (local x) coordinate of wals start points and ends
        local_points[ind_start_face] = self.get_average_x_coords(points_start)
        local_points[ind_end_face] = self.get_average_x_coords(points_end)
        # transform back to global coordinates
        points = self.transform_to_global(local_points)
        self.set_points(points)

    def set_points_without_inner_wall_points(self):
        """ Sets wall points excluding inner wall points.
        
            Returns
            -------
            points : numpy.ndarray, shape (M, 3)
                The wall points excluding inner wall points."""
        
        points_l = self.transform_to_local(self.points)
        min_y = np.min(points_l[:, 1])
        max_y = np.max(points_l[:, 1])
        # Keep only points in the outer surface of the wall
        ind = np.isclose(points_l[:, 1], min_y, atol=0.001) | np.isclose(points_l[:, 1], max_y, atol=0.001)
        points = self.set_points(self.points[ind])
        return points

    def set_openings(self):
        """ Identifies and sets openings in the wall.
        
            Returns
            -------
            openings : SolidObjectSet or None
                A set of solid objects representing openings in the wall, or None if no openings are found."""
        
        if self.openings.objects != []:
            openings = self.openings
            return openings
        ind_openings = self.get_index_of_opening_points()
        if ind_openings is None:
          return None
        n_open = int(ind_openings.shape[0] / 8)

        for i in range(n_open):
            ind_open = ind_openings[8*i:8*i+8]
            self.openings.add_object(SolidObject(self.points[ind_open]))
        openings = self.openings
        return openings

    def get_openings(self):
        """ Returns the set of openings in the wall.
        
            Returns
            -------
            openings : SolidObjectSet or None
                A set of solid objects representing openings in the wall, or None if no openings are found."""
        
        if self.openings.objects != []:
          openings = self.openings
          return openings
        else:
          self.set_openings()
          openings = self.openings
          return openings

    def get_net_volume(self):
        """ Returns the net volume of the wall, accounting for openings.
        
            Returns
            -------
            V : float
                The net volume of the wall after subtracting the volumes of openings."""
        
        # Full wall volume
        V = self.get_volume()
        # Volume of opening
        V_o = 0
        Os = self.get_openings()
        if Os is not None:
          for O_i in Os.objects:
            V_o += self.get_volume(points=O_i.points)
          V = V - V_o
          return V
        return V

    def get_half_net_volumes(self, z_cut=None):
        """ Returns the net volumes of the lower and upper halves of the wall.

          Parameters
          ----------
          z_cut : float, optional
              The height at which to cut the wall into lower and upper halves. If None, cuts at mid-height. Default is None.

          Returns
          -------
          V_b : float
              The net volume of the lower half of the wall.

          V_t : float
              The net volume of the upper half of the wall."""
        
        # z_cut is the height of plane cutting the wall to upper and lower volumes
        if z_cut is None:
          z_max = self.get_top_height()
          z_min = self.get_bottom_height()
          z_cut = (z_max-z_min)/2
          c = 0.5
        else:
          # ratio of volumes
          c = (z_cut-z_min)/(z_max-z_min)
        V = self.get_volume()
        # Wall's bottom volume
        V_b = c*V
        # Wall's top volume
        V_t = (1.-c)*V
        # Subtract area of openings
        # Volume of opening
        V_ob = 0
        V_ot = 0
        Os = self.get_openings()
        if Os is not None:
          for O_i in Os.objects:
            V_o = self.get_volume(points=O_i.points)
            z_o_min = O_i.get_bottom_height()
            z_o_max = O_i.get_top_height()
            if z_cut > z_o_min and z_cut < z_o_max:
              c_o = (z_cut-z_o_min)/(z_o_max-z_o_min)
            elif z_cut>z_o_max:
              c_o = 1.
            elif z_cut<z_o_min:
              c_o = 0
            V_ob += c_o*V_o
            V_ot += (1-c_o)*V_o
          return V_b-V_ob, V_t-V_ot
        return V_b, V_t

    def project_opening_points_to_wall_bottom_and_top(self):
        """ Projects opening points to the wall's bottom and top surfaces.

            Returns
            -------
            points : numpy.ndarray, shape (N, 3) or None
                The wall points with opening points projected to bottom and top surfaces, or None if no openings are found."""
        
        ind_opening = self.get_index_of_opening_points()
        points = self.points.copy()
        # Maybe the next should be the local one, not sure if it can happen that the two does not match????
        z_bottom = self.get_bottom_height()
        z_top = self.get_top_height()
        if ind_opening is None:
            return None
        elif ind_opening.shape[0] % 8 != 0:
            raise ValueError("Number of opening points must be a multiple of 8, because only rectangular openings can be handled now.")
        else:
            n_open = int(ind_opening.shape[0] / 8)
            for i in range(n_open):
                # indices of starting and ending faces of the openings
                ind_start_open_b = ind_opening[8*i:8*i+2]
                ind_start_open_t = ind_opening[8*i+2:8*i+4]
                ind_end_open_b = ind_opening[8*i+4:8*i+6]
                ind_end_open_t = ind_opening[8*i+6:8*i+8]
                 # project coordinates
                points[ind_start_open_b, 2] = z_bottom
                points[ind_start_open_t, 2] = z_top
                points[ind_end_open_b, 2] = z_bottom
                points[ind_end_open_t, 2] = z_top
            return points

    def is_edge_on_wall_top(self, e_points, face_points=None, tol_z=0.02, tol_y=None):
        """ Checks if an edge is on the top face of the wall.

          Parameters
          ----------
          e_points : numpy.ndarray, shape (2, 3)
              The start and end points of the edge to check.

          face_points : numpy.ndarray, shape (M, 3), optional
              An array of points defining the top face of the wall. If None, uses the wall's upper corner points. Default is None.

          tol_z : float, optional
              Tolerance for vertical (Z) alignment check. Default is 0.02.

          tol_y : float, optional
              Tolerance for horizontal (Y) alignment check. If None, no Y alignment check is performed. Default is None.

          Returns
          -------
          is_on_top : bool
              True if the edge is on the top face of the wall, False otherwise.

          contact_info : dict or None
              A dictionary containing contact information if the edge is on the top face, or None otherwise."""
        
        is_on_top = False
        contact_info = None

        # Find the points of the top face of the wall (global coordinate)
        if face_points is None:
          ind = self.get_index_of_upper_corner_points()
          face_points = self.points[ind]
        # Check if vertical (Z) alignment is within tolerance
        if not np.isclose(np.max(face_points[:, 2]), np.min(e_points[:, 2]), atol=tol_z):
            return is_on_top, contact_info
        # Find correct order of upper face points to define polygon
        ind = self.sort_points_by_local_coordinates(face_points, sort_by_axis='xyz')
        # Get polygon of supporting wall (XY only)
        poly_xy = face_points[ind[[0,1,3,2]], :2]
        # Get edge of loading wall (XY only)
        edge_xy = e_points[:, :2]

        # 2) Check if edge intersects polygon (orthogonal or parallel cases)
        intersection_segments, unsupported_segments = \
          get_line_polygon_intersection_and_gaps(edge_xy, poly_xy)
        if intersection_segments:
          if not len(intersection_segments) == 1:
            print(f"Wall {self.id} upper face polygon is note convex")
          try:
            contact_length = np.linalg.norm(intersection_segments[0][1] - intersection_segments[0][0])
          except Exception as e:
            print("Intersectin failed")
            print("Error:", e)
            print (intersection_segments)
            print ("Length", len(intersection_segments))
            print (self.id)
          from_point = np.append(intersection_segments[0][0], e_points[0,2])
          to_point = np.append(intersection_segments[0][1], e_points[1,2])
          ns_segments=[]
          for sec_i in unsupported_segments:
            point_start = np.append(sec_i[0], e_points[0,2])
            point_end = np.append(sec_i[1], e_points[1,2])
            ns_segments.append({"StartPoint": point_start,
                              "EndPoint": point_end})

          is_on_top = True
          contact_info = {
              "ContactLength": contact_length,
              "ContactStart": from_point,
              "ContactEnd": to_point,
              "UnsupportedEdgeSections": ns_segments
          }
          return is_on_top, contact_info

        return is_on_top, contact_info

  ##################################################################
  #                Methods to get mechanical properties
  ##################################################################
    def get_upper_half_weight(self):
        """ Returns the weight of the upper half of the wall.
        
          Returns
          -------
          G_t : float
              The weight of the upper half of the wall."""
        
        V_b, V_t = self.get_half_net_volumes()
        rho = self.properties.density
        G_t = V_t* 9.81 * rho
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        if self.properties.g1 is not None:
                G_t += self.properties.g1 * self.get_dimensions()["Width"]* self.get_dimensions()["Length"]
        if self.properties.g2 is not None:
                G_t += self.properties.g2 * self.get_dimensions()["Width"]* self.get_dimensions()["Length"]
        for cat_i in categories:
          q_i = "q_"+ cat_i
          
          if hasattr(self, q_i) and getattr(self, q_i) is not None:
            val_i = getattr(self, q_i) 
            G_t += self.get_psi_2_factor(cat_i) * val_i
        return G_t 

    def get_lower_half_weight(self):
        """ Returns the weight of the lower half of the wall.
        
          Returns
          -------
          G_b : float
              The weight of the lower half of the wall."""
        
        V_b, V_t = self.get_half_net_volumes()
        rho = self.properties.density
        G_b = V_b* 9.81 * rho
        return G_b     
    
    def get_distributed_loads_from_slabs(self):
        """ Returns distributed loads on the wall from slab partitions.

            Returns
            -------
            p_g1 : float or None
                The permanent distributed load (force/length) from slabs on the wall, or None if undefined.

            p_g2 : float or None
                The non-permanent distributed load (force/length) from slabs on the wall, or None if undefined.

            p_q : dict
                A dictionary with variable distributed loads (force/length) from slabs on the wall for different categories."""
          
        # This method computes a uniform distributed load (force/lenggth) 
        # equally distributed along the wall length acting on the top of the wall

        # Length of the wall
        L = self.get_dimensions()["Length"]
        # Initiate loading on top of the wall distributed along the axis
        p_g1 = 0
        p_g2 = 0
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        p_q = {cat_i: 0 for cat_i in categories}
        # Add load from slabs
        if self.loaded_by_slab_partition != []:
          for load_i in self.loaded_by_slab_partition:
            SP_i = load_i["Object"]
            L_i = load_i["ContactLength"]
            if None not in (p_g1, SP_i.p_g1) and L != 0:
              p_g1 = p_g1 + SP_i.p_g1*L_i/L   #force/m linear loading
            else:
              p_g1 = None
            if SP_i.p_g2 is not None and L != 0:
              p_g2 = p_g2 + SP_i.p_g2*L_i/L
            
            # Add different category q values
            for cat_j, val_j in SP_i.p_q.items():
              if val_j is not None and L != 0:
                p_q[cat_j] += val_j*L_i/L

        return p_g1, p_g2, p_q

    def set_transmitting_distributed_loads(self):
        """ Sets the distributed loads transmitted by the wall.
        
            This method computes and sets the total distributed loads (permanent, non-permanent, and variable)
            acting on the wall, considering contributions from slabs and above walls, as well as the wall's own weight
            and any additional loads defined in its properties."""
        # Compute loads from slabs
        p_g1_S, p_g2_S, p_q_S = self.get_distributed_loads_from_slabs()
        # Compute loads from above walls
        p_g1_W, p_g2_W, p_q_W = self.get_distributed_loads_from_above_walls()
        # Compute wall's total weight
        V = self.get_net_volume()
        rho = self.properties.density
        L = self.get_dimensions()["Length"]
        # Some up structural pernament loads
        if None not in (p_g1_S, p_g1_W, rho) and L != 0:
            self.p_g1 = p_g1_S + p_g1_W + 9.81 * V * rho / L
            # add additional wall loads if defined by the IFC
            if self.properties.g1 is not None and self.p_g1 is not None:
                self.p_g1 += self.properties.g1 * self.get_dimensions()["Width"]
        else:
            self.p_g1 = None
        # Some up structural non-pernament loads
        self.p_g2 = 0
        if p_g2_S is not None:
          self.p_g2 +=  p_g2_S
        if p_g2_W is not None:
          self.p_g2 += p_g2_W
        if self.properties.g2 is not None:
          self.p_g2 += self.properties.g2 * self.get_dimensions()["Width"]

        # Initialize dictionary of variable loads
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.p_q = {cat_i:0. for cat_i in categories}
        # Sum up loads from slabs
        for cat_i, val_i in p_q_S.items():
          if val_i is not None:
              self.p_q[cat_i] += val_i
        # From upper walls
        for cat_i, val_i in p_q_W.items():
          if val_i is not None:
            self.p_q[cat_i] += val_i
        # From direct wall load
        for cat_i, val_i in self.properties.get_q_dict().items():
          if val_i is not None:
            self.p_q[cat_i] += val_i * self.get_dimensions()["Width"]
  

    def get_distributed_loads_from_above_walls(self):
        """ Returns distributed loads on the wall from above walls.
        
            Returns
            -------
            p_g1 : float or None
                The permanent distributed load (force/length) from above walls on the wall, or None if undefined.

            p_g2 : float or None
                The non-permanent distributed load (force/length) from above walls on the wall, or None if undefined.

            p_q : dict
                A dictionary with variable distributed loads (force/length) from above walls on the wall for different categories."""
        
        loads = self.loaded_by_walls
        # Initiate loading on top of the wall distributed along the axis
        p_g1 = 0
        p_g2 = 0
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        p_q = {cat_i:0. for cat_i in categories}
        # Add load from walls
        if loads != []:
          L = self.get_dimensions()["Length"]
          for load_i in loads:
            W_i = load_i["Object"]
            L_i = load_i["ContactLength"]
            if p_g1 is not None:
              p_g1 =  p_g1 + W_i.p_g1 * L_i / L if W_i.p_g1 is not None else None
            if p_g2 is not None:
              p_g2 =  p_g2 + W_i.p_g2 * L_i / L if W_i.p_g2 is not None else None
            for cat_i, val_i in W_i.p_q.items():
              if val_i is not None:
                p_q[cat_i] += val_i * L_i / L 
        return p_g1, p_g2, p_q

    def get_limit_state_stress(self, limit_state="SLE"):
        """ Returns the stress in the wall for a given limit state.
        
            Parameters
            ----------
            limit_state : str, optional
                The limit state for which to calculate the stress. Must be either "SLE" (Serviceability Limit State) or "SLU" (Ultimate Limit State). Default is "SLE".

            Returns
            -------
            sigma : float or None
                The calculated stress in the wall for the specified limit state, or None if the required distributed loads are undefined."""
        
        if None in (self.p_g1, self.p_g2, self.p_q):
          return None
        w = self.get_dimensions()["Width"]
        sigma_g1 = self.p_g1/w
        sigma_g2 = self.p_g2/w
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        sigma_q = {cat_i: self.p_q[cat_i]/w for cat_i in categories}
        
        if limit_state == "SLU":
          sigma = 1.1 * sigma_g1 + 1.3 * sigma_g2 
          for cat_i, val_i in sigma_q.items():
            sigma += 1.5 * val_i
        elif limit_state == "SLE":
          sigma =  1 * (sigma_g1 + sigma_g2)
          for cat_i, val_i in sigma_q.items():
            sigma += self.get_psi_2_factor(cat_i) * val_i
        else:
          raise(ValueError("Limit state must be SLE or SLU"))
        return sigma



  ##################################################################
  #Method related to contacts with other elements, wall partitions
  ##################################################################

    def get_WP_support_info(self, points_start, points_end):
        """ Returns support information for a wall partition.

          Parameters
          ----------
          points_start : numpy.ndarray, shape (M, 3)
              An array of points defining the starting face of the wall partition.

          points_end : numpy.ndarray, shape (M, 3)
              An array of points defining the ending face of the wall partition.

          Returns
          -------
          supporting_WPs : list
              A list of wall partitions that support the wall partition.

          contact_length : float
              The total contact length between the wall partition and its supporting walls.

          unsupported_sections : list
              A list of unsupported sections of the wall partition, where each section is represented as a tuple with start and end points."""
        
        WP_points = np.concatenate((points_start, points_end), axis=0)
        WP_axis = self.get_lower_face_axis_points(points=WP_points)
        WP_length = np.linalg.norm(points_end - points_start)
        contact_length = 0
        unsupported_sections = []  #list for final sections, elements of list are tuple with start point and end point
        supporting_WPs = []
        for sup_i in self.supported_by_walls:
          W_i = sup_i["Object"]
          for WP_ij in W_i.wall_partitions.objects:
            ind = W_i.get_index_of_upper_corner_points(points=WP_ij.points)
            poly_points = WP_ij.points[ind]
            is_supported, contact_info = W_i.is_edge_on_wall_top(WP_axis, face_points=poly_points)
            if is_supported:
              supporting_WPs.append(WP_ij)
              if contact_info["UnsupportedEdgeSections"] == []:   # if whole edge is supoprted
                return supporting_WPs, contact_info["ContactLength"], contact_info["UnsupportedEdgeSections"]
              else:
                contact_length += contact_info["ContactLength"]
                if np.isclose(contact_length, WP_length):
                  return supporting_WPs, contact_length, []
                else:
                  # check if new unsupported edge overlaps with old one
                  not_in_contact = []   #store start and end points of unsupported sections
                  for sec_i in contact_info["UnsupportedEdgeSections"]:
                    start_point = sec_i["StartPoint"]
                    end_point = sec_i["EndPoint"]
                    x_start = self.transform_to_local(start_point)[0]
                    x_end = self.transform_to_local(end_point)[0]
                    not_in_contact.append((x_start, x_end))
                  if unsupported_sections == []:   # if this is the first unsupported section stored
                    unsupported_sections = not_in_contact
                  else:   # otherwise, check overlapping sections that are still not supported by anything
                    overlapping_sections = []
                    for sec_k in not_in_contact:
                      for sec_j in unsupported_sections:
                        overlap = segments_overlap(sec_k, sec_j)
                        if overlap is not None:
                          overlapping_sections.append(overlap)
                    unsupported_sections = overlapping_sections
        return supporting_WPs, contact_length, unsupported_sections

    def register_wall_partition(self, points_start, points_end, linked_WP_points_begin=None, linked_WP_points_end=None ):
        """ Registers a wall partition for the wall.

          Parameters
          ----------
          points_start : numpy.ndarray, shape (M, 3)
              An array of points defining the starting face of the wall partition.

          points_end : numpy.ndarray, shape (M, 3)
              An array of points defining the ending face of the wall partition.

          linked_WP_points_begin : numpy.ndarray, shape (K, 3), optional
              An array of points defining a linked structural unit at the beginning of the wall partition. Default is None.

          linked_WP_points_end : numpy.ndarray, shape (K, 3), optional
              An array of points defining a linked structural unit at the end of the wall partition. Default is None.

          Returns
          -------
          WPs : list or None
              A list of registered wall partitions, or None if no partition was created."""
              
        n = len(self.wall_partitions.objects)

        if self.floor == 0:
          points = np.concatenate((points_start, points_end), axis=0)
          linked_WP_points = [p for p in [linked_WP_points_begin, linked_WP_points_end] if p is not None]
          try:
            WP = StructuralUnit(
                points, f"{self.id}_{n+1}",
                linked_struct_points=linked_WP_points,
                struct_type="LoadBearing")
            self.wall_partitions.add_object(WP)
            WPs = [WP]
            return WPs
          except QhullError:
            print(f"Error computing convex hull for wall partition of ground floor element {self.id}")

        supporting_WPs, contact_length, unsupported_sections = self.get_WP_support_info(points_start, points_end)

        if supporting_WPs == []:  # If WP is not supported and not on ground floor, don't create WP
          return None
        else:
          # First merge unsupported sections if the distance between unsupported sections is very small
          # make sure, that the sections are ordered
          unsupported_sections = self.merge_sections_and_ignore_small_ones(unsupported_sections, tol_merge=0.1, tol_ignore=0.2)

          if unsupported_sections == []:
            linked_WP_points = [p for p in [linked_WP_points_begin, linked_WP_points_end] if p is not None]
            points = np.concatenate((points_start, points_end), axis=0)
            try:
              WP = StructuralUnit(
                  points, f"{self.id}_{n+1}",
                  linked_struct_points=linked_WP_points,
                  struct_type="LoadBearing",
                  supporting_elements=supporting_WPs)
              self.wall_partitions.add_object(WP)
            except QhullError:
              print(f"Error computing convex hull for wall partition of element {self.id}")
          else:   # if WP has to be partitioned because of unsupported sections
            # Exclude unsupported sections
            WPs = []
            points_start_l = self.transform_to_local(points_start)
            points_end_l = self.transform_to_local(points_end)
            x_start = np.min(points_start_l[:, 0])
            x_end = np.max(points_end_l[:, 0])
            for i, sec_i in enumerate(unsupported_sections):
              # if start point of WP is no close to the next hole create WP from x_start to beginning of next hole
              if not np.isclose(x_start, sec_i[0], atol=0.07):  #if not starting with a hole
                if i == 0:  #attach linked above window element to first element
                  linked_WP_points = linked_WP_points_begin
                else:
                  linked_WP_points = None
                p_s_l = points_start_l.copy()
                p_s_l[:,0] = x_start
                p_e_l = points_end_l.copy()
                p_e_l[:,0] = sec_i[0]
                points = self.transform_to_global(np.concatenate((p_s_l, p_e_l), axis=0))
                sup_WPs, _, _ = self.get_WP_support_info(points[:4], points[-4:])
                WP = StructuralUnit(
                    points, f"{self.id}_{n+1}",
                    linked_struct_points=linked_WP_points,
                    struct_type="LoadBearing",
                    supporting_elements=sup_WPs)
                self.wall_partitions.add_object(WP)
                WPs.append(WP)
                n += 1
              elif i == 0 and linked_WP_points_begin is not None:
                print(f"Wall part above window of wall {self.id} can not be supported by the beginning of wall part")
              x_start = sec_i[1]
            if not np.isclose(x_start, x_end, atol=0.07):
              linked_WP_points = linked_WP_points_end
              p_s_l = points_start_l.copy()
              p_s_l[:,0] = x_start
              points = self.transform_to_global(np.concatenate((p_s_l, points_end_l), axis=0))
              sup_WPs, _, _ = self.get_WP_support_info(points[:4], points[-4:])
              WP = StructuralUnit(
                  points, f"{self.id}_{n+1}",
                  linked_struct_points=linked_WP_points_end,
                  struct_type="LoadBearing",
                  supporting_elements=sup_WPs)  #TODO: this is not really true, it has to be checked whether in this partitioned WP which WP is supporting
              self.wall_partitions.add_object(WP)
              WPs.append(WP)
            elif linked_WP_points_end is not None:
              print(f"Wall part above window of wall {self.id} can not be supported by the end of wall part")
            return WPs

    def merge_sections_and_ignore_small_ones(self, sections, tol_merge=0.05, tol_ignore=0.2):
        """ Merges close sections and ignores small ones.

            Parameters
            ----------
            sections : list
                A list of sections represented as tuples with start and end points.

            tol_merge : float, optional
                Tolerance for merging close sections. Default is 0.05.

            tol_ignore : float, optional
                Tolerance for ignoring small sections. Default is 0.2.

            Returns
            -------
            new_sections : list
                A list of merged and filtered sections."""
        
        if len(sections)>1:
          # Sort by section start (made for unsupported sections of wall parts)
          sections.sort(key=lambda interval: interval[0])
          # Merge sections if gap between them too small
          new_sections = []
          current_start, current_end = sections[0]

          for next_start, next_end in sections[1:]:
              # If the gap between current_end and next_start is less than or equal to tolerance
              if next_start - current_end <= tol_merge:
                  # Extend the current interval
                  current_end = max(current_end, next_end)
              else:
                  # Save the current interval and start a new one
                  new_sections.append((current_start, current_end))
                  current_start, current_end = next_start, next_end

          # Append the last interval
          new_sections.append((current_start, current_end))
        else:
          new_sections = sections
        # Get rid of too small sections
        if tol_ignore is not None:
          new_sections = [s for s in new_sections if s[1]-s[0] > tol_ignore]
        return new_sections

    def set_wall_partitions(self):
        """ Sets wall partitions for the wall.

            This method identifies and creates wall partitions based on the wall's geometry and openings.
            It handles both walls with and without openings, creating appropriate structural units for each partition.

            Returns
            -------
            wall_partitions : StructuralUnitSet
                A set of wall partitions created for the wall."""
        
        if self.wall_partitions.objects != []:
          wall_partitions = self.wall_partitions
          return wall_partitions
        # Get corner nodes and nodes that define openings
        # print(f"Creating WPs for wall {self.id}")
        ind_start_face, ind_end_face = self.get_index_of_corner_points()
        ind_openings = self.get_index_of_opening_points()
        # No openings
        if ind_openings is None:
          points_start = self.points[ind_start_face]
          points_end = self.points[ind_end_face]
          self.register_wall_partition(points_start, points_end)
          wall_partitions = self.wall_partitions
          return wall_partitions

        # Handle openings
        if ind_openings.shape[0] % 8 != 0:
            raise ValueError(f"Number of opening points of wall {self.id} must be a multiple of 8, because only rectangular openings can be handled now.")

        n_open = int(ind_openings.shape[0] / 8)
        projected_points = self.project_opening_points_to_wall_bottom_and_top()

        for i in range(n_open):
            ind_start_open = ind_openings[8*i:8*i+4]
            ind_end_open = ind_openings[8*i+4:8*i+8]

            # Wall part before opening
            if i == 0:
                points_start = self.points[ind_start_face]
                # no linked points at the start
                l_points_begin = None
            else:
                points_start = projected_points[prev_ind_end_open]
                l_points_begin = np.concatenate((p_e_o_t_half, p_e_p_t_half, p_e_o_t, p_e_p_t), axis=0)

            points_end = projected_points[ind_start_open]

            # Coordinates of vertices of cuboid above opening
            p_s_o_t = self.points[ind_start_open[2:4]]
            p_s_p_t = projected_points[ind_start_open[2:4]]
            p_e_o_t = self.points[ind_end_open[2:4]]
            p_e_p_t = projected_points[ind_end_open[2:4]]

            # Coordinates in the plane halfing the part above opening
            p_e_o_t_half = (p_s_o_t + p_e_o_t) / 2
            p_e_p_t_half = (p_s_p_t + p_e_p_t) / 2

            # Add loading cuboid vertices of part above actual opening
            if (i == n_open-1) and (p_e_o_t[0,0] ==  self.points[ind_end_face][0,0]):
               # for the last opening, if there is no wall after opening
               l_points_end = np.concatenate((p_s_o_t, p_s_p_t, p_e_o_t, p_e_p_t), axis=0)
            else:
               l_points_end = np.concatenate((p_s_o_t, p_s_p_t, p_e_o_t_half, p_e_p_t_half), axis=0)

            # Create load bearing partition with added loading section
            if i == 0 and np.isclose(self.transform_to_local(points_start)[0, 0],self.transform_to_local(points_end)[0,0], atol=0.05): #for first wall part, if length of the wall is 0 
              p_e_o_t_half = p_s_o_t  # then put whole loading part on the continuing part
              p_e_p_t_half = p_s_p_t
            else:
              self.register_wall_partition(
                  points_start, points_end,
                  linked_WP_points_begin=l_points_begin,
                  linked_WP_points_end=l_points_end)

            # Store index of opening for next loop
            if n_open > 1:
                prev_ind_end_open = ind_end_open

        # Last wall segment after final opening
        points_start = projected_points[ind_end_open]
        points_end = self.points[ind_end_face]
        p_trans_elem = np.concatenate((p_e_o_t_half, p_e_p_t_half, p_e_o_t, p_e_p_t), axis=0)
        if not np.isclose(self.transform_to_local(points_start)[0, 0],self.transform_to_local(points_end)[0,0], atol=0.05):  # if there is wall part after opening
          self.register_wall_partition(
              points_start, points_end,
              linked_WP_points_begin=p_trans_elem)
        wall_partitions = self.wall_partitions
        return wall_partitions
    
    def get_wall_partitions(self):
        """ Returns the wall partitions for the wall.

            Returns
            -------
            wall_partitions : StructuralUnitSet
                A set of wall partitions created for the wall."""
        
        if self.wall_partitions.objects == []:
          self.set_wall_partitions()
        wall_partitions = self.wall_partitions
        return wall_partitions

 
    def __repr__(self):
        """ Returns a string representation of the Wall object."""
        
        return f"<Wall #{self.ifc_obj.id()}>"
