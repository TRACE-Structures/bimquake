# Define class of wall created from the IFC product
from .IFC_objects import IfcObject
from .structural_partitions import StructuralUnitSet, WallPartition
import numpy as np
from .simple_objects import SolidObject, SolidObjectSet
from scipy.spatial import QhullError
from .utils import compute_limit_state_values, get_line_polygon_intersection_and_gaps, segments_overlap
from .frame_element import MaterialProperties, Linkage, Linkages

class Wall(IfcObject):
    """ Class representing a wall created from the IFC product. 
    
        Attributes
        ----------
        wall_partitions : StructuralUnitSet
            The set of wall partitions belonging to this wall.
        supported_by_walls : list
            The list of walls supporting this wall.
        loaded_by_slab_partition : list
            The list of slab partitions loading this wall.
        loaded_by_walls : list
            The list of walls loading this wall.
        openings : SolidObjectSet
            The set of openings in this wall. """
    
    def __init__(self, ifc_object):
        """ Initializes a Wall instance.

            Parameters
            ----------
            ifc_object : IfcObject
                The IFC object representing the wall. """
        
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
                A dictionary containing information about the supporting wall contact. """
        
        self.supported_by_walls.append(contact_dict)

    def add_loading_slab_partition(self, contact_dict):
        """ Adds a slab partition that loads this wall.

            Parameters
            ----------
            contact_dict : dict
                A dictionary containing information about the slab partition contact. """
        
        self.loaded_by_slab_partition.append(contact_dict)

    def add_loading_wall(self, contact_dict):
        """ Adds a wall that loads this wall.

            Parameters
            ----------
            contact_dict : dict
                A dictionary containing information about the loading wall contact. """
        
        self.loaded_by_walls.append(contact_dict)

  ##################################################################
  #                Methods to get geometrical properties
  ##################################################################
    def get_average_x_coords(self, coords):
        """ Adjusts the x-coordinates of the given points to their average value.

            Parameters
            ----------
            coords : array-like
                The coordinates of the points.

            Returns
            -------
            coords : array-like
                The coordinates with adjusted x-values. """
        
        if not np.allclose(coords[:,0], coords[0,0]):
          coords[:, 0] = np.mean(coords[:, 0])
        return coords

    def get_index_of_corner_points(self, points=None, warn_flag=True):
      """ Gets the indices of the corner points of the wall.

          Parameters
          ----------
          points : array-like, optional
              The coordinates of the wall points (default is None, which uses self.points).
          warn_flag : bool, optional
              Whether to print a warning if finding the bounding box corner indices fails (default is True).

          Returns
          -------
          ind_start_face : array-like
              The indices of the corner points of the start face.
          ind_end_face : array-like
              The indices of the corner points of the end face. """
      
      if points is None:
        points = self.points
      # Get index of four corner points of the wall (order first by x, then z, then y)
      #ind_sort = self.sort_points_by_local_coordinates(points=points, sort_by_axis="xzy")
      #ind_start_face = ind_sort[:4]
      #ind_end_face = ind_sort[-4:]
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
       """ Gets the indices of the upper corner points of the wall.

           Parameters
           ----------
           points : array-like, optional
               The coordinates of the wall points (default is None, which uses self.points).

           Returns
           -------
           ind_upper_face : array-like
               The indices of the corner points of the upper face. """
       
       if points is None:
         points = self.points
       ind_corner = self.get_bounding_box_corner_indices(points=points, tol=0.02) 
       ind_sort = self.sort_points_by_coordinates(points=points[ind_corner], sort_by_axis="zxy")
       ind_upper_face = ind_corner[ind_sort[-4:]]
       return ind_upper_face

    def get_upper_face_axis_points(self, points=None):
      """ Gets the axis points of the upper face of the wall.

          Parameters
          ----------
          points : array-like, optional
              The coordinates of the wall points (default is None, which uses self.points).

          Returns
          -------
          axis_points : array-like
              The coordinates of the axis points of the upper face. """
      
      if points is None:
        points = self.points
      ind_u = self.get_index_of_upper_corner_points(points=points)
      point_l_u = self.transform_to_local(points[ind_u])
      ind_sort = self.sort_points_by_coordinates(points=point_l_u, sort_by_axis="xyz")

      p1 = np.mean(point_l_u[ind_sort[:2]], axis=0)
      p2 = np.mean(point_l_u[ind_sort[2:4]], axis=0)
      p = np.array([p1, p2])
      axis_points = self.transform_to_global(p)
      return axis_points

    def get_lower_face_axis_points(self, points=None):
      """ Gets the axis points of the lower face of the wall.

          Parameters
          ----------
          points : array-like, optional
              The coordinates of the wall points (default is None, which uses self.points).

          Returns
          -------
          axis_points : array-like
              The coordinates of the axis points of the lower face. """
      
      if points is None:
        points = self.points
      ind_start, ind_end = self.get_index_of_corner_points(points=points)
      # Local coords of bottom points of start and end face
      point_b_start = self.transform_to_local(points[ind_start[:2]])
      point_b_end = self.transform_to_local(points[ind_end[:2]])
      p1 = np.mean(point_b_start, axis=0)
      p2 = np.mean(point_b_end, axis=0)

      p = np.array([p1, p2])
      axis_points = self.transform_to_global(p)
      return axis_points

    def get_index_of_opening_points(self):
      """ Gets the indices of the opening points of the wall.

          Returns
          -------
          ind_openings : array-like
              The indices of the opening points of the wall. """
      
      n = self.points.shape[0]
      if n > 8:
        all_indices = np.arange(n)
        ind_corner = self.get_bounding_box_corner_indices(tol=0.02)
        ind_openings = np.setdiff1d(all_indices, ind_corner)
        ind_openings = self.filter_opening_points(ind_openings)
        # Sort array by x then by z, then by y
        # ind_sort = self.sort_points_by_local_coordinates(self.points)
        # Sort array by x then by z, then by y
      #  ind_sort = self.sort_points_by_local_coordinates(sort_by_axis="xzy")
      #  ind_openings = ind_sort[4:-4]
      #else:
      #  ind_openings = None
      else:
        ind_openings=None
      return ind_openings

    def filter_opening_points(self, ind_openings, x_tol=0.01, y_tol=0.01, z_tol=0.01):
        """ Filters the opening points of the wall based on their local coordinates.

            Parameters
            ----------
            ind_openings : array-like
                The indices of the opening points of the wall.
            x_tol : float, optional
                The tolerance for grouping x coordinates (default is 0.01).
            y_tol : float, optional
                The tolerance for grouping y coordinates (default is 0.01).
            z_tol : float, optional
                The tolerance for grouping z coordinates (default is 0.01).

            Returns
            -------
            filtered_openings : array-like
                The filtered indices of the opening points that form rectangular shapes. """
        
        o_points = self.points[ind_openings]
        local_points = self.transform_to_local(o_points)
        ind_sort = self.sort_points_by_coordinates(local_points)
        points = local_points[ind_sort]
        # Round x-values to group similar values
        x_rounded = np.round(points[:, 0] / x_tol) * x_tol
        # Get unique rounded x values and their inverse indices
        x_vals, inv_ind = np.unique(x_rounded, return_inverse=True)
        filtered_indices = []
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
                filtered_indices += [ind_sort[int(i)] for i in ind_same_x]
              #TODO: it should be also checked if the groups of two subsequent four points shape a rectangular shape 
        filtered_openings = ind_openings[filtered_indices]
        return filtered_openings

    def orthogonalize_end_faces(self):
        """ Orthogonalizes the end faces of the wall by adjusting the local x-coordinates of the corner points. """

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
        """ Sets the points of the wall without the inner wall points, keeping only the outer surface points. 
        
            Returns
            -------
            points : array-like
                The points of the wall without the inner wall points. """
        
        points_l = self.transform_to_local(self.points)
        min_y = np.min(points_l[:, 1])
        max_y = np.max(points_l[:, 1])
        # Keep only points in the outer surface of the wall
        ind = np.isclose(points_l[:, 1], min_y, atol=0.001) | np.isclose(points_l[:, 1], max_y, atol=0.001)
        points = self.set_points(self.points[ind])
        return points

    def set_openings(self):
        """ Sets the openings of the wall if they are not already set.

            Returns
            -------
            openings : SolidObjectCollection
                The collection of openings of the wall. """
        
        if self.openings.objects != []:
            return self.openings
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
        """ Gets the openings of the wall.

            Returns
            -------
            openings : SolidObjectCollection
                The collection of openings of the wall. """
        
        if self.openings.objects != []:
          openings = self.openings
        else:
          self.set_openings()
          openings = self.openings
        return openings


        #for i in range(n_open):
        #    ind_open = ind_openings[8*i:8*i+8]
        #    self.openings.add_object(SolidObject(self.points[ind_open]))
        # return self.openings

    def get_net_volume(self):
        """ Gets the net volume of the wall, subtracting the volume of openings.

            Returns
            -------
            net_volume : float
                The net volume of the wall after subtracting the volume of openings. """
        
        # Full wall volume
        V = self.get_volume()
        # Volume of opening
        V_o = 0
        Os = self.get_openings()
        if Os is not None:
          for O_i in Os.objects:
            V_o += self.get_volume(points=O_i.points)
          net_volume = V-V_o
          return net_volume
        
        net_volume = V
        return net_volume
    
    def get_half_net_volumes(self, z_cut=None):
        """ Gets the net volumes of the upper and lower halves of the wall, optionally cutting at a specified height.

            Parameters
            ----------
            z_cut : float, optional
                The height at which to cut the wall. If None, the wall is cut at its midpoint.

            Returns
            -------
            V_b : float
                The net volume of the bottom half of the wall.
            V_t : float
                The net volume of the top half of the wall. """
        
      # z_cut is the height of plane cutting the wall to upper and lower volumes
        if z_cut is None:
          z_max = self.get_top_height()
          z_min = self.get_bottom_height()
          z_cut = (z_max-z_min)/2 + z_min
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
        """ Projects the opening points of the wall to the bottom and top faces of the wall.

            Returns
            -------
            points : np.ndarray
                The coordinates of the wall points with the opening points projected to the bottom and top faces.
                Returns None if there are no opening points. """
        
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
        """ Checks if a given edge is on the top face of the wall.

            Parameters
            ----------
            e_points : np.ndarray
                The coordinates of the edge points to check.
            face_points : np.ndarray, optional
                The coordinates of the top face points of the wall. If None, the top face points are obtained from the wall.
            tol_z : float, optional
                The tolerance for Z alignment. Default is 0.02.
            tol_y : float, optional
                The tolerance for Y alignment. Default is None.

            Returns
            -------
            is_on_top : bool
                True if the edge is on the top face of the wall, False otherwise.
            contact_info : dict or None
                Information about the contact if the edge is on the top face, None otherwise. """
        
        # Find the points of the top face of the wall (global coordinate)
        if face_points is None:
          ind = self.get_index_of_upper_corner_points()
          face_points = self.points[ind]
        # Check if vertical (Z) alignment is within tolerance
        if not np.isclose(np.max(face_points[:, 2]), np.min(e_points[:, 2]), atol=tol_z):
            return False, None
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
          ns_segments=[];
          for sec_i in unsupported_segments:
            point_start = np.append(sec_i[0], e_points[0,2])
            point_end = np.append(sec_i[1], e_points[1,2])
            ns_segments.append({"StartPoint": point_start,
                              "EndPoint": point_end})

          
          contact_info = {
              "ContactLength": contact_length,
              "ContactStart": from_point,
              "ContactEnd": to_point,
              "UnsupportedEdgeSections": ns_segments
          }
          return True, contact_info

        return False, None

  ##################################################################
  #                Methods to get mechanical properties
  ##################################################################
    def get_upper_half_weight(self):
        """ Computes the weight of the upper half of the wall.

            Returns
            -------
            G_t : float
                The weight of the upper half of the wall. """
        
        V_b, V_t = self.get_half_net_volumes()
        rho = self.properties.density
        G_t = V_t* 9.81 * rho
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        A =  self.get_dimensions()["Width"]* self.get_dimensions()["Length"]
        if self.properties.g1 is not None:
                G_t += self.properties.g1 * A
        if self.properties.g2 is not None:
                G_t += self.properties.g2 * A
        for cat_i in categories:
          q_i = "q_"+ cat_i
          
          if hasattr(self, q_i) and getattr(self, q_i) is not None:
            val_i = getattr(self, q_i) * A
            G_t += self.get_psi_2_factor(cat_i) * val_i
        return G_t 

    def get_lower_half_weight(self):
        """ Computes the weight of the lower half of the wall.

            Returns
            -------
            G_b : float
                The weight of the lower half of the wall. """
        
        V_b, V_t = self.get_half_net_volumes()
        rho = self.properties.density
        G_b = V_b* 9.81 * rho
        return G_b     
    
    def get_distributed_loads_from_slabs(self):
        """ Computes the distributed loads on the wall from the slabs above it.

            Returns
            -------
            p_g1 : float
                The distributed load from g1 on the top of the wall.
            p_g2 : float
                The distributed load from g2 on the top of the wall.
            p_q : dict
                The distributed loads from different q categories on the top of the wall. """
        
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
        """ Sets the transmitting distributed loads on the wall from slabs and above walls. """

        # Compute loads from slabs
        p_g1_S, p_g2_S, p_q_S = self.get_distributed_loads_from_slabs()
        # Compute loads from above walls
        p_g1_W, p_g2_W, p_q_W = self.get_distributed_loads_from_above_walls()
        # Set loading on top of the wall
        if None not in (p_g1_S, p_g1_W):
            self.p_g1_top = p_g1_S + p_g1_W
            # add additional wall loads if defined by the IFC
            if self.properties.g1 is not None and self.p_g1_top is not None:
                self.p_g1_top += self.properties.g1 * self.get_dimensions()["Width"]
        else:
            self.p_g1_top = None
        # Compute wall's total weight
        V = self.get_net_volume()
        rho = self.properties.density
        L = self.get_dimensions()["Length"]
        # Some up structural pernament loads

        if None not in (self.p_g1_top, rho) and L != 0:
            self.p_g1 = self.p_g1_top + 9.81 * V * rho / L
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

    #----------------------------------------------------------------------
    def set_stress_in_wall_partitions(self):
        """ Sets the stress in the wall partitions based on the applied loads. 

            This method calculates the normal stresses at the top and half-height of each wall partition
            considering the permanent and variable loads applied to the wall.

            The computed stresses are then assigned to the respective wall partitions. """
    
        # Wall loading at the top of the wall [N/m]
        g1 = self.p_g1_top
        g2 = self.p_g2
        q = self.p_q
        # wall density
        rho = self.properties.density
        
        # wall partitions
        WPs = self.get_wall_partitions()
        if WPs is not None or not WPs.objects==[]:
            for i, WP_i in enumerate(WPs.objects):   # loop over wall partitions
                WP_points = WP_i.points
                # Get geometrical properties of WP
                dim_i = self.get_dimensions(points=WP_points)
                L_i = dim_i["Length"]
                w_i = dim_i["Width"]
                h_i = dim_i["Height"]
                V_i = L_i*w_i*h_i               
                
                # computed total volume and length of linked parts
                L_LP_i = WP_i.get_total_length_of_linked_parts()
                V_LP_i = WP_i.get_total_volume_of_linked_parts()

                # compute normal stresses at the top of the wall partition
                sigma = {}
                sigma["top"] = {
                  "ULS": compute_limit_state_values(g1, g2, q, limit_state = "ULS")/w_i,
                  "SLS": compute_limit_state_values(g1, g2, q, limit_state = "SLS")/w_i}
              
                # compute normal stresses at the wall partition half height
                if None not in (g1, rho):
                    # print("Computing force acting on", self.id)
                    G1 = g1*(L_i + L_LP_i) + 9.81 * (V_i/2+V_LP_i) * rho   # half of the WP weight + total weight of linked parts
                else:
                    G1 = None
                G2 = g2*(L_i + L_LP_i)
                
                s_g1 = G1/(w_i*L_i)
                s_g2 = G2/(w_i*L_i)
                s_q = {cat_i: q[cat_i]*(L_i + L_LP_i)/(w_i*L_i) for cat_i in q.keys()}
                
                # Compute limit state sress
                sigma["half-height"] = {
                  "ULS": compute_limit_state_values(s_g1, s_g2, s_q, limit_state = "ULS"),
                  "SLS": compute_limit_state_values(s_g1, s_g2, s_q, limit_state = "SLS")}

                G1 = G1 + 9.81 * (V_i/2) * rho
                s_g1 = G1/(w_i*L_i)

                sigma["bottom"] = {
                  "ULS": compute_limit_state_values(s_g1, s_g2, s_q, limit_state = "ULS"),
                  "SLS": compute_limit_state_values(s_g1, s_g2, s_q, limit_state = "SLS")}

                # Set stress values
                WP_i.set_stress_values(sigma)
    #---------------------------------------------------------------------
  

    def get_distributed_loads_from_above_walls(self):
        """ Computes the distributed loads on the wall from the walls above it.

            Returns
            -------
            p_g1 : float
                The distributed load from g1 on the top of the wall.
            p_g2 : float
                The distributed load from g2 on the top of the wall.
            p_q : dict
                The distributed loads from different q categories on the top of the wall. """
        
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


  ##################################################################
  #Method related to contacts with other elements, wall partitions
  ##################################################################

    def get_WP_support_info(self, points_start, points_end):
        """ Gets the support information for the wall partition defined by the start and end points.

            Parameters
            ----------
            points_start : np.ndarray
                The start points of the wall partition edges.
            points_end : np.ndarray
                The end points of the wall partition edges.

            Returns
            -------
            supporting_WPs : list
                List of wall partitions supporting the current wall partition.
            contact_length : float
                The total contact length of the supporting wall partitions.
            unsupported_sections : list
                List of unsupported sections, each represented as a tuple with start and end points.
            contact_length_list : list
                List of contact lengths for each supporting wall partition. """
        
        WP_points = np.concatenate((points_start, points_end), axis=0)
        WP_axis = self.get_lower_face_axis_points(points=WP_points)
        WP_length = np.linalg.norm(points_end - points_start)
        contact_length = 0
        unsupported_sections = []  #list for final sections, elements of list are tuple with start point and end point
        supporting_WPs = []
        contact_length_list = []
        for sup_i in self.supported_by_walls:
          W_i = sup_i["Object"]
          for WP_ij in W_i.wall_partitions.objects:
            #axis_ij = W_i.get_upper_face_axis_points(points=WP_ij.points)
            ind = W_i.get_index_of_upper_corner_points(points=WP_ij.points)
            poly_points = WP_ij.points[ind]
            is_supported, contact_info = W_i.is_edge_on_wall_top(WP_axis, face_points=poly_points)
            if is_supported:
              supporting_WPs.append(WP_ij)
              contact_length_list.append(contact_info["ContactLength"])
              if contact_info["UnsupportedEdgeSections"] == []:   # if whole edge is supoprted
                return supporting_WPs, contact_info["ContactLength"], contact_info["UnsupportedEdgeSections"], contact_length_list
              else:
                contact_length += contact_info["ContactLength"]
                if np.isclose(contact_length, WP_length):
                  return supporting_WPs, contact_length, [], contact_length_list
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
        return supporting_WPs, contact_length, unsupported_sections, contact_length_list

   
    def register_wall_partition(self, points_start, points_end, linked_WP_points_begin=None, linked_WP_points_end=None ):
        """ Registers a wall partition with the given start and end points.

            Parameters
            ----------
            points_start : np.ndarray
                The start points of the wall partition edges.
            points_end : np.ndarray
                The end points of the wall partition edges.
            linked_WP_points_begin : list, optional
                List of linked wall partition points at the beginning of the wall partition.
            linked_WP_points_end : list, optional
                List of linked wall partition points at the end of the wall partition.

            Returns
            -------
            WPs : list or None
                The registered wall partition object if successfully created, otherwise None. """
              
        n = len(self.wall_partitions.objects)

        if self.floor == 0:
          points = np.concatenate((points_start, points_end), axis=0)
          linked_WP_points = [p for p in [linked_WP_points_begin, linked_WP_points_end] if p is not None]
          try:
            WP = WallPartition(
                points, f"{self.id}_{n+1}",
                self,
                linked_struct_points=linked_WP_points,
                struct_type="LoadBearing")
            self.wall_partitions.add_object(WP)
            return [WP]
          except QhullError:
            print(f"Error computing convex hull for wall partition of ground floor element {self.id}")

        supporting_WPs, contact_length, unsupported_sections, contact_length_list = self.get_WP_support_info(points_start, points_end)

        if supporting_WPs == []:  # If WP is not supported and not on ground floor, don't create WP
          return None
        else:
          # First merge unsupported sections if the distance between unsupported sections is very small
          # make sure, that the sections are ordered
          unsupported_sections = self.merge_sections_and_ignore_small_ones(unsupported_sections, tol_merge=0.1, tol_ignore=0.2)

          if unsupported_sections == []:  # if whole part can be registered (no unsupported sections)
            linked_WP_points = [p for p in [linked_WP_points_begin, linked_WP_points_end] if p is not None]
            points = np.concatenate((points_start, points_end), axis=0)
            # Define supporting linkages
            sup_links = Linkages()
            for WP_sup_i, l_i in zip(supporting_WPs, contact_length_list):
              sup_links.add(Linkage(element=WP_sup_i, contact_length=l_i)) 
            
            try:
              WP = WallPartition(
                  points, f"{self.id}_{n+1}",
                  self,
                  linked_struct_points=linked_WP_points,
                  struct_type="LoadBearing",
                  supporting_elements=sup_links)
              self.wall_partitions.add_object(WP)

              load_links = Linkages()
              for  l_i in contact_length_list:
                load_links.add(Linkage(element=WP, contact_length=l_i)) 

              for WPsup_i in supporting_WPs:
                WPsup_i.add_loading_elements(load_links)
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
              # if start point of WP is not close to the next hole create WP from x_start to beginning of next hole
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
                sup_WPs, _, _, c_lengths = self.get_WP_support_info(points[:4], points[-4:])
                # Define supporting linkages
                sup_links = Linkages()
                for WP_sup_i, l_i in zip(sup_WPs, c_lengths):
                  sup_links.add(Linkage(element=WP_sup_i, contact_length=l_i))
                WP = WallPartition(
                    points, f"{self.id}_{n+1}",
                    self,
                    linked_struct_points=linked_WP_points,
                    struct_type="LoadBearing",
                    supporting_elements=sup_links)
                self.wall_partitions.add_object(WP)
                WPs.append(WP)
                n += 1

                load_links_i = Linkages()
                for l_i in c_lengths:
                  load_links_i.add(Linkage(element=WP, contact_length=l_i))
                for WPsup_i in sup_WPs:
                  WPsup_i.add_loading_elements(load_links_i)
                
                
              elif i == 0 and linked_WP_points_begin is not None:
                print(f"Wall part above window of wall {self.id} can not be supported by the beginning of wall part")
              x_start = sec_i[1]
            if not np.isclose(x_start, x_end, atol=0.07):
              linked_WP_points = linked_WP_points_end
              p_s_l = points_start_l.copy()
              p_s_l[:,0] = x_start
              points = self.transform_to_global(np.concatenate((p_s_l, points_end_l), axis=0))
              sup_WPs, _, _, c_lengths = self.get_WP_support_info(points[:4], points[-4:])
              # Define supporting linkages
              sup_links = Linkages()
              for WP_sup_i, l_i in zip(sup_WPs, c_lengths):
                sup_links.add(Linkage(element=WP_sup_i, contact_length=l_i))
          
              WP = WallPartition(
                  points, f"{self.id}_{n+1}",
                  self,
                  linked_struct_points=linked_WP_points_end,
                  struct_type="LoadBearing",
                  supporting_elements=sup_links)  #TODO: this is not really true, it has to be checked whether in this partitioned WP which WP is supporting
              self.wall_partitions.add_object(WP)
              WPs.append(WP)
              load_links_i = Linkages()
              for l_i in c_lengths:
                load_links_i.add(Linkage(element=WP, contact_length=l_i)) 
              for WPsup_i in sup_WPs:
                  WPsup_i.add_loading_elements([WP])
            elif linked_WP_points_end is not None:
              print(f"Wall part above window of wall {self.id} can not be supported by the end of wall part")
            return WPs


    def merge_sections_and_ignore_small_ones(self, sections, tol_merge=0.05, tol_ignore=0.2):
      """ Merges overlapping or closely spaced sections and ignores sections that are too small.

          Parameters
          ----------
          sections : list
              List of sections, each represented as a tuple with start and end points.
          tol_merge : float, optional
              Tolerance for merging sections that are close to each other. Default is 0.05.
          tol_ignore : float, optional
              Tolerance for ignoring sections that are too small. Default is 0.2.

          Returns
          -------
          new_sections : list
              List of merged and filtered sections. """
      
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
        """ Sets the wall partitions for the current wall.

            Returns
            -------
            wall_partitions : list
                List of wall partitions for the current wall. """
        
        if self.wall_partitions.objects != []:
          return self.wall_partitions
        # Get corner nodes and nodes that define openings
        # print(f"Creating WPs for wall {self.id}")
        ind_start_face, ind_end_face = self.get_index_of_corner_points()  #index of points of starting face of the wall, ending face of the wall
        ind_openings = self.get_index_of_opening_points()
        # No openings
        if ind_openings is None:
          points_start = self.points[ind_start_face]
          points_end = self.points[ind_end_face]
          self.register_wall_partition(points_start, points_end)
          return self.wall_partitions

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
            p_s_o_t = self.points[ind_start_open[2:4]] #points of top of opening start
            p_s_p_t = projected_points[ind_start_open[2:4]] #points of top of opening start projected to the top face of the wall
            p_e_o_t = self.points[ind_end_open[2:4]] #points of top of opening end
            p_e_p_t = projected_points[ind_end_open[2:4]] #points of top of opening end projected to the top face of the wall

            # Coordinates in the plane halfing the part above opening
            p_e_o_t_half = (p_s_o_t + p_e_o_t) / 2
            p_e_p_t_half = (p_s_p_t + p_e_p_t) / 2

            # Add loading cuboid vertices of part above actual opening
            # Local coordinates of end of opening and end of wall (to check if they don't match)
            x_l_open_end = self.transform_to_local(p_e_o_t[0])
            x_l_wall_end = self.transform_to_local(self.points[ind_end_face[0]])
            if (i == n_open-1) and (x_l_open_end[0] ==  x_l_wall_end[0]):  #if last opening's end is at the end of the wall
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
        return self.wall_partitions

    def get_wall_partitions(self):
        """ Gets the wall partitions for the current wall.

            Returns
            -------
            wall_partitions : list
                List of wall partitions for the current wall. """
        
        if self.wall_partitions.objects == []:
          self.set_wall_partitions()
        return self.wall_partitions

    def get_material_properties(self, mu = 1.5):
        """ Gets the material properties for the current wall.

            Parameters
            ----------
            mu : float, optional
                Friction coefficient. Default is 1.5.

            Returns
            -------
            mat_props : MaterialProperties
                Material properties of the current wall. """
        
        rho = self.properties.density
        mat_props = MaterialProperties(
          E = self.properties.E,         # Young modulus
          G = self.properties.G,        # Shear modulus
          #f_u = self.properties.f_u,    # Ultimate tensile strength
          tau = self.properties.f_u/1.5, # Shear strength
          f_m = self.properties.f_c,     # Masonry compressive strength
          nu = self.properties.nu,       # Poisson ratio
          mu = mu,                       # Friction coefficients
          gamma = rho*9.81
        )
        return mat_props

 
    def __repr__(self):
        """ Returns a string representation of the wall object. """
        return f"<Wall #{self.ifc_obj.id()}>"
