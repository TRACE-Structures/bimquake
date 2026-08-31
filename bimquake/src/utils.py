import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon   # for matching wall below wall connections
import numpy as np

def get_psi_2_factor(category):
    """ Return the psi_2 factor based on the category of the load.
    
    Parameters
    ----------
    category : str
        The category of the load (e.g., "A", "B", "C", "D", "E", "F", "G", "H").

    Returns
    -------
    psi_2 : float
        The corresponding psi_2 factor for the given category.
    """

    factor ={
      "A":0.3,
      "B":0.3,
      "C":0.6,
      "D":0.6,
      "E":0.8,
      "F":0.6,
      "G":0.3,
      "H":0.
    }

    psi_2 = factor[category]
    return psi_2


def compute_limit_state_values(g1, g2, q, limit_state="ULS"):
    """ Compute the total load based on the limit state.

        Parameters
        ----------
        g1 : float
            The first permanent load.
        
        g2 : float
            The second permanent load.

        q : dict
            A dictionary containing variable loads with their categories as keys and load values as values.

        limit_state : str, optional
            The limit state to consider ("ULS" or "SLS"). Default is "ULS".

        Returns
        -------
        p : float
            The computed total load based on the specified limit state.
        """
    
    if limit_state == "ULS":
      p = 1.1 * g1 + 1.3 * g2
      for cat_i, val_i in q.items():
        p += 1.5 * val_i
    elif limit_state == "SLS":
      p =  1 * (g1 + g2)
      for cat_i, val_i in q.items():
        p += get_psi_2_factor(cat_i) * val_i
    else:
      raise(ValueError("Limit state must be SLS or ULS"))
    return p


def plot_by_plotly(data, title, showlegend=True):
    """ Create a 3D plot using Plotly.
    
        Parameters
        ----------
        data : list
            A list of Plotly trace objects to be plotted.
            
        title : str
            The title of the plot.
            
        showlegend : bool, optional
            Whether to show the legend in the plot. Default is True.
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            The Plotly figure object containing the 3D plot."""
    
    fig = go.Figure(data=data)
    fig.update_layout(
        title = title,
        showlegend=showlegend,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
          xaxis_title='X',
          yaxis_title='Y',
          zaxis_title='Z',
          aspectmode='data'
          )
        )
    return fig


def get_line_polygon_intersection_and_gaps(line_points, poly_points, tol=1e-3):
    """ Check the intersection of a line with a polygon and return the contact segments and unsupported segments.

        Parameters
        ----------
        line_points : list of list
            A list of two points defining the line (e.g., [[x1, y1], [x2, y2]]).

        poly_points : list of list
            A list of points defining the polygon (e.g., [[x1, y1], [x2, y2], ..., [xn, yn]]).

        tol : float, optional
            Tolerance for geometric calculations. Default is 1e-3.

        Returns
        -------
        contact_segments : list of np.ndarray
            A list of segments where the line intersects the polygon.

        unsupported_segments : list of np.ndarray
            A list of segments of the line that do not intersect the polygon. """
    
    # Create polygon
    poly = Polygon([tuple(pt) for pt in poly_points])
    if not poly.is_valid:
        poly = poly.buffer(0)
        if not poly.is_valid:
            raise ValueError("Invalid polygon that cannot be fixed")

    # Create line
    line = LineString([tuple(pt) for pt in line_points])
    if len(line.coords) != 2:
        raise ValueError("Line must have exactly two distinct points")

    # Check intersection
    inter = line.intersection(poly)
    contact_segments = []
    if not inter.is_empty:
      if inter.geom_type == "LineString":
          contact_segments = [np.array(inter.coords)]
      elif inter.geom_type == "MultiLineString":
          contact_segments = [np.array(seg.coords) for seg in inter.geoms]

    diff = line.difference(poly)
    unsupported_segments = []
    if not diff.is_empty:
      if diff.geom_type == "LineString":
          unsupported_segments.append(np.array(diff.coords))
      elif diff.geom_type == "MultiLineString":
          unsupported_segments.extend([np.array(seg.coords) for seg in diff.geoms])

    # --- Step 3: aligned with polygon sides (if no intersection detected) ---
    if not contact_segments:
        line_xy = np.array(line_points)[:, :2]
        for i in range(len(poly_points)):
            poly_seg = np.vstack([poly_points[i], poly_points[(i + 1) % len(poly_points)]])
            length, overlap = segments_overlap_2d(line_xy, poly_seg, tol=tol)
            if length > 0:
                contact_segments.append(overlap)

                # Add non-overlapping parts to unsupported_segments
                # Before overlap
                if np.linalg.norm(overlap[0] - line_xy[0]) > tol:
                    unsupported_segments.append(np.array([line_xy[0], overlap[0]]))
                # After overlap
                if np.linalg.norm(overlap[1] - line_xy[1]) > tol:
                    unsupported_segments.append(np.array([overlap[1], line_xy[1]]))
    
    return contact_segments, unsupported_segments


def segments_overlap(seg1, seg2, tol=1e-3):
    """
    Return the overlap (start, end) of two 1D segments if they intersect.

    Parameters
    ----------
    seg1 : tuple
        A tuple representing the first segment (start1, end1).

    seg2 : tuple
        A tuple representing the second segment (start2, end2).

    tol : float, optional
        Tolerance for determining overlap. Default is 1e-3.

    Returns
    -------
    overlap : tuple or None
        A tuple representing the overlapping segment (overlap_start, overlap_end) if they intersect,
        or None if there is no overlap.

    """
    start1, end1 = sorted([seg1[0], seg1[1]])
    start2, end2 = sorted([seg2[0], seg2[1]])
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start > overlap_end:
      return None
    else:
      return (overlap_start, overlap_end)


def segments_overlap_2d(seg1, seg2, tol=1e-6):
    """ Check if two 2D segments overlap and return the overlapping segment if they do.

    Parameters
    ----------
    seg1 : np.ndarray
        A (2, 2) array representing the first segment in XY coordinates.

    seg2 : np.ndarray
        A (2, 2) array representing the second segment in XY coordinates.

    tol : float, optional
        Tolerance for determining overlap. Default is 1e-6.

    Returns
    -------
    overlap_length : float
        The length of the overlapping segment. Returns 0 if there is no overlap.
            
    overlap_segment : np.ndarray or None
        A (2, 2) array representing the overlapping segment if they overlap, or None if they do not overlap. """

    p1, p2 = seg1
    q1, q2 = seg2

    v1 = p2 - p1
    v2 = q2 - q1

    L = np.linalg.norm(v1)
    if L < tol:
        return 0.0, None

    # Normalized first segment
    v1n = v1 / L  # unit direction vector

    # --- Parallel check --- 
    #(extend to 3d array, lieing in the XY plane, the cross product will be parallel to the Z axis)
    cross = np.cross(np.append(v1, 0), np.append(v2, 0))
    # if segments are not parallel
    if abs(cross[2]) > tol:   # length of the z coordinate is directly the length of the vector, if equals 0, the two segments are parallel
        return 0.0, None  # no overlap (overlap length=0.)

    # --- Distance between the parallel segments ---
    dist = abs(np.cross(v1, q1 - p1)) / L
    # If two segments are not aligned
    if dist > tol:
        return 0.0, None

    # ---- 1D projection onto the line ----
    t_p1 = 0.0
    t_p2 = L
    t_q1 = np.dot(q1 - p1, v1n)
    t_q2 = np.dot(q2 - p1, v1n)

    overlap = segments_overlap(
        (t_p1, t_p2),
        (t_q1, t_q2),
        tol=tol
    )

    if overlap is None:
        return 0.0, None

    t0, t1 = overlap
    overlap_seg = np.vstack([
        p1 + v1n * t0,
        p1 + v1n * t1
    ])

    return t1 - t0, overlap_seg