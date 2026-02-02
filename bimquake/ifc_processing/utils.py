import plotly.graph_objects as go
from shapely.geometry import LineString, Polygon   # for matching wall below wall connections
import numpy as np

def plot_by_plotly(data, title, showlegend=True):
    """ Creates a Plotly 3D figure with the given data and title.

        Parameters
        ----------
        data : list
            A list of Plotly graph objects to be included in the figure.

        title : str
            The title of the figure.

        showlegend : bool, optional
            Whether to display the legend in the figure. Default is True.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A Plotly figure object containing the provided data and layout."""
    
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
    """ Determine the intersection and unsupported segments of a line with respect to a polygon.

        Parameters
        ----------
        line_points : array-like, shape (2, 2) or (2, 3)
            The start and end points of the line segment.

        poly_points : array-like, shape (N, 2) or (N, 3)
            The vertices of the polygon in order.

        tol : float, optional
            Tolerance for geometric calculations. Default is 1e-3.

        Returns
        -------
        contact_segments : list of np.ndarray
            List of line segments (as arrays of points) that are in contact with the polygon.

        unsupported_segments : list of np.ndarray
            List of line segments (as arrays of points) that are not supported by the polygon."""
    
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
    """ Return the overlap (start, end) of two 1D segments if they intersect.

        Parameters
        ----------
        seg1 : tuple of float
            (start, end) of the first segment.

        seg2 : tuple of float
            (start, end) of the second segment.

        tol : float, optional
            Tolerance for determining overlap. Default is 1e-3.

        Returns
        -------
        overlap : tuple of float or None
            (start, end) of the overlapping segment, or None if there is no overlap. """
    
    overlap = None
    start1, end1 = sorted([seg1[0], seg1[1]])
    start2, end2 = sorted([seg2[0], seg2[1]])
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start > overlap_end:
      return overlap
    else:
      overlap = (overlap_start, overlap_end)
      return overlap

def segments_overlap_2d(seg1, seg2, tol=1e-6):
    """ Return the overlap length and overlapping segment of two 2D segments if they are collinear and overlap.

        Parameters
        ----------
        seg1 : numpy.ndarray, shape (2, 2)
            The start and end points of the first segment.

        seg2 : numpy.ndarray, shape (2, 2)
            The start and end points of the second segment.

        tol : float, optional
            Tolerance for geometric calculations. Default is 1e-6.

        Returns
        -------
        overlap_length : float
            The length of the overlapping segment.

        overlap_segment : numpy.ndarray, shape (2, 2) or None
            The start and end points of the overlapping segment in XY plane, or None if there is no overlap."""
    
    overlap_length = 0.0
    overlap_segment = None
    p1, p2 = seg1
    q1, q2 = seg2

    v1 = p2 - p1
    v2 = q2 - q1

    L = np.linalg.norm(v1)
    if L < tol:
        return overlap_length, overlap_segment

    # Normalized first segment
    v1n = v1 / L  # unit direction vector

    # --- Parallel check --- 
    #(extend to 3d array, lieing in the XY plane, the cross product will be parallel to the Z axis)
    cross = np.cross(np.append(v1, 0), np.append(v2, 0))
    # if segments are not parallel
    if abs(cross[2]) > tol:   # length of the z coordinate is directly the length of the vector, if equals 0, the two segments are parallel
        return overlap_length, overlap_segment  # no overlap (overlap length=0.)

    # --- Distance between the parallel segments ---
    dist = abs(np.cross(v1, q1 - p1)) / L
    # If two segments are not aligned
    if dist > tol:
        return overlap_length, overlap_segment

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
        return overlap_length, overlap_segment

    t0, t1 = overlap
    overlap_segment = np.vstack([
        p1 + v1n * t0,
        p1 + v1n * t1
    ])
    overlap_length = t1 - t0

    return overlap_length, overlap_segment
