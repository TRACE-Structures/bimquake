import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import random

class SolidObject:
    """ Represents a 3D solid object defined by a set of points in space. 
    
        Attributes
        ----------
        points : np.ndarray
            A 2D array of shape (n, 3) representing the coordinates of the points in 3D space.
            
        faces : np.ndarray
            A 2D array representing the triangular faces of the convex hull of the points. """
    
    def __init__(self, coords):
        """ Initialize the SolidObject with a set of 3D coordinates.
        
            Parameters
            ----------
            coords : np.ndarray
                A 2D array of shape (n, 3) representing the coordinates of the points in 3D space."""
        
        self.points = coords
        self.faces = self.get_faces_from_point_coords()
       
    def get_bottom_height(self):
        """ Get the minimum height (z-coordinate) of the points in the solid object.
        
            Returns
            -------
            bottom_height : float
                The minimum z-coordinate of the points. """

        bottom_height = np.min(self.points[:, 2])
        return bottom_height

    def get_top_height(self):
        """ Get the maximum height (z-coordinate) of the points in the solid object.
        
            Returns
            -------
            top_height : float
                The maximum z-coordinate of the points. """
        
        top_height = np.max(self.points[:, 2])
        return top_height

    def get_faces_from_point_coords(self, points=None):
        """ Compute the triangular faces of the convex hull of the points.
        
            Parameters
            ----------
            points : np.ndarray, optional
                A 2D array of shape (n, 3) representing the coordinates of the points in 3D space.
                If None, uses the object's own points.
                
            Returns
            -------
            faces : np.ndarray
                A 2D array representing the triangular faces of the convex hull. """
        
        if points is None:
            points = self.points
        # Calculate the hull of the points
        hull = ConvexHull(points)
        faces = hull.simplices  # These are the triangle indices
        return faces

    def get_plotly_scatterpoints(self, points=None, name=None, mode='markers', color='green', size=2):
        """ Generate a Plotly Scatter3d object for the points in the solid object.
        
            Parameters
            ----------
            points : np.ndarray, optional
                A 2D array of shape (n, 3) representing the coordinates of the points in 3D space.
                If None, uses the object's own points.
                
            name : str, optional
                Name for the scatter plot trace.
                
            mode : str, optional
                Mode for the scatter plot (e.g., 'markers', 'lines', etc.). Default
                is 'markers'.
                
            color : str, optional
                Color for the markers. Default is 'green'.
                
            size : int, optional
                Size of the markers. Default is 2."""
        
        if points is None:
          points = self.points
        plotly_points = go.Scatter3d(
          x=points[:,0], y=points[:,1], z=points[:,2],
            mode='markers',
            marker=dict(size=size, color=color),
          name=name
        )
        return plotly_points

    def get_mesh(self, name=None, color='blue', opacity=1):
        """ Generate a Plotly Mesh3d object for the solid object.

            Parameters
            ----------
            name : str, optional
                Name for the mesh plot trace.
                
            color : str, optional
                Color for the mesh. Default is 'blue'.
                
            opacity : float, optional
                Opacity of the mesh. Default is 1.
                
            Returns
            -------
            mesh : go.Mesh3d
                A Plotly Mesh3d object representing the solid object."""
        
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        i, j, k = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=name
        )
        return mesh


    def __repr__(self):
        return f"<SolidObject with {len(self.points)} points and {len(self.faces)} faces>"

class SolidObjectSet:
    """A collection of SolidObject instances.
    
        Attributes
        ----------
        objects : list of SolidObject
            List of solid objects in the collection.
        points : list of np.ndarray
            List of points from all solid objects.
        meshes : list of go.Mesh3d
            List of Plotly Mesh3d objects for all solid objects. """
    
    def __init__(self):
        """ Initialize an empty SolidObjectSet instance."""

        self.objects = []
        self.points = []
        self.meshes = []

    def add_object(self, obj):
        """ Add a SolidObject to the collection.

            Parameters
            ----------
            obj : SolidObject
                The solid object to add.

            Returns
            -------
            objects : list of SolidObject
                The updated list of solid objects in the collection."""
        
        if isinstance(obj, SolidObject):
            self.objects.append(obj)
        else:
            print(type(obj))
            print(obj)
            raise TypeError("Only SolidObject objects can be added.")
        objects = self.objects
        return objects

    def __getitem__(self, index):
        """ Get a SolidObject from the collection by its index.

            Parameters
            ----------
            index : int
                The index of the solid object to retrieve.

            Returns
            -------
            obj : SolidObject
                The solid object at the specified index. """
        obj = self.objects[index]
        return obj
    
    def __len__(self):
        """ Get the number of solid objects in the collection.

            Returns
            -------
            length : int
                The number of solid objects in the collection. """
        
        length = len(self.objects)
        return length

    def __iter__(self):
        """ Get an iterator for the solid objects in the collection.

            Returns
            -------
            iterator : iterator
                An iterator over the solid objects in the collection. """
        iterator = iter(self.objects)
        return iterator

    def remove_object_by_index(self, ind):
        """ Remove a SolidObject from the collection by its index.

            Parameters
            ----------
            ind : int
                The index of the solid object to remove."""
        
        del self.objects[ind]

    def get_all_points(self):
        """ Get all points from all solid objects in the collection.

            Returns
            -------
            points : np.ndarray
                Array of points from all solid objects. """
        
        for obj in self.objects:
            self.points.extend(obj.points)
        self.points = np.array(self.points)
        points = self.points
        return points

    def random_pastel(self):
        """ Generate a random pastel color.

            Returns
            -------
            color : str
                The generated pastel color in RGB format. """
        r = lambda: random.randint(100, 255)
        color = f'rgb({r()},{r()},{r()})'
        return color

    def get_plotly_scatterpoints(self, points=None, mode='markers', color='green', size=2):
        """ Get a Plotly Scatter3d object for the given points.

            Parameters
            ----------
            points : np.ndarray, optional
                Array of points to plot. If None, all points from the collection are used.
            mode : str, optional
                The mode for the scatter plot (default is 'markers').
            color : str, optional
                The color of the markers (default is 'green').
            size : int, optional
                The size of the markers (default is 2).

            Returns
            -------
            plotly_points : go.Scatter3d
                The Plotly Scatter3d object representing the points. """
        if points is None:
          points = self.get_all_points()
        plotly_points = go.Scatter3d(
          x=points[:,0], y=points[:,1], z=points[:,2],
            mode='markers',
            marker=dict(size=size, color=color)
        )
        return plotly_points

    def get_meshes(self, opacity=0.8):
        """ Get the mesh representations of all solid objects in the collection.

            Parameters
            ----------
            opacity : float, optional
                The opacity of the meshes (default is 0.8).

            Returns
            -------
            meshes : list
                List of mesh objects representing the solid objects. """
        for obj in self.objects:
            self.meshes.append(obj.get_mesh(color=self.random_pastel(), opacity=opacity))
        meshes = self.meshes
        return meshes
