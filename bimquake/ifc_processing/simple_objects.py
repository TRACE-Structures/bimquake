
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import random

class SolidObject:
    """ A class representing a solid 3D object defined by its point coordinates.
    
        Attributes
        ----------
        points : numpy.ndarray
            An array of point coordinates defining the object.
            
        faces : numpy.ndarray
            An array of faces defined by indices of the points.
            
        Methods
        -------
        get_bottom_height()
            Returns the minimum z-coordinate (bottom height) of the object.
            
        get_top_height()
            Returns the maximum z-coordinate (top height) of the object.
            
        get_faces_from_point_coords(points=None)
            Computes the faces of the object using the convex hull of the points.
            
        get_plotly_scatterpoints(points=None, name=None, mode='markers', color='green', size=2)
            Generates a Plotly Scatter3d object for visualizing the points.
            
        get_mesh(name=None, color='blue', opacity=1)
            Generates a Plotly Mesh3d object for visualizing the solid object."""
    
    def __init__(self, coords):
        """ Initializes the SolidObject with given point coordinates.
        
            Parameters
            ----------
            coords : numpy.ndarray
                An array of point coordinates defining the object. """
        
        self.points = coords
        self.faces = self.get_faces_from_point_coords()
       
    def get_bottom_height(self):
        """ Returns the minimum z-coordinate (bottom height) of the object.
        
            Returns
            -------
            bottom_height : float
                The minimum z-coordinate of the object's points. """
        
        bottom_height = np.min(self.points[:, 2])
        return bottom_height

    def get_top_height(self):
        """ Returns the maximum z-coordinate (top height) of the object.
        
            Returns
            -------
            top_height : float
                The maximum z-coordinate of the object's points. """
        
        top_height = np.max(self.points[:, 2])
        return top_height

    def get_faces_from_point_coords(self, points=None):
        """ Computes the faces of the object using the convex hull of the points.
        
            Parameters
            ----------
            points : numpy.ndarray, optional
                An array of point coordinates. If None, uses the object's points.
                
            Returns
            -------
            faces : numpy.ndarray
                An array of faces defined by indices of the points. """
        
        if points is None:
            points = self.points
        # Calculate the hull of the points
        hull = ConvexHull(points)
        faces = hull.simplices  # These are the triangle indices
        return faces

    def get_plotly_scatterpoints(self, points=None, name=None, mode='markers', color='green', size=2):
        """ Generates a Plotly Scatter3d object for visualizing the points.

            Parameters
            ----------
            points : numpy.ndarray, optional
                An array of point coordinates. If None, uses the object's points.
            name : str, optional
                The name of the scatter plot.
            mode : str, optional
                The mode of the scatter plot (default is 'markers').
            color : str, optional
                The color of the markers (default is 'green').
            size : int, optional
                The size of the markers (default is 2).
                
            Returns
            -------
            plotly_points : plotly.graph_objects.Scatter3d
                A Plotly Scatter3d object representing the points. """
        
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
        """ Generates a Plotly Mesh3d object for visualizing the solid object.
        
            Parameters
            ----------
            name : str, optional
                The name of the mesh plot.

            color : str, optional
                The color of the mesh (default is 'blue').

            opacity : float, optional
                The opacity of the mesh (default is 1).

            Returns
            -------
            mesh : plotly.graph_objects.Mesh3d
                A Plotly Mesh3d object representing the solid object. """
        
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
        """ Returns a string representation of the SolidObject. """

        return f"<SolidObject with {len(self.points)} points and {len(self.faces)} faces>"

class SolidObjectSet:
    """ A class representing a collection of SolidObject instances.

        Attributes
        ----------
        objects : list
            A list of SolidObject instances in the collection.

        points : numpy.ndarray
            An array of all points from the SolidObject instances.

        meshes : list
            A list of Plotly Mesh3d objects for visualizing the SolidObject instances.

        Methods
        -------
        add_object(obj)
            Adds a SolidObject to the collection.
            
        remove_object_by_index(ind)
            Removes a SolidObject from the collection by index.
            
        get_all_points()
            Retrieves all points from the SolidObject instances.
            
        random_pastel()
            Generates a random pastel color.
            
        get_plotly_scatterpoints(points=None, mode='markers', color='green', size=2)
            Generates a Plotly Scatter3d object for visualizing the points.
            
        get_all_meshes(opacity=0.8)
            Generates Plotly Mesh3d objects for visualizing the SolidObject instances."""
    
    def __init__(self):
        """ Initializes an empty SolidObjectSet. """

        self.objects = []
        self.points = []
        self.meshes = []

    def add_object(self, obj):
        """ Adds a SolidObject to the collection.
        
            Parameters
            ----------
            obj : SolidObject
                The SolidObject instance to be added.
                
            Returns
            -------
            objects : list
                The updated list of SolidObject instances in the collection. """
        
        if isinstance(obj, SolidObject):
            self.objects.append(obj)
        else:
            print(type(obj))
            print(obj)
            raise TypeError("Only SolidObject objects can be added.")
        objects = self.objects
        return objects

    def remove_object_by_index(self, ind):
        """ Removes a SolidObject from the collection by index.
        
            Parameters
            ----------
            ind : int
                The index of the SolidObject to be removed. """
        
        del self.objects[ind]

    def get_all_points(self):
        """ Retrieves all points from the SolidObject instances.
        
            Returns
            -------
            points : numpy.ndarray
                An array of all points from the SolidObject instances. """
        
        for obj in self.objects:
            self.points.extend(obj.points)
        self.points = np.array(self.points)
        points = self.points
        return points

    def random_pastel(self):
        """ Generates a random pastel color.
        
            Returns
            -------
            color : str
                A string representing a random pastel RGB color. """
        
        r = lambda: random.randint(100, 255)
        color = f'rgb({r()},{r()},{r()})'
        return color

    def get_plotly_scatterpoints(self, points=None, mode='markers', color='green', size=2):
        """ Generates a Plotly Scatter3d object for visualizing the points.

            Parameters
            ----------
            points : numpy.ndarray, optional
                An array of point coordinates. If None, retrieves all points from the collection.

            mode : str, optional
                The mode of the scatter plot (default is 'markers').

            color : str, optional
                The color of the markers (default is 'green').

            size : int, optional
                The size of the markers (default is 2).
                
            Returns
            -------
            plotly_points : plotly.graph_objects.Scatter3d
                A Plotly Scatter3d object representing the points. """
        
        if points is None:
          points = self.get_all_points()
        plotly_points = go.Scatter3d(
          x=points[:,0], y=points[:,1], z=points[:,2],
            mode='markers',
            marker=dict(size=size, color=color)
        )
        return plotly_points

    def get_all_meshes(self, opacity=0.8):
        """ Generates Plotly Mesh3d objects for visualizing the SolidObject instances.
        
            Parameters
            ----------
            opacity : float, optional
                The opacity of the meshes (default is 0.8).

            Returns
            -------
            meshes : list
                A list of Plotly Mesh3d objects representing the SolidObject instances. """
        
        for obj in self.objects:
            self.meshes.append(obj.get_mesh(color=self.random_pastel(), opacity=opacity))
        meshes = self.meshes
        return meshes