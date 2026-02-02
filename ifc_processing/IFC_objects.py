from simple_objects import SolidObject
import ifcopenshell
import numpy as np
import math
from itertools import product
import plotly.graph_objects as go
import warnings

class IfcObjectProperties:
    """ Class to extract and manage IFC object properties, including material and general properties.
    
        Attributes
        ----------
        ifc_obj : ifcopenshell.entity_instance
            The IFC object instance.
            
        materials : list of dict
            List of associated materials with their properties.
            
        general : list of dict
            List of general properties of the IFC object.
            
        Methods
        -------
        show()
            Displays the key properties of the IFC object.
            
        map_attribute_to_ifc_property(attr_name)
            Maps attribute names to IFC property names.
            
        get_default_unit(prop_name)
            Returns the default unit for a given property name.
            
        get_q_dict()
            Returns a dictionary of variable load properties.
            
        find_material_property_val(property_name)
            Finds the value of a material property.
            
        find_general_property_val(property_name, verbose=True)
            Finds the value of a general property.
            
        find_property(props, property_name)
            Searches for a property in a list of properties.
            
        is_property_declared(property_name)
            Checks if a property is declared in the IFC object.
            
        get_material_properties(mat)
            Extracts material properties from an IFC material.
            
        ifc_property_to_dict(prop)
            Converts an IFC property to a dictionary.
            
        get_associated_material_properties()
            Retrieves associated material properties of the IFC object.
            
        get_properties()
            Retrieves general properties of the IFC object.
            
        show_ifc_general_and_material_properties()
            Displays all general and material properties of the IFC object."""
    
    def __init__(self, ifc_obj):
        """ Initializes the IfcObjectProperties instance by extracting properties from the IFC object.

            Parameters
            ----------
            ifc_obj : ifcopenshell.entity_instance
                The IFC object instance from which to extract properties."""
        
        self.ifc_obj = ifc_obj
        self.materials = self.get_associated_material_properties() # List of dict of associated materials with" propertiess", that is a list of dict of ifc single properties)
        self.general = self.get_properties() # List of dict of IFCSingle properties (name, value, unit, dexription..)
        attr2prop = self.map_attribute_to_ifc_property
        if ifc_obj.is_a("IfcSlab"):
            self.load_direction = self.find_general_property_val(attr2prop("load_direction"))  # in angle
            self.load_distribution = self.find_general_property_val(attr2prop("load_distribution")) #'OneWay' or 'TwoWays"
            self.g1 = self.find_general_property_val(attr2prop("g1"))
             # Nonstructural pernament surface loading
            self.g2 = self.find_general_property_val(attr2prop("g2"))
            # Variable surface loading
            for category in ["A", "B", "C", "D", "E", "F", "G", "H"]:
              q_name = f"q_{category}"
              setattr(self, q_name, self.find_general_property_val(attr2prop(q_name)))
           
        elif ifc_obj.is_a("IfcWall"):
            self.density = self.find_material_property_val(attr2prop("density"))  # in [kg/m3]  #
            self.E = self.find_material_property_val(attr2prop("E"))  # YoungModulus'in [Pa]
            self.f_u  = self.find_material_property_val(attr2prop("f_u")) # 'TensileStrength'[PA]
            self.f_c =  self.find_material_property_val(attr2prop("f_c"))  # 'ConcreteCompression'
            self.G = self.find_material_property_val(attr2prop("G"))   # 'ShearModulus' [Pa]
            self.nu = self.find_material_property_val(attr2prop("nu"))   #'PoissonRatio'
            # Additional structural pernament loading (other then from slabs and walls)
            self.g1 = self.find_general_property_val(attr2prop("g1"), verbose=False)
            # Nonstructural pernament surface loading (other then from slabs and walls)
            self.g2 = self.find_general_property_val(attr2prop("g2"), verbose=False)
            # Additional variable loading (other then from slabs and walls)
            for category in ["A", "B", "C", "D", "E", "F", "G", "H"]:
              q_name = f"q_{category}"
              setattr(self, q_name, self.find_general_property_val(attr2prop(q_name)))
           
    def show(self):
        """ Displays the key properties of the IFC object. """

        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        if self.ifc_obj.is_a("IfcSlab"):
          for attr in ["load_direction",
                       "load_distribution",
                       "g1", "g2"] + ["q_"+ cat_i for cat_i in categories]:
              val = getattr(self, attr, 'Not found')
              unit = self. get_default_unit(self.map_attribute_to_ifc_property(attr))[0]
              print(f"{attr}: {val} [{unit}]")
        elif self.ifc_obj.is_a("IfcWall"):
          for attr in ["density",
                       "E", "G",
                       "f_u", "f_c", "nu",
                        "g1", "g2"] + ["q_"+ cat_i for cat_i in categories]:
              val = getattr(self, attr, 'Not found')
              unit = self. get_default_unit(self.map_attribute_to_ifc_property(attr))[0]
              print(f"{attr}: {val} [{unit}]")

    def map_attribute_to_ifc_property(self, attr_name):
        """ Maps attribute names to IFC property names.

            Parameters
            ----------
            attr_name : str
                The attribute name to map.

            Returns
            -------
            ifc_property : str or None
                The corresponding IFC property name, or None if not found."""
        
        map = {
            "load_direction" : "LoadDirection",
            "load_distribution": "LoadDistribution",
            "density" : "MassDensity",
            "g1": "PermanentStructuralLoad",
            "g2": "PermanentNonStructuralLoad",
            "q_A":  "VariableLoadCategoryA",
            "q_B":  "VariableLoadCategoryB",
            "q_C":  "VariableLoadCategoryC",
            "q_D":  "VariableLoadCategoryD",
            "q_E":  "VariableLoadCategoryE",
            "q_F":  "VariableLoadCategoryF",
            "q_G":  "VariableLoadCategoryG",
            "q_H":  "VariableLoadCategoryH",
            "g": "PernamentSurfaceLoad",
            "E": "YoungModulus",
            "f_u": "TensileStrength",
            "f_c": "CompressiveStrength",
            "G": "ShearModulus",
            "nu": "PoissonRatio"
        }

        ifc_property = map.get(attr_name, None)
        return ifc_property

    def get_default_unit(self, prop_name):
        """ Returns the default unit for a given property name.

            Parameters
            ----------
            prop_name : str
                The property name.

            Returns
            -------
            unit : list or None
                A list of acceptable units for the property, or None if not found."""
        
        Pa =  ["Pa", "N/sqm", "N/m^2", "N/m2", None]
        def_units = {
            "LoadDirection": ["degree", None],
            "LoadDistribution": ["-", None],
            "MassDensity": ["kg/m3", "kg/cum", None],
            "PermanentNonStructuralLoad":  Pa,
            "PermanentStructuralLoad":  Pa,
            "YoungModulus": Pa,
            "TensileStrength": Pa,
            "CompressiveStrength": Pa,
            "ShearModulus": Pa,
            "PoissonRatio": ["-", None]
        }
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for cat_i in categories:
          def_units[f"VariableLoadCategory{cat_i}"] = Pa

        if prop_name not in def_units:
          prop_name = self.map_attribute_to_ifc_property(prop_name)
          if prop_name is None:
            return None
        unit = def_units[prop_name]
        return unit

    def get_q_dict(self):
        """ Returns a dictionary of variable load properties.

            Returns
            -------
            q_dict : dict
                A dictionary with variable load categories as keys and their corresponding values."""
        
        categories = ["A", "B", "C", "D", "E", "F", "G", "H"]
        q_dict = {cat_i: getattr(self, f"q_{cat_i}")  for cat_i in categories}
        return q_dict

    def find_material_property_val(self, property_name):
        """ Finds the value of a material property.

            Parameters
            ----------
            property_name : str
                The name of the material property to find.

            Returns
            -------
            value : float or None
                The value of the material property, or None if not found."""
        
        prop = []
        for mat_i in self.materials:
          prop_i = self.find_property(mat_i["Properties"], property_name)
          if prop_i is None:
            continue
          else:
            prop.append(prop_i["Value"])

        if len(prop)== 0:
          value = self.find_general_property_val(property_name)
        elif len(prop) == 1:
          value = prop[0]
        else:
          warnings.warn(f"Several Materials have property{property_name} so it is averaged")
          value = np.mean(prop)

        return value

    def find_general_property_val(self, property_name, verbose=True):
        """ Finds the value of a general property.
        
            Parameters
            ----------
            property_name : str
                The name of the general property to find.
            verbose : bool, optional
                Whether to print a warning if the property is not found (default is True).

            Returns
            -------
            value : float or None
                The value of the general property, or None if not found."""
        
        prop = self.find_property(self.general, property_name)
        if prop is None:
          if verbose and ("VariableLoadCategory" not in property_name and "PermanentNonStructuralLoad"!= property_name):
            print(f"property {property_name} is not defined for the properties of element {self.ifc_obj.__repr__()[:37] + '...'}")
          value = None
        else:
          value = prop["Value"]
        return value

    def find_property(self, props, property_name):
        """ Searches for a property in a list of properties.

            Parameters
            ----------
            props : list of dict
                The list of properties to search.

            property_name : str
                The name of the property to find.

            Returns
            -------
            prop : dict or None
                The property dictionary if found, otherwise None."""
        
        for prop in props:
            if prop["Name"] == property_name:
                unit = prop["Unit"]
                def_unit = self.get_default_unit(property_name)
                if def_unit is None:
                  return prop
                elif prop["Unit"] in def_unit:
                  prop["Unit"] = def_unit[0]
                  return prop
                else:
                  warnings.warn(f"Unit [{unit}] does not match the default unit {def_unit[0]}, conversion not implemented yet " )
        prop = None
        return prop

    def is_property_declared(self, property_name):
        """ Checks if a property is declared in the IFC object.

            Parameters
            ----------
            property_name : str
                The name of the property to check.

            Returns
            -------
            declared : bool
                True if the property is declared, False otherwise.

            prop_type : str or None
                The type of property ("General Property" or "Material Property") if declared, otherwise None.

            prop_value : dict, list of tuples, or None
                The property value(s) if declared, otherwise None."""
        
        # Check if property is in general properties
        prop = self.find_property(self.general, property_name)
        # Otherwise check if its in material properties
        if prop is None:
          prop = []
          mat_name = []
          for mat_i in self.materials:
            prop_i = self.find_property(mat_i["Properties"], property_name)
            if prop_i is None:
              continue
            else:
              prop.append(prop_i)
              mat_name.append(mat_i["MaterialName"])
          if len(prop)== 0:
            declared = False
            prop_type = None
            prop_value = None
          else:
            declared = True
            prop_type = "Material Property"
            prop_value = list(zip(mat_name, prop))
        else:
            declared = True
            prop_type = "General Property"
            prop_value = prop

        return declared, prop_type, prop_value

    def get_material_properties(self, mat):
        """ Extracts material properties from an IFC material.

            Parameters
            ----------
            mat : ifcopenshell.entity_instance
                The IFC material instance.

            Returns
            -------
            mat_props_list : list of dict
                A list of material properties as dictionaries."""
        
        mat_props_list = []
        if hasattr(mat, "HasProperties"):
            for mat_prop in mat.HasProperties:
                if mat_prop.is_a("IfcMaterialProperties"):
                    for p in mat_prop.Properties:
                        if p.is_a("IfcPropertySingleValue"):
                            prop_dict = self.ifc_property_to_dict(p)
                            mat_props_list.append(prop_dict)
        return mat_props_list

    def ifc_property_to_dict(self, prop):
        """ Converts an IFC property to a dictionary.

            Parameters
            ----------
            prop : ifcopenshell.entity_instance
                The IFC property instance.

            Returns
            -------
            prop_dict : dict
                A dictionary representation of the IFC property."""
        
        prop_dict = {}
      
        if prop.is_a("IfcPropertySingleValue"):
          prop_dict = {
              "Name": prop.Name,
              "Value": prop.NominalValue.wrappedValue,
              "Unit": prop.Unit,
              "Description": prop.Description
        }

        elif prop.is_a("IfcPropertyBoundedValue"):
          prop_dict = {
              "Name": prop.Name,
              "Value": prop.SetPointValue,
              "LowerBound": prop.LowerBoundValue,
              "UpperBound": prop.UpperBoundValue,
              "Unit": prop.Unit,
              "Description": prop.Description
          }
        return prop_dict

    def get_associated_material_properties(self):
        """ Retrieves associated material properties of the IFC object.

            Returns
            -------
            materials : list of dict or None
                A list of associated materials with their properties, or None if no materials are found."""
        
        materials = None 
        for assoc in self.ifc_obj.HasAssociations:
            if assoc.is_a("IfcRelAssociatesMaterial"):
                material = assoc.RelatingMaterial

                def get_mat_properties_from_matset(matset, matset_type=None):
                    mat_prop_list = []
                    for mat_i in matset:
                      mat = mat_i.Material
                      entry = {
                          "TypeMaterialSet": matset_type,
                          "MaterialName": getattr(mat_i, "Name", None),
                          "Thickness": getattr(mat_i, "Thickness",None),
                          "Category": getattr(mat, "Category", None),
                          "Description": getattr(mat, "Description", None),
                          "Properties":
                          self.get_material_properties(mat) if mat else []
                          }
                      mat_prop_list.append(entry)
                    return mat_prop_list

                if material.is_a("IfcMaterialConstituentSet"):
                  materials = get_mat_properties_from_matset(
                      material.MaterialConstituents, material.is_a())

                elif material.is_a("IfcMaterialLayerSetUsage"):
                    materials = get_mat_properties_from_matset(
                        material.ForLayerSet.MaterialLayers, material.is_a())

                elif material.is_a("IfcMaterial"):
                    materials = [{
                        "TypeMaterialSet": "SingleMaterial",
                        "MaterialName": material.Name,
                        "Thickness": None,
                        "Category": getattr(material, "Category", None),
                        "Description": getattr(material, "Description", None),
                        "Properties": self.get_material_properties(material)
                    }]
                else:
                    warnings.warn(f"Unknown material type: {material.is_a()}")
                    materials = None
                break
            else:
                materials = None
        return materials

    def get_properties(self):
        """ Retrieves general properties of the IFC object.

            Returns
            -------
            props : list of dict
                A list of general properties as dictionaries."""
        
        props = []
        for prop_i in self.ifc_obj.IsDefinedBy:
            if prop_i.is_a("IfcRelDefinesByProperties"):
                prop_def = prop_i.RelatingPropertyDefinition
                if prop_def.is_a("IfcPropertySet"):
                    for prop_ij in prop_def.HasProperties:
                        if prop_ij.is_a("IfcPropertySingleValue") or prop_ij.is_a("IfcPropertyBoundedValue"):
                            prop_dict = self.ifc_property_to_dict(prop_ij)
                            props.append(prop_dict)
                else:
                    # Skip or handle other types like IfcElementQuantity, IfcPropertySetTemplate, etc.
                    continue
        return props

    def show_ifc_general_and_material_properties(self):
        """ Displays all general and material properties of the IFC object. """

        try:
            mat_summary = []
            for mat_i in self.materials:
                mat_name = mat_i.get("MaterialName", "Unknown")
                props = mat_i.get("Properties", [])
                prop_lines = "\n".join(
                    f"    {prop['Name']}: {prop['Value']} [{prop['Unit']}]" for prop in props
                )
                mat_summary.append(f"Material: {mat_name}\n{prop_lines}")
            mat_summary_str = "\n\n".join(mat_summary)
        except Exception as e:
            mat_summary_str = f"Could not parse material properties: {e}"

        try:
            gen_summary = "\n".join(
                f"{prop['Name']}: {prop['Value'] }[{prop['Unit']}]" for prop in self.general
            )
        except Exception as e:
            gen_summary = f"Could not parse general properties: {e}"

        print (
            f"<IfcObjectProperties for {type(self.ifc_obj)} #{self.ifc_obj.id()}>\n\n"
            f"General Properties:\n{gen_summary}\n\n"
            f"Material Properties:\n{mat_summary_str}"
        )
        
        
# Define class of wall created from the IFC product
class IfcObject(SolidObject):
    """ Class representing an IFC object, such as a wall or slab, with geometric and property information.
        
        Attributes
        ----------
        ifc_obj : ifcopenshell.entity_instance
            The IFC object instance.

        id : int
            The unique identifier of the IFC object.

        name : str
            The name of the IFC object.

        shape : ifcopenshell.geom.shape
            The geometric shape of the IFC object.

        points : numpy.ndarray
            The global coordinates of the points defining the IFC object.

        faces : numpy.ndarray
            The faces of the IFC object.

        origin : numpy.ndarray
            The origin of the local coordinate system of the IFC object.

        x_axis : numpy.ndarray
            The local x-axis unit vector of the IFC object.

        y_axis : numpy.ndarray
            The local y-axis unit vector of the IFC object.

        z_axis : numpy.ndarray
            The local z-axis unit vector of the IFC object.

        properties : IfcObjectProperties
            The properties of the IFC object.

        floor : int or None
            The floor number the IFC object belongs to.

        Methods
        -------
        get_shape_from_ifc()
            Retrieves the geometric shape from the IFC object.

        get_points_from_shape()
            Extracts the global coordinates of the points from the shape.

        set_points(points)
            Sets the points of the IFC object.

        set_floor(floor)
            Sets the floor number of the IFC object.

        get_faces_from_shape()
            Extracts the faces from the shape.

        set_faces(faces)
            Sets the faces of the IFC object.

        get_index_of_corner_points(points=None, tol=0.05)
            Identifies the indices of the corner points of the IFC object.

        get_bounding_box_corner_indices(points=None, tol=0.05)
            Identifies the indices of the bounding box corner points.

        get_bounding_box_points(coord='local', points=None)
            Retrieves the bounding box corner points.
            
        get_origin_and_local_axes()
            Computes the origin and local axes of the IFC object.
            
        get_transformation_matrix()
            Computes the transformation matrix from global to local coordinates.
            
        get_bottom_height()
            Computes the bottom height of the IFC object.
            
        get_top_height()
            Computes the top height of the IFC object.
            
        transform_to_local(global_coords)
            Transforms global coordinates to local coordinates.
            
        transform_to_global(local_coords)
            Transforms local coordinates to global coordinates.
            
        get_dimensions(points=None)
            Computes the dimensions of the IFC object.
            
        get_volume(points=None)
            Computes the volume of the IFC object.
            
        get_upper_face_area(points=None)
            Computes the area of the upper face of the IFC object.

        sort_points_by_coordinates(points=None, sort_by_axis="xzy")
            Sorts points based on their global coordinates.

        sort_points_by_local_coordinates(points=None, sort_by_axis="xzy")
            Sorts points based on their local coordinates.

        get_angle_of_axis()
            Computes the angle of a given axis in the local coordinate system.

        get_mesh(color='blue', opacity=0.8, name=None)
            Generates a 3D mesh representation of the IFC object for visualization.

        get_plotly_local_coord()
            Generates a 3D plot of the IFC object in local coordinates using Plotly.

        get_psi_2_factor(category)
            Computes the psi_2 factor for a given load category."""
    
    def __init__(self, ifc_object):
        """ Initializes the IfcObject instance by extracting geometric and property information from the IFC object.

            Parameters
            ----------
            ifc_object : ifcopenshell.entity_instance
                The IFC object instance to be represented."""
        
        self.ifc_obj = ifc_object
        self.id = self.ifc_obj.id()
        if ifc_object.is_a("IfcWall"):
          self.name ="Wall "+ str(self.id)
        elif ifc_object.is_a("IfcSlab"):
          self.name = "Slab " + str(self.id)
        self.shape = self.get_shape_from_ifc()
        self.points = self.get_points_from_shape()  # global coordinates of the points
        self.faces = self.get_faces_from_shape()
        self.origin, self.x_axis, self.y_axis, self.z_axis = self.get_origin_and_local_axes()
        self.properties = IfcObjectProperties(self.ifc_obj)
        self.floor = None

    def get_shape_from_ifc(self):
        """ Retrieves the geometric shape from the IFC object.

            Returns
            -------
            shape : ifcopenshell.geom.shape
                The geometric shape of the IFC object."""
        
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        shape = ifcopenshell.geom.create_shape(settings, self.ifc_obj)
        return shape

    def get_points_from_shape(self):
        """ Extracts the global coordinates of the points from the shape.

            Returns
            -------
            points : numpy.ndarray
                The global coordinates of the points defining the IFC object."""
        
        points = np.array(self.shape.geometry.verts).reshape(-1,3)
        r_points = np.round(points, decimals=4)
        _, u_ind = np.unique(r_points, axis=0, return_index=np.True_)
        points = points[sorted(u_ind)]
        return points

    def set_points(self, points):
        """ Sets the points of the IFC object.

            Parameters
            ----------
            points : numpy.ndarray
                The global coordinates of the points to set."""
        
        self.points = points

    def set_floor(self, floor):
        """ Sets the floor number of the IFC object.

            Parameters
            ----------
            floor : int
                The floor number to set."""
        
        self.floor = floor

    def get_faces_from_shape(self):
        """ Extracts the faces from the shape.

            Returns
            -------
            faces : numpy.ndarray
                The faces of the IFC object."""
        
        faces = np.array(self.shape.geometry.faces).reshape(-1, 3)
        return faces

    def set_faces(self, faces):
        """ Sets the faces of the IFC object.

            Parameters
            ----------
            faces : numpy.ndarray
                The faces to set."""
        
        self.faces = faces

    def get_index_of_corner_points(self, points=None, tol=0.05):
        """ Identifies the indices of the corner points of the IFC object.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to consider (default is None, which uses the object's points).

            tol : float, optional
                The tolerance for identifying corner points (default is 0.05).

            Returns
            -------
            ind_start_face : numpy.ndarray
                The indices of the corner points of the start face.

            ind_end_face : numpy.ndarray
                The indices of the corner points of the end face."""
        
        if points is None:
          points = self.points
        # The following function only works if front and end faces have only 4 points
        # Transform point coordinates to local coordinates
        # local_points = self.transform_to_local(self.points)
        # Get index of four corner points of the wall (order first by x, then z, then y)
        # ind_sort = self.sort_points_by_local_coordinates(sort_by_axis="xzy")
        # ind_start_face = ind_sort[:4]
        # ind_end_face = ind_sort[-4:]
        ind_corner = self.get_bounding_box_corner_indices(points=points, tol=tol)
        ind_sort = self.sort_points_by_local_coordinates(points=points[ind_corner], sort_by_axis="xzy")
        ind_start_face = ind_corner[ind_sort[:4]]
        ind_end_face = ind_corner[ind_sort[-4:]]

        return ind_start_face, ind_end_face

    def get_bounding_box_corner_indices(self, points=None, tol=0.05):
        """ Identifies the indices of the bounding box corner points.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to consider (default is None, which uses the object's points).

            tol : float, optional
                The tolerance for identifying corner points (default is 0.05).

            Returns
            -------
            ind : numpy.ndarray
                The indices of the bounding box corner points."""
        
        #Get bounding box corner points
        if points is None:
          points = self.points
        local_points = self.transform_to_local(points)
        bbox_corners =  self.get_bounding_box_points(coord='local', points=points)
        ind = []
        for corner in bbox_corners:
            # Find index of the point in `points` that matches the corner
            dists = np.linalg.norm(local_points - corner, axis=1)
            match_idx = np.argmin(dists)
            if dists[match_idx] < tol:
                ind.append(match_idx)
                if dists[match_idx] > 0.1:
                  print(f"Distance of bounding box point {np.round(corner)} and closest element point is {dists[match_idx]}>0.1, meaning the element {self.id} is not rectangular, which can cause inaccuracies ")
            else:
                raise ValueError(f"No point found for corner {corner} of element {self.id} within tolerance {tol}. Minimum distance was {dists[match_idx]}. Probably this element is not rectangular")
        ind = np.array(ind, dtype=int)
        return ind

    def get_bounding_box_points(self, coord='local', points=None):
        """ Retrieves the bounding box corner points.

            Parameters
            ----------
            coord : str, optional
                The coordinate system to return the bounding box points in ('local' or 'global', default is 'local').

            points : numpy.ndarray, optional
                The global coordinates of the points to consider (default is None, which uses the object's points).

            Returns
            -------
            bbox_points : numpy.ndarray
                The coordinates of the bounding box corner points."""
        
        if points is None:
          points = self.points
        local_points = self.transform_to_local(points)

        # Get min and max for each axis
        mins = np.min(local_points, axis=0)
        maxs = np.max(local_points, axis=0)

        # Generate all 8 combinations of min/max x, y, z
        bbox_points = np.array(list(product(*zip(mins, maxs))))
        if coord == 'global':
          bbox_points = self.transform_to_global(bbox_points)
        return bbox_points

    # Function to get the local x, y, z direction of the wall
    def get_origin_and_local_axes(self):
        """ Returns the wall's local X, Y, Z unit vectors and origin as numpy arrays.
        
            Returns
            -------
            origin : numpy.ndarray
                The origin of the local coordinate system of the IFC object.
                
            x_dir : numpy.ndarray
                The local x-axis unit vector of the IFC object.
                
            y_dir : numpy.ndarray
                The local y-axis unit vector of the IFC object.
                
            z_dir : numpy.ndarray
                The local z-axis unit vector of the IFC object."""
        
        placement = self.ifc_obj.ObjectPlacement.RelativePlacement
        origin = np.array(placement.Location.Coordinates)

        # Default directions if not specified
        x_dir = np.array(placement.RefDirection.DirectionRatios) if placement.RefDirection else np.array([1, 0, 0])
        z_dir = np.array(placement.Axis.DirectionRatios) if placement.Axis else np.array([0, 0, 1])
        y_dir = np.cross(z_dir, x_dir)

        x_dir, y_dir, z_dir = x_dir.astype(float), y_dir.astype(float), z_dir.astype(float)

        # Normalize all
        x_dir /= np.linalg.norm(x_dir)
        y_dir /= np.linalg.norm(y_dir)
        z_dir /= np.linalg.norm(z_dir)

        return origin, x_dir, y_dir, z_dir

    def get_transformation_matrix(self):
        """ Computes the transformation matrix from global to local coordinates.

            Returns
            -------
            TR : numpy.ndarray
                The transformation matrix (3x3) from global to local coordinates."""
        
        # Build transformation matrix (world → local)
        TR = np.vstack([self.x_axis, self.y_axis, self.z_axis]).T  # 3x3
        return TR

    def get_bottom_height(self):
        """ Computes the bottom height of the IFC object.

            Returns
            -------
            bottom_height : float
                The bottom height of the IFC object."""
        
        bottom_height = np.min(self.points[:, 2])
        return bottom_height

    def get_top_height(self):
        """ Computes the top height of the IFC object.

            Returns
            -------
            top_height : float
                The top height of the IFC object."""
        top_height = np.max(self.points[:, 2])
        return top_height

    def transform_to_local(self, global_coords):
        """ Transforms global coordinates to local coordinates.

            Parameters
            ----------
            global_coords : numpy.ndarray
                The global coordinates to transform.

            Returns
            -------
            local_coords : numpy.ndarray
                The transformed local coordinates."""
        
        # Build transformation matrix (world → local)
        TR = self.get_transformation_matrix()
        # Project into wall space (@ is the same as np.matmul)
        local_coords = (global_coords-self.origin) @ TR
        return local_coords

    def transform_to_global(self, local_coords):
        """ Transforms local coordinates to global coordinates.

            Parameters
            ----------
            local_coords : numpy.ndarray
                The local coordinates to transform.

            Returns
            -------
            global_coords : numpy.ndarray
                The transformed global coordinates."""
        
        # Build transformation matrix (world → local)
        TR = self.get_transformation_matrix()
        # Project into wall space (@ is the same as np.matmul)
        global_coords = local_coords @ TR.T  + self.origin
        return global_coords

        # Function to get the main properties of the object

    def get_dimensions(self, points=None):
        """ Computes the dimensions of the IFC object.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to consider (default is None, which uses the object's points).

            Returns
            -------
            dimensions : dict
                A dictionary containing the Length, Width, Height, and CenterPoint coordinates of the IFC object."""
        
        if points is None:
            points = self.points

        # Transformation from world coordinates to local reference coords
        local_points = self.transform_to_local(points)

        # Compute bounding box in local space
        min_coords = local_points.min(axis=0)
        max_coords = local_points.max(axis=0)
        size = max_coords - min_coords
        center_local = (min_coords + max_coords) / 2
        center_global = self.transform_to_global(center_local)  # Back to world coords
        
        dimensions = {
            "Length": size[0],  # X in local space
            "Width": size[1],   # Y in local space
            "Height": size[2],  # Z in local space
            "CenterPointX": center_global[0],
            "CenterPointY": center_global[1],
            "CenterPointz": center_global[2]
            }
        return dimensions

    def get_volume(self, points=None):
        """ Computes the volume of the IFC object.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to consider (default is None, which uses the object's points).

            Returns
            -------
            V : float
                The volume of the IFC object."""
        
        if points is None:
            points = self.points

        # Transformation from world coordinates to local reference coords
        local_points = self.transform_to_local(points)

        # Compute bounding box in local space
        min_coords = local_points.min(axis=0)
        max_coords = local_points.max(axis=0)
        size = max_coords - min_coords
        V = size[0]*size[1]*size[2]
        return V

    def get_upper_face_area(self, points=None):
        """ Computes the area of the upper face of the IFC object.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to consider (default is None, which uses the object's points).

            Returns
            -------
            A : float
                The area of the upper face of the IFC object."""
        
        if points is None:
            points = self.points

        # Transformation from world coordinates to local reference coords
        local_points = self.transform_to_local(points)

        # Compute bounding box in local space
        min_coords = local_points.min(axis=0)
        max_coords = local_points.max(axis=0)
        size = max_coords - min_coords
        A = size[0]*size[1]
        return A

    def sort_points_by_coordinates(self, points = None, sort_by_axis="xzy"):
        """ Sorts points based on their global coordinates.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to sort (default is None, which uses the object's points).

            sort_by_axis : str, optional
                The order of axes to sort by (default is "xzy").

            Returns
            -------
            ind_sort : numpy.ndarray
                The indices that would sort the points based on the specified axes."""
        
        if points is None:
            points = self.points

        r_points = np.round(points, decimals=10)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_order = [axis_map[char] for char in sort_by_axis]

        # Sort local coordinates
        ind_sort = np.lexsort((r_points[:, axis_order[2]],
                               r_points[:, axis_order[1]],
                               r_points[:, axis_order[0]]))
        return ind_sort

    def sort_points_by_local_coordinates(self, points=None, sort_by_axis="xzy"):
        """ Sorts points based on their local coordinates.

            Parameters
            ----------
            points : numpy.ndarray, optional
                The global coordinates of the points to sort (default is None, which uses the object's points).

            sort_by_axis : str, optional
                The order of axes to sort by (default is "xzy").

            Returns
            -------
            ind_sort : numpy.ndarray
                The indices that would sort the points based on the specified local axes."""
        
        if points is None:
            points = self.points

        # trasform_to_local_coordinates
        local_points = self.transform_to_local(points)
        ind_sort = self.sort_points_by_coordinates(local_points, sort_by_axis)

        return ind_sort

    def get_angle_of_axis(self):
        """ Computes the angle of a given axis in the local coordinate system.

            Returns
            -------
            angle_deg : float
                The angle of the local x-axis in degrees relative to the global X-axis in the XY plane."""
        
        x_axis = self.x_axis
        # Calculate angle
        # Project to XY plane
        projected = np.array([x_axis[0], x_axis[1]])
        norm = np.linalg.norm(projected)
        if norm == 0:
            return 0.0
        projected /= norm
        if projected[0] < 0 or projected[0]==0 and projected[1]==-1:
          projected *= -1
        angle_rad = math.acos(np.clip(np.dot(projected, [1, 0]), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        # Adjust for direction
        if projected[1] < 0:
            angle_deg = - angle_deg
        return angle_deg

    def get_mesh(self, color='blue', opacity=0.8, name=None):
        """ Generates a 3D mesh representation of the IFC object for visualization.

            Parameters
            ----------
            color : str, optional
                The color of the mesh (default is 'blue').
            
            opacity : float, optional
                The opacity of the mesh (default is 0.8).

            name : str, optional
                The name of the mesh (default is None, which uses the object's name).

            Returns
            -------
            mesh : vedo.Mesh
                The 3D mesh representation of the IFC object."""
        
        if name is None:
          name = f"{self.name}"
        mesh = super().get_mesh(name=name, color=color, opacity=opacity)
        return mesh

    def get_plotly_local_coord(self):
        """ Generates a 3D plot of the IFC object in local coordinates using Plotly.
        
            Returns
            -------
            arrow_x_z : plotly.graph_objects.Cone
                A Plotly cone object representing the local x and z directions of the IFC object."""
        
        origin, x_dir, y_dir, z_dir = self.get_origin_and_local_axes()
        x_o, y_o, z_o = origin

        # Create cone (arrow) that points in the direction of the local x and z direction
        arrow_x_z = go.Cone(
            x=[x_o, x_o],
            y=[y_o, y_o],
            z=[z_o, z_o],
            u=[x_dir[0], z_dir[0]],
            v=[x_dir[1], z_dir[1]],
            w=[x_dir[2], z_dir[2]],
            sizemode="absolute",
            sizeref=0.2,  # Adjust size here
            anchor="tail",  # Anchor cone at the tail (starting point)
            showscale=False,
            colorscale=[[0, 'green'], [1, 'red']]
         )
        return arrow_x_z

    def get_psi_2_factor(self, category):
        """ Computes the psi_2 factor for a given load category.

            Parameters
            ----------
            category : str
                The load category (A, B, C, D, E, F, G, H).

            Returns
            -------
            factor : float
                The psi_2 factor corresponding to the load category."""
        
        factors ={
          "A":0.3,
          "B":0.3,
          "C":0.6,
          "D":0.6,
          "E":0.8,
          "F":0.6,
          "G":0.3,
          "H":0.
        }
        factor = factors[category]
        return factor

    def __repr__(self):
        """ Returns a string representation of the IFC object. """
        
        return f"<IFC object {type(self.ifc_obj)} #{self.ifc_obj.id()}>"

