import json
import folium
import importlib.resources
from scipy.io import loadmat
import scipy.interpolate as si
import pandas as pd
import numpy as np
import reverse_geocoder as rg
import math

def get_Parameters(latitude, longitude):
    """ Retrieve earthquake hazard parameters based on geographic coordinates.

        Parameters
        ----------
        latitude : float
            Latitude of the location.

        longitude : float
            Longitude of the location.

        Returns
        -------
        ParaTR : pd.DataFrame or None
            DataFrame containing return period parameters if available, else None.

        city : str
            Name of the city corresponding to the coordinates.

        country_code : str
            Country code corresponding to the coordinates. """

    city, country_code = _coordinate_check(latitude, longitude)
    with importlib.resources.open_text("bimquake.hazard_data", "countries.json") as countries_file:
        countries = json.load(countries_file)
        country_codes = countries.keys()
        if country_code in country_codes:
            with importlib.resources.path("bimquake.hazard_data", "HazardNTCgrid_{}.mat".format(country_code)) as grid_file:
                hazard_data = loadmat(grid_file)['lonlatsism']
                ParaTR = _get_ParaTR(longitude, latitude, hazard_data)
                ParaTR = np.concatenate((np.array([30, 50, 72, 101, 140, 201, 475, 975, 2475]).reshape(-1, 1), ParaTR), axis=1)
                columns = ['Return Period', 'ag', 'Fo', 'Tc*']
                ParaTR = pd.DataFrame(ParaTR, columns=columns)
        else:
            ParaTR = None
            print("Earthquake hazard calculation is not supported in the selected region. Please add Italian coordinates or upload the return period parameters in a CSV file below.")
        return ParaTR, city, country_code
        
def _format_coordinates(coordinate, type="latitude"):
    """ Format geographic coordinates into degrees, minutes, and seconds.

        Parameters
        ----------
        coordinate : float
            The geographic coordinate (latitude or longitude).

        type : str, optional
            Type of coordinate ("latitude" or "longitude"). Default is "latitude".

        Returns
        -------
        coord_string : str
            Formatted coordinate string in degrees, minutes, and seconds. """
    
    abs_degrees = abs(coordinate)
    degrees = math.floor(abs_degrees)
    minutes = math.floor(60*(abs_degrees-degrees))
    seconds = round(3600 * (abs_degrees-degrees) - 60*minutes)
    coord_string = ""
    if type == "latitude":
        if coordinate < 0:
            coord_string = """{}° {}' {}" S""".format(degrees, minutes, seconds)
            return coord_string
        else:
            coord_string = """{}° {}' {}" N""".format(degrees, minutes, seconds)
            return coord_string
    else:
        if coordinate < 0:
            coord_string = """{}° {}' {}" W""".format(degrees, minutes, seconds)
            return coord_string
        else:
            coord_string = """{}° {}' {}" E""".format(degrees, minutes, seconds)
            return coord_string

def get_map(latitude, longitude, city, country):
    """ Generate an interactive map with a marker at the specified coordinates.

        Parameters
        ----------
        latitude : float
            Latitude of the location.

        longitude : float
            Longitude of the location.

        city : str
            Name of the city.

        country : str
            Country code.

        Returns
        -------
        f : folium.Figure
            Folium figure object containing the map with the marker. """
    

    f = folium.Figure(width=500, height=500)
    m = folium.Map(location=[latitude, longitude], zoom_start=10).add_to(f)
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup('<b>{}, {}</b><br>({} {})'.format(city, country, _format_coordinates(latitude, "latitude"), _format_coordinates(longitude, "longitude")), max_width=400, min_width=80), # pop-up label for the marker
        icon=folium.Icon()
    ).add_to(m)
    return f

def _coordinate_check(latitude, longitude):
    """ Check and retrieve city and country information based on coordinates.
    
        Parameters
        ----------
        latitude : float
            Latitude of the location.

        longitude : float
            Longitude of the location.

        Returns
        -------
        city : str
            Name of the city corresponding to the coordinates.

        country : str
            Country code corresponding to the coordinates. """
    
    coordinates = (latitude, longitude)
    location = rg.search(coordinates)
    country = location[0]['cc']
    city = location[0]['name']
    return city, country


def _get_ParaTR(longitude, latitude, hazard_data):
    """ Interpolate hazard parameters based on geographic coordinates.

        Parameters
        ----------
        longitude : float
            Longitude of the location.

        latitude : float
            Latitude of the location.

        hazard_data : np.ndarray
            Array containing hazard data.

        Returns
        -------
        ParaTR : np.ndarray
            Array of interpolated hazard parameters for different return periods. """
    
    coordinates =  [longitude, latitude]
    ParaTR = np.zeros((9, 3))
    for i in range(9):
        ParaTR[i,0] = si.griddata(hazard_data[:,:2], hazard_data[:,1+(i)*3+1], coordinates)/10
        ParaTR[i,1] = si.griddata(hazard_data[:,:2], hazard_data[:,1+(i)*3+2], coordinates)
        ParaTR[i,2] = si.griddata(hazard_data[:,:2], hazard_data[:,1+(i)*3+3], coordinates)
    return ParaTR
