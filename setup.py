from setuptools import setup, find_packages

setup(
    name='bimquake',
    version='0.1.0',
    author='Filippo Landi, Giada Bartolini, Áron Friedman, Bence Popovics, Noémi Friedman',
    author_email='filippo.landi@ing.unipi.it, giada.bartolini@strath.ac.uk, friedrron@gmail.com, popbence@sztaki.hun-ren.hu, n.friedman@ilab.sztaki.hu',
    description='Python package for earthquake resilience assessment of buildings, integrating IFC (Industry Foundation Classes) building information models with seismic hazard analysis and structural evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TRACE-Structures/bimquake/',
    packages=find_packages(),
    py_modules=['bimquake'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'pandas',
        'matplotlib',
        'reverse_geocoder',
        'folium',
        'shapely',
        'ifcopenshell',
        'plotly',
        'openpyxl',
        'nbformat'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
)