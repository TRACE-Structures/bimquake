# bimquake

A Python package for earthquake resilience assessment of buildings, integrating IFC (Industry Foundation Classes) building information models with seismic hazard analysis and structural evaluation.

- Website: [https://buildchain.ilab.sztaki.hu/](https://buildchain.ilab.sztaki.hu/)
- Source code: [https://github.com/TRACE-Structures/bimquake](https://github.com/TRACE-Structures/bimquake)
- Bug reports: [https://github.com/TRACE-Structures/bimquake/issues](https://github.com/TRACE-Structures/bimquake/issues)

## Overview

BIMQuake combines Building Information Modeling (BIM) with earthquake engineering to provide comprehensive seismic analysis capabilities. The package enables users to:

- Extract structural data from IFC building models
- Retrieve seismic hazard parameters based on geographic location
- Perform linear static analysis for earthquake loading
- Visualize building layouts and structural wall resistance
- Generate interactive maps and 2D floor plans

## Installation

```bash
pip install bimquake
```

## Features

### IFC Processing
- **Building Model Extraction**: Parse and process IFC files to extract structural elements (walls, slabs, columns)
- **Geometric Analysis**: Extract geometric properties, dimensions, and spatial relationships
- **Material Properties**: Retrieve material characteristics from IFC models
- **3D Visualization**: Generate meshes for 3D visualization of building components

### Earthquake Hazard Analysis
- **Seismic Parameter Retrieval**: Get earthquake hazard parameters based on latitude/longitude coordinates
- **Italian Seismic Code**: Built-in support for Italian NTC seismic hazard grid data
- **Return Period Analysis**: Calculate parameters for various return periods (30, 50, 72, 101, 140, 201, 475, 975, 2475 years)
- **Interactive Mapping**: Generate folium-based maps showing building location

### Structural Analysis
- **Linear Static Analysis**: Perform seismic analysis according to Italian building codes
- **Wall Resistance Calculation**: Evaluate structural wall capacity under seismic loading
- **Floor-by-Floor Analysis**: Analyze building response at each floor level
- **Failure Mechanism Visualization**: Visualize potential failure modes

## Core Components

### `eq_resilience_library`
Main module containing earthquake resilience analysis functions:
- `get_Parameters()`: Retrieve seismic hazard parameters
- `get_map()`: Generate interactive location map
- `run_linear_static_analysis()`: Perform linear static analysis
- `run_pushover_analysis()`: Perform pushover analysis
- `get_2D_layout()`: Generate 2D floor plan visualization
- `plot_wall_resistence()`: Visualize wall resistance and failure mechanisms

### `ifc_processing`
Subpackage for processing IFC building models:
- `building.py`: Main building model handler with `IfcObjectSet` class
- `wall.py`: Wall element extraction and analysis
- `slab.py`: Slab element extraction and analysis
- `IFC_objects.py`: Base IFC object definitions
- `simple_objects.py`: Simplified geometric representations

## Data Formats

### Input Excel File Format
The analysis requires building data in Excel format with the following sheets:
- **Description**: Building overview (number of floors, height, etc.)
- **Floor_0, Floor_1, ...**: Individual floor data including wall positions, dimensions, materials

### IFC Requirements
- Compatible with IFC 2x3 and IFC 4 formats
- Requires structural elements to have properly defined geometry
- Material properties should be assigned to structural elements

## Current Limitations

- Seismic hazard data currently available only for Italy
- Analysis follows Italian building codes (NTC 2018)

## Authors and acknowledgment

The code is developed by Filippo Landi, Giada Bartolini, Bence Popovics, Noémi Friedman and Áron Friedman in the TRACE-Structures group.

This work has been funded by the European Commission Horizon Europe Innovation Action project 101092052 [BUILDCHAIN](https://buildchain-project.eu/)

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-only). See the [LICENSE](https://github.com/TRACE-Structures/bimquake/tree/main?tab=GPL-3.0-1-ov-file) file for details.

## Related Projects

- [uncertain_variables](https://github.com/TRACE-Structures/uncertain_variables/): Probabilistic variable management
- [gPCE_model](https://github.com/TRACE-Structures/gPCE_model/): Generalized Polynomial Chaos Expansion
- [digital_twinning](https://github.com/TRACE-Structures/digital_twinning/): Digital Twinning

## Support

For issues, questions, or contributions, please refer to the project repository or contact the authors.
