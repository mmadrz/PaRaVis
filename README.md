![LOGO](https://github.com/mmadrz/PaRaVis/assets/117746151/7ce42d2a-020e-4d2e-95b8-33b1395b92dc)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)](https://semver.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

***
# Overview
_PaRaVis_ is a powerful GUI designed for extracting, visualizing, and analyzing Rao's Q based on VIs. The GUI comprises three main sections, each enhancing usability and functionality for different process aspects.

### 1. Vegetation Index Analysis
The code provides a user-friendly interface for importing stacked raster datasets, selecting specific bands, and calculating multiple indices simultaneously. spatial visualizations with customization options enhance the exploration of vegetation indices. Users can also select and save calculated indices (based on their CV value) as GeoTIFF files for future analysis.

### 2. Rao's Q Index Computation
This section focuses on the computation of Rao's Q index from raster files in both unidimensional and multidimensional modes. offering  parameter customization options. parallelization using the Ray framework optimizes computational efficiency, reducing processing time significantly. The code employs the "tqdm" library to monitor processing time.

### 3. Visualization and Analysis
This part emphasizes visualizing and analyzing Rao's Q outputs through an intuitive interface. Various widgets facilitate seamless file selection, output path definition, and customization of plot settings. The tool generates insightful plots, histograms, difference heatmaps, and split plots, making complex operations accessible to users.

__For more information about the _PaRaVis_ tool, please refer to the [__Documentation__](Documentation/Documentation.md).__

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
