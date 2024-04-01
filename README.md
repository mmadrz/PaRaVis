![LOGO](https://github.com/mmadrz/PaRaVis/assets/117746151/7ce42d2a-020e-4d2e-95b8-33b1395b92dc)
[![Version](https://img.shields.io/badge/Version-1.0.1-blue.svg)](https://semver.org)
[![DOI](https://zenodo.org/badge/728080000.svg)](https://zenodo.org/doi/10.5281/zenodo.10396919)
[![Python Version](https://img.shields.io/badge/Python-3.8|3.9|3.10|3.11-yellow.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/paravis.svg)](https://pypi.org/project/paravis/1.0.1/)



***
## Overview
_PaRaVis_ is a powerful Graphical package designed for extracting, visualizing, and analyzing Rao's Q based on VIs and raster datasets. It comprises three main sections, each enhancing usability and functionality for different process aspects.

### 1. Vegetation Index Analysis
provides a user-friendly interface for importing stacked raster datasets, selecting specific bands, and calculating multiple indices simultaneously. spatial visualizations with customization options enhance the exploration of vegetation indices. Users can also select and save calculated indices (based on their CV value) as GeoTIFF files for future analysis.

### 2. Rao's Q Index Computation
Focuses on the computation of Rao's Q index from raster files in both unidimensional and multidimensional modes. offering  parameter customization options. parallelization using the Ray framework optimizes computational efficiency, reducing processing time significantly. The code employs the "tqdm" library to monitor processing time.

### 3. Visualization and Analysis
Emphasizes visualizing and analyzing Rao's Q outputs through an intuitive interface. Various widgets facilitate seamless file selection, output path definition, and customization of plot settings. The tool generates insightful plots, histograms, difference heatmaps, and split plots, making complex operations accessible to users.

## Installation and Usage
You can effortlessly install _PaRaVis_ from PyPI using the following command:
```bash
pip install paravis
```
<br/>

If you are using _PaRaVis_ on Debian-based Linux distributions like Ubuntu operating system, you should also install the following package for tkinter support befor using _PaRaVis_ :
```bash
sudo apt-get install python3-tk
```
<br/>

> [!NOTE]
> To customize theme and cell size for a better experience within Jupyter Notebook, use the following magic command from your jupyter notebook, and remember to refresh for changes to take effect (use F5):
```bash
!jt -t grade3 -cellw 100% -N -T -kl
```
<br/>

> [!IMPORTANT]
> If you are using Jupyter Notebook Version 7 and above, you might encounter compatibility issues, as it has undergone changes that break backward compatibility. In such cases, it is recommended to use [nbclassic](https://github.com/jupyter/nbclassic) instead of Jupyter Notebook.

In Jupyter Notebook or Jupyter Lab you can import different modules of _PaRaVis_ as follows:

|Module| Import Statement| Description|
|------------------------------|---------------------------------|-------------------------------------|
| Vegetation Index Analysis| ```from paravis import Indices```| Calculate and visually represent vegetation indices.|
| Rao's Q Index Computation| `from paravis import Raos`| Perform Rao's Q index computation with customizable options.|
| Visualization and Analysis| `from paravis import Visualize`| Visualize, analyze, and compare outputs using this module.|
<br/>

__For more information about the _PaRaVis_ tool, please refer to the [__Documentation__](Documentation/Documentation.md).__

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
