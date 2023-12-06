#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## import os
# List of libraries required by the script
needed_libraries = [
    "tkinter",
    "ipywidgets",
    "xarray",
    "rioxarray",
    "rasterio",
    "numpy",
    "spyndex",
    "pandas",
    "matplotlib",
    "seaborn",
    "ray",
    "tqdm",
]

# List to store missing libraries
missing_libraries = []

# Loop through the required libraries and try importing them
for library in needed_libraries:
    try:
        __import__(library)
    except ImportError:
        # If the library is not found, add it to the list of missing libraries
        missing_libraries.append(library)
        # Use os.system to run 'pip install' to install the missing library
        os.system(f"pip install {library}")
# Check if any libraries were missing and print the results
if missing_libraries:
    print("Installed the following missing libraries:")
    for library in missing_libraries:
        print(library)
else:
    print("All required libraries are already installed.")


# In[ ]:


try:
    import jupyterthemes

    themes_installed = True
except ImportError:
    themes_installed = False

if themes_installed:
    print("Jupyter Themes is already installed.")
else:
    # Install jupyterthemes package
    get_ipython().system('pip install jupyterthemes')
    print("Jupyter Themes is installed.")

# Change cell width and apply a theme
if themes_installed:
    # Apply the 'grade3' theme with a cell width of 100%
    get_ipython().system('jt -t grade3 -cellw 100% -N -T')
    # -t grade3: Applies the 'grade3' theme.
    # -cellw 100%: Sets the cell width to 100%.
    # -T: Changes the toolbar appearance.
    print("Themes have been changed. Please refresh this page (press F5).")


# In[ ]:


# Import libraries
import math
import os
import platform
import subprocess
import tkinter as tk
import warnings
from itertools import combinations
from tkinter import Tk, filedialog

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rasterio
import ray
import rioxarray
import seaborn as sns
import spyndex
import xarray as xr
from IPython.display import clear_output, display
from ipywidgets import interactive, widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.io import MemoryFile
from tqdm import tqdm


# In[ ]:


# Disable all warnings
warnings.filterwarnings("ignore")
plt.rcParams[
    "font.family"
] = "DejaVu Sans"  # Use a font that supports Unicode characters

# Create the main Tkinter window (desplay on top)
root = tk.Tk()
root.wm_attributes("-topmost", 1)
root.eval("tk::PlaceWindow . center")
root.withdraw()

# Define a dic for datasets
indices_dict = {}


# Function to open and reproject GeoTIFF datasets
def open_gd(bt):
    clear_output(wait=True)  # Clear the output area
    # Select GeoTIFF datasets using Tkinter
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        filetypes=(("GeoTIFF files", "*.tif;*.tiff"), ("All files", "*.*"))
    )
    global dataset_dict
    dataset_dict = {}  # For saving selected datasets
    for file_path in file_paths:
        global file_name
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        dataset = xr.open_rasterio(file_path).astype("float64")

        # Reprojection datasets
        dst_crs = "EPSG:4326"  # Define the target CRS (WGS 84, EPSG:4326)
        dataset = dataset.rio.reproject(dst_crs)  # Reproject the dataset to EPSG:4326
        dataset = dataset.where(dataset != 0, np.nan)  # Replace 0 values with NaN
        bounds = dataset.rio.transform()
        dataset_dict[file_name] = {"dataset": dataset, "bounds": bounds}
        print("GeoTIFF datasets loaded:", dataset_dict.keys())
        display(widgets_row)  # Display the main container
        return dataset_dict


# Normalize the bands to the range [0, 1]
def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return (band - band_min) / ((band_max - band_min))


# For calculating desire indices
def calculate_indices(b):
    if len(index_selection.value) == 0:
        print("Error: Please select one or more indices to campute.")
        return
    selected_bands = [selection_widget.value for selection_widget in band_selection]
    selected_indices = (
        list(index_selection.value)
        if len(index_selection.value) > 1
        else list(index_selection.value)[0]
    )
    global indices_dict
    indices_dict = {}
    for file_name, data in dataset_dict.items():
        dataset = data["dataset"]
        bounds = data["bounds"]
        snt = dataset / 10000
        index = index_names

        # Specify parameters for indices calculation
        params = {
            "B": snt.sel(band=selected_bands[0]),
            "G": snt.sel(band=selected_bands[1]),
            "R": snt.sel(band=selected_bands[2]),
            "RE1": snt.sel(band=selected_bands[3]),
            "N": snt.sel(band=selected_bands[4]),
            "N2": snt.sel(band=selected_bands[5]),
            "L": 0.5,
            "g": 2.5,
            "C1": 6,
            "C2": 7.5,
            "kNN": 0.5,
            "kNR": spyndex.computeKernel(
                kernel="RBF",
                params={
                    "a": snt.sel(band=selected_bands[4]),
                    "b": snt.sel(band=selected_bands[2]),
                    "sigma": snt.sel(band=[selected_bands[4], selected_bands[2]]).mean(
                        "band"
                    ),
                },
            ),
        }

        # Calculate the indices and place them in the defined dictionary
        idx = spyndex.computeIndex(index=selected_indices, params=params)
        indices_dict[file_name] = {}

        # Add bands to dictionary as seperate Gtiff
        for band in snt.band:
            selected_band_data = snt.sel(band=band)
            bands_data = normalize(selected_band_data)
            indices_dict[file_name][f"Band{band.item()}"] = bands_data
        if len(index_selection.value) > 1:
            for i in selected_indices:
                indices_data = idx.sel(index=f"{i}")
                normalized_data = normalize(indices_data)
                indices_dict[file_name][i] = normalized_data
        else:
            indices_data = idx
            normalized_data = normalize(indices_data)
            indices_dict[file_name][selected_indices] = normalized_data

    # Calculate and add CV values
    for dataset, indices in indices_dict.items():
        for index_name, index_data in indices.items():
            mean_value = np.mean(index_data)
            std_deviation = np.std(index_data)
            cv_percentage = (std_deviation / mean_value) * 100  # CV in percentage

            # Add to existing dictionary
            indices[index_name]["cv_percentage"] = cv_percentage
    print("Indices calculation finished!")
    dataset_up_dr()


# For Ploting figures
def plot_figure(button):
    clear_output(wait=True)  # Clear the output for multiple attempts

    # Get values from widgets
    selected_dataset = dataset_dropdown.value
    selected_indices = indices_dropdown.value
    selected_cmap = cmap_dropdown.value
    cv_value = float(indices_dict[selected_dataset][selected_indices]["cv_percentage"])
    bounds = dataset_dict[selected_dataset]["bounds"]
    selected_indices_data = indices_dict[selected_dataset][selected_indices]
    normalized_indices_data = normalize(selected_indices_data)

    # Figure options
    plt.figure(figsize=(20, 5))
    plt.imshow(normalized_indices_data, cmap=selected_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(pad=0.01, label="Value")
    cbar.ax.get_yaxis().label.set_fontsize(12)  # Set font size for the colorbar label
    cbar.ax.get_yaxis().label.set_fontweight(
        "bold"
    )  # Set font weight for colorbar label
    plt.title(
        f"{selected_indices} - {selected_dataset} - CV: {cv_value:.2f}%\n",
        size=14,
        fontweight="semibold",
    )
    x_ticks = np.linspace(
        bounds[2], (bounds[2] + selected_indices_data.shape[1] * bounds[0]), num=5
    )
    y_ticks = np.linspace(
        bounds[5], (bounds[5] + selected_indices_data.shape[0] * bounds[4]), num=5
    )
    plt.xticks(
        np.linspace(0, selected_indices_data.shape[1], num=5),
        ["{:.2f}\u00b0 {}".format(tick, "W" if tick < 0 else "E") for tick in x_ticks],
        size=10,
        fontweight="semibold",
        ha="left",
    )
    plt.yticks(
        np.linspace(0, selected_indices_data.shape[0], num=5),
        ["{:.2f}\u00b0 {}".format(tick, "S" if tick < 0 else "N") for tick in y_ticks],
        size=10,
        fontweight="semibold",
    )
    plt.gca().xaxis.get_major_ticks()[4].label1.set_visible(False)
    plt.gca().xaxis.get_major_ticks()[4].tick1line.set_visible(False)
    plt.gca().yaxis.get_major_ticks()[4].label1.set_visible(False)
    plt.gca().yaxis.get_major_ticks()[4].tick1line.set_visible(False)

    plt.xlabel("Longitude", size=12, fontweight="semibold")
    plt.ylabel("Latitude", size=12, fontweight="semibold")
    plt.xticks(rotation=0)  # Rotate the x-axis ticks
    plt.yticks(rotation=90)  # Rotate the y-axis ticks
    display(widgets_row)
    plt.show()


# For selecting the Path
def select_path(button):
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.askdirectory()
    save_path_text.value = save_path
    save_path_text_2.value = save_path


# For saving the figure
def save_figure(button):

    # Get values from widgets
    selected_dataset = dataset_dropdown.value
    selected_indices = indices_dropdown.value
    selected_cmap = cmap_dropdown.value
    save_path = save_path_text.value
    save_name = save_name_text.value
    dpi = dpi_dropdown.value
    file_format = format_dropdown.value
    cv_value = float(indices_dict[selected_dataset][selected_indices]["cv_percentage"])
    bounds = dataset_dict[selected_dataset]["bounds"]
    selected_indices_data = indices_dict[selected_dataset][selected_indices]
    normalized_indices_data = normalize(selected_indices_data)

    # Figure options
    plt.figure(figsize=(20, 5))
    plt.imshow(normalized_indices_data, cmap=selected_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(pad=0.01, label="Value")
    cbar.ax.get_yaxis().label.set_fontsize(12)
    cbar.ax.get_yaxis().label.set_fontweight("bold")
    plt.title(
        f"{selected_indices} - {selected_dataset} - CV: {cv_value:.2f}%\n",
        size=14,
        fontweight="semibold",
    )
    x_ticks = np.linspace(
        bounds[2], (bounds[2] + selected_indices_data.shape[1] * bounds[0]), num=5
    )
    y_ticks = np.linspace(
        bounds[5], (bounds[5] + selected_indices_data.shape[0] * bounds[4]), num=5
    )
    plt.xticks(
        np.linspace(0, selected_indices_data.shape[1], num=5),
        ["{:.2f}\u00b0 {}".format(tick, "W" if tick < 0 else "E") for tick in x_ticks],
        size=10,
        fontweight="semibold",
        ha="left",
    )
    plt.yticks(
        np.linspace(0, selected_indices_data.shape[0], num=5),
        ["{:.2f}\u00b0 {}".format(tick, "S" if tick < 0 else "N") for tick in y_ticks],
        size=10,
        fontweight="semibold",
    )
    plt.gca().xaxis.get_major_ticks()[4].label1.set_visible(False)
    plt.gca().xaxis.get_major_ticks()[4].tick1line.set_visible(False)
    plt.gca().yaxis.get_major_ticks()[4].label1.set_visible(False)
    plt.gca().yaxis.get_major_ticks()[4].tick1line.set_visible(False)
    plt.xlabel("Longitude", size=12, fontweight="semibold")
    plt.ylabel("Latitude", size=12, fontweight="semibold")
    plt.yticks(rotation=90)  # Rotate the y-axis ticks
    save_figure_path = f"{save_path}/{save_name}.{file_format}"
    plt.savefig(save_figure_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Figure saved as {save_figure_path}")


# For saving Gtiff of selected VI files
def save_indices(button):
    # Get user-selected inputs
    selected_dataset = dataset_dropdown_2.value
    selected_indices_with_cv = index_tosave.value
    save_path = save_path_text.value
    dataset_info = dataset_dict[selected_dataset]

    # Iterate over selected indices and save selected indices GeoTIFFs
    for index_with_cv in selected_indices_with_cv:
        index_name = index_with_cv.split(" ---------- ")[0]
        index_data = indices_dict[selected_dataset][f"{index_name}"]
        output_path = f"{save_path}/{selected_dataset}_{index_name}.tif"
        original_dataset = dataset_info["dataset"]
        index_data.rio.write_crs(original_dataset.rio.crs, inplace=True)
        index_data.rio.to_raster(output_path, compress="lzw")
        print(f"Index '{index_name}' saved as {output_path}")


# For updating the dataset dropdowns
def dataset_up_dr():

    # First dataset dropdown
    dataset_dropdown.options = list(indices_dict.keys())
    dataset_dropdown.value = list(dataset_dict.keys())[0]
    indices_dropdown.options = list(indices_dict[file_name].keys())

    # Second dataset dropdown
    dataset_dropdown_2.options = list(dataset_dict.keys())
    dataset_dropdown_2.value = list(dataset_dict.keys())[0]


# For updating the index_tosave widget with indices sorted by CV value
def indices_up_dr(change):
    selected_location = change.new
    selected_indices = indices_dict.get(selected_location, {})
    sorted_indices = sorted(
        selected_indices.items(), key=lambda x: x[1]["cv_percentage"], reverse=True
    )  # Sort indices based on CV value
    index_tosave.options = [
        f"{index_name} ---------- CV value: {index_info['cv_percentage']:.2f}%"
        for index_name, index_info in sorted_indices
    ]


# Create header widgets
header_widget1 = widgets.HTML(
    "<h3 style='font-family: Arial, sans-serif; color: white; font-weight:semibold; background-color: blue; text-align: center;'>Bands Order</h3>"
)
header_widget2 = widgets.HTML(
    "<h3 style='font-family: Arial, sans-serif; color: white; font-weight: semibold; background-color: blue; text-align: center;'>Indices Selection</h3>"
)
header_widget3 = widgets.HTML(
    "<h3 style='font-family: Arial, sans-serif; color: white; font-weight: semibold; background-color: blue; text-align: center;'>Indices Visualization</h3>"
)
header_widget4 = widgets.HTML(
    "<h3 style='font-family: Arial, sans-serif; color: white; font-weight: semibold; background-color: blue; text-align: center;'>Saving Indices</h3>"
)

# Create a button widget for loading
load_button = widgets.Button(description="Load Datasets")

# Create GUI for band selection
band_names = ["Blue", "Green", "Red", "Red Edge", "Near Infrared 1", "Near Infrared 2"]
default_band_numbers = [2, 3, 5, 6, 7, 8]  # Default band numbers (for WV-2 dataset)
band_selection = []
for i in range(len(band_names)):
    selection_widget = widgets.Dropdown(
        options=list(range(1, 10)),
        value=default_band_numbers[i],  # Set default value
        description=band_names[i] + ":",
    )
    band_selection.append(selection_widget)

# Indices list
index_names = [
    "ARI",
    "ARI2",
    "BAI",
    "BCC",
    "BNDVI",
    "CIG",
    "CIRE",
    "CVI",
    "DVI",
    "EVI",
    "EVI2",
    "ExG",
    "ExGR",
    "ExR",
    "FCVI",
    "GARI",
    "GBNDVI",
    "GCC",
    "GLI",
    "GNDVI",
    "GOSAVI",
    "GRNDVI",
    "GRVI",
    "GSAVI",
    "IKAW",
    "IPVI",
    "MCARI",
    "MCARI1",
    "MCARI2",
    "MCARIOSAVI",
    "MGRVI",
    "MNLI",
    "MRBVI",
    "MSAVI",
    "MSR",
    "MTVI1",
    "MTVI2",
    "NDREI",
    "NDTI",
    "NDVI",
    "NDWI",
    "NDYI",
    "NGRDI",
    "NIRv",
    "NLI",
    "NormG",
    "NormNIR",
    "NormR",
    "OSAVI",
    "RCC",
    "RDVI",
    "RGBVI",
    "RGRI",
    "RI",
    "SARVI",
    "SAVI",
    "SI",
    "SR",
    "SR2",
    "SR3",
    "SeLI",
    "TCARI",
    "TCARIOSAVI",
    "TCI",
    "TDVI",
    "TGI",
    "TVI",
    "TriVI",
    "VARI",
    "VARI700",
    "VI700",
    "VIG",
    "kIPVI",
    "kNDVI",
    "kRVI",
]

# Create multiselection for selecting indices to calculate
index_selection = widgets.SelectMultiple(
    options=index_names, layout=widgets.Layout(width="300px", height="190px")
)

# Create multiselection box for indices we want to save
index_tosave = widgets.SelectMultiple(
    layout=widgets.Layout(width="300px", height="90px")
)

# For indices calculation button
calculate_button = widgets.Button(description="Calculate Indices")
calculate_button.layout.margin = "0px 0px 0px 140px"  # Adjust the margins
calculate_button.layout.width = "150px"

# For plotting and saving the figure
plot_button = widgets.Button(description="Plot Figure")
save_button = widgets.Button(description="Save Figure")
path_button = widgets.Button(description="Output Path")

# For entering the save path and plot name
save_path_text = widgets.Text(
    description="Output Path:", placeholder="Enter output path here"
)
save_name_text = widgets.Text(
    description="Output Name:", placeholder="Enter plot name here", value="index_fig"
)

# For DPI, Format and Colormap
dpi_dropdown = widgets.Dropdown(
    description="DPI:", options=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
)
format_dropdown = widgets.Dropdown(
    description="Format:", options=["png", "jpg", "svg", "tiff", "tif"]
)
cmap_dropdown = widgets.Dropdown(
    description="Colormap:", options=plt.colormaps(), value="nipy_spectral"
)

# Dataset and Indices dropdowns
indices_dropdown = widgets.Dropdown(options=[], description="Indices:")
dataset_dropdown = widgets.Dropdown(
    options=list(indices_dict.keys()), description="Dataset:"
)

# For dataset selection
dataset_dropdown_2 = widgets.Dropdown(
    options=list(indices_dict.keys()), description="Dataset:"
)

# For save path
save_path_text_2 = widgets.Text(
    description="Output Path:", placeholder="Enter output path here"
)
path_button_2 = widgets.Button(description="Output Path")
save_button_2 = widgets.Button(description="Save Indices")
path_button_2.layout.margin = "4px 4px 0px 155px"
save_button_2.layout.margin = "6px 6px 0px 155px"

# ordering Containers for widgets
plot_container = widgets.HBox([cmap_dropdown, plot_button])
path_container = widgets.HBox([save_path_text, path_button])
dropdowns_container = widgets.VBox([dpi_dropdown, format_dropdown])
save_container = widgets.HBox([save_name_text, save_button])
main_container = widgets.VBox(
    [
        dataset_dropdown,
        indices_dropdown,
        plot_container,
        dropdowns_container,
        path_container,
        save_container,
    ]
)
widgets_container = widgets.VBox(
    [dataset_dropdown_2, index_tosave, save_path_text_2, path_button_2, save_button_2]
)
index_widget = widgets.VBox([index_selection, calculate_button])
vbox_margin = "10px"
widgets_row = widgets.HBox(
    [
        load_button,
        widgets.VBox([header_widget1] + band_selection),
        widgets.VBox([header_widget2, index_widget]),
        widgets.VBox([header_widget3, main_container]),
        widgets.VBox([header_widget4, widgets_container]),
    ],
    layout=widgets.Layout(justify_content="space-between", margin=f"0px {vbox_margin}"),
)
display(widgets_row)

# Attach the update function to the dropdown's value change event
dataset_dropdown_2.observe(indices_up_dr, names="value")

# Attach functions to buttons
load_button.on_click(open_gd)
calculate_button.on_click(calculate_indices)
plot_button.on_click(plot_figure)
save_button.on_click(save_figure)
path_button.on_click(select_path)
path_button_2.on_click(select_path)
save_button_2.on_click(save_indices)


# In[ ]:


# For updating widgets base on uni or multidimensional mode
def update_input(mode_value):
    if mode_value == "unidimensional":
        input_text.description = "Input File:"
        input_text.rows = 1  # Limit selection to one file
        input_button.description = "Input"
    else:
        input_text.description = "Input Files:"
        input_text.rows = len(index_names)  # Display more rows for multi-file selection
        input_button.description = "Inputs"


# For selecting the input(s)
def select_input(btn):
    root = tk.Tk()
    root.withdraw()
    if mode.value == "unidimensional":
        file_path = filedialog.askopenfilename(
            filetypes=[("TIFF Files", "*.tif" or "*.ttif")]
        )
        if file_path:
            input_text.options = [file_path]
            input_text.value = os.path.basename(
                file_path
            )  # Assign the file name without the path
            global input_indice
            input_indice = [rasterio.open(file_path)]
    else:
        file_paths = filedialog.askopenfilenames(
            filetypes=[("TIFF Files", "*.tif" or "*.ttif")]
        )
        if file_paths:
            input_text.options = list(file_paths)
            input_text.value = "\n".join(
                [os.path.basename(fp) for fp in file_paths]
            )  # Display file names only
            input_indices.clear()
            input_indices.extend([rasterio.open(fp) for fp in file_paths])


# For selecting output path
def select_path(button):
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.askdirectory()
    output_text.value = save_path


# for getting available Memory
def get_memory_usage():
    system_platform = platform.system()

    # Linux-specific memory monitoring
    if system_platform == "Linux":
        try:
            output = subprocess.check_output(["free", "-b"])
            lines = output.decode("utf-8").split("\n")
            data = lines[1].split()
            available_memory = int(data[6])
            return available_memory / (1024**3)  # Convert to GB
        except Exception as e:
            print("Error:", str(e))
            return None

    # Windows-specific memory monitoring
    elif system_platform == "Windows":
        try:
            response = os.popen('systeminfo | find "Available Physical Memory"').read()
            available_memory = int(
                response.split(":")[1].strip().split(" ")[0].replace(",", "")
            ) / (
                1024
            )  # Convert to GB
            return available_memory
        except Exception as e:
            print("Error:", str(e))
            return None
    else:
        print("Unsupported operating system:", system_platform)
        return None


# Get the sudo pass if the platform is linux
if platform.system() == "Linux":
    sudo_password = input("Enter your sudo password: ")
    clear_output(wait=True)  # Clear the output


# Call rao function when a button is clicked
def click(btn):
    ray.shutdown()  # for running again
    p_minkowski = p_minkowskii.value  # Get the degree param for minkowski

    # Compute Euclidean distance between two vectors
    def euclidean_dist(pair_list):
        return math.sqrt(sum([(x[0] - x[1]) ** 2 for x in pair_list]))

    # Compute Manhattan distance between two vectors
    def manhattan_dist(pair_list):
        return sum([abs(x[0] - x[1]) for x in pair_list])

    # Compute Chebyshev distance between two vectors
    def chebyshev_dist(pair_list):
        return max([abs(x[0] - x[1]) for x in pair_list])

    # Compute Jaccard distance between two vectors
    def jaccard_dist(pair_list):
        dists = []
        for x in pair_list:
            numerator = min(x[0], x[1])
            denominator = max(x[0], x[1])
            dists.append(1 - (numerator / denominator))
        return sum(dists)

    # Compute canberra distance between two vectors
    def canberra_dist(pair_list):
        dists = []
        for x in pair_list:
            numerator = abs(x[0] - x[1])
            denominator = abs(x[0]) + abs(x[1])
            dists.append(numerator / denominator)
        return sum(dists)

    # Compute Minkowski distance between two vectors with parameter p
    def minkowski_dist(pair_list, p_minkowski):
        return sum(
            [(abs(x[0] - x[1]) ** p_minkowski) ** (1 / p_minkowski) for x in pair_list]
        )

    # Convert TIFF input(s) to NumPy array
    def tiff_to_np(tiff_input):
        matrix1 = tiff_input.read()
        matrix1 = matrix1.reshape((matrix1.shape[1]), matrix1.shape[2])
        minNum = -999
        matrix1[matrix1 == minNum] = np.nan
        return matrix1

    # Write the computation output to a GeoTIFF file
    def export_geotiff(naip_meta, output_rao, output_path):
        naip_meta["count"] = 1
        naip_meta["dtype"] = "float64"
        with rasterio.open(output_path, "w", **naip_meta) as dst:
            dst.write(output_rao, 1)

    # Computes Rao's Q index for a specified range of rows and columns
    @ray.remote
    def compute_raoq_range(
        row_start,
        row_end,
        col_start,
        col_end,
        trastersm_list,
        window,
        distance_m,
        na_tolerance,
    ):
        w = int(
            (window - 1) / 2
        )  # Number of neighbors from the central pixel to the edge of the window
        raoq_values = []  # Initialize a list to store computed Rao's Q values

        # iterate through rows and columns
        for rw in range(row_start, row_end):
            for cl in range(col_start, col_end):

                # Create a list of border condition results for all input arrays
                borderCondition = [
                    np.sum(
                        np.invert(np.isnan(x[rw - w : rw + w + 1, cl - w : cl + w + 1]))
                    )
                    < np.power(window, 2) - ((np.power(window, 2)) * na_tolerance)
                    for x in trastersm_list
                ]

                # Check if any array satisfies the border condition
                if True in borderCondition:
                    raoq_values.append(np.nan)
                else:

                    # Extract sub-windows for all input arrays
                    tw = [
                        x[rw - w : rw + w + 1, cl - w : cl + w + 1]
                        for x in trastersm_list
                    ]
                    lv = [x.ravel() for x in tw]  # Flatten the sub-windows

                    # Generate combinations of sub-window pairs
                    vcomb = combinations(range(lv[0].shape[0]), 2)
                    vcomb = list(vcomb)
                    vout = []

                    # Calculate  selected distances for all sub-window pairs
                    for comb in vcomb:
                        lpair = [[x[comb[0]], x[comb[1]]] for x in lv]
                        if distance_m == "euclidean":
                            out = euclidean_dist(lpair)
                        elif distance_m == "manhattan":
                            out = manhattan_dist(lpair)
                        elif distance_m == "chebyshev":
                            out = chebyshev_dist(lpair)
                        elif distance_m == "canberra":
                            out = canberra_dist(lpair)
                        elif distance_m == "minkowski":
                            out = minkowski_dist(lpair, p_minkowski)
                        elif distance_m == "Jaccard":
                            out = jaccard_dist(lpair)
                        vout.append(out)

                    # Rescale the computed distances and calculate Rao's Q value
                    vout_rescaled = [x * 2 for x in vout]
                    vout_rescaled[:] = [x / window**4 for x in vout_rescaled]
                    raoq_value = np.nansum(vout_rescaled)
                    raoq_values.append(raoq_value)

        # Return the results for the specified row and column range
        return row_start, row_end, col_start, col_end, raoq_values

    # Parallelizes the computation of Rao's Q
    def parallel_raoq(
        data_input,
        output_path,
        distance_m="euclidean",
        window=9,
        na_tolerance=0.0,
        batch_size=100,
    ):
        if window % 2 == 1:
            w = int((window - 1) / 2)
        else:
            raise Exception(
                "The size of the moving window must be an odd number. Exiting..."
            )

        # Convert input data to NumPy arrays
        numpy_data = [tiff_to_np(data) for data in data_input]

        # Initialize raoq array with NaN values
        raoq = np.zeros(shape=numpy_data[0].shape)
        raoq[:] = np.nan

        # Create a list of transformed arrays for each input
        trastersm_list = []
        for mat in numpy_data:
            trasterm = np.zeros(shape=(mat.shape[0] + 2 * w, mat.shape[1] + 2 * w))
            trasterm[:] = np.nan
            trasterm[w : w + mat.shape[0], w : w + mat.shape[1]] = mat
            trastersm_list.append(trasterm)

        # Adjust batch size to fit all pixels
        max_rows = numpy_data[0].shape[0] - 2 * w + 1
        max_cols = numpy_data[0].shape[1] - 2 * w + 1
        batch_size = min(batch_size, max_rows, max_cols)

        # Adjust row and column batches
        rows = numpy_data[0].shape[0]
        cols = numpy_data[0].shape[1]
        row_batches = range(w, rows + w, batch_size)
        col_batches = range(w, cols + w, batch_size)

        # Adjust the last batch
        row_batches = list(row_batches)
        col_batches = list(col_batches)
        if row_batches[-1] != rows + w:
            row_batches.append(rows + w)
        if col_batches[-1] != cols + w:
            col_batches.append(cols + w)

        # Use Ray to parallelize the computation
        ray_results = []
        for row_start, row_end in zip(row_batches[:-1], row_batches[1:]):
            for col_start, col_end in zip(col_batches[:-1], col_batches[1:]):
                pixel_data = (
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                    trastersm_list,
                    window,
                    distance_m,
                    na_tolerance,
                )
                ray_results.append(compute_raoq_range.remote(*pixel_data))

        # Update raoq array with the computed values
        with tqdm(total=len(ray_results)) as pbar:
            for result in ray_results:
                row_start, row_end, col_start, col_end, raoq_values = ray.get(result)
                raoq[
                    row_start - w : row_end - w, col_start - w : col_end - w
                ] = np.array(raoq_values).reshape(
                    row_end - row_start, col_end - col_start
                )
                pbar.update(1)

        # Export the computed Rao's Q index as a TIFF file
        info = data_input[0].profile
        export_geotiff(info, raoq, output_path)

    # Use the obtained password with the sudo command for linux platform
    if platform.system() == "Linux":
        command = f'echo "{sudo_password}" | sudo -S mount -o remount,size={memory_slider.value}G /dev/shm'
        os.system(command)

    # Initialize Ray
    ray.init(
        num_cpus=num_cpu_cores.value, object_store_memory=memory_slider.value * 10**9
    )
    output = parallel_raoq(
        data_input=(input_indice if mode.value == "unidimensional" else input_indices),
        output_path=(
            r"{}/{}.tif".format(output_text.value, output_filename_text.value)
        ),
        distance_m=distance_options.value,
        window=window.value,
        na_tolerance=na_tolerance.value,
        batch_size=batch_size.value,
    )
    # Shutdown Ray
    ray.shutdown()


input_indices = []  # List to store input rasterio.DatasetReader instances

# Create widgets for input parameters
mode = widgets.ToggleButtons(
    options=["unidimensional", "multidimensional"],
    description="Mode:",
    value="unidimensional",
)

# Define the toggle buttons
distance_options = widgets.ToggleButtons(
    options=["euclidean", "manhattan", "chebyshev", "Jaccard", "canberra", "minkowski"],
    description="Distance:",
    value="euclidean",
)

# Create the layout using GridBox
buttons_layout = widgets.GridBox(
    children=[distance_options],
    layout=widgets.Layout(grid_template_columns="repeat(1, 1fr)"),
)

# Create widgets for input parameters
output_text = widgets.Text(
    description="Output Path:", placeholder="Enter output path here"
)
output_button = widgets.Button(description="Output Path")
output_filename_text = widgets.Text(description="Output name:", value="Rao")
p_minkowskii = widgets.BoundedIntText(description="degree:", value=2, min=2, max=5000)
window = widgets.BoundedIntText(description="Window:", min=1, max=333, step=2, value=3)
na_tolerance = widgets.BoundedFloatText(
    description="NA Tolerance:", min=0, max=1, step=0.1, value=0.0
)
batch_size = widgets.BoundedIntText(
    description="Batch size:", min=10, max=10000, step=10, value=100
)
input_text = widgets.Textarea(
    value="", description="Input File:", placeholder="Input file(s) name"
)

# Create a Dropdown for workers
import multiprocessing

num_cpu_cores = widgets.Dropdown(
    options=list(range(1, multiprocessing.cpu_count() + 1)),
    description="CPU workers:",
    value=4,
)

# Get available memory
available_memory = get_memory_usage()
if available_memory is not None:
    # Calculate the maximum value for the slider (85% of available memory)
    max_slider_value = int(available_memory * 0.85)
else:
    print("Unable to retrieve available memory information.")
    max_slider_value = 10

# Create a slider for available memory
memory_slider = widgets.IntSlider(
    value=0, min=1, max=max_slider_value, step=1, description=f"Memory (GB):"
)

# Buttons
input_button = widgets.Button(description="Input")
run_button = widgets.Button(description="Run")

# adjustment
input_button.layout.margin = "0px 0px 0px 50px"
p_minkowskii.layout.margin = "0px 0px 0px 50px"
run_button.layout.margin = "0px 0px 0px 170px"
input_text.layout.height = "90px"
input_text.layout.width = "350px"

# Update data input widget when mode is changed
mode.observe(lambda change: update_input(change["new"]), names="value")

# Header
header_widget = widgets.HTML(
    "<h3 style='font-family: Arial, sans-serif; color: white; font-weight: semibold;background-color: blue; text-align: center;'>Extract Rao’s-Q diversity</h3>"
)
header_widget.layout.width = "300px"
header_widget.layout.height = "25px"
display(header_widget)

# Create box layouts for widgets
input_widgets = widgets.VBox(
    [
        widgets.HBox(
            [
                mode,
                input_button,
                output_text,
                output_button,
                output_filename_text,
                input_text,
            ]
        ),
        widgets.HBox([window, na_tolerance, batch_size, num_cpu_cores, memory_slider]),
        widgets.HBox([buttons_layout, p_minkowskii, run_button]),
    ],
    layout=widgets.Layout(justify_content="space-between", margin=f"0px {vbox_margin}"),
)

# Attach functions to buttons
input_button.on_click(select_input)
output_button.on_click(select_path)
run_button.on_click(click)

# Display the input widgets
display(input_widgets)


# In[ ]:


plt.rcParams[
    "font.family"
] = "DejaVu Sans"  # Use a font that supports Unicode characters for ticks

# To open and add selected files to the dictionary
def add_files_to_dict(i):
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        filetypes=(("GeoTIFF files", "*.tif;*.tiff"), ("All files", "*.*"))
    )
    files = []
    if file_paths:
        for file_path in file_paths:
            file_name = os.path.basename(file_path).split(".")[0]
            files.append((file_name, file_path))
        for file_name, src in files:
            file_dict[file_name] = src
            print(f"File '{file_name}' added to the dictionary.")
        update_file_dropdown()
        update_files_select_multiple()
    else:
        print("No file selected.")


# To update dropdown widgets with file names
def update_file_dropdown():
    file_dropdown.options = list(file_dict.keys())


# To update the SelectMultiple widget with file names
def update_files_select_multiple():
    files_select_multiple.options = list(file_dict.keys())


# To select an output directory
def select_path(b):
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    if directory:
        output_directory_textbox.value = directory


# To normalize data (0,1)
def normalize_data(data):
    if np.isnan(np.nanmin(data)) or np.isnan(np.nanmax(data)):
        # Handle cases where all values are NaN
        normalized_data = np.zeros_like(data)
    else:
        normalized_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    return normalized_data


# Changing ticks format to dms
def decimal_degrees_to_dms(dd):
    degrees = int(dd)
    minutes = int((dd - degrees) * 60)
    seconds = ((dd - degrees) * 60 - minutes) * 60
    return degrees, minutes, seconds


# Function to plot an individual file with matplotlib
def plot_individual_file(b):
    clear_output(wait=True)
    display(header_widget, widget_box)
    file_name = file_dropdown.value
    file_path = file_dict.get(file_name, None)  # Get the file_path from the dict
    colormap = colormap_dropdown.value
    output_dir = output_directory_textbox.value
    output_file_name = output_filename_textbox.value
    dpi = dp_dropdown.value
    file_format = fmt_dropdown.value
    output_filename = os.path.join(
        output_dir, f"{file_name}_{output_file_name}.{file_format}"
    )  # For saving
    if file_path and colormap and output_dir and output_file_name:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            transform = src.transform

            # Extract latitude and longitude from the georeferencing information
            lat, long = transform * (
                0,
                0,
            )  # Extract the coordinates from the top-left corner
            lat_deg, lat_min, lat_sec = decimal_degrees_to_dms(lat)
            long_deg, long_min, long_sec = decimal_degrees_to_dms(long)
        num_rows, num_cols = data.shape
        x = np.arange(0, num_cols) * transform.a + transform.c
        y = np.arange(0, num_rows) * transform.e + transform.f

        plt.figure(figsize=(8, 6))
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=1))

        im = plt.imshow(
            normalize_data(data),
            cmap=colormap,
            extent=[x.min(), x.max(), y.min(), y.max()],
        )
        cbar = plt.colorbar(im, label="Value", pad=0.01)
        cbar.ax.get_yaxis().label.set_fontsize(
            12
        )  # Set font size for the colorbar label
        cbar.ax.get_yaxis().label.set_fontweight(
            "bold"
        )  # Set font weight for colorbar label
        plt.title(f"{file_name}\n", fontsize=14, fontweight="bold")
        x_ticks = np.linspace(x.min(), x.max(), num=len(plt.xticks()[0]))
        y_ticks = np.linspace(y.min(), y.max(), num=len(plt.yticks()[0]))

        # Set the x and y tick labels as DMS format with symbols
        plt.gca().set_xticklabels(
            [
                "%.0f° %.0f' %.0f\" %s"
                % (*decimal_degrees_to_dms(val), "W" if val < 0 else "E")
                for val in x_ticks
            ]
        )
        plt.gca().set_yticklabels(
            [
                "%.0f° %.0f' %.0f\" %s"
                % (*decimal_degrees_to_dms(val), "S" if val < 0 else "N")
                for val in y_ticks
            ]
        )
        plt.xticks(rotation=0, fontsize=10, ha=ha_widget.value, fontweight="bold")
        plt.yticks(rotation=90, fontsize=10, va=va_widget.value, fontweight="bold")
        plt.xlabel("Longitude", fontsize=12, fontweight="bold")
        plt.ylabel("Latitude", fontsize=12, fontweight="bold")
        plt.grid(False)
        if save_checkbox.value:
            plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
            print(f"Individual plot saved as '{output_filename}'.")
        plt.show()
    else:
        print(
            "Error: Please select a file, colormap, output directory, and output file name."
        )


# To plot statistics in IQR range for selected files
def plot_selected_files(b):
    clear_output(wait=True)
    display(header_widget, widget_box)
    selected_files = files_select_multiple.value
    if selected_files:
        colormap = colormap_dropdown.value
        output_dir = output_directory_textbox.value
        output_file_name = output_filename_textbox.value
        dpi = dp_dropdown.value
        file_format = fmt_dropdown.value
        plot_type = plot_types_dropdown.value
        color_palette = color_palette_dropdown.value
        output_filename = os.path.join(
            output_dir, f"{plot_type}_{output_file_name}.{file_format}"
        )
        sns.set_palette(sns.palettes.SEABORN_PALETTES[color_palette])

        # Creating the figure
        plt.figure(figsize=(8, 6))
        data_dict = {"Files": [], "Pixel Values": []}
        for file_name in selected_files:
            file_path = file_dict.get(file_name, None)
            if file_path:
                with rasterio.open(file_path) as src:
                    data = normalize_data(src.read(1).flatten())
                data_dict["Files"].extend([file_name] * len(data))
                data_dict["Pixel Values"].extend(data)
        data_df = pd.DataFrame(data_dict)

        # Define the lower and upper bounds to identify outliers and filter them
        IQR = data_df["Pixel Values"].quantile(0.75) - data_df["Pixel Values"].quantile(
            0.25
        )
        lower_bound = data_df["Pixel Values"].quantile(0.25) - 1.5 * IQR
        upper_bound = data_df["Pixel Values"].quantile(0.75) + 1.5 * IQR
        data_df = data_df[
            (data_df["Pixel Values"] >= lower_bound)
            & (data_df["Pixel Values"] <= upper_bound)
        ]

        # Different statistic plots
        if plot_type == "Box Plot":
            sns.violinplot(
                data=data_df,
                x="Files",
                y="Pixel Values",
                inner="quartile",
                bw=0.02,
                palette=sns.color_palette(),
                flierprops=dict(marker=""),
                width=0.4,
                dodge=0.2,
            )
            sns.boxplot(
                data=data_df,
                x="Files",
                y="Pixel Values",
                palette=color_palette_dropdown.value,
                flierprops=dict(marker=""),
                width=0.5,
                dodge=0.2,
            )
            sns.stripplot(
                data=data_df,
                x="Files",
                y="Pixel Values",
                color="black",
                alpha=0.002,
                jitter=0.35,
                size=0.7,
                marker="D",
                palette=color_palette_dropdown.value,
            )
        elif plot_type == "Histogram":
            sns.histplot(
                data=data_df,
                x="Pixel Values",
                kde=False,
                stat="density",
                hue="Files",
                common_norm=False,
                palette=color_palette_dropdown.value,
            )
        elif plot_type == "KDE Plot":
            sns.kdeplot(
                data=data_df,
                x="Pixel Values",
                hue="Files",
                color="black",
                lw=1,
                linestyle="--",
            )
        elif plot_type == "Violin Plot":
            sns.violinplot(
                data=data_df,
                x="Files",
                y="Pixel Values",
                inner="quartile",
                bw=0.02,
                palette=sns.color_palette(),
                flierprops=dict(marker=""),
                width=0.4,
                dodge=0.2,
            )
            sns.stripplot(
                data=data_df,
                x="Files",
                y="Pixel Values",
                color="black",
                alpha=0.002,
                jitter=0.35,
                size=0.7,
                marker="D",
                palette=color_palette_dropdown.value,
            )

        # Customize the plot appearance
        plt.title(f"{plot_type}\n", fontsize=14, fontweight="bold")
        plt.xlabel(
            "Rao's Value"
            if plot_type == "Histogram"
            else "Rao's Value"
            if plot_type == "KDE Plot"
            else "Files",
            fontsize=12,
            fontweight="bold",
        )
        plt.ylabel(
            "Rao's Value"
            if plot_type == "Box Plot"
            else "Density"
            if plot_type == "KDE Plot"
            else "Rao's Value",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.5)
        sns.despine(trim=True, offset=5)
        plt.xticks(
            rotation=45, fontsize=10, fontweight="bold"
        )  # Rotate x-axis labels horizontally
        plt.yticks(
            rotation=45, fontsize=10, fontweight="bold"
        )  # Rotate x-axis labels horizontally

        # For saving figure
        if save_checkbox.value == True:
            plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
            print(f"selected files plot saved as '{output_filename}'.")
        plt.show()
    else:
        print("Error: Please select one or more files and a plot type.")


# Compute and plot the difference between two selected files
def difference_files(b):
    clear_output(wait=True)
    display(header_widget, widget_box)
    colormap = colormap_dropdown.value
    output_dir = output_directory_textbox.value
    output_file_name = output_filename_textbox.value
    dpi = dp_dropdown.value
    file_format = fmt_dropdown.value
    selected_files = files_select_multiple.value

    # Error for selecting more or less than 2 files
    if len(selected_files) != 2:
        print("Error: Please select exactly 2 files for differencing.")
        return
    file1_name, file1_path = selected_files[0], file_dict.get(selected_files[0], None)
    file2_name, file2_path = selected_files[1], file_dict.get(selected_files[1], None)
    output_filename = os.path.join(
        output_dir, f"{file1_name}_{file2_name}_{output_file_name}.{file_format}"
    )
    if file1_path and file2_path:
        with rasterio.open(file1_path) as src1, rasterio.open(file2_path) as src2:
            data1 = normalize_data(src1.read(1))
            data2 = normalize_data(src2.read(1))
            transform = src2.transform  # Get the transform object for one of 2 file

            # Extract latitude and longitude from the georeferencing information
            lat, long = transform * (
                0,
                0,
            )  # Extract the coordinates from the top-left corner
            lat_deg, lat_min, lat_sec = decimal_degrees_to_dms(lat)
            long_deg, long_min, long_sec = decimal_degrees_to_dms(long)
        global subtraction_data
        subtraction_data = normalize_data(abs(data1 - data2))
        # Create a meshgrid of coordinates
        num_rows, num_cols = data2.shape
        global x, y
        x = np.arange(0, num_cols) * transform.a + transform.c
        y = np.arange(0, num_rows) * transform.e + transform.f
        # Creating the figure
        plt.figure(figsize=(8, 6))
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        im = plt.imshow(
            subtraction_data, cmap=colormap, extent=[x.min(), x.max(), y.min(), y.max()]
        )
        cbar = plt.colorbar(im, label="Value", pad=0.01)
        cbar.ax.get_yaxis().label.set_fontsize(
            12
        )  # Set font size for the colorbar label
        cbar.ax.get_yaxis().label.set_fontweight(
            "bold"
        )  # Set font weight for colorbar label
        plt.title(
            f"Difference plot\n{file1_name}----{file2_name}\n",
            fontsize=14,
            fontweight="bold",
        )
        x_ticks = np.linspace(x.min(), x.max(), num=len(plt.xticks()[0]))
        y_ticks = np.linspace(y.min(), y.max(), num=len(plt.yticks()[0]))

        # Set the x and y tick labels as DMS format with symbols
        plt.gca().set_xticklabels(
            [
                "%.0f° %.0f' %.0f\" %s"
                % (*decimal_degrees_to_dms(val), "W" if val < 0 else "E")
                for val in x_ticks
            ]
        )
        plt.gca().set_yticklabels(
            [
                "%.0f° %.0f' %.0f\" %s"
                % (*decimal_degrees_to_dms(val), "S" if val < 0 else "N")
                for val in y_ticks
            ]
        )
        plt.xticks(rotation=0, fontsize=10, ha=ha_widget.value, fontweight="bold")
        plt.yticks(rotation=90, fontsize=10, va=va_widget.value, fontweight="bold")
        plt.xlabel("Longitude", fontsize=12, fontweight="bold")
        plt.ylabel("Latitude", fontsize=12, fontweight="bold")
        plt.grid(False)
        if save_checkbox.value == True:
            plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
            print(f"Difference plot saved as '{output_filename}'.")
        plt.show()
    else:
        print("Error: Unable to subtract files. Please ensure both files are selected.")


# To generate a generative heatmap plot for the difference between two selected files with an slidebar
def heatmap_files(b):
    clear_output(wait=True)
    display(header_widget, widget_box)
    colormap = colormap_dropdown.value
    output_dir = output_directory_textbox.value
    output_file_name = output_filename_textbox.value
    dpi = dp_dropdown.value
    file_format = fmt_dropdown.value
    selected_files = files_select_multiple.value
    if len(selected_files) != 2:
        print("Error: Please select exactly 2 files for differencing.")
        return
    if "subtraction_data" not in globals():
        print("Please generate Difference Plot first.")
        return
    file1_name, file1_path = selected_files[0], file_dict.get(selected_files[0], None)
    file2_name, file2_path = selected_files[1], file_dict.get(selected_files[1], None)

    file_path = file_dict.get(selected_files, None)
    window_size = int(window_size_slider.value)

    output_filename = os.path.join(
        output_dir,
        f"Difference_heatmap_{file1_name}_{file2_name}_window{window_size}_{output_file_name}.{file_format}",
    )
    if file1_path and file2_path:
        with rasterio.open(file1_path) as src:
            transform = src.transform  # Get the transform object for one of the files
            data = normalize_data(subtraction_data)
            nrows, ncols = data.shape

            # Extract latitude and longitude from the georeferencing information
            lat, long = transform * (
                0,
                0,
            )  # Extract the coordinates from the top-left corner
            lat_deg, lat_min, lat_sec = decimal_degrees_to_dms(lat)
            long_deg, long_min, long_sec = decimal_degrees_to_dms(long)

            # Calculate the grid cell size based on the selected window size
            grid_size_rows = max(1, nrows // window_size)
            grid_size_cols = max(1, ncols // window_size)
            grid_means = np.zeros((grid_size_rows, grid_size_cols))
            for i in range(grid_size_rows):
                for j in range(grid_size_cols):
                    row_start = i * window_size
                    row_end = min((i + 1) * window_size, nrows)
                    col_start = j * window_size
                    col_end = min((j + 1) * window_size, ncols)
                    grid = data[row_start:row_end, col_start:col_end]
                    valid_pixels = np.isfinite(grid)  # Find non-NaN values
                    grid_mean = np.nanmean(
                        grid[valid_pixels]
                    )  # Calculate mean for valid pixels
                    grid_means[i, j] = grid_mean

            # Create x and y arrays based on the transform
            x = np.arange(0, ncols) * transform.a + transform.c
            y = np.arange(0, nrows) * transform.e + transform.f
            plt.figure(figsize=(8, 6))
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
            im = plt.imshow(
                data, cmap=colormap, extent=[x.min(), x.max(), y.min(), y.max()]
            )

            # Grid lines
            for i in range(grid_size_rows):
                plt.axhline(y[i * window_size], color="white", linewidth=0.5)
            for j in range(grid_size_cols):
                plt.axvline(x[j * window_size], color="white", linewidth=0.5)

            # Calculate the number of grid lines in x and y directions
            num_x_lines = grid_size_cols + 1
            num_y_lines = grid_size_rows + 1

            # Calculate the difference between adjacent grid lines
            grid_cell_width = x[1] - x[0]
            grid_cell_height = y[1] - y[0]

            # Calculate font size based on the number of grid lines
            min_grid_line_count = min(num_x_lines, num_y_lines)
            font_size = (
                100000
                * np.mean([abs(grid_cell_width), abs(grid_cell_height)])
                * 60
                / min_grid_line_count
            )  # Adjust 60

            # Display grid means as text with adjusted font size
            for i in range(grid_size_rows):
                for j in range(grid_size_cols):
                    plt.text(
                        x[j * window_size] + (window_size / 2) * transform.a,
                        y[i * window_size] + (window_size / 2) * transform.e,
                        f"{round(grid_means[i, j], 2)}",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=font_size,  # Adjusted font size
                        fontweight="bold",
                    )
            cbar = plt.colorbar(im, label="Value", pad=0.01)
            cbar.ax.get_yaxis().label.set_fontsize(
                12
            )  # Set font size for the colorbar label
            cbar.ax.get_yaxis().label.set_fontweight(
                "bold"
            )  # Set font weight for colorbar label
            plt.title(
                f"Difference Heatmap Plot\nFiles: {file1_name}-{file2_name}   Grid Size: {window_size}\n",
                fontsize=14,
                fontweight="bold",
            )
            x_ticks = np.linspace(x.min(), x.max(), num=len(plt.xticks()[0]))
            y_ticks = np.linspace(y.min(), y.max(), num=len(plt.yticks()[0]))

            # Set the x and y tick labels as DMS format with symbols
            plt.gca().set_xticklabels(
                [
                    "%.0f° %.0f' %.0f\" %s"
                    % (*decimal_degrees_to_dms(val), "W" if val < 0 else "E")
                    for val in x_ticks
                ]
            )
            plt.gca().set_yticklabels(
                [
                    "%.0f° %.0f' %.0f\" %s"
                    % (*decimal_degrees_to_dms(val), "S" if val < 0 else "N")
                    for val in y_ticks
                ]
            )
            plt.xticks(rotation=0, fontsize=10, ha=ha_widget.value, fontweight="bold")
            plt.yticks(rotation=90, fontsize=10, va=va_widget.value, fontweight="bold")
            plt.xlabel("Longitude", fontsize=12, fontweight="bold")
            plt.ylabel("Latitude", fontsize=12, fontweight="bold")
            if save_checkbox.value:
                plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
                print(f"Difference heatmap plot saved as '{output_filename}'.")
            plt.show()
    else:
        print("Error: Please select a file to generate the heatmap plot.")


# To create an split_plot as an attractive plot for comparing to selected files
def split_plot(_):
    clear_output(wait=True)
    display(header_widget, widget_box)
    colormap = colormap_dropdown.value
    output_dir = output_directory_textbox.value
    output_file_name = output_filename_textbox.value
    dpi = dp_dropdown.value
    file_format = fmt_dropdown.value
    selected_files = files_select_multiple.value
    if len(selected_files) != 2:
        print("Error: Please select exactly 2 files to compare.")
        return
    file1_name, file1_path = selected_files[0], file_dict.get(selected_files[0], None)
    file2_name, file2_path = selected_files[1], file_dict.get(selected_files[1], None)
    if file1_path and file2_path:
        with rasterio.open(file1_path) as src1, rasterio.open(file2_path) as src2:
            left_data = normalize_data(src1.read(1))
            right_data = normalize_data(src2.read(1))
            right_crs = src2.crs
            right_transform = (
                src2.transform
            )  # Get the transform object for one of 2 file

    # To create the combined raster from to selected files
    def create_combined_raster(split_size):
        combined_data = np.copy(left_data)
        # Calculate the split column index
        split_position = int(split_size * left_data.shape[1])
        # Combine the left and right rasters
        combined_data[:, split_position:] = right_data[:, split_position:]
        return combined_data

    # To display and save the split plot
    def display_split_view(split_size):
        combined_data = create_combined_raster(split_size)

        # Create a new rasterio dataset for the combined data
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=combined_data.shape[0],
                width=combined_data.shape[1],
                count=1,
                dtype=combined_data.dtype,
                crs=right_crs,
                transform=right_transform,
            ) as dataset:
                dataset.write(combined_data, 1)

            # Plot the combined image with the split line
            with memfile.open() as combined_ds:
                combined_data = combined_ds.read(1)
                combined_transform = (
                    combined_ds.transform
                )  # Get the transform object for one of 2 file

                # Extract latitude and longitude from the georeferencing information
                lat, long = combined_transform * (
                    0,
                    0,
                )  # Extract the coordinates from the top-left corner
                lat_deg, lat_min, lat_sec = decimal_degrees_to_dms(lat)
                long_deg, long_min, long_sec = decimal_degrees_to_dms(long)

                # Create a meshgrid of coordinates
                num_rows, num_cols = combined_data.shape
                global x, y  # Define x and y as global variables
                x = np.arange(0, num_cols) * combined_transform.a + combined_transform.c
                y = np.arange(0, num_rows) * combined_transform.e + combined_transform.f
                plt.figure(figsize=(8, 6))
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
                im = plt.imshow(combined_data, cmap=colormap, origin="upper")
                cbar = plt.colorbar(im, label="Value", pad=0.01)
                cbar.ax.get_yaxis().label.set_fontsize(
                    12
                )  # Set font size for the colorbar label
                cbar.ax.get_yaxis().label.set_fontweight(
                    "bold"
                )  # Set font weight for colorbar label
                plt.title(
                    f"Split Plot\nLeft: {file1_name} ------------ Right: {file2_name}\n",
                    fontsize=14,
                    fontweight="bold",
                )
                x_ticks = np.linspace(x.min(), x.max(), num=len(plt.xticks()[0]))
                y_ticks = np.linspace(y.min(), y.max(), num=len(plt.yticks()[0]))
                y_ticks = y_ticks[::-1]

                # Set the x and y tick labels as DMS format with symbols
                plt.gca().set_xticklabels(
                    [
                        "%.0f° %.0f' %.0f\" %s"
                        % (*decimal_degrees_to_dms(val), "W" if val < 0 else "E")
                        for val in x_ticks
                    ]
                )
                plt.gca().set_yticklabels(
                    [
                        "%.0f° %.0f' %.0f\" %s"
                        % (*decimal_degrees_to_dms(val), "S" if val < 0 else "N")
                        for val in y_ticks
                    ]
                )
                plt.xticks(
                    rotation=0, fontsize=10, ha=ha_widget.value, fontweight="bold"
                )
                plt.yticks(
                    rotation=90, fontsize=10, va=va_widget.value, fontweight="bold"
                )

                # Set the axis labels as latitude and longitude
                plt.xlabel("Longitude", fontsize=12, fontweight="bold")
                plt.ylabel("Latitude", fontsize=12, fontweight="bold")
                plt.grid(False)

                # Add the split line
                split_position = int(split_size * left_data.shape[1])
                plt.axvline(x=split_position, color="red", linestyle="--", linewidth=1)
                if save_checkbox.value == True:
                    output_filename = os.path.join(
                        output_dir,
                        f"SplitPlot_{file1_name}_{file2_name}_{output_file_name}.{file_format}",
                    )
                    plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
                    print(f"Split plot saved as '{output_filename}'.")
                plt.show()

    # To save the split plot with slider changes
    def save_plot(split_size):
        combined_data = create_combined_raster(split_size)
        colormap = colormap_dropdown.value
        output_dir = output_directory_textbox.value
        output_file_name = output_filename_textbox.value
        dpi = dp_dropdown.value
        file_format = fmt_dropdown.value

        # Create a new rasterio dataset for the combined data
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=combined_data.shape[0],
                width=combined_data.shape[1],
                count=1,
                dtype=combined_data.dtype,
                crs=right_crs,
                transform=right_transform,
            ) as dataset:
                dataset.write(combined_data, 1)

            # Plot the combined image with the split line
            with memfile.open() as combined_ds:
                combined_data = combined_ds.read(1)
                combined_transform = combined_ds.transform

                # Extract latitude and longitude from the georeferencing information
                lat, long = combined_transform * (
                    0,
                    0,
                )  # Extract the coordinates from the top-left corner
                lat_deg, lat_min, lat_sec = decimal_degrees_to_dms(lat)
                long_deg, long_min, long_sec = decimal_degrees_to_dms(long)

                # Create a meshgrid of coordinates
                num_rows, num_cols = combined_data.shape
                x = np.arange(0, num_cols) * combined_transform.a + combined_transform.c
                y = np.arange(0, num_rows) * combined_transform.e + combined_transform.f
                y = y[::-1]
                plt.figure(figsize=(8, 6))
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
                im = plt.imshow(combined_data, cmap=colormap, origin="upper")
                cbar = plt.colorbar(im, label="Value", pad=0.01)
                cbar.ax.get_yaxis().label.set_fontsize(
                    12
                )  # Set font size for the colorbar label
                cbar.ax.get_yaxis().label.set_fontweight(
                    "bold"
                )  # Set font weight for colorbar label
                plt.title(
                    f"Split Plot\nLeft: {file1_name} ------------ Right: {file2_name}\n",
                    fontsize=14,
                    fontweight="bold",
                )
                x_ticks = np.linspace(x.min(), x.max(), num=len(plt.xticks()[0]))
                y_ticks = np.linspace(y.min(), y.max(), num=len(plt.yticks()[0]))
                y_ticks = y_ticks[::-1]
                # Set the x and y tick labels as DMS format with symbols
                plt.gca().set_xticklabels(
                    [
                        "%.0f° %.0f' %.0f\" %s"
                        % (*decimal_degrees_to_dms(val), "W" if val < 0 else "E")
                        for val in x_ticks
                    ]
                )
                plt.gca().set_yticklabels(
                    [
                        "%.0f° %.0f' %.0f\" %s"
                        % (*decimal_degrees_to_dms(val), "S" if val < 0 else "N")
                        for val in y_ticks
                    ]
                )
                plt.xticks(
                    rotation=0, fontsize=10, ha=ha_widget.value, fontweight="bold"
                )
                plt.yticks(
                    rotation=90, fontsize=10, va=va_widget.value, fontweight="bold"
                )
                plt.xlabel("Longitude", fontsize=12, fontweight="bold")
                plt.ylabel("Latitude", fontsize=12, fontweight="bold")
                plt.grid(False)

                # Add the split line
                split_position = int(split_size * left_data.shape[1])
                plt.axvline(x=split_position, color="red", linestyle="--", linewidth=1)

                # Save the plot
                output_filename = os.path.join(
                    output_dir,
                    f"SplitPlot_{file1_name}_{file2_name}_{output_file_name}.{file_format}",
                )
                plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
                print(f"Split plot saved as '{output_filename}'.")
                plt.show()
                clear_output(wait=True)
                display(header_widget, widget_box)
                # Display the interactive plot
                display(widgets.HBox([interactive_plot, save_gif_button]))

    # Function to generate and save a GIF with a moving split line using Matplotlib
    def save_gif():
        file_format = fmt_dropdown.value
        global x, y
        images = []
        fig, ax = plt.subplots(figsize=(8, 6))
        split_line = ax.axvline(
            x=0, color="red", linestyle="--", linewidth=1
        )  # Initialize the line for the split
        cbar = plt.colorbar(
            ax.imshow(left_data, cmap=colormap, origin="upper"), label="Value", pad=0.01
        )
        cbar.ax.get_yaxis().label.set_fontsize(
            12
        )  # Set font size for the colorbar label
        cbar.ax.get_yaxis().label.set_fontweight(
            "bold"
        )  # Set font weight for colorbar label

        def animate(split_position):
            ax.clear()
            combined_data = create_combined_raster(split_position)
            im = ax.imshow(combined_data, cmap=colormap, origin="upper")
            ax.set_title(
                f"Left: {file1_name} ------------ Right: {file2_name}\n",
                fontsize=14,
                fontweight="bold",
            )
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
            x_ticks = np.linspace(x.min(), x.max(), num=len(plt.xticks()[0]))
            y_ticks = np.linspace(y.min(), y.max(), num=len(plt.yticks()[0]))
            y_ticks = y_ticks[::-1]
            # Set the x and y tick labels as DMS format with symbols
            plt.gca().set_xticklabels(
                [
                    "%.0f° %.0f' %.0f\" %s"
                    % (*decimal_degrees_to_dms(val), "W" if val < 0 else "E")
                    for val in x_ticks
                ]
            )
            plt.gca().set_yticklabels(
                [
                    "%.0f° %.0f' %.0f\" %s"
                    % (*decimal_degrees_to_dms(val), "S" if val < 0 else "N")
                    for val in y_ticks
                ]
            )
            plt.xticks(rotation=0, fontsize=10, ha=ha_widget.value, fontweight="bold")
            plt.yticks(rotation=90, fontsize=10, va=va_widget.value, fontweight="bold")
            ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
            ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
            ax.grid(False)
            # Add the split line
            split_position_pixel = int(split_position * left_data.shape[1])
            ax.axvline(x=split_position_pixel, color="red", linestyle="--", linewidth=1)
            # Update the split line position
            split_position_pixel = int(split_position * left_data.shape[1])
            split_line.set_xdata([split_position_pixel, split_position_pixel])
            # Append the current frame to the list of images
            images.append([split_line])

        # Create the animation
        anim = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, 1.01, 0.01), interval=100
        )

        # Save the animation as a GIF
        gif_filename = os.path.join(
            output_dir,
            f"SplitPlotAnimation_{file1_name}_{file2_name}_{output_file_name}.gif",
        )
        anim.save(
            gif_filename,
            writer="pillow",
            dpi=dpi,
            fps=10,
            savefig_kwargs={"format": file_format},
        )  # Specify the frame format)  # Adjust fps as needed
        clear_output(wait=True)
        display(header_widget, widget_box)

        # Display the interactive plot
        display(widgets.HBox([interactive_plot, save_gif_button]))
        print(f"Split plot animation saved as '{gif_filename}'.")

    # Create an interactive widget to adjust the split size
    split_slider = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.01,
        description="Split Size:",
        continuous_update=False,
    )

    # Callback function to handle slider value changes
    def on_slider_value_change(change):
        split_size = change["new"]
        if save_checkbox.value:
            save_plot(split_size)

    # Attach the callback function to the slider's value attribute
    split_slider.observe(on_slider_value_change, names="value")

    # Create the interactive plot
    interactive_plot = interactive(display_split_view, split_size=split_slider)

    # Button to save the plot as a GIF
    save_gif_button = widgets.Button(description="Generate GIF")
    save_gif_button.on_click(lambda x: save_gif())

    # Display the button
    display(widgets.HBox([interactive_plot, save_gif_button]))


# For imported datasets
file_dict = {}

# GUI components
select_files_button = widgets.Button(description="Select Raster Files")
select_files_button.on_click(add_files_to_dict)
output_directory_textbox = widgets.Text(
    description="Output Path:", placeholder="Enter output path here"
)
output_directory_button = widgets.Button(description="Output Path")
output_directory_button.on_click(select_path)
output_filename_textbox = widgets.Text(description="Output Name:", value="RaoVis")
file_dropdown = widgets.Dropdown(
    options=list(file_dict.keys()), description="Select File:"
)
files_select_multiple = widgets.SelectMultiple(
    options=list(file_dict.keys()), description="Select Files:"
)
files_select_multiple.layout.height = "90px"
files_select_multiple.layout.width = "350px"
plot_types_dropdown = widgets.Dropdown(
    options=["Box Plot", "Histogram", "KDE Plot", "Violin Plot"],
    description="Plot Type:",
)
color_palette_dropdown = widgets.Dropdown(
    options=list(sns.palettes.SEABORN_PALETTES.keys()),
    value=list(sns.palettes.SEABORN_PALETTES.keys())[
        7
    ],  # Set the seven palette as default
    description="Color Palette:",
)
dp_dropdown = widgets.Dropdown(
    description="DPI:",
    options=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    value=500,
)
fmt_dropdown = widgets.Dropdown(
    description="Format:", options=["png", "jpg", "svg", "tiff", "tif"], value="jpg"
)
colormap_dropdown = widgets.Dropdown(
    options=plt.colormaps(), description="Colormap:", value="nipy_spectral"
)
save_checkbox = widgets.Checkbox(description="save plot")
plot_individual_button = widgets.Button(description="Plot")
plot_individual_button.on_click(plot_individual_file)
plot_selected_button = widgets.Button(description="Plot")
plot_selected_button.on_click(plot_selected_files)
difference_files_button = widgets.Button(description="Difference Plot")
difference_files_button.on_click(difference_files)
window_size_slider = widgets.IntSlider(
    value=600, min=1, max=5000, step=1, description="Grid Size:", readout_format=".1f"
)
from ipywidgets import Layout, interact, widgets

# Create widget sliders for HA and VA
ha_widget = widgets.Dropdown(
    options=["center", "right", "left"],
    value="center",
    description="H-Alignment:",
    layout=Layout(width="140px"),  # Adjust the width to make it smaller
)

va_widget = widgets.Dropdown(
    options=["center", "top", "bottom", "baseline"],
    value="center",
    description="V-Alignment:",
    layout=Layout(width="140px"),
)
heatmap_button = widgets.Button(description="Difference Heatmap")
heatmap_button.on_click(heatmap_files)
split_button = widgets.Button(description="Split Plot")
split_button.on_click(split_plot)

# Arrange the widgets in a VBox for display
widget_box = widgets.HBox(
    [
        widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.HBox([select_files_button, save_checkbox]),
                        widgets.HBox(
                            [output_directory_textbox, output_directory_button]
                        ),
                        output_filename_textbox,
                        widgets.HBox([ha_widget, va_widget]),
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                file_dropdown,
                                colormap_dropdown,
                                dp_dropdown,
                                widgets.HBox([fmt_dropdown, plot_individual_button]),
                            ]
                        )
                    ]
                ),
                widgets.HBox(
                    [
                        files_select_multiple,
                        widgets.VBox(
                            [
                                color_palette_dropdown,
                                plot_types_dropdown,
                                window_size_slider,
                            ]
                        ),
                        widgets.VBox(
                            [
                                plot_selected_button,
                                difference_files_button,
                                heatmap_button,
                                split_button,
                            ]
                        ),
                    ]
                ),
            ]
        )
    ]
)

header_widget = widgets.HTML(
    "<h3 style='font-family: Arial, sans-serif; color: white; font-weight: semibold; background-color: blue; text-align: center;'>Rao’s-Q Visualization</h3>"
)
header_widget.layout.width = "300px"
header_widget.layout.height = "25px"
display(header_widget, widget_box)

