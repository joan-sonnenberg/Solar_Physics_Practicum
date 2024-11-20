"""
Created on Wed Nov 20

@author: Joan Sonnenberg

Code plots ratio of the two datasets to observe interesting features in the data. Note: pixel direction has been inerted to grow in wavelength.

"""

import numpy as np
import os
import glob
from astropy.io import fits
import numpy as np
from scipy.signal import find_peaks_cwt
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path
from tqdm import tqdm

# insert local data folder paths
folder_data_sun = 'C:/Users/joans/Desktop/Natuur- en Sterrenkunde/Year 2/Period 2/N&S Practicum 2/Practicum/Solar_Physics_Practicum/Solar_Physics_Practicum/Flux_raw_sun_centre'
folder_data_sunspot = 'C:/Users/joans/Desktop/Natuur- en Sterrenkunde/Year 2/Period 2/N&S Practicum 2/Practicum/Solar_Physics_Practicum/Solar_Physics_Practicum/Flux_raw_sunspot3477'

# Create a folder named 'photos' in the current working directory
output_folder = "Ratios"
os.makedirs(output_folder, exist_ok=True)

# Loop through the data files
for n in range(0, 24, 1):
    N_order = n
    # Load data for the sun and sunspot
    data_order_N_sun = np.loadtxt(os.path.join(folder_data_sun, f"data_raw_order_{N_order}.csv"), delimiter=',')
    data_order_N_sunspot = np.loadtxt(os.path.join(folder_data_sunspot, f"data_raw_order_{N_order}.csv"), delimiter=',')

    # Extract pixel values and flux data
    x_pixelvalues_sun = np.arange(len(data_order_N_sun[0]))
    x_pixelvalues_sunspot = np.arange(len(data_order_N_sunspot[0]))
    flux_object_sun = data_order_N_sun[4]
    flux_object_sunspot = data_order_N_sunspot[4]

    # Compute the ratio and smooth it
    ratio = gaussian_filter1d(flux_object_sun / flux_object_sunspot, sigma=10)
    inverse = ratio[::-1]

    # Create the plot
    fig4 = plt.figure()
    plt.plot(inverse, label="ratio", color="red")
    plt.xlabel("Pixels")
    plt.ylabel("Ratio")
    plt.title(f"Order {N_order}")

    # Save the plot in the 'photos' folder
    output_path = os.path.join(output_folder, f"Ratio_{N_order}.jpg")
    plt.savefig(output_path, dpi=500)

    # Close the figure to free up memory
    plt.close(fig4)