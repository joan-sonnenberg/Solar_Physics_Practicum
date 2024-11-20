"""
Created on Wed Nov 20

@author: Joan Sonnenberg

Code normalizes the data for sun and sunspot for an order and plots. Allows to compare and note differences.

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
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path
from tqdm import tqdm

# insert local data folder paths
folder_data_sun = 'C:/Users/joans/Desktop/Natuur- en Sterrenkunde/Year 2/Period 2/N&S Practicum 2/Practicum/Solar_Physics_Practicum/Solar_Physics_Practicum/Flux_raw_sun_centre'
folder_data_sunspot = 'C:/Users/joans/Desktop/Natuur- en Sterrenkunde/Year 2/Period 2/N&S Practicum 2/Practicum/Solar_Physics_Practicum/Solar_Physics_Practicum/Flux_raw_sunspot3477'


def Normalization(N_order, main_folder):
    
    
    wavelength_list_complete = [[], [], [], [], [], [], [], [], [], [], [], [], [5187.7462, 5162.2845, 5176.961], [], [], [], [], [], [], [], [], [], [], []]

    x_list_complete = [[], [], [], [], [], [], [], [], [], [], [], [], [4230, 4859, 4485], [], [], [], [], [], [], [], [], [], [], []]

    uncertainty_x_complete = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]

    
    data_order_N = np.loadtxt(os.path.join(main_folder, "data_raw_order_{}.csv").format(N_order),  delimiter=',')

    x_pixelvalues = np.arange(len(data_order_N[0]))
    thar = data_order_N[0]
    tungstenflat = data_order_N[1]
    bias = data_order_N[2]
    dark = data_order_N[3]
    flux_object = data_order_N[4]
    SNR = data_order_N[5]
    darkflat = data_order_N[6]


    plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
    plt.plot(x_pixelvalues,thar, label = 'ThAr')
    plt.plot(x_pixelvalues,tungstenflat, label = 'Tungsten')
    plt.plot(x_pixelvalues,bias, label = 'Bias')
    plt.plot(x_pixelvalues,dark, label = 'Dark')
    plt.plot(x_pixelvalues,flux_object, label = 'Object')
    plt.plot(x_pixelvalues,SNR, label = 'SNR')
    plt.plot(x_pixelvalues,darkflat, label = 'darkflat')
    plt.legend()
    #plt.show()
    
    wavelength_list = wavelength_list_complete[N_order]
    x_list = x_list_complete[N_order]
    uncertainty_x = uncertainty_x_complete[N_order]
    plt.plot(x_pixelvalues, thar)
    plt.scatter(x_list, thar[x_list], c='red', label = 'calibration points' )
    for index in range(len(x_list)):
        plt.text(x_list[index]+20, thar[x_list][index]+20, wavelength_list[index], size=8)
    plt.legend()
    #plt.show()
    
    fit_order = 2
    fit_1 = np.polynomial.polynomial.polyfit(x_list,wavelength_list,fit_order,w=uncertainty_x)

    # x & y coordinaten van de fit
    wavelength_object = []
    for x in x_pixelvalues:
        y = 0
        # Calculate y_coordinate
        for n in range(len(fit_1)):
            y += fit_1[n] * (x)**n       
        # Save coordinates
        wavelength_object.append(y)   


    #  Residuals berekenen

    residuals = []
    for i, x_value in enumerate(x_list):
        # Bereken de voorspelde waarde met de fit-coëfficiënten
        predicted_wavelength = sum(fit_1[n] * (x_value)**n for n in range(len(fit_1)))
        
        # Bereken het residual door het verschil te nemen tussen de werkelijke en voorspelde waarde
        residual = wavelength_list[i] - predicted_wavelength
        residuals.append(residual)
        
    # lekker plotten:

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [7, 2]})
    fig.subplots_adjust(hspace=0)

    yerr = abs(uncertainty_x*np.array(fit_1[1]))

    ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength)")
    ax1.plot(x_pixelvalues, wavelength_object)
    ax1.set_ylabel("Wavelength [Angstrom]")
    ax1.errorbar(x_list, wavelength_list, yerr=yerr, fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
    ax1.scatter(x_list,wavelength_list, c='blue')



    ax2.errorbar(x_list, residuals, yerr=yerr, fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
    ax2.scatter(x_list,residuals)
    ax2.set_ylabel("Pixels")
    ax2.set_ylabel("Residuals [Angstrom]")
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, label = 'model')
    ax2.axhline(fit_1[1], color='gray', linestyle='--', linewidth=1, label = '1 pixel difference')
    ax2.axhline(-1*fit_1[1], color='gray', linestyle='--', linewidth=1)
    for index in range(len(x_list)):
        ax2.text(x_list[index], residuals[index], wavelength_list[index], size=8)
    plt.legend()
    #plt.show()
    
    plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
    plt.plot(wavelength_object,(flux_object-dark)/(tungstenflat-darkflat))
    plt.ylim(0,)
    plt.xlim(6560, 6565)
    #plt.show()
    
    fit_degree = 20
    fit_coeffs = np.polyfit(wavelength_object, (flux_object - dark) / (tungstenflat - darkflat), fit_degree)
    polynomial_fit = np.poly1d(fit_coeffs)
    fitted_flux = polynomial_fit(wavelength_object)
    normalized_flux_o = (flux_object - dark) / (tungstenflat - darkflat) / fitted_flux

    plt.figure(dpi=300)
    plt.plot(wavelength_object, (flux_object - dark) / (tungstenflat - darkflat), label='Original Flux')
    plt.plot(wavelength_object, fitted_flux, label=f'Polynomial Fit (Degree {fit_degree})', linestyle='--')
    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux', alpha=0.8)
    plt.ylim(0,)
    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title("Flux Normalization using Polynomial Fit")
    #plt.show()

    #############################################################

    plt.figure(dpi=300)

    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux', alpha=0.8)
    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title("Flux Normalization using Polynomial Fit")
    #plt.show()

    fit_degree = 20


    from scipy.signal import find_peaks


    total_flux = (flux_object - dark) / (tungstenflat - darkflat)


    interval = 100
    peak_indices = []

    for start in range(0, len(total_flux), interval):
        end = min(start + interval, len(total_flux))
        segment = total_flux[start:end]
        peaks, properties = find_peaks(segment, height=0)  
        if peaks.size > 0:
            
            highest_peak_index = start + peaks[np.argmax(properties['peak_heights'])]
            
            if not (6540 < wavelength_object[highest_peak_index] < 6542):
                peak_indices.append(highest_peak_index)

    wavelength_peaks = np.array(wavelength_object)[peak_indices]
    flux_peaks = total_flux[peak_indices]


    fit_coeffs = np.polyfit(wavelength_peaks, flux_peaks, fit_degree)


    polynomial_fit = np.poly1d(fit_coeffs)


    fitted_flux = polynomial_fit(wavelength_object)


    normalized_flux = total_flux / fitted_flux


    plt.figure(figsize=(16.5, 11.7), dpi=300)
    plt.plot(wavelength_object, total_flux, label='Original Flux')
    plt.plot(wavelength_object, fitted_flux, label=f'Polynomial Fit (Degree {fit_degree})', linestyle='--')
    plt.scatter(wavelength_peaks, flux_peaks, color='red', label='Selected Peaks')
    #plt.plot(wavelength_object, normalized_flux, label='Normalized Flux', alpha=0.8)
    plt.ylim(0,)
    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title("Flux Normalization using Polynomial Fit through Selected Peaks")
    #plt.savefig("dots.jpg", dpi=500)
    #plt.show()



    plt.figure(dpi=300)
    plt.plot(wavelength_object, normalized_flux, label='Normalized Flux peaks', alpha=0.8, color="green")
    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux all data', alpha=0.8, color="blue")
    plt.plot([6450, 6750], [1, 1], linestyle='--', color="black")
    plt.title("this is")
    plt.legend()
    #plt.savefig("normalize.jpg", dpi=500)
    #plt.show()
    
    return wavelength_object, normalized_flux

order = 12
wavelength_object_sun, normalized_flux_o_sun = Normalization(order, folder_data_sun)
wavelength_object_sunspot, normalized_flux_sunspot = Normalization(order, folder_data_sunspot)

plt.figure(dpi=300)
plt.plot(wavelength_object_sun, normalized_flux_o_sun, label='Sun', alpha=0.8, color="red", linestyle='--')
plt.plot(wavelength_object_sunspot, normalized_flux_sunspot, label='Sunspot', alpha=0.8, color="blue")

plt.axhline(y=1, color='black', linestyle='--')
plt.xlim(5150+17, 5175+6)
plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Normalized Flux")
plt.title(f"Normalized flux Order {order}")
plt.legend()
plt.savefig(f"Normalized_line_compare_order_{order}.jpg", dpi=500)