"""
Created on Wed Nov 20

@author: Joan Sonnenberg



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
from scipy.optimize import minimize

# insert local data folder paths
folder_data_sun = 'C:/Users/joans/Desktop/Natuur- en Sterrenkunde/Year 2/Period 2/N&S Practicum 2/Practicum/Solar_Physics_Practicum/Solar_Physics_Practicum/Flux_raw_sun_centre'
folder_data_sunspot = 'C:/Users/joans/Desktop/Natuur- en Sterrenkunde/Year 2/Period 2/N&S Practicum 2/Practicum/Solar_Physics_Practicum/Solar_Physics_Practicum/Flux_raw_sunspot3477'

with open('ThArCal.csv', 'r') as MyData:
    lambda_cal, pixel_cal = [], []
    i = 0
    for line in MyData:
        i += 1
        if i > 1:                      
            data_cut = line.split(',')
            lambda_cal.append([float(data_cut[2]), float(data_cut[3]), float(data_cut[4])])
            pixel_cal.append([float(data_cut[5]), float(data_cut[6]), float(data_cut[7])])


def Normalization(N_order, main_folder):
    
    wavelength_list_complete = lambda_cal
    x_list_complete = pixel_cal
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


    # Create a 1x2 grid of subplots with shared x-axis
    fig = plt.figure(figsize=(10, 7))  # Adjust figure size as needed

    # Plot on the left (original plot)
    
    plt.plot(x_pixelvalues, tungstenflat, label='Tungsten lamp spectrum', zorder=1, color="purple")
    plt.plot(x_pixelvalues, flux_object, label='Sunspot spectrum', zorder=2, color="orange")
    plt.plot(x_pixelvalues, thar, label='Echelle spectrum of Th/Ar lamp', zorder=3, color="green")
    plt.scatter([4230, 4859, 4485], [1.175 * 100000, 6.14 * 10000, 2.82 * 10000], color = "blue", zorder=4, label="Pixel-Wavelength calibration lines")
    #axes[0].plot(x_pixelvalues, bias, label='Bias')
    plt.xlim(0, 6000)
    plt.ylim(20000, 500000)
    plt.yscale("log")  # Set log scale for the left plot
    #axes[0].set_title("Left Plot (Log Scale)")
    plt.legend()
    ticks = [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000]  # Example: ticks at 10, 100, 1000, ...
    plt.yticks(ticks, labels=["2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "40", "50", "60"])  # Add labels
    plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
    plt.tick_params(axis='both', which='minor', labelsize=11)
    
    """axes[1].plot(x_pixelvalues, dark, label='Dark')
    axes[1].plot(x_pixelvalues, darkflat, label='Dark Flat')
    axes[1].set_xlim(0, 6000)
    axes[1].legend()"""

    # Plot on the right (commented-out elements)
    
    #axes[1].plot(x_pixelvalues, SNR, label='SNR')
    
    """axes[1].set_xlim(0, 6000)
    axes[1].set_yscale("log")
    #axes[2].set_title("Right Plot (Linear Scale)")
    axes[1].legend()"""

    # Save the figure
    plt.savefig("Thorium.jpg", dpi=1000)
    
    
    wavelength_list = wavelength_list_complete[N_order]
    x_list = x_list_complete[N_order]
    uncertainty_x = uncertainty_x_complete[N_order]
    plt.plot(x_pixelvalues, thar)
    #plt.scatter(x_list, thar[x_list], c='red', label = 'calibration points' )
    """for index in range(len(x_list)):
        plt.text(x_list[index]+20, thar[x_list][index]+20, wavelength_list[index], size=8)"""
    plt.legend()
    #plt.show()
    plt.savefig("Pixel_values.jpg", dpi=300)
    plt.close()
    
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
    #fig.subplots_adjust(hspace=0.05) 
    fig.subplots_adjust(hspace=0)

    yerr = abs(uncertainty_x*np.array(fit_1[1]))

    #ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength)")
    ax1.plot(x_pixelvalues, wavelength_object, color="red", linestyle="--", linewidth=2)
    #ax1.set_ylabel("Wavelength [Angstrom]")
    ax1.errorbar(x_list, wavelength_list, yerr=yerr, fmt='o', color='blue', capsize=3, label='Calibration points')
    ax1.scatter(x_list,wavelength_list, color='blue', s=25)



    ax2.errorbar(x_list, residuals, yerr=yerr, fmt='o', color='blue', capsize=3, label='Residuals with error bars')
    ax2.scatter(x_list,residuals)
    #ax2.set_xlabel("Pixels")
    #ax2.set_ylabel("Residuals [Angstrom]")
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label = 'Model')
    ax2.axhline(fit_1[1], color='purple', linestyle='--', linewidth=2, label = '1 pixel difference')
    ax2.axhline(-1*fit_1[1], color='purple', linestyle='--', linewidth=2)
    #for index in range(len(x_list)):
        #ax2.text(x_list[index], residuals[index], wavelength_list[index], size=8)
    plt.legend(loc=2)
    #plt.show()
    plt.xlim(0, 6000)
    ax1.set_ylim(5100, 5450)
    plt.savefig("subplots.jpg", dpi=1000)
    plt.close(fig)
    
    plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
    plt.plot(wavelength_object,(flux_object-dark)/(tungstenflat-darkflat))
    plt.ylim(0,)
    #plt.xlim(6560, 6565)
    #plt.show()
    plt.savefig("short.jpg", dpi=300)
    plt.close()
    
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
    plt.savefig("Fit1.jpg", dpi=300)
    plt.close()
    #############################################################

    plt.figure(dpi=300)

    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux', alpha=0.8)
    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title("Flux Normalization using Polynomial Fit")
    #plt.show()
    plt.savefig("o_flux.jpg", dpi=300)
    plt.close()
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


    plt.figure(figsize=(10, 7))
    plt.plot(wavelength_object, total_flux, label='Sunspot spectrum', color="black", linewidth=1)
    plt.scatter(wavelength_peaks, flux_peaks, color='blue', label='Selected peaks', s=25)
    plt.plot(wavelength_object, fitted_flux, label=f'Polynomial fit (degree {fit_degree})', linestyle='--', color="red", linewidth=3)
    
    #plt.plot(wavelength_object, normalized_flux, label='Normalized Flux', alpha=0.8)
    plt.ylim(0, 0.5)
    plt.xlim(5150, 5450)
    #plt.xlabel("Wavelength [Angstrom]")
    #plt.ylabel("Normalized Flux")
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
    plt.tick_params(axis='both', which='minor', labelsize=11)
    #plt.title("Flux Normalization using Polynomial Fit through Selected Peaks")
    #plt.savefig("dots.jpg", dpi=500)
    #plt.show()
    plt.savefig("Polynomials.jpg", dpi=1000)
    plt.close()


    plt.figure(dpi=300)
    plt.plot(wavelength_object, normalized_flux, label='Normalized Flux peaks', alpha=0.8, color="green")
    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux all data', alpha=0.8, color="blue")
    plt.plot([6450, 6750], [1, 1], linestyle='--', color="black")
    plt.title("this is")
    plt.legend()
    #plt.savefig("normalize.jpg", dpi=500)
    #plt.show()
    plt.savefig("Polynomials_normalization.jpg", dpi=300)
    plt.close()
    return wavelength_object, normalized_flux

order = 12
start_wav = 5166
end_wav = 5168
line_center_measured = 5167.023
gaussian_amplitude = 0.35
loop_start = 0.3
loop_end = 0.6
lambda_rest = 5167.3216 #line_center[order]

g_lande_1 = 1.5
g_lande_2 = 1.75
j_1 = 5
j_2 = 3
big_g = ((g_lande_1 + g_lande_2)/2) + ((g_lande_1 - g_lande_2)/4)*(j_1*(j_1 + 1) - j_2*(j_2 + 1))

g = 2 #big_g


wavelength_object_sun, normalized_flux_o_sun = Normalization(order, folder_data_sun)
wavelength_object_sunspot, normalized_flux_sunspot = Normalization(order, folder_data_sunspot)

plt.figure(figsize=(10, 7))
plt.plot(wavelength_object_sun, normalized_flux_o_sun, label='Sun spectrum (Heliostat APO)', color="black", linestyle='--')
plt.plot(wavelength_object_sunspot, normalized_flux_sunspot, label='Sunspot spectrum (Heliostat APO)', color="black")

#plt.axhline(y=1, color='black', linestyle='--')
#plt.xlim(6301, 6303)
#plt.ylim(0.6, 1)
plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Normalized Flux")
#plt.title(f"Normalized flux Order {order}")
plt.legend()
#plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
plt.tick_params(axis='both', which='minor', labelsize=11)
plt.savefig(f"Normalized_line_compare_order_{order}.jpg", dpi=1000)
plt.close()



def absorption_line(x, y, order):
    start = [start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav,start_wav]
    end = [end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav,end_wav]
    
    x_data = []
    y_data = []
    indexes = []
    
    for i in range(len(x)):
        if x[i] > start[order] and x[i] < end[order]:
            x_data.append(x[i])
            y_data.append(y[i])
            indexes.append(i)
            
    """start_index = indexes[0]
    end_index = indexes[-1]"""           
            
    return x_data, y_data #, start_index, end_index
    
    
"""line_x, line_y_sun, start_index, end_index = absorption_line(wavelength_object_sun, normalized_flux_o_sun, order)  
line_x, line_y_sunspot, start_index, end_index = absorption_line(wavelength_object_sunspot, normalized_flux_sunspot * 5/3, order)""" 

line_x, line_y_sun = absorption_line(wavelength_object_sun, normalized_flux_o_sun, order)  
line_x, line_y_sunspot = absorption_line(wavelength_object_sunspot, normalized_flux_sunspot, order)  

plt.figure(dpi=300)
plt.plot(line_x, line_y_sun, label='Sun', alpha=0.8, color="red", linestyle='--')
plt.plot(line_x, line_y_sunspot, label='Sunspot', alpha=0.8, color="blue")

plt.axhline(y=1, color='black', linestyle='--')
#plt.xlim(5150+17, 5175+6)
#plt.xlim(5188, 5178)
plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Normalized Flux")
plt.title(f"Normalized flux Order {order}")
plt.legend()
plt.grid(True)
plt.savefig(f"Trim_{order}.jpg", dpi=500)  
plt.close()


"""print(f"start: {start_index}")
print(f"end: {end_index}")  
"""
def gaussian(x, amp, cen, wid, dy):
    return amp * np.exp(-(x-cen)**2 / wid) + dy


wavelength = line_x
flux = line_y_sun

#5183 order 12
#6562.7 order 3
#5895.92 order 7
#4861.34 order 15

line_name = [0,0,0,"Fraunhofer line C: H_α",0,"VI",0,"Fraunhofer line D_2: Na I (5889.95 Å)",0,0,0,0,"Fraunhofer line E_2: Fe",0,0,"Fraunhofer line F: H_β", "Mn I",0,0,0,0,0]
line_center = line_center_measured

init_vals = [-0.5, line_center, 0.5, 1]  # for [amp, cen, wid]
best_vals, covar = curve_fit(gaussian, wavelength, flux, p0=init_vals)
best_vals_spot, covar_spot = curve_fit(gaussian, wavelength, line_y_sunspot, p0=init_vals)

print(best_vals)
h = best_vals[0] / 2 + best_vals[3]

def scaled_gaussian(x, amp, cen, wid, scale):
    return scale * gaussian(x, amp, cen, wid)

wavelength_array = np.array(wavelength)
red_flux = np.array(flux)  # Y-values for the red spectrum
blue_flux = np.array(line_y_sunspot)  # Y-values for the blue spectrum

def parallelism_metric(scale):
    # Fit the blue flux with the current scaling factor
    scaled_blue_flux = scale * blue_flux
    best_vals_blue, _ = curve_fit(gaussian, wavelength, scaled_blue_flux, p0=init_vals)
    
    # Compute Gaussian fits for both datasets
    red_fit = gaussian(wavelength, *best_vals)
    blue_fit = gaussian(wavelength, *best_vals_blue)
    
    # Define the absorption line region
    left_region = (wavelength_array < line_center - gaussian_amplitude)
    right_region = (wavelength_array > line_center + gaussian_amplitude)
    
    # Compute differences in the left and right regions
    left_diff = np.abs(red_fit[left_region] - blue_fit[left_region])
    right_diff = np.abs(red_fit[right_region] - blue_fit[right_region])
    
    # Return the sum of squared differences
    return np.sum(left_diff**2) + np.sum(right_diff**2)

# Minimize the metric to find the optimal scaling factor
result = minimize(parallelism_metric, x0=1.0, bounds=[(0.1, 10)])
optimal_scale = result.x[0]

# Apply the optimal scaling factor
scaled_blue_flux = optimal_scale * blue_flux

# Fit the scaled blue flux
best_vals_blue, _ = curve_fit(gaussian, wavelength, scaled_blue_flux, p0=init_vals)

# Generate Gaussian fits for plotting
red_fit = gaussian(wavelength, *best_vals)
blue_fit = gaussian(wavelength, *best_vals_blue)



up_or_down = scaled_blue_flux[-1] - line_y_sunspot[-1]
blue_fit = blue_fit - up_or_down

distance = 0
counter = 0
x_solutions = []
solution_values = []
y_value_list=[]

for y_value in np.arange(loop_start, loop_end, 0.01):
    
    amplitude = best_vals[0]
    centrum = best_vals[1]
    wide = best_vals[2]
    dy_value = best_vals[3]
    
    amplitude_s = best_vals_blue[0]
    centrum_s = best_vals_blue[1]
    wide_s = best_vals_blue[2]
    dy_value_s = best_vals_blue[3]
    
    x_value_1_sun = centrum + np.sqrt(-wide * np.log((y_value - dy_value) / amplitude))
    x_value_2_sun = centrum - np.sqrt(-wide * np.log((y_value - dy_value) / amplitude))

    x_value_1_sunspot = centrum_s + np.sqrt(-wide_s * np.log((y_value + up_or_down - dy_value_s) / amplitude_s))
    x_value_2_sunspot = centrum_s - np.sqrt(-wide_s * np.log((y_value + up_or_down - dy_value_s) / amplitude_s))
    
    dx_1 = abs(x_value_1_sun - x_value_1_sunspot)
    dx_2 = abs(x_value_2_sun - x_value_2_sunspot)
    
    solution_values.append(dx_1)
    solution_values.append(dx_2)

    average_dx = (dx_1 + dx_2) / 2
    distance = distance + dx_1 + dx_2
    counter = counter + 2
    x_solutions.append(x_value_1_sun)
    x_solutions.append(x_value_2_sun)
    x_solutions.append(x_value_1_sunspot)
    x_solutions.append(x_value_2_sunspot)
    y_value_list.append(y_value)
    y_value_list.append(y_value)
    y_value_list.append(y_value)
    y_value_list.append(y_value)


"""plt.figure(figsize=(10, 7))
#plt.plot(wavelength, red_flux, color="red", label="sun", linewidth=3)
#plt.plot(wavelength, scaled_blue_flux - up_or_down, color="blue", label="sunspot", linewidth=3)
#plt.plot(wavelength,gaussian(wavelength, best_vals[0],best_vals[1],best_vals[2], best_vals[3]), color="orange", label="fit sun")
plt.axhline(y=0.3, color='gray', linestyle='--', alpha = 0.8, label = "Range for δ measurements")
plt.axhline(y=0.6, color='gray', linestyle='--', alpha = 0.8)



# Gaussian fits
plt.plot(np.array(wavelength) + 0.26, red_fit, '-', label="Mg I line fit (sun spectrum)", color="red")
plt.plot(np.array(wavelength) + 0.26, blue_fit, '-', label="Mg I line scaled fit (sunspot spectrum)", color="blue")
#plt.plot(wavelength,gaussian(wavelength, best_vals_spot[0], best_vals_spot[1], best_vals_spot[2], best_vals_spot[3]), color="green", label="fit sunspot")
#plt.scatter(x_solutions, y_value_list, color="black", label="shift calc. points", s=12, zorder=3)
plt.plot([5166.950, 5167.581], [h, h], color='orange', linestyle='--', alpha = 0.9, label = "FWHM")

label_size_font = 15
#plt.xlabel("Wavelength (Å)", fontsize=label_size_font)
#plt.ylabel("Normalized Flux", fontsize=label_size_font)
#plt.title(f"{line_name[order]}. Blue Scaled {optimal_scale}. Order_{order}.", fontsize=20)
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
plt.tick_params(axis='both', which='minor', labelsize=11)
plt.legend()
plt.ylim(0.1, 0.8)
plt.xlim(5166.6, 5168)
plt.savefig(f"gaussian_order_now.jpg", dpi=1000)"""



plt.figure(figsize=(10, 7))
plt.plot(np.array(wavelength) + 0.26, red_flux, color="red", label="Mg I line (sun spectrum)", linewidth=1)

#plt.plot(wavelength,gaussian(wavelength, best_vals[0],best_vals[1],best_vals[2], best_vals[3]), color="orange", label="fit sun")
#plt.axhline(y=0.3, color='gray', linestyle='--', alpha = 0.8, label = "Range for δ measurements")
#plt.axhline(y=0.6, color='gray', linestyle='--', alpha = 0.8)
#plt.plot(np.array(line_x) + 0.26, line_y_sun, label='Sun', alpha=0.7, color="red", linestyle='--')
plt.plot(np.array(line_x) + 0.26, line_y_sunspot, label='Mg I line (sunspot spectrum)', alpha=0.7, color="blue")
plt.plot(np.array(wavelength) + 0.26, scaled_blue_flux - up_or_down - 0.009, color="blue", label="Mg I line inflated (sunspot spectrum)", linewidth=1, linestyle="--")
# Gaussian fits
#plt.plot(np.array(wavelength) + 0.26, red_fit, '-', label="Mg I line fit (sun spectrum)", color="red")
#plt.plot(np.array(wavelength) + 0.26, blue_fit, '-', label="Mg I line scaled fit (sunspot spectrum)", color="blue")
#plt.plot(wavelength,gaussian(wavelength, best_vals_spot[0], best_vals_spot[1], best_vals_spot[2], best_vals_spot[3]), color="green", label="fit sunspot")
#plt.scatter(x_solutions, y_value_list, color="black", label="shift calc. points", s=12, zorder=3)
#plt.plot([5166.950, 5167.581], [h, h], color='orange', linestyle='--', alpha = 0.9, label = "FWHM")

label_size_font = 15
#plt.xlabel("Wavelength (Å)", fontsize=label_size_font)
#plt.ylabel("Normalized Flux", fontsize=label_size_font)
plt.title(f"{line_name[order]}. Blue Scaled {optimal_scale}. Order_{order}.", fontsize=20)
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
plt.tick_params(axis='both', which='minor', labelsize=11)
plt.legend()
plt.ylim(0.1, 0.9)
plt.xlim(5166.25, 5168.25)
plt.savefig(f"gaussian_order_now.jpg", dpi=1000)





    






W_o= 2*best_vals[2]* np.sqrt(np.log(2))


W = W_o * (10 **(-10))
perr = np.sqrt(np.diag(covar))

print(best_vals[2])
print(perr)
print("W", W_o)
print(perr*2*np.sqrt(np.log(2)))


line_x, line_y_sun = absorption_line(wavelength_object_sun, normalized_flux_o_sun, order)  
line_x, line_y_sunspot = absorption_line(wavelength_object_sunspot, normalized_flux_sunspot * 1.45, order) #1.45


# TESTING
#best_vals, covar = curve_fit(gaussian, wavelength, flux, p0=init_vals)
best_vals_spot, covar_spot = curve_fit(gaussian, wavelength, line_y_sunspot, p0=init_vals)
#best_vals_spot, covar_spot = curve_fit(gaussian, wavelength, line_y_sunspot, p0=[-0.5,6257.87,0.5,1])






#print(best_vals)


######################################################################################################################################################


       


sun_hand = [4920.486, 4920.127, 4920.391, 4920.244]
sunspot_hand = [4920.727, 4920.035, 4920.591, 4920.113]

longitude = 0
for m in range(len(sun_hand)):
    longitude = longitude + abs(sun_hand[m] - sunspot_hand[m])
avg_long = longitude / len(sunspot_hand)
    
method = 1
    
if method == 1:
    shift = distance / counter
    result_dx = 0.83 * np.sqrt(W_o * shift)
else:
    shift = avg_long
    result_dx = 0.83 * np.sqrt(W_o * avg_long)
    solutions = []
    for j in range(len(sun_hand)):
        solutions.append(abs(sun_hand[j] - sunspot_hand[j]))
        
    solution_values = solutions
    
sumation = 0
for value in solution_values:
    add = (value - shift)**2
    sumation = sumation + add
sdv = np.sqrt(sumation / (len(solution_values) - 1))
error_shift = sdv / np.sqrt(len(solution_values))
error_W = perr[2]
print(f"shiffi {shift}")
print(f"{error_shift}")
derivative_1 = 0.83 * 0.5 * ((W_o * shift)**(-0.5)) * shift
derivative_2 = 0.83 * 0.5 * ((W_o * shift)**(-0.5)) * W_o
error_result_dx = np.sqrt((derivative_1 * error_W)**2 + (derivative_2 * error_shift)**2)

##################################################################

m_e = 9.10938 * (10**(-31))
c = 299792458
e = 1.60217663 * (10**(-19))

#big_g = ((g_lande_1[order] + g_lande_2[order])/2) + ((g_lande_1[order] - g_lande_2[order])/4)*(j_1[order]*(j_1[order] + 1) - j_2[order]*(j_2[order] + 1))


#1.425 for 5328.051

magnetic_field = (((result_dx * (10**(-10))) * 4 * np.pi * m_e * c) / (e * g * ((lambda_rest * (10**(-10)))**2))) * 10000
magnetic_field_error = (((error_result_dx * (10**(-10))) * 4 * np.pi * m_e * c) / (e * g * ((lambda_rest * (10**(-10)))**2))) * 10000

print(f"Lambda rest: {lambda_rest}")
print(f"B = {round(magnetic_field,1)} +/- {round(magnetic_field_error,1)} G")
print(f"G: {g}")



#print(f"Shift avg: {shift} +/- {error_shift} A")
print(f"Delta lambda: {round(result_dx,3)} +/- {round(error_result_dx,3)} A")
#print(avg_long)
#print(perr)
#print(W_o)


plt.figure(figsize=(10,6))
plt.scatter(wavelength, flux / max(flux), color="red", label="sun")
plt.scatter(wavelength, line_y_sunspot / max(line_y_sunspot), color="blue", label="sunspot")
plt.plot(wavelength,gaussian(wavelength, best_vals[0],best_vals[1],best_vals[2], best_vals[3]) / max(flux), color="orange", label="fit sun")
#plt.plot(wavelength,gaussian(wavelength, best_vals_spot[0], best_vals_spot[1], best_vals_spot[2], best_vals_spot[3]), color="green", label="fit sunspot")
plt.scatter(x_solutions, y_value_list, color="black", label="shift calc. points", s=12, zorder=3)



plt.xlabel("Wavelength (Å)", fontsize=label_size_font)
plt.ylabel("Normalized Flux", fontsize=label_size_font)
plt.title(f"{line_name[order]}. Scaled. Order_{order}", fontsize=20)
plt.grid(True)
plt.legend()
plt.savefig(f"Final_plot_order_{order}", dpi=500)



import matplotlib.pyplot as plt
import numpy as np

from lmfit import models



Lines = ["Mg I", "Mg I", "Mg I", "Fe I", "Na I", "Mg I", "Na I", "Fe I", "V I", "Ca I", "Ca I", "Fe I", "Fe I"]
Gs = [1.25, 2, 1.75, 1.2, 1.33, 1, 1.33, 1.25, 3.33, 1, 1.45, 0.74, 0.5]
Deltas = [0.147, 0.087, 0.095, 0.154, 0.173, 0.07, 0.252, 0.19, 0.145, 0.114, 0.188, 0.158, 0.068]
Deltas_errors = [0.009, 0.005, 0.007, 0.005, 0.012, 0.003, 0.004, 0.004, 0.02, 0.008, 0.009, 0.007, 0.005]
Rests = [5183.6042, 5167.3216, 5172.684, 4920.5028, 5895.924, 4702.9909, 5889.9509, 4957.613, 6258.571, 6717.69, 6439.07, 5270.39, 5168.91]
Bs= [9375.8, 3474.4, 4368.3, 11348.3, 7977.4, 6738.4, 11687.1, 13262.4, 2380.1, 5408.8, 6692.2, 16457.6, 10978.3]
B_errors = [551.3, 180.9, 303.4, 366, 551.7, 268.9, 204.1, 290.7, 334.3, 392.6, 312.3, 710.9, 805.7] 




x_axis = []
y_axis = []
Deltas_errors_units = []
x_axis_new = []

for i in range(len(Gs)):
    #if "V" in Lines[i]:
    #if i == 0 or i == 1 or i == 2 or i == 4 or i == 6 or i == 7 or i == 11 or i == 12: # F lines
    if i != 8: # 2 or i == 1:
        x = Gs[i] * ((Rests[i] * (10**(-10)))**2) 
        
        y = Deltas[i] * (10**(-10)) 
        x_axis.append(x)
        y_axis.append(y)
        Deltas_errors_units.append(Deltas_errors[i] * (10**(-10)))
        x = Gs[i] * ((Rests[i] )**2) 
        x_axis_new.append(x / 10000000)
    
"""plt.errorbar(x_axis, y_axis, yerr=Deltas_errors, fmt='o')
plt.ylabel("Δλ (Å)")
plt.xlabel("Gλ² (Å²)")

plt.close()"""






m_e = 9.10938 * (10**(-31))
c = 299792458
e = 1.60217663 * (10**(-19))

# Define fit function
def fit_function (x, B):
    Delta = (x * e * B) / (4 * np.pi * c * m_e)
    return Delta

# Create a model from the fit function
MI_model = models.Model(fit_function, name="MI_model")

errors = np.array(Deltas_errors_units)

# Perform the fit
fit_result = MI_model.fit(y_axis, x=x_axis, weights=1/errors, B=0.3)

print(fit_result.fit_report())





y_axis = np.array(y_axis)* (10**(13))
Deltas_errors_units = np.array(Deltas_errors_units) * (10**(13))
    
 
    



###########################################################

for_fit_deltas = []
for_fit_x = []
for i in range(len(Gs)):
    #if "V" in Lines[i]:
    #if i == 0 or i == 1 or i == 2 or i == 4 or i == 6 or i == 7 or i == 11 or i == 12: # F lines
    if i != 8: # 2 or i == 1:
        rest_wavelength = Rests[i] * (10**(-10))
        field_desired = 0.25
        delta_computed = (e * Gs[i] * field_desired * (rest_wavelength)**2) / (4 * np.pi * m_e * c)
        for_fit_deltas.append(delta_computed)
        x = Gs[i] * ((Rests[i] * (10**(-10)))**2) 
        for_fit_x.append(x)
        

# Create a model from the fit function
MI_model_expected = models.Model(fit_function, name="MI_model")

errors_expected = np.array(Deltas_errors_units)

# Perform the fit
fit_result_expected = MI_model_expected.fit(for_fit_deltas, x=for_fit_x, weights=1/errors_expected, B=0.25)

print(fit_result_expected.fit_report())

###############################################################################################

for_fit_deltas_2 = []
for_fit_x_2 = []
for i in range(len(Gs)):
    #if "V" in Lines[i]:
    #if i == 0 or i == 1 or i == 2 or i == 4 or i == 6 or i == 7 or i == 11 or i == 12: # F lines
    if i != 8: # 2 or i == 1:
        rest_wavelength = Rests[i] * (10**(-10))
        field_desired = 0.4
        delta_computed = (e * Gs[i] * field_desired * (rest_wavelength)**2) / (4 * np.pi * m_e * c)
        for_fit_deltas_2.append(delta_computed)
        x = Gs[i] * ((Rests[i] * (10**(-10)))**2) 
        for_fit_x_2.append(x)
        

# Create a model from the fit function
MI_model_expected_2 = models.Model(fit_function, name="MI_model")

errors_expected_2 = np.array(Deltas_errors_units)

# Perform the fit
fit_result_expected_2 = MI_model_expected_2.fit(for_fit_deltas_2, x=for_fit_x_2, weights=1/errors_expected_2, B=0.25)

print(fit_result_expected_2.fit_report())





# Plot data with error bars and the fit
plt.figure(figsize=(10, 7))
plt.errorbar(x_axis_new, y_axis, yerr=Deltas_errors_units, fmt='o', color="black", capsize=5, markersize=5, label="Broadening Δλ measurements")
plt.plot(x_axis_new, fit_result.best_fit* (10**(13)), color="blue", alpha=0.8, label="Best linear fit over measurements")
plt.plot(x_axis_new, fit_result_expected.best_fit* (10**(13)), color="red", alpha=0.5, label="Fit for field of B = 2.5 kG")
plt.plot(x_axis_new, fit_result_expected_2.best_fit* (10**(13)), color="orange", alpha=0.5, label="Fit for field of B = 4 kG")




plt.ylabel("Δλ (mÅ)")
plt.xlabel("Gλ² (Å²) · 10^-7")
plt.xlim(1, 7)
plt.ylim(0,300)
plt.legend(loc=2)
plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
plt.tick_params(axis='both', which='minor', labelsize=11)
#plt.savefig("final_plot.jpg", dpi=1000)
plt.show()





