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
    plt.close()
    
    wavelength_list = wavelength_list_complete[N_order]
    x_list = x_list_complete[N_order]
    uncertainty_x = uncertainty_x_complete[N_order]
    plt.plot(x_pixelvalues, thar)
    """plt.scatter(x_list, thar[x_list], c='red', label = 'calibration points' )
    for index in range(len(x_list)):
        plt.text(x_list[index]+20, thar[x_list][index]+20, wavelength_list[index], size=8)"""
    plt.legend()
    #plt.show()
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
    plt.close(fig)
    
    plt.subplots(1, 1, figsize=(16.5, 11.7), dpi=300)
    plt.plot(wavelength_object,(flux_object-dark)/(tungstenflat-darkflat))
    plt.ylim(0,)
    plt.xlim(6560, 6565)
    #plt.show()
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
    plt.close()
    #############################################################

    plt.figure(dpi=300)

    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux', alpha=0.8)
    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title("Flux Normalization using Polynomial Fit")
    #plt.show()
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
    plt.close()


    plt.figure(dpi=300)
    plt.plot(wavelength_object, normalized_flux, label='Normalized Flux peaks', alpha=0.8, color="green")
    plt.plot(wavelength_object, normalized_flux_o, label='Normalized Flux all data', alpha=0.8, color="blue")
    plt.plot([6450, 6750], [1, 1], linestyle='--', color="black")
    plt.title("this is")
    plt.legend()
    #plt.savefig("normalize.jpg", dpi=500)
    #plt.show()
    plt.close()
    return wavelength_object, normalized_flux

order = 7
wavelength_object_sun, normalized_flux_o_sun = Normalization(order, folder_data_sun)
wavelength_object_sunspot, normalized_flux_sunspot = Normalization(order, folder_data_sunspot)

plt.figure(dpi=300)
plt.plot(wavelength_object_sun, normalized_flux_o_sun, label='Sun', alpha=0.8, color="red", linestyle='--')
plt.plot(wavelength_object_sunspot, normalized_flux_sunspot, label='Sunspot', alpha=0.8, color="blue")

plt.axhline(y=1, color='black', linestyle='--')
#plt.xlim(5150+17, 5175+6)
#plt.xlim(5188, 5178)
plt.xlabel("Wavelength [Angstrom]")
plt.ylabel("Normalized Flux")
plt.title(f"Normalized flux Order {order}")
plt.legend()
plt.grid(True)
plt.savefig(f"Normalized_line_compare_order_{order}.jpg", dpi=500)
plt.show()



def absorption_line(x, y, order):
    start = [0,0,0,6560,0,6257,0,5888,0,0,0,5268,5178,5168,0,4860,4837]
    end = [0,0,0,6566,0,6259,0,5892,0,0,0,5270.35,5188,5169.2,0,4863,4840.2]
    
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

line_name = [0,0,0,"Fraunhofer line C: H_α",0,"VI",0,"Fraunhofer line D_2: Na I (5889.95 Å)",0,0,0,0,"Fraunhofer line E_2: Fe",0,0,"Fraunhofer line F: H_β", "Mn I"]
line_center = [0,0,0,6562.7,0,6258,0,5889.95,0,0,0,5270.39,5183,0,0,4861.34,4838.56]

init_vals = [-0.5, line_center[order], 0.5, 1]  # for [amp, cen, wid]
best_vals, covar = curve_fit(gaussian, wavelength, flux, p0=init_vals)
best_vals_spot, covar_spot = curve_fit(gaussian, wavelength, line_y_sunspot, p0=init_vals)

plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, color="red", label="sun", linewidth=4)
plt.plot(wavelength, line_y_sunspot, color="blue", label="sunspot", linewidth=4)
#plt.plot(wavelength,gaussian(wavelength, best_vals[0],best_vals[1],best_vals[2], best_vals[3]), color="orange", label="fit sun")
#plt.plot(wavelength,gaussian(wavelength, best_vals_spot[0], best_vals_spot[1], best_vals_spot[2], best_vals_spot[3]), color="green", label="fit sunspot")

label_size_font = 15
plt.xlabel("Wavelength (Å)", fontsize=label_size_font)
plt.ylabel("Normalized Flux", fontsize=label_size_font)
plt.title(f"{line_name[order]}. Og Data. Order_{order}.", fontsize=20)
plt.grid(True)
plt.legend()
plt.savefig(f"gaussian_order_{order}.jpg", dpi=1000)

W_o= 2*best_vals[2]*np.log(2)

W = W_o * (10 **(-10))
perr = np.sqrt(np.diag(covar))


line_x, line_y_sun = absorption_line(wavelength_object_sun, normalized_flux_o_sun, order)  
line_x, line_y_sunspot = absorption_line(wavelength_object_sunspot, normalized_flux_sunspot * 1.45, order) #1.45


# TESTING
best_vals, covar = curve_fit(gaussian, wavelength, flux, p0=init_vals)
best_vals_spot, covar_spot = curve_fit(gaussian, wavelength, line_y_sunspot, p0=init_vals)
#best_vals_spot, covar_spot = curve_fit(gaussian, wavelength, line_y_sunspot, p0=[-0.5,6257.87,0.5,1])






#print(best_vals)


######################################################################################################################################################
distance = 0
counter = 0
x_solutions = []
solution_values = []
y_value_list=[]
for y_value in np.arange(0.35, 0.6, 0.01):

    amplitude = best_vals[0]
    centrum = best_vals[1]
    wide = best_vals[2]
    dy_value = best_vals[3]

    amplitude_s = best_vals_spot[0]
    centrum_s = best_vals_spot[1]
    wide_s = best_vals_spot[2]
    dy_value_s = best_vals_spot[3]

    x_value_1_sun = centrum + np.sqrt(-wide * np.log((y_value* max(flux) - dy_value) / amplitude))
    x_value_2_sun = centrum - np.sqrt(-wide * np.log((y_value* max(flux) - dy_value) / amplitude))

    x_value_1_sunspot = centrum_s + np.sqrt(-wide_s * np.log((y_value - dy_value_s) / amplitude_s))
    x_value_2_sunspot = centrum_s - np.sqrt(-wide_s * np.log((y_value - dy_value_s) / amplitude_s))

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
 
   

    
    
sunspot_hand_11 = [5269.7519, 5269.7044, 5270.2553, 5270.3387, 5269.6670]
sun_hand_11 = [5269.8310, 5269.7893, 5270.0956, 5270.1374, 5269.7663]

sunspot_hand_7 = [5890.302, 5890.175, 5889.508, 5889.424, 5889.628, 5890.398]
sun_hand_7 = [5890.175, 5890.073, 5889.603, 5889.520, 5889.702, 5890.302]

sunspot_hand_16 = [4838.305, 4838.342]
sun_hand_16 = [4838.343, 4838.395]

sun_hand = sun_hand_11
sunspot_hand = sunspot_hand_11

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

derivative_1 = 0.83 * 0.5 * ((W_o * shift)**(-0.5)) * shift
derivative_2 = 0.83 * 0.5 * ((W_o * shift)**(-0.5)) * W_o
error_result_dx = np.sqrt((derivative_1 * error_W)**2 + (derivative_2 * error_shift)**2)

##################################################################

m_e = 9.10938 * (10**(-31))
c = 299792458
e = 1.60217663 * (10**(-19))
lambda_rest = line_center[order]

g_lande = [1,1,1,1,1,3.3333,1,2.265,1,1,1,1.835,1,1,1,1,2.6667]
g = g_lande[order]



magnetic_field = (((result_dx * (10**(-10))) * 4 * np.pi * m_e * c) / (e * g * ((lambda_rest * (10**(-10)))**2))) * 10000
magnetic_field_error = (((error_result_dx * (10**(-10))) * 4 * np.pi * m_e * c) / (e * g * ((lambda_rest * (10**(-10)))**2))) * 10000

print(f"B = {round(magnetic_field,1)} +/- {round(magnetic_field_error,1)} G")


print(f"Shift avg: {shift} +/- {error_shift} A")
print(f"Delta lambda: {result_dx} +/- {error_result_dx} A")
#print(avg_long)
#print(perr)
#print(W_o)


plt.figure(figsize=(10,6))
plt.scatter(wavelength, flux / max(flux), color="red", label="sun")
plt.scatter(wavelength, line_y_sunspot / max(line_y_sunspot), color="blue", label="sunspot")
plt.plot(wavelength,gaussian(wavelength, best_vals[0],best_vals[1],best_vals[2], best_vals[3]) / max(flux), color="orange", label="fit sun")
plt.plot(wavelength,gaussian(wavelength, best_vals_spot[0], best_vals_spot[1], best_vals_spot[2], best_vals_spot[3]), color="green", label="fit sunspot")
plt.scatter(x_solutions, y_value_list, color="black", label="shift calc. points", s=12, zorder=3)



plt.xlabel("Wavelength (Å)", fontsize=label_size_font)
plt.ylabel("Normalized Flux", fontsize=label_size_font)
plt.title(f"{line_name[order]}. Scaled. Order_{order}", fontsize=20)
plt.grid(True)
plt.legend()
plt.savefig(f"Final_plot_order_{order}", dpi=1000)
plt.show()
