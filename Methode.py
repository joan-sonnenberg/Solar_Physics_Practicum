# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:10:11 2024

@author: noahr
"""


import numpy as np
import os
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
import warnings
from astropy.modeling import models
from astropy import units as u
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


main_folder = 'C:/Users/noahr/OneDrive/Bureaublad/School/School/NSP2'
data_cal = np.loadtxt(os.path.join(main_folder,
                                   "ThArCal.csv"), delimiter=',', skiprows=1)
orde = data_cal[:, 0]
atlas_orde = data_cal[:, 1]
lambda_1 = data_cal[:, 2]
lambda_2 = data_cal[:, 3]
lambda_3 = data_cal[:, 4]
pixel_1 = data_cal[:, 5]
pixel_2 = data_cal[:, 6]
pixel_3 = data_cal[:, 7]

uncertainty_x = [0.5, 0.5, 0.5]

# %% Plot residuals residualplot()


def residualplot(x_list, wavelength_list, uncertainty_x, x_pixelvalues, wavelength_object):
    residuals = []
    for i, x_value in enumerate(x_list):
        # Bereken de voorspelde waarde met de fit-coëfficiënten
        predicted_wavelength = sum(fit_1[n]
                                   * (x_value)**n for n in range(len(fit_1)))

        # Bereken residual door verschil werkelijke en voorspelde waarde
        residual = wavelength_list[i] - predicted_wavelength
        residuals.append(residual)

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [7, 2]})
    fig.subplots_adjust(hspace=0)

    yerr = abs(uncertainty_x*np.array(fit_1[1]))

    ax1.set_title("Wavelength calibration fit (x-pixels vs wavelength) Orde {}".format(N_order))
    ax1.plot(x_pixelvalues, wavelength_object)
    ax1.set_ylabel("Wavelength [Angstrom]")
    ax1.errorbar(x_list, wavelength_list, yerr=yerr, fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
    ax1.scatter(x_list, wavelength_list, c='blue')

    ax2.errorbar(x_list, residuals, yerr=yerr, fmt='o', ecolor='red', capsize=3, label='Residuals with error bars')
    ax2.scatter(x_list, residuals)
    ax2.set_ylabel("Pixels")
    ax2.set_ylabel("Residuals [Angstrom]")
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, label='model')
    ax2.axhline(fit_1[1], color='gray', linestyle='--', linewidth=1, label='1 pixel difference')
    ax2.axhline(-1*fit_1[1], color='gray', linestyle='--', linewidth=1)
    for index in range(len(x_list)):
        ax2.text(x_list[index], residuals[index], wavelength_list[index], size=8)
    plt.legend()
    plt.show()
    return

# %% Plot callibrated spectra plotcal()


def plotcal():

    plt.plot(wavelength_object, (flux_object-dark)/(tungstenflat-darkflat))
    plt.ylim(0,)
    plt.show()
    return


# %% Callibration+ Normalization

normal_sun_flux = np.empty((0, 6248), float)
normal_spot_flux = np.empty((0, 6248), float)
normal_sun_wavelength = np.empty((0, 6248), float)
normal_spot_wavelength = np.empty((0, 6248), float)
normal_spot_flux_err = np.empty((0, 6248), float)
normal_sun_flux_err = np.empty((0, 6248), float)

for N_order in range(1, 23):
    suncenternorm = []
    corrected_flux = []
    fitflux = []
    SNR = []

    for i in ('gold', 'saddlebrown'):
        """Laad data van orde N"""
        if i == 'gold':
            data_order_N = np.loadtxt(os.path.join(main_folder, "Flux_raw_sun_centre/data_raw_order_{}.csv").format(N_order), delimiter=',')
        if i == 'saddlebrown':
            suncenternorm = corrected_flux/fitflux
            suncentererr = SNR/fitflux
            data_order_N = np.loadtxt(os.path.join(main_folder, "Flux_raw_sunspot3477/data_raw_order_{}.csv").format(N_order), delimiter=',')

        x_pixelvalues = np.arange(len(data_order_N[0]))
        thar = data_order_N[0]
        tungstenflat = data_order_N[1]
        bias = data_order_N[2]
        dark = data_order_N[3]
        flux_object = data_order_N[4]
        SNR = data_order_N[5]
        darkflat = data_order_N[6]

        """Callibratie meth Th-Ar Spectrum"""
        wavelength_list = [lambda_1[N_order],
                           lambda_2[N_order],
                           lambda_3[N_order]]
        x_list = [pixel_1[N_order],
                  pixel_2[N_order],
                  pixel_3[N_order]]

        fit_order = 2
        fit_1 = np.polynomial.polynomial.polyfit(x_list, wavelength_list,
                                                 fit_order, w=uncertainty_x)

        # x & y coordinaten van de fit
        wavelength_object = []
        for x in x_pixelvalues:
            y = 0
            # Calculate y_coordinate
            for n in range(len(fit_1)):
                y += fit_1[n] * (x)**n
            # Save coordinates
            wavelength_object.append(y)

        corrected_flux = (flux_object-dark)/(tungstenflat-darkflat)
        # residualplot(x_list, wavelength_list, uncertainty_x, x_pixelvalues, wavelength_object)

        """Normaliseer Flux"""

        spectrum = Spectrum1D(flux=corrected_flux*u.Jy, spectral_axis=wavelength_object*u.um)
        with warnings.catch_warnings():  # Ignore warnings
            warnings.simplefilter('ignore')
            g1_fit = fit_generic_continuum(spectrum)
        y_continuum_fitted = np.array(g1_fit(wavelength_object*u.um))

        peak_indexes = find_peaks_cwt(corrected_flux, 1)[0:]

        lesser_peaks_y = np.array(corrected_flux)[peak_indexes]
        lesser_peaks_x = np.array(wavelength_object)[peak_indexes]
        greater_peaks_y = []
        greater_peaks_x = []

        for j in range(len(lesser_peaks_x)):
            if lesser_peaks_y[j] > y_continuum_fitted[peak_indexes][j]:
                greater_peaks_y.append(lesser_peaks_y[j])
                greater_peaks_x.append(lesser_peaks_x[j])

        fit_degree = 50
        fit_3 = np.polynomial.polynomial.polyfit(greater_peaks_x, greater_peaks_y, fit_degree)

        fitflux = []
        for x in wavelength_object:
            y = 0
            # Calculate y_coordinate
            for n in range(len(fit_3)):
                y += fit_3[n] * (x)**n       
            # Save coordinates
            fitflux.append(y)

    normal_sun_flux = np.vstack([normal_sun_flux, suncenternorm])
    normal_sun_flux_err = np.vstack([normal_spot_flux_err, suncenternorm/suncentererr])
    normal_spot_flux = np.vstack([normal_spot_flux, corrected_flux/fitflux])
    normal_spot_flux_err = np.vstack([normal_spot_flux_err, (corrected_flux/fitflux)/SNR])
    normal_sun_wavelength = np.vstack([normal_sun_wavelength, wavelength_object])
    normal_spot_wavelength = np.vstack([normal_spot_wavelength, wavelength_object])

# %% W vinden

#[4108:4226] h alpha
#[3600:3700] 5895.92

spot = 14  # spot=orde-1

flux = normal_sun_flux[spot]
wavelength = normal_sun_wavelength[spot]
flux_err = normal_sun_flux_err[spot]

spotflux = normal_spot_flux[spot]
spotwavelength = normal_spot_wavelength[spot]
spotflux_err = normal_spot_flux_err[spot]

cen = 4920.837  # Centrale golflengte spectraallijn
pdx = 0.5  # Plus verschuiving van centrum
mdx = 0.5  # Min verschuving van centrum
indices = np.where((wavelength >= cen - mdx) & (wavelength <= cen + pdx))

plt.figure()
plt.plot(wavelength[indices], flux[indices])
plt.axhline(y=1, color='black', linestyle='--')


def gaussian(x, amp, cen, wid, dy):
    return amp * np.exp(-(x-cen)**2 / wid) + dy


init_vals = [-0.5, cen, 2, 1]  # for [amp, cen, wid, dy]
best_vals_sun, covar_sun = curve_fit(gaussian, wavelength[indices], flux[indices], p0=init_vals,  sigma=(flux_err[indices]))
best_vals_spot, covar_spot = curve_fit(gaussian, spotwavelength[indices], spotflux[indices], p0=init_vals,  sigma=(spotflux_err[indices]))

scale = best_vals_sun[3]/best_vals_spot[3]

plt.scatter(wavelength[indices], flux[indices])
plt.plot(wavelength[indices], gaussian(wavelength[indices], best_vals_sun[0], best_vals_sun[1], best_vals_sun[2], best_vals_sun[3]))
plt.scatter(spotwavelength[indices], scale * spotflux[indices])
plt.plot(spotwavelength[indices], scale * gaussian(spotwavelength[indices], best_vals_spot[0], best_vals_spot[1], best_vals_spot[2], best_vals_spot[3]))

perr = np.sqrt(np.diag(covar_sun))
W = 2*best_vals_sun[2]*np.log(2)
dW = 2*perr[2]*np.log(2)
print(W, dW)
print(perr)

# %% Distance


x_values = np.linspace(best_vals_sun[1]-2*W, best_vals_sun[1]+2*W, 100)
g1 = gaussian(x_values, best_vals_sun[0], best_vals_sun[1], best_vals_sun[2], best_vals_sun[3])
g2 = gaussian(x_values, best_vals_spot[0], best_vals_sun[1], best_vals_spot[2], best_vals_spot[3])
distance = np.mean(np.abs(g1 - g2))
d_err = np.std(np.abs(g1 - g2))
print(distance)
print(d_err)


g = 1.2

Magnetic = (0.83 * np.sqrt(W*distance)) / ((cen**2) * (4.67 * (10**(-13))) * g)
print(Magnetic)
# plt.figure()
# plt.plot(x_values, g1)
# plt.plot(x_values, g2)
# plt.show()

# %% Plotjes
spot = 14

plt.figure()
plt.title("Orde {}".format(spot+1))
plt.plot(normal_sun_wavelength[spot], normal_sun_flux[spot], label='sun')
plt.plot(normal_spot_wavelength[spot], normal_spot_flux[spot], label='spot')
plt.scatter(normal_sun_wavelength[spot], normal_sun_flux[spot], label='sun')
plt.scatter(normal_spot_wavelength[spot], normal_spot_flux[spot], label='spot')
plt.ylim(0, 2)
plt.legend()
plt.show()

# %% test




