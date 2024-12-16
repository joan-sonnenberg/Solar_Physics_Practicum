import matplotlib.pyplot as plt
import numpy as np
with open('paper_solid_line.csv', 'r') as MyData:
    wavelength_solid, flux_solid, = [], []

    for line in MyData:                     
        data_cut = line.split(',')
        wavelength_solid.append(float(data_cut[0]))
        flux_solid.append(float(data_cut[1]))
        
with open('paper_dot_line.csv', 'r') as MyData:
    wavelength_dot, flux_dot, = [], []

    for line in MyData:                     
        data_cut = line.split(',')
        wavelength_dot.append(float(data_cut[0]))
        flux_dot.append(float(data_cut[1]))
        
    
wavelength_solid = np.array(wavelength_solid)
flux_solid = np.array(flux_solid)
wavelength_dot = np.array(wavelength_dot)
flux_dot = np.array(flux_dot)

plt.figure(figsize=(10, 7))
plt.plot(wavelength_solid, flux_solid, color="black", label="Sunspot spectrum (ESPARTACO)")
plt.plot(wavelength_dot, flux_dot, color="black", linestyle="--", label="Sun spectrum (ESPARTACO)")
plt.tick_params(axis='both', which='major', labelsize=11)  # Change 'major' ticks
plt.tick_params(axis='both', which='minor', labelsize=11)
plt.xlim(6301, 6303)
plt.ylim(1, 4)
plt.legend()
plt.savefig("paper.jpg", dpi=1000)