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

for i in range(len(Gs)):
    #if "Mg" in Lines[i]:
    #if i == 0 or i == 1 or i == 2 or i == 4 or i == 6 or i == 7 or i == 11 or i == 12: # F lines
        x = Gs[i] * ((Rests[i])**2) 
        
        y = Deltas[i]
        x_axis.append(x)
        y_axis.append(y)
        Deltas_errors_units.append(Deltas_errors[i])
    
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



# Plot data with error bars and the fit
plt.scatter(x_axis, y_axis)
plt.errorbar(x_axis, y_axis, yerr=Deltas_errors_units, fmt='o', color="blue", capsize=5)
plt.plot(x_axis, fit_result.best_fit, 'r-', label="Best Fit")
plt.ylabel("Δλ (Å)")
plt.xlabel("Gλ² (Å²)")
#plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("results.jpg")
plt.show()
