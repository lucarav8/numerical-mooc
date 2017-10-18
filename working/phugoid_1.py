#! /usr/bin/python3
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


# Equation to solve (constants start with capitol letters)
# z''(t) + G z(t) / Z_t = G
#
# Can be written as:
# z'(t) = b(t)
# b'(t)= G ( 1 - z(t) / Z_t)
#
# Which can be solved by Euler's method:
# f(x + dt) = f(x) + dt*f'(x)


# Calculation parameters
t0 = 0
tf = 200
dt = 0.01
G = 9.81

# Initial conditions
z0 = 100.  #altitude
b0  = 10.  #upward velocity resulting from gust
Zt = 100.

# Define function to run Euler's method
def euler_solv_phugoid(t0, tf, dt, Zt, z0 = 100., b0 = 10.):

    # Initialize array of time points
    t = np.arange(t0, tf + dt, dt)

    # Make an array of the initial conditions
    u = np.array([z0, b0])

    # Initialize an array to hold the changing elevation values
    z = np.zeros(len(t))
    z[0] = z0

    # Numerical solution
    for n in range(1, len(t)):
        u = u + dt * np.array([u[1], G * (1 - u[0] / Zt)])
        z[n] = u[0]

    # Analytical solution
    z_exact = b0 * (Zt / G) ** .5 * np.sin((G / Zt) ** .5 * t) + \
              (z0 - Zt) * np.cos((G / Zt) ** .5 * t) + Zt

    # Calculate the error as sum of the absolute value of the difference at every point
    error = dt * np.sum(np.abs(z - z_exact))

    return t, z, z_exact, error

# Time-loop using Euler's method
t, z, z_exact, error = euler_solv_phugoid(t0, tf, dt, Zt, z0, b0)

# Plot the results
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.plot(t,z, 'k-')
ax.plot(t,z_exact, 'r-')
ax.legend(['Numerical Solution','Analytical Solution'], fontsize = 6);
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('t', fontsize=14) #x label
ax.set_ylabel('z', fontsize=14) #y label
ax.set_ylim(0, 200)
fig.subplots_adjust(bottom=0.2, left=0.15)

fig.show()
input("Press enter to continue") #Hack to keep plot open


#Calculate error with different steps to check convergence of the Euler method

# time-increment and error arrays
dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0001])
error_values = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):
    t, z, z_exact, error_values[i] = euler_solv_phugoid(t0, tf, dt, Zt, z0, b0)

# Plot the results
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.loglog(dt_values, error_values, 'ko')
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('dt', fontsize=14) #x label
ax.set_ylabel('Error', fontsize=14) #y label
# ax.set_ylim(0, 200)
ax.grid(color='k', linestyle='-', linewidth=0.5)
plt.axis("equal")   #make axes scale equally;
fig.subplots_adjust(bottom=0.2, left=0.15)

fig.show()
input("Press enter to continue") #Hack to keep plot open
