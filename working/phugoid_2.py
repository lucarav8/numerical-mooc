#! /usr/bin/python3
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from math import sin, cos, log, ceil
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equations.
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """

    return u + dt * f(u)

def f(u):
    """Returns the right-hand side of the phugoid system of equations.

    Parameters
    ----------
    u : array of float
        array containing the solution at time n.

    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """

    v = u[0]
    theta = u[1]
    x = u[2]
    y = u[3]
    return np.array([-g * sin(theta) - C_D / C_L * g / v_t ** 2 * v ** 2,
                        -g * cos(theta) / v + g / v_t ** 2 * v,
                        v * cos(theta),
                        v * sin(theta)])


def get_diffgrid(u_current, u_fine, dt):
    """Returns the difference between one grid and the fine one using L-1 norm.

    Parameters
    ----------
    u_current : array of float
        solution on the current grid.
    u_finest : array of float
        solution on the fine grid.
    dt : float
        time-increment on the current grid.

    Returns
    -------
    diffgrid : float
        difference computed in the L-1 norm.
    """

    N_current = len(u_current[:, 0])
    N_fine = len(u_fine[:, 0])

    grid_size_ratio = ceil(N_fine / N_current)

    diffgrid = dt * np.sum(np.abs( \
        u_current[:, 2] - u_fine[::grid_size_ratio, 2]))

    return diffgrid


# model parameters:
g = 9.8      # gravity in m s^{-2}
v_t = 4.9   # trim velocity in m s^{-1}
C_D = 1/5  # drag coefficient --- or D/L if C_L=1
C_L = 1   # for convenience, use C_L = 1

### set initial conditions ###
v0 = v_t     # start at the trim velocity (or add a delta)
theta0 = 0 # initial angle of trajectory
x0 = 0     # horizotal position is arbitrary
y0 = 1000  # initial altitude

T = 100  # final time
dt = 0.1  # time increment
N = int(T / dt) + 1  # number of time-steps

# initialize the array containing the solution for each time-step
u = np.empty((N, 4))
u[0] = np.array([v0, theta0, x0, y0])  # fill 1st element with initial values

# time loop - Euler method
for n in range(N - 1):
    u[n + 1] = euler_step(u[n], f, dt)

# get the glider's position with respect to the time
x = u[:,2]
y = u[:,3]

# visualization of the path
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'k-')
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('x', fontsize=14) #x label
ax.set_ylabel('y', fontsize=14) #y label
ax.set_title('Glider trajectory, flight time = %.2f' % T, fontsize=14)
# ax.set_ylim(0, 200)
ax.grid(color='k', linestyle='--', linewidth=0.25)
# plt.axis("equal")   #make axes scale equally;
fig.subplots_adjust(bottom=0.2, left=0.15)

fig.show()
input("Press enter to continue") #Hack to keep plot open

# Compare the errors calculating with different time grids
dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])

u_values = np.empty_like(dt_values, dtype=np.ndarray)

for i, dt in enumerate(dt_values):

    N = int(T / dt) + 1  # number of time-steps

    # initialize the array containing the solution for each time-step
    u = np.empty((N, 4))
    u[0] = np.array([v0, theta0, x0, y0])

    # time loop
    for n in range(N - 1):
        u[n + 1] = euler_step(u[n], f, dt)  ### call euler_step() ###

    # store the value of u related to one grid
    u_values[i] = u


# compute difference between one grid solution and the finest one
diffgrid = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):
    print('dt = {}'.format(dt))

    ### call the function get_diffgrid() ###
    diffgrid[i] = get_diffgrid(u_values[i], u_values[-1], dt)



# Plot the results
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.loglog(dt_values[:-1], diffgrid[:-1], color='k', ls='', lw=2, marker='o')
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('$\Delta t$', fontsize=14) #x label
ax.set_ylabel('$L_1$-norm of the grid differences', fontsize=10) #y label
# ax.set_ylim(0, 200)
ax.grid(color='k', linestyle='-', linewidth=0.5)
plt.axis("equal")   #make axes scale equally;
fig.subplots_adjust(bottom=0.2, left=0.15)

fig.show()
input("Press enter to continue") #Hack to keep plot open

# Calculate the order of convergence of the Euler's method

r = 2
h = 0.001

dt_values2 = np.array([h, r * h, r ** 2 * h])
u_values2 = np.empty_like(dt_values2, dtype=np.ndarray)
diffgrid2 = np.empty(2)

for i, dt in enumerate(dt_values2):

    N = int(T / dt) + 1  # number of time-steps
    # initialize the array containing the solution for each time-step
    u = np.empty((N, 4))
    u[0] = np.array([v0, theta0, x0, y0])

    # time loop
    for n in range(N - 1):
        u[n + 1] = euler_step(u[n], f, dt)  ### call euler_step() ###

    # store the value of u related to one grid
    u_values2[i] = u

# calculate f2 - f1
diffgrid2[0] = get_diffgrid(u_values2[1], u_values2[0], dt_values2[1])
# calculate f3 - f2
diffgrid2[1] = get_diffgrid(u_values2[2], u_values2[1], dt_values2[2])
# calculate the order of convergence
p = (log(diffgrid2[1]) - log(diffgrid2[0])) / log(r)

print('The order of convergence is p = {:.3f}'.format(p));