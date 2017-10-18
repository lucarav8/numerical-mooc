#! /usr/bin/python3
import numpy as np
import matplotlib as mpl
import pandas as pd
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

def plot_matrix(ax, dataframe, theta_values, v_values):
    max = dataframe.values.max()
    min = dataframe.values.min()
    if (max >= 0 and min >= 0):
        val_M = max
        val_m = 0
        color = 'Reds'
    elif (max <= 0 and min <= 0):
        val_M = 0
        val_m = min
        color = 'Blues_r'
    else:
        if abs(max) >= abs(min):
            val_M = max
            val_m = -max
            color = 'bwr'
        else:
            val_M = -min
            val_m = min
            color = 'bwr'

    cax = ax.matshow(dataframe, cmap=color, vmax=val_M, vmin=val_m)
    cbar = ax.get_figure().colorbar(cax, shrink=0.75)
    cbar.ax.tick_params(which='both', axis='y', direction='out',
                   labelsize=7, pad=2)

    ax.set_xticks(np.arange(0, dataframe.shape[1], 1))
    ax.set_yticks(np.arange(0, dataframe.shape[0], 1))
    ax.set_yticklabels(theta_values)
    ax.set_xticklabels(["%.2f" % v for v in v_values])
    ax.tick_params(which='both', axis='both', direction='out',
                    labelsize=6, bottom='off', right='off', pad=2)
    ax.tick_params(which='both', axis='x', pad=2,)
    ax.tick_params(which='minor', top='off', left='off')
    plt.xticks(rotation=90)
    # ax.grid(which='minor')



# model parameters:
g = 9.8      # gravity in m s^{-2}
v_t = 4.9   # trim velocity in m s^{-1}
C_D = 1/5  # drag coefficient --- or D/L if C_L=1
C_L = 1   # for convenience, use C_L = 1

### set initial conditions ###
v0 = 1000     # start at the trim velocity (or add a delta)
theta0 = 0 # initial angle of trajectory
x0 = 0     # horizotal position is arbitrary
y0 = 10  # initial altitude

# T = 100  # final time
dt = 0.001  # time increment
# N = int(T / dt) + 1  # number of time-steps

# initialize the array containing the solution for each time-step
u = np.empty((1, 4))
u[0] = np.array([v0, theta0, x0, y0])  # fill 1st element with initial values


# # time loop - Euler method
# for n in range(N - 1):
#     u[n + 1] = euler_step(u[n], f, dt)
theta_max = 45
theta_min = 0
theta_step = 5
theta_N = int((theta_max - theta_min)/theta_step) + 1
theta_values = np.linspace(theta_min, theta_max, theta_N)

v_max = 10
v_min = 1
v_step = 1
v_N = int((v_max - v_min)/v_step) + 1
v_values = np.linspace(v_min, v_max, v_N)


distance = np.empty((theta_N, v_N))
cycle = 1
for num_t, theta0 in enumerate(theta_values):
    for num_v, v0 in enumerate(v_values):
        u = np.empty((1, 4))
        u[0] = np.array([v0, theta0, x0, y0])
        n = 0
        while u[n][3] > 0:
            u = np.vstack((u, np.array(euler_step(u[n], f, dt))))

            n += 1
        distance[num_t][num_v] = u[n][2]
    print("The plane flew for {} m at cycle {}".format(distance[num_t][num_v], cycle))
    cycle += 1

max = distance.max()
i,j = np.where(distance==max)
best_theta = theta_values[i][0]
best_v = v_values[j][0]

print("\nBest flight data: \n"
      " - The plane flew for {} m, \n"
      " - Initial angle: {} degres, \n"
      " - Initial velocity: {} m/s.\n".format(max, best_theta, best_v))
# max_theta, max_v = np.


# visualization of the matrix of distances
print("Printing the matrix of flight tests...")
fig = plt.figure(figsize=(3, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
plot_matrix(ax, pd.DataFrame(distance), theta_values, v_values)
ax.tick_params(axis='both', labelsize=5) #increase font size for ticks
ax.set_xlabel('v', fontsize=14) #x label
ax.set_ylabel('Theta', fontsize=14) #y label
title = ax.set_title('Matrix of distances \n in flight tests', fontsize=8)
title.set_y(1.1)
fig.subplots_adjust(top=0.8)

fig.show()
input("Press enter to continue") #Hack to keep plot open


# Recalculate the trajectory for the best case
theta0 = best_theta
v0 = best_v
u = np.empty((1, 4))
u[0] = np.array([v0, theta0, x0, y0])
n = 0
while u[n][3] > 0:
    u = np.vstack((u, np.array(euler_step(u[n], f, dt))))
    n += 1
x = u[:,2]
y = u[:,3]

# visualization of the path
print("\nPrinting the trajectory of the best flight...")
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, 'k-')
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('x', fontsize=14) #x label
ax.set_ylabel('y', fontsize=14) #y label
ax.set_title('Glider trajectory, flight distance = %.2f m' % max, fontsize=12)
ax.set_ylim(0, 2*y0)
ax.grid(color='k', linestyle='--', linewidth=0.25)
fig.subplots_adjust(bottom=0.2, left=0.15)

fig.show()
input("Press enter to continue") #Hack to keep plot open

