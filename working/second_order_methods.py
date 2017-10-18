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


def rk2_step(u, f, dt):
    """Returns the solution at the next time-step using 2nd-order Runge-Kutta.

    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        solution at the next time step.
    """
    u_star = u + 0.5 * dt * f(u)
    return u + dt * f(u_star)


def lf_step(unm1, u, f, dt):
    """Returns the solution time-step n+1) using leapfrog's method.

    Parameters
    ----------
    unm1 : array of float
        solution at time-step n-1.
    u : array of float
        solution at time-step n.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.

    Returns
    -------
    u_n_plus_1 : array of float
        solution at time-step n+1.
    """
    return unm1 + 2.0 * dt * f(u)

# model parameters:
g = 9.8      # gravity in m s^{-2}
v_t = 4.9    # trim velocity in m s^{-1}
C_D = 1/5.0  # drag coefficient --- or D/L if C_L=1
C_L = 1.0    # for convenience, use C_L = 1

### set initial conditions ###
v0 = 6.5     # start at the trim velocity (or add a delta)
theta0 = -0.1 # initial angle of trajectory
x0 = 0.0     # horizotal position is arbitrary
y0 = 25.0     # initial altitude

# set time-increment and discretize the time
T  = 36.0                           # final time
dt = 0.01                             # set time-increment
N  = int(T/dt) + 1                   # number of time-steps

# set initial conditions
u_euler = np.empty((N, 4))
u_rk = np.empty((N, 4))
u_lf = np.empty((N, 4))

# initialize the array containing the solution for each time-step
u_euler[0] = np.array([v0, theta0, x0, y0])
u_rk[0] = np.array([v0, theta0, x0, y0])
u_lf[0] = np.array([v0, theta0, x0, y0])


# use a for loop to call the function rk2_step()
for n in range(N - 1):
    u_euler[n + 1] = euler_step(u_euler[n], f, dt)
    u_rk[n + 1] = rk2_step(u_rk[n], f, dt)
    if n == 0:
        u_lf[n + 1] = rk2_step(u_rk[n], f, dt)
    else:
        u_lf[n + 1] = lf_step(u_lf[n - 1], u_lf[n], f, dt)


# get the glider's position with respect to the time,
# according to the two different methods
x_euler = u_euler[:,2]
y_euler = u_euler[:,3]
x_rk = u_rk[:,2]
y_rk = u_rk[:,3]
x_lf = u_lf[:,2]
y_lf = u_lf[:,3]



# get the index of element of y where altitude becomes negative
idx_negative_euler = np.where(y_euler < 0.0)[0]
if len(idx_negative_euler) == 0:
    idx_ground_euler = N - 1
    print ('Euler integration has not touched ground yet!')
else:
    idx_ground_euler = idx_negative_euler[0]

idx_negative_rk = np.where(y_rk < 0.0)[0]
if len(idx_negative_rk) == 0:
    idx_ground_rk = N - 1
    print ('Runge-Kutta integration has not touched ground yet!')
else:
    idx_ground_rk = idx_negative_rk[0]

idx_negative_lf = np.where(y_lf < 0.0)[0]
if len(idx_negative_lf) == 0:
    idx_ground_lf = N - 1
    print ('Leapfrog integration has not touched ground yet!')
else:
    idx_ground_lf = idx_negative_lf[0]

# check to see if the paths match (default 10^-5 difference)
print('Are the x-values close? {}'.format(np.allclose(x_euler, x_rk)))
print('Are the y-values close? {}'.format(np.allclose(y_euler, y_rk)))




# visualization of the path
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_euler[:idx_ground_euler], y_euler[:idx_ground_euler], 'k-', label='Euler')
ax.plot(x_rk[:idx_ground_rk], y_rk[:idx_ground_rk], 'r--', label='RK2')
ax.plot(x_lf[:idx_ground_lf], y_lf[:idx_ground_lf], 'r--', label='LF')
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('x', fontsize=14) #x label
ax.set_ylabel('y', fontsize=14) #y label
ax.set_title('Trajectories', fontsize=12)
ax.legend(fontsize = 6)
ax.grid(color='k', linestyle='--', linewidth=0.25)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x_euler[:idx_ground_euler], y_euler[:idx_ground_euler], 'k-', label='Euler')
ax2.plot(x_rk[:idx_ground_rk], y_rk[:idx_ground_rk], 'r--', label='RK2')
ax2.plot(x_lf[:idx_ground_lf], y_lf[:idx_ground_lf], 'g:', label='LF')
ax2.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax2.set_xlabel('x', fontsize=14) #x label
ax2.set_ylabel('y', fontsize=14) #y label
ax2.legend(fontsize = 6)#['Euler','Runge-Kutta 2'], fontsize = 6)
ax2.set_xlim(0, 10)
ax2.set_ylim(23, 26)
ax2.grid(color='k', linestyle='--', linewidth=0.25)
fig.subplots_adjust(bottom=0.2, wspace = 0.4)

fig.show()
input("Press enter to continue") #Hack to keep plot open







# Compare the errors calculating with different time grids
# dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
dt_values = np.array([0.01, 0.005, 0.001]) #Leapfrog explodes!

u_e_values = np.empty_like(dt_values, dtype=np.ndarray)
u_rk_values = np.empty_like(dt_values, dtype=np.ndarray)
u_lf_values = np.empty_like(dt_values, dtype=np.ndarray)

for i, dt in enumerate(dt_values):

    N = int(T / dt) + 1  # number of time-steps

    ### discretize the time t ###
    t = np.linspace(0.0, T, N)

    # initialize the array containing the solution for each time-step
    u_e = np.empty((N, 4))
    u_e[0] = np.array([v0, theta0, x0, y0])
    u_rk = np.empty((N, 4))
    u_rk[0] = np.array([v0, theta0, x0, y0])
    u_lf = np.empty((N, 4))
    u_lf[0] = np.array([v0, theta0, x0, y0])

    # time loop
    for n in range(N - 1):
        u_e[n + 1] = euler_step(u_e[n], f, dt)  ### call euler_step() ###
        u_rk[n + 1] = rk2_step(u_rk[n], f, dt)
        if n == 0:
            u_lf[n + 1] = rk2_step(u_rk[n], f, dt)
        else:
            # print(u_lf[n-1], u_lf[n], f, dt)
            u_lf[n + 1] = lf_step(u_lf[n-1], u_lf[n], f, dt)

    # store the value of u related to one grid
    u_e_values[i] = u_e
    u_rk_values[i] = u_rk
    u_lf_values[i] = u_lf


# compute difference between one grid solution and the finest one
diffgrid_e = np.empty_like(dt_values)
diffgrid_rk = np.empty_like(dt_values)
diffgrid_lf = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):
    # print('dt = {}'.format(dt))

    ### call the function get_diffgrid() ###
    diffgrid_e[i] = get_diffgrid(u_e_values[i], u_e_values[-1], dt)
    diffgrid_rk[i] = get_diffgrid(u_rk_values[i], u_rk_values[-1], dt)
    diffgrid_lf[i] = get_diffgrid(u_lf_values[i], u_lf_values[-1], dt)





# Calculate the order of convergence of the method's

r = 2
h = 0.001

dt_values2 = np.array([h, r * h, r ** 2 * h])
u_e_values2 = np.empty_like(dt_values2, dtype=np.ndarray)
u_rk_values2 = np.empty_like(dt_values2, dtype=np.ndarray)
u_lf_values2 = np.empty_like(dt_values2, dtype=np.ndarray)
diffgrid_e2 = np.empty(2)
diffgrid_rk2 = np.empty(2)
diffgrid_lf2 = np.empty(2)

for i, dt in enumerate(dt_values2):

    N = int(T / dt) + 1  # number of time-steps
    # initialize the array containing the solution for each time-step
    u_e2 = np.empty((N, 4))
    u_e2[0] = np.array([v0, theta0, x0, y0])
    u_rk2 = np.empty((N, 4))
    u_rk2[0] = np.array([v0, theta0, x0, y0])
    u_lf2 = np.empty((N, 4))
    u_lf2[0] = np.array([v0, theta0, x0, y0])

    # time loop
    for n in range(N - 1):
        u_e2[n + 1] = euler_step(u_e2[n], f, dt)  ### call euler_step() ###
        u_rk2[n + 1] = rk2_step(u_rk2[n], f, dt)
        if n == 0:
            u_lf2[n + 1] = rk2_step(u_rk2[n], f, dt)
        else:
            u_lf2[n + 1] = lf_step(u_lf2[n-1], u_lf2[n], f, dt)

    # store the value of u related to one grid
    u_e_values2[i] = u_e2
    u_rk_values2[i] = u_rk2
    u_lf_values2[i] = u_lf2

# calculate f2 - f1
diffgrid_e2[0] = get_diffgrid(u_e_values2[1], u_e_values2[0], dt_values2[1])
diffgrid_rk2[0] = get_diffgrid(u_rk_values2[1], u_rk_values2[0], dt_values2[1])
diffgrid_lf2[0] = get_diffgrid(u_lf_values2[1], u_lf_values2[0], dt_values2[1])
# calculate f3 - f2
diffgrid_e2[1] = get_diffgrid(u_e_values2[2], u_e_values2[1], dt_values2[2])
diffgrid_rk2[1] = get_diffgrid(u_rk_values2[2], u_rk_values2[1], dt_values2[2])
diffgrid_lf2[1] = get_diffgrid(u_lf_values2[2], u_lf_values2[1], dt_values2[2])
# calculate the order of convergence
p_e = (log(diffgrid_e2[1]) - log(diffgrid_e2[0])) / log(r)
p_rk = (log(diffgrid_rk2[1]) - log(diffgrid_rk2[0])) / log(r)
p_lf = (log(diffgrid_lf2[1]) - log(diffgrid_lf2[0])) / log(r)

print(diffgrid_e)
# Plot the results
fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
ax.loglog(dt_values[:-1], diffgrid_e[:-1], color='k', ls='', lw=2, marker='o', label='Euler: p = {:.3f}'.format(p_e))
ax.loglog(dt_values[:-1], diffgrid_rk[:-1], color='r', ls='', lw=2, marker='o', label='Runge-Kutta: p = {:.3f}'.format(p_rk))
ax.loglog(dt_values[:-1], diffgrid_lf[:-1], color='g', ls='', lw=2, marker='o', label='Leapfrog: p = {:.3f}'.format(p_lf))
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('$\Delta t$', fontsize=14) #x label
ax.set_ylabel('$L_1$-norm of the grid differences', fontsize=10) #y label
ax.legend(fontsize=6)
# ax.set_ylim(0, 200)
ax.grid(color='k', linestyle='-', linewidth=0.5)
plt.axis("equal")   #make axes scale equally;
fig.subplots_adjust(bottom=0.2, left=0.15)

fig.show()
input("Press enter to continue") #Hack to keep plot open


print('The order of convergence for Euler method is p = {:.3f}'.format(p_e))
print('The order of convergence for Runge-Kutta 2 method is p = {:.3f}'.format(p_rk))
print('The order of convergence for Leapfrog method is p = {:.3f}'.format(p_lf))