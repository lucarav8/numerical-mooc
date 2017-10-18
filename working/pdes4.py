import numpy as np
import sympy

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

def linearconv(nx):
    """Solve the linear convection equation.

    Solves the equation d_t u + c d_x u = 0 where
    * the wavespeed c is set to 1
    * the domain is x \in [0, 2]
    * 20 timesteps are taken, with \Delta t computed using the CFL 0.5
    * the initial data is the hat function

    Produces a plot of the results

    Parameters
    ----------

    nx : integer
        number of internal grid points

    Returns
    -------

    None : none"""

    dx = 2/(nx-1)
    nt = 20
    c = 1
    sigma = .5
    x = np.linspace(0,2,nx)

    dt = sigma*dx

    x = np.linspace(0,2,nx)

    u = np.ones(nx)
    lbound = np.where(x >= 0.5)
    ubound = np.where(x <= 1)
    u[np.intersect1d(lbound, ubound)]=2

    un = np.ones(nx)

    for n in range(nt):
        un = u.copy()
        u[1:] = un[1:] -c*dt/dx*(un[1:] -un[0:-1])
        u[0] = 1.0


    plot_2dline(x,u,title="Linear convection")

# A function to make a fast plot
def plot_2dline(x,y,xlabel="X",ylabel="Y",title="Title"):
    fig = plt.figure(figsize=(5, 3), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, 'k-')
    ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
    ax.set_xlabel(xlabel, fontsize=14) #x label
    ax.set_ylabel(ylabel, fontsize=14) #y label
    ax.set_title(title, fontsize=12)
    # ax.set_ylim(0, 2*y0)
    # ax.grid(color='k', linestyle='--', linewidth=0.25)
    fig.subplots_adjust(bottom=0.2, left=0.15)

    fig.show()


#Setup the initial conditions!
###############################
#Solve an equation simbolically with sympy!

x, nu, t = sympy.symbols('x nu t')
phi = sympy.exp(-(x-4*t)**2/(4*nu*(t+1))) + \
sympy.exp(-(x-4*t-2*np.pi)**2/(4*nu*(t+1)))
# print(phi)

#Differenciate with respect to x
phiprime = phi.diff(x)
# print(phiprime)

#Find u
u = -2*nu*(phiprime/phi)+4
# print(u)

#Convert the symbolic equation in an
#usable code!
from sympy.utilities.lambdify import lambdify
u_lamb = lambdify((t, x, nu), u)
# print("The value of u at t=1, x=4, nu=3 is {}.".format(u_lamb(1,4,3)))

#Burger's equation
##################

###variable declarations
nx = 101
nt = 100
dx = 2*np.pi/(nx-1)
nu = .07
sigma = .1
dt = sigma*dx**2/nu

x = np.linspace(0, 2*np.pi, nx)
un = np.empty(nx)
t = 0

u = np.asarray([u_lamb(t, x0, nu) for x0 in x])

# plot_2dline(x, u, title="Analytical Burger's Equation")

#Apply periodic boundary conditions and calculate numerically
for n in range(nt):
    un = u.copy()

    u[1:-1] = un[1:-1] - un[1:-1] * dt/dx * (un[1:-1] - un[:-2]) + nu*dt/dx**2*\
                    (un[2:] - 2*un[1:-1] + un[:-2])

    u[0] = un[0] - un[0] * dt/dx * (un[0] - un[-1]) + nu*dt/dx**2*\
                (un[1] - 2*un[0] + un[-1])
    u[-1] = un[-1] - un[-1] * dt/dx * (un[-1] - un[-2]) + nu*dt/dx**2*\
                (un[0]- 2*un[-1] + un[-2])

u_analytical = np.asarray([u_lamb(nt*dt, xi, nu) for xi in x])

#Plot the two curves
# fig = plt.figure(figsize=(5, 3), dpi=150)
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x,u, 'k-')
# ax.plot(x,u_analytical, 'r-')
# ax.legend(['Numerical Solution','Analytical Solution'], fontsize = 6);
# ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
# ax.set_xlabel('t', fontsize=14) #x label
# ax.set_ylabel('z', fontsize=14) #y label
# # ax.set_ylim(0, 200)
# fig.subplots_adjust(bottom=0.2, left=0.15)
# # plt.show()
# plt.close()

#Prepare the animation
from matplotlib import animation
u = np.asarray([u_lamb(t, x0, nu) for x0 in x])


fig = plt.figure(figsize=(5, 3), dpi=150)
ax = fig.add_subplot(1, 1, 1)
line = ax.plot([],[], 'k-')[0]
line2 = ax.plot([],[], 'r-')[0]
ax.legend(['Numerical Solution','Analytical Solution'], fontsize = 6);
ax.tick_params(axis='both', labelsize=10) #increase font size for ticks
ax.set_xlabel('t', fontsize=14) #x label
ax.set_ylabel('z', fontsize=14) #y label
ax.set_ylim(0,10)
ax.set_xlim(0,2*np.pi)
fig.subplots_adjust(bottom=0.2, left=0.15)

def burgers(n):

    un = u.copy()

    u[1:-1] = un[1:-1] - un[1:-1] * dt/dx * (un[1:-1] - un[:-2]) + nu*dt/dx**2*\
                    (un[2:] - 2*un[1:-1] + un[:-2])

    u[0] = un[0] - un[0] * dt/dx * (un[0] - un[-1]) + nu*dt/dx**2*\
                (un[1] - 2*un[0] + un[-1])
    u[-1] = un[-1] - un[-1] * dt/dx * (un[-1] - un[-2]) + nu*dt/dx**2*\
                (un[0]- 2*un[-1] + un[-2])

    u_analytical = np.asarray([u_lamb(n*dt, xi, nu) for xi in x])
    line.set_data(x,u)
    line2.set_data(x, u_analytical)


anim = animation.FuncAnimation(fig, burgers,
                        frames=nt, interval=100)
#Save the animation
anim.save('burger_wave.mp4')
#Run the animation!
plt.show()
