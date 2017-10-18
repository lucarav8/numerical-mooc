import numpy as np

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
    print("aaa")
    input("Press enter to continue..")

#Define variables
nx = 41
dx = 2./(nx-1)
nt = 20
nu = 0.3   #the value of viscosity
sigma = .2
dt = sigma*dx**2/nu

x = np.linspace(0,2,nx)
ubound = np.where(x >= 0.5)
lbound = np.where(x <= 1)

u = np.ones(nx)
u[np.intersect1d(lbound, ubound)] = 2

un = np.ones(nx)

#Diffusion equation
for n in range(nt):
    un = u.copy()
    u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:] -2*un[1:-1] +un[0:-2])

# plot_2dline(x, u, "Diffusion equation")

#Setup an animation
from matplotlib import animation

#Reset initial conditions
nt = 50

u = np.ones(nx)
u[np.intersect1d(lbound, ubound)] = 2

un = np.ones(nx)

#Set an initial empty figure
fig = plt.figure(figsize=(8,5))
ax = plt.axes(xlim=(0,2), ylim=(1,2.5))
line = ax.plot([], [], color='#003366', ls='--', lw=3)[0]

#Define a propagation equation
def diffusion(i):
    line.set_data(x,u)

    un = u.copy()
    u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:] -2*un[1:-1] +un[0:-2])

#Set the animation object
anim = animation.FuncAnimation(fig, diffusion,
                               frames=nt, interval=100)
#Run the animation!
plt.show()
