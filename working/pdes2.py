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
# nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
# dx = 2/(nx-1)
# nt = 25
# dt = .02
# c = 1      #assume wavespeed of c = 1
# x = np.linspace(0,2,nx)
#
# #Define the square wavespeed
# u = np.ones(nx)      #np function ones()
# lbound = np.where(x >= 0.5)
# ubound = np.where(x <= 1)
#
# bounds = np.intersect1d(lbound, ubound)
# u[bounds]=2  #setting u = 2 between 0.5 and 1 as per our I.C.s

linearconv(800)
