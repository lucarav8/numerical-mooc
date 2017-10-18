import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


#Define variables
nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
dx = 2/(nx-1)
nt = 25
dt = .02
c = 1      #assume wavespeed of c = 1
x = np.linspace(0,2,nx)

#Define the square wavespeed
u = np.ones(nx)      #numpy function ones()
lbound = np.where(x >= 0.5)
ubound = np.where(x <= 1)

bounds = np.intersect1d(lbound, ubound)
u[bounds]=2  #setting u = 2 between 0.5 and 1 as per our I.C.s
print(u)

#Plot
fig = plt.figure(figsize=(5,3), dpi=150)
ax = fig.add_subplot(1,1,1)
ax.plot(x, u, color='#003366', ls='--', lw=3)
ax.set_ylim(0,2.5);
plt.show()

#Linear case
for n in range(1,nt):
    un = u.copy()
    for i in range(1,nx):
        u[i] = un[i]-c*dt/dx*(un[i]-un[i-1])

fig = plt.figure(figsize=(5,3), dpi=150)
ax = fig.add_subplot(1,1,1)
ax.plot(x, u, color='#003366', ls='--', lw=3)
ax.set_ylim(0,2.5);
plt.show()

####################
#Nonlinear case
####################
##problem parameters
nx = 41
dx = 2/(nx-1)
nt = 10
dt = .02

##initial conditions
u = np.ones(nx)
u[np.intersect1d(lbound, ubound)]=2

fig = plt.figure(figsize=(5,3), dpi=150)
ax = fig.add_subplot(1,1,1)
ax.plot(x, u, color='#003366', ls='--', lw=3)
ax.set_ylim(0,2.5);
plt.show()

for n in range(1, nt):
    un = u.copy()
    u[1:] = un[1:]-un[1:]*dt/dx*(un[1:]-un[0:-1])


fig = plt.figure(figsize=(5,3), dpi=150)
ax = fig.add_subplot(1,1,1)
ax.plot(x, u, color='#003366', ls='--', lw=3)
ax.set_ylim(0,2.5);
plt.show()
