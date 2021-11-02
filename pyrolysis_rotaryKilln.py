#----------------------------------------------------------------------------------------------------------------------#
# Algorithm for the solution of slow biomass pyrolysis in a rotary kiln with particle residence time distribution      #
#                                                                                                                      #
# When using the script, please cite:                                                                                  #
# ------------------------------------                                                                                 #
# Mario Pichler, Bahram Haddadi, Christian Jordan, Hamidreza Norouzi and Michael Harasek:                              #
# Influence of Particle Residence Time Distribution on the Biomass Pyrolysis in a Rotary Kiln,                         #
# Journal of Analytical and Applied Pyrolysis 2021, 158, 105171, DOI: 10.1016/j.jaap.2021.105171.                      #
#                                                                                                                      #
# This program is free software: you can redistribute it and/or modify                                                 #
# it under the terms of the GNU Affero General Public License as                                                       #
# published by the Free Software Foundation, either version 3 of the                                                   #
# License, or (at your option) any later version.                                                                      #
#                                                                                                                      #
# This program is distributed in the hope that it will be useful,                                                      #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                                       #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                                        #
# GNU Affero General Public License for more details.                                                                  #
#                                                                                                                      #
# You should have received a copy of the GNU Affero General Public License                                             #
# along with this program.  If not, see <https://www.gnu.org/licenses/>.                                               #
#                                                                                                                      #
# Imlementation:                                                                                                       #
# Dipl. Ing. Mario Pichler                                                                                             #
# Copyright (C) 2020  Mario Pichler                                                                                    #
#----------------------------------------------------------------------------------------------------------------------#

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import scipy.integrate as spi
import os

#----------------------------------------------------------------------------------------------------------------------#
# Defining needed general functions:                                                                                   #
# ca_in(t,nt) ... pulse function                                                                                       #
# dispersionModel(nx, te, nt, Pe, tau) ... axial dispersion model                                                      #
# standardPlot(..) ... function for creating plots                                                                     #
#----------------------------------------------------------------------------------------------------------------------#

def ca_in(t, nt):
    t_in = 1/(1.1*nt)
    if t <= t_in:
        ca_in = 1/t_in
    else:
        ca_in = 0

    return ca_in

def dispersionModel(nx, te, nt, Pe, tau):
    print('nx=',nx,'nt=',nt,'Pe=',Pe,'tau=',round(tau,2))
    dx = 1/nx
    dt = te/nt

    # Define Matrix A
    A = np.zeros((nx, nx))

    e = 1/tau
    a = e * dt/dx * (1/(Pe*dx) + 1/2)
    b = 1 - 2*e*dt/(Pe*dx**2)
    c = e * dt/dx * (1/(Pe*dx) - 1/2)

    # Define diagonal entries
    for i in range(1, nx-1):
        A[i, i - 1] = a
        A[i, i] = b
        A[i, i + 1] = c

    # define boundary conditions
    A[0, 0] = b - 2*dx*Pe*a
    A[0, 1] = a + c

    A[-1, -2] = a + c
    A[-1, -1] = b

    # Solve the system
    t_plot = np.zeros((nt+2, 1))
    ca_plot = np.zeros((nt+2, 1))
    ca = np.zeros((nx, 1))
    t = 0
    ca[0, 0] = ca_in(t, nt)
    perc_old = 0
    for i in range(nt):
        perc_new = round(t / te * 100,2)
        if perc_new % 1 == 0 and perc_old != perc_new:
            perc_old = round(t / te * 100,2)
            print('ADM, Pe =', Pe, ',', perc_old, '%')
        ca = np.dot(A, ca)
        ca[0, 0] = ca[0, 0] + ca_in(t, nt) * 2 * dx * Pe * a
        t = t + dt
        t_plot[i, 0] = t
        ca_plot[i, 0] = ca[-1, 0]

    t_plot[-2, 0] = t_plot[-3, 0]
    t_plot[-1, 0] = 100000
    t_plot = t_plot

    return t_plot, ca_plot

def standardPlot(xData, yData, xL, xU, yL, yU, color=None, title='', xlabel='$x$', ylabel='$y$', style='-', lWidth=2.0, 
                 mSize=5.0, figNr=1, figSize=(7, 6), grid=False, xFormatter='%.2f', yFormatter='%.2f'):
    # Create empty figure
    fig = plt.figure(figNr, figSize)

    # Add plot to figure
    myPlot = fig.add_subplot(1, 1, 1)

    # Set plot title
    plt.title(title, size=22)

    # Adjust plot in figure, set margins
    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.20, right=0.95)

    # Create plot

    myPlot.plot(xData, yData, style, linewidth=lWidth, markersize=mSize, color=color)

    ax = myPlot.axes
    # Set upper and lower bound for the x- and y-axis
    ax.set_ybound(lower=yL, upper=yU)
    ax.set_xbound(lower=xL, upper=xU)

    # Set axis labels, size, and position
    plt.xlabel(xlabel, size=22, position=(0.5, 0.0))
    plt.ylabel(ylabel, size=22, position=(0.0, 0.5))

    # Set ticklabels, position, and label size
    ax.set_xticklabels(ax.get_xticks(), size=18, position=(0.0, -0.05))
    ax.set_yticklabels(ax.get_yticks(), size=18, position=(-0.02, 0.0))
    ax.grid(grid)

    # Change linewidth or ticks and frame
    for line in myPlot.get_xticklines() + myPlot.get_yticklines():
        line.set_markersize(10)
        line.set_markeredgewidth(1.5)
    myPlot.patch.set(linewidth=1.5, linestyle='solid', edgecolor='black')
    ax.xaxis.set_major_formatter(FormatStrFormatter(xFormatter))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yFormatter))

    return ax

#----------------------------------------------------------------------------------------------------------------------#
# Defining initial and operating conditions                                                                            #
#----------------------------------------------------------------------------------------------------------------------#

subSample = 100 # for plotting ... plotting subSample number of points

# boundary conditions killn
beta_grad = 3.           # angle of inclanation in °
x_max = 2.               # kiln length in m
R = 200/1000             # drum radius in m
n = 2/60                 # rotational speed in rotations per second
phi_grad = 37.           # dynamic angle of repose in °

# initial conditions
H0 = 2.5/100            # filling height at outlet in m, is dam height or particle diameter
T0 = 25+273.15          # ambient temperature in K
p = 1.013*10**5         # pressure in kiln, Pa
F_N2 = 1/3600           # Flow rate nitrogen in kg/s
F_fg = 175/3600         # Flow rate flue gas in kg/s
T_fg_in = 560+273.15    # Flue gas temperature in K 560°C
Feed = 30/3600          # Feed flow wood kg/s

H2O = 1*10**(-3)        # Massfraction of water in Feed, min 10**-3 (0.1%)
rho_H2O_0 = 1000        # density of water at inlet conditions in kg/m**3

t_max = 1400            # maximum time for RTD integration, in seconds
nt = 50                 # nr of solids in simulation
Feed_limit = 0.01       # limit for feed flow to include in simulation in %
Pe = 10                 # Peclet Number for ADM > 50 'plug flow'

# Model Parameter
rho_p = 650         # particle density in kg/m**3
epsilon_b = 0.57    # bed porosity
d_p = 1.5/100       # particle diameter in m
k_b1 = 0.11          # bed conductivity in W/m K
k_g = 0.026         # gas conductivity in W/m K
nu = 4*10**(-6)     # kinematic viscosity in m**2/s
h_ins = 2.5         # heat transfer coefficient insulation in W/m**2 K
emm_ew = 0.65       # emmisivity exposed wall
emm_eb = 0.82       # emmisivity exposed bed

# Inert Material
Feed_inert = 0/3600          # Feed flow wood kg/s
rho_p_inert = 2500            # particle density in kg/m**3
a_cp_inert = [1500, 1, 0]; b_cp_inert = [1, 2] # inert particle cp coefficient

# mesh
nx = 10001            # number of kiln sections
nx_save = 101

# ADM
Pe_list = [5]         # Peclet Number for ADM
nxADM_list = [100]       # Mesh for ADM
nt_list = [40]         # nr of solids in simulation

# directory name and log file name
saveDirectory = './MyFirstResults/' # end with /
logFileName = 'log_'

#----------------------------------------------------------------------------------------------------------------------#
# calculating bed height and mean particle residence time                                                              #
#----------------------------------------------------------------------------------------------------------------------#
# create directories
if not os.path.exists(saveDirectory):
    os.mkdir(saveDirectory)
if not os.path.exists(saveDirectory + 'logFiles/'):
    os.mkdir(saveDirectory + 'logFiles/')

Q_fixed = Feed*(1-H2O)/(rho_p*(1-epsilon_b)) + Feed*H2O/(rho_H2O_0*(1-epsilon_b)) + \
          Feed_inert/(rho_p_inert*(1-epsilon_b))     # volumetric feed flow in m**3/s

# define plot size
l_plot = 12*1.5
h_plot = 2*R/x_max*l_plot*2

# solving for filling height and mean residence time. Seaman's model
def f(h, x):
    H = h[0]
    phi = phi_grad * math.pi / 180
    beta = beta_grad * math.pi / 180
    a = 3 * np.tan(phi) * Q_fixed / (math.pi * 4 * n)

    h_x = (a*(R**2-(H-R)**2)**(-3/2)-np.tan(beta)/np.cos(phi))
    return h_x

# initial conditions
y0 = [H0]                                # dam height at outlet
x = np.linspace(0, x_max, nx)            # create grid
x_save = np.linspace(0, x_max, nx_save)           # create grid

# solve Seaman's model
soln = odeint(f, y0, x)
h = np.flipud(soln[:, 0])                # flip coordinate system

# calculating bed cross section over length and bed volume
s_bed = np.zeros((len(x), 1))   # initializing bed cross section
s_gas = np.zeros((len(x), 1))   # initializing gas cross section
s_bed[0,0] = R**2*math.acos((R - h[0])/R) - (R - h[0])*math.sqrt(2*R*h[0] - h[0]**2)
volume_bed = 0                  # initializing bed volume

for i in range(1, len(s_bed)):
    s_bed[i, 0] = R**2*math.acos((R - h[i])/R) - (R - h[i])*math.sqrt(2*R*h[i] - h[i]**2)
    s_gas[i, 0] = R**2*math.pi - s_bed[i,0]
    volume_bed = volume_bed + x_max/(nx*2)*(s_bed[i,0] + s_bed[i-1,0])

tau = volume_bed/Q_fixed # mean residence time,s
print('Bed Volume: ', round(volume_bed*10**6,1), ' mm^3')
print('Mean Residence Time: ', round(tau,1), ' s')
print('Mean Residence Time: ', round(tau/60,2), ' min')

# plot kiln geometry
standardPlot([0,x_max], [0,0], -x_max*0.01, x_max*1.01, -2*R*0.05, 2*R*1.05, style='-', color='k', figNr=1,
                xFormatter='%.2f', yFormatter='%.2f', xlabel='Kiln Length, m', ylabel='Filling Height, m', grid=True,
                figSize=(l_plot, h_plot))
standardPlot([0,x_max], [R,R], 0, x_max, -0.01, 2*R*0.1, style='-.', color='k', figNr=1)
standardPlot([0,x_max], [2*R,2*R], 0, x_max, -0.01, 2*R*0.1, style='-', color='k', figNr=1)
standardPlot([0,x_max], [H0,H0], 0, x_max, -0.01, 2*R*0.1, style='--', color='r', figNr=1)
standardPlot([0,0], [0,2*R], 0, x_max, -0.01, 2*R*0.1, style='-', color='k', figNr=1)
standardPlot([x_max,x_max], [0,2*R], 0, x_max, -0.01, 2*R*0.1, style='-', color='k', figNr=1)

# fill areas
plt.fill_between(x, h, step="pre", color='b', alpha=0.4)
plt.fill_between(x,h,2*R, step="pre", color='k', alpha=0.1)
plt.fill_between(x, H0, facecolor='none', hatch='/', edgecolor='g', linewidth=0.0)

# plot bed height
standardPlot(x, h, -x_max*0.01, x_max*1.01, -2*R*0.05, 2*R*1.05, xlabel='Kiln Length, m', ylabel='H, m',
                style='-', figNr=1, grid=True, color='b', figSize=(l_plot, h_plot),xFormatter='%.2f', yFormatter='%.2f')
plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.3)
plt.savefig(saveDirectory+'bedHeight.png')
plt.close()

#----------------------------------------------------------------------------------------------------------------------#
# define some E(t), only one should be uncommented                                                                     #
# use E(t) for ADM                                                                                                     #
#----------------------------------------------------------------------------------------------------------------------#
'''
# Perfectly Stirred Reactor (PSR)
def E(t):
    E = math.exp(-(t)/tau)/tau
    return E

# laminar pipe flow reactor (LPF)
def E(t):
    if t<tau/2:
        E = 10**(-20)
    else:
        E = tau**2/(2*t**3)
    return E
'''
def Feed_factor(t0, t1):
    Feed_factor = spi.quad(lambda t: E(t), t0, t1)[0]
    return Feed_factor

def t_factor(t0, t1):
    t_factor = (spi.quad(lambda t: E(t)*t, t0, t1)[0]/spi.quad(lambda t: E(t), t0, t1)[0])/tau
    return t_factor

def E(t):
    E = g(t)
    if E <= 0:
        E = 10 ** (-20)
    return E

#----------------------------------------------------------------------------------------------------------------------#
# reactions parameters                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#
# preexponential factors A in s**(-1); activation energies E in kJ/mol, reaction enthalpies h in J/kg
A_t = 1.08*10**10; E_t = 148; h_t = 80*10**3
A_g = 4.38*10**9; E_g = 152.7; h_g = 80*10**3
A_g2 = 4.28*10**6; E_g2 = 108; h_g2 = -42*10**3
A_is = 3.75*10**6; E_is = 111.7; h_is = 80*10**3
A_c = 1.38*10**10; E_c = 161.0; h_c = -300*10**3
A_c2 = 10**5; E_c2 = 108.0; h_c2 = -42*10**3
A_H2O = 5.13*10**10; E_H2O = 88.0

# Molar masses kg/mol
M_t = 0.078
M_it = 0.031
M_N2 = 0.028
M_g = 0.031
M_g2 = 0.031
M_is = 0.078
M_c = 0.012
M_c2 = 0.012
M_H2O = 0.018
gasConstant = 8.314462618  # J/mol K

Psi = 0.57+0.06*(n*60)   # tar factor omega in 1/s, 1/min or rad/s

# Arrhenius
def Ki(Ai, Ei,T):
    Ki = Ai*math.exp(-Ei*1000/(gasConstant*T))
    return Ki

# heat of vaporization water
def dh_W(T):
    dh_W = (-10328.34 + 134.92*T + (-0.51388155)*T**2 + 8.39*10**(-4)*T**3 + (-5.0848289*10**(-7))*T**4)*10**3
    return dh_W

#----------------------------------------------------------------------------------------------------------------------#
# heat capacity parameters                                                                                             #
#----------------------------------------------------------------------------------------------------------------------#

# spezific heat capacities, J/kg K
a_cp_w = [1500, 1, 0, 0]; b_cp_w = [1, 2, 0]
a_cp_is = [1500, 1, 0, 0]; b_cp_is = [1, 2, 0]
a_cp_c = [420, 2.09, 6.85*10**(-3), 0]; b_cp_c = [1, 2, 0]
a_cp_rt = [-100, 4.4, -1.57*10**(-3), 0]; b_cp_rt = [1, 2, 0]
a_cp_it = [-100, 4.4, -1.57*10**(-3), 0]; b_cp_it = [1, 2, 0]
a_cp_g = [770, 6.29*10**(-1), -1.91*10**(-4), 0]; b_cp_g = [1, 2, 0]
a_cp_N2 = [971.28, 1.494*10**(-1), 0, 0]; b_cp_N2 = [1, 2, 0]
a_cp_fg = [983.24, 2.605*10**(-1), 18.5*10**6, 0]; b_cp_fg = [1, -2, 0]
a_cp_H2O_l = [10083.46, -50.86889, 0.14419, -1.340619*10**(-4)]; b_cp_H2O_l = [1, 2, 3]
a_cp_H2O_g = [1608.267, 0.8371189, -1.3478*10**(-4), 7.1222*10**(-9)]; b_cp_H2O_g = [1, 2, 3]

# heat capacity f(T)
def cpi(a, b, T):
    cpi = a[0] + a[1]*T**b[0] + a[2]*T**b[1] + a[3]*T**b[2]
    return cpi

# thermal conductivity of the bed based on water content
def k_bed(rho_w,rho_H2O_l):
    if rho_H2O_l < 10 ** (-6):
        rho_H2O_l = 0
    k_bed = (0.11+0.5*rho_H2O_l/rho_w)
    return k_bed

#----------------------------------------------------------------------------------------------------------------------#
# defining equatino system. Solves over t (=x, kiln length) for initial conditions y                                   #
#----------------------------------------------------------------------------------------------------------------------#
def eq_system(t,y):
    # massflow solids
    F_w = []
    F_is = []
    F_c = []
    F_rt = []
    F_w_wet = []
    for i in range(n_solid):
        F_w.append(y[i])
        F_is.append(y[i+n_solid])
        F_c.append(y[i+n_solid*2])
        F_rt.append(y[i+n_solid*3])
        F_w_wet.append(y[i+n_solid*4])
    F_solid = [F_w, F_is, F_c, F_rt, F_w_wet]
    # massflow gas
    F_it = y[n_solid*5]
    F_g = y[n_solid*5+1]
    F_H2O_g = y[n_solid*5+2]
    # Temperature bed, gas, wall
    T_b = y[n_solid*5+3]
    T_g = y[n_solid*5+4]
    T_w = y[n_solid*5+5]
    # bed height
    H = y[n_solid*5+6]

    # kiln angle and angle of repose for h(x)
    phi = phi_grad * math.pi / 180
    beta = beta_grad * math.pi / 180
    a = 3 * np.tan(phi) * Q_fixed / (math.pi * 4 * n)

    # reactions
    # density of solid phases in kg/m**3
    rho_w = []
    rho_is = []
    rho_c = []
    rho_rt = []
    rho_w_wet = []
    rho_H2O_l = []  # density of the moisture fraction for reaction rate
    for i in range(n_solid):
        rho_w.append(F_w[i] / ((1 - epsilon_b) * Q_fixed/Q_w_factor[i]))
        rho_is.append(F_is[i] / ((1 - epsilon_b) * Q_fixed/Q_w_factor[i]))
        rho_c.append(F_c[i] / ((1 - epsilon_b) * Q_fixed/Q_w_factor[i]))
        rho_rt.append(F_rt[i] / ((1 - epsilon_b) * Q_fixed/Q_w_factor[i]))
        rho_w_wet.append(F_w_wet[i] / ((1 - epsilon_b) * Q_fixed/Q_w_factor[i]))
        rho_H2O_l.append(F_w_wet[i]*H2O / ((1 - epsilon_b) * Q_fixed/Q_w_factor[i]))
    rho_solid = [rho_w, rho_is, rho_c, rho_rt, rho_w_wet]

    cp_solid_w_wet = (1-H2O)*cpi(a_cp_w, b_cp_w, T_b) + H2O*cpi(a_cp_H2O_l, b_cp_H2O_l, T_b)
    cp_solid = [cpi(a_cp_w, b_cp_w, T_b), cpi(a_cp_is, b_cp_is, T_b), cpi(a_cp_c, b_cp_c, T_b),
                cpi(a_cp_rt, b_cp_rt, T_b), cp_solid_w_wet]

    # rate factors, modelled using Arrhenius law
    r_w = []
    r_is = []
    r_c = []
    r_rt = []
    r_H2O_l = []
    for i in range(n_solid):
        r_w.append((-Ki(A_g, E_g, T_b) - Ki(A_t, E_t, T_b) - Ki(A_is, E_is, T_b)) * rho_w[i] + \
                   Ki(A_H2O, E_H2O, T_b) * rho_H2O_l[i]*(1-H2O))
        r_is.append(Ki(A_is, E_is, T_b) * rho_w[i] - Ki(A_c, E_c, T_b) * rho_is[i])
        r_c.append(Ki(A_c, E_c, T_b) * rho_is[i] + Ki(A_c2, E_c2, T_b) * rho_rt[i])
        r_rt.append((1 - Psi) * Ki(A_t, E_t, T_b) * rho_w[i] - (Ki(A_g2, E_g2, T_b) + Ki(A_c2, E_c2, T_b)) * rho_rt[i])
        r_H2O_l.append(-Ki(A_H2O, E_H2O, T_b) * rho_H2O_l[i])
    r_it = Psi * Ki(A_t, E_t, T_b) * np.sum(rho_w)
    r_g = Ki(A_g, E_g, T_b) * np.sum(rho_w) + Ki(A_g2, E_g2, T_b) * np.sum(rho_rt)
    r_H2O_g = Ki(A_H2O, E_H2O, T_b) * np.sum(rho_H2O_l)*H2O

    # reaction energy
    G_r_w = 0
    G_r_is = 0
    G_r_rt = 0
    G_r_H2O_l = 0
    for i in range(n_solid):
        G_r_w = G_r_w -(Ki(A_g, E_g, T_b)*h_g + Ki(A_t, E_t, T_b)*h_t + Ki(A_is, E_is, T_b)*h_is) * rho_w[i]
        G_r_is = G_r_is - Ki(A_c, E_c, T_b)*h_c * rho_is[i]
        G_r_rt = G_r_rt - (Ki(A_g2, E_g2, T_b)*h_g2 + Ki(A_c2, E_c2, T_b)*h_c2) * rho_rt[i]
        G_r_H2O_l = G_r_H2O_l - Ki(A_H2O, E_H2O, T_b) * 2500*10**3 * rho_H2O_l[i]*H2O # *dh_W(T_b)

    # heat production due to reaction
    G = np.sum(G_r_w) + np.sum(G_r_is) + np.sum(G_r_rt) + np.sum(G_r_H2O_l)

    # total heat capacity flow gas
    FCp_g = F_N2 * cpi(a_cp_N2, b_cp_N2, T_g) + F_g * cpi(a_cp_g, b_cp_g, T_g) + F_it * cpi(a_cp_it, b_cp_it, T_g) + \
            F_H2O_g * cpi(a_cp_H2O_g, b_cp_H2O_g, T_g)

    # heat needed for heating up gas from bed to gas temperature
    Cp_dT_g, Cp_dT_g_error = spi.quad(lambda T: cpi(a_cp_g, b_cp_g, T), T_b, T_g)
    Cp_dT_it, Cp_dT_it_error = spi.quad(lambda T: cpi(a_cp_it, b_cp_it, T), T_b, T_g)
    Cp_dT_H2O_g, Cp_dT_H2O_g_error = spi.quad(lambda T: cpi(a_cp_H2O_g, b_cp_H2O_g, T), T_b, T_g)

    r_Cp_dT_gas = r_g * Cp_dT_g + r_it * Cp_dT_it + r_H2O_g * Cp_dT_H2O_g

    # heat capacity flow of solid, total heat capacity of solid
    FCp_b = 0
    rho_cp_b = 0
    for i in range(len(F_solid)):
        for j in range(n_solid):
            FCp_b = FCp_b + cp_solid[i] * F_solid[i][j]
            rho_cp_b = rho_cp_b + cp_solid[i] * rho_solid[i][j]
    if Feed_inert != 0:
        FCp_b = FCp_b + cpi(a_cp_inert, b_cp_inert, T_b) * Feed_inert
        rho_cp_b = rho_cp_b + cpi(a_cp_inert, b_cp_inert, T_b) * rho_p_inert

    # heat transfer
    # bed wall resistence
    k_b = 0
    for i in range(n_solid):
         k_b = k_b + k_bed(rho_w_wet[i]*(1-H2O), rho_w_wet[i]*H2O)/n_solid

    ksi = 0.012
    phi_small = math.acos((R - H) / R)
    h_cw_cb = 1/((ksi*d_p)/k_g + 1/2*(2*k_b*rho_cp_b*2*math.pi*n/phi_small)**(-1/2))  # k = thermal conductivity

    # convective heat transfer
    d_e = R * (2 * math.pi - phi_small + math.sin(phi_small)) / (
    math.pi - phi_small / 2 + math.sin(phi_small / 2))  # hydrodynamic diameter

    S_bed = R ** 2 * math.acos((R - H) / R) - (R - H) * math.sqrt(2 * R * H - H ** 2)  # bed crossection area
    S_gas = R ** 2 * math.pi - S_bed  # gas crossection area

    u_g = (F_N2 / M_N2 + F_g / M_g + F_it / M_it + F_H2O_g / M_H2O) / S_gas * (gasConstant * T_g)/p  # gas velocity
    Re_g = u_g * d_e / nu  # axial gas Reynolds number
    Re_omega = n * d_e ** 2 / nu  # rotational gas Reynolds number

    eta = (phi_small - math.sin(phi_small) / (2 * math.pi))  # relative filling level
    # convective heat transfer coefficient exposed wall-gas:
    h_ew_g = 1.54 * Re_g ** 0.575 * Re_omega ** (-0.292) * k_g / d_e
    # convective heat transfer coefficient exposed bed-gas:
    h_g_eb = 0.46 * Re_g ** 0.535 * Re_omega ** 0.104 * eta ** (-0.341) * k_g / d_e

    Lambda_eb = 2 * math.sqrt(H * (2 * R - H))  # surface area per kiln length exposed bed
    Lambda_cw = 2 * R * math.acos((R - H) / R)  # surface area per kiln length covered wall
    Lambda_ew = 2 * math.pi * R - Lambda_cw  # surface area per kiln length exposed wall

    # radiative heat transfer
    sigma = 5.67 * 10 ** (-8)  # Stefan-Boltzmann Konstante W/(m**2K**4)
    f_ew_eb = Lambda_eb / Lambda_ew
    f_ew_g = 1
    f_g_eb = 1

    Em_g = sigma * T_g ** 4
    Em_ew = sigma * T_w ** 4
    Em_eb = sigma * T_b ** 4

    # solve linear system for J_ew and J_eb
    const1 = (1 - emm_ew) / (emm_ew * Lambda_ew)
    const2 = (1 - emm_eb) / (emm_eb * Lambda_eb)

    b1 = -(Em_ew / const1 + Em_g * Lambda_ew)
    b2 = -(Em_eb / const2 + Em_g * Lambda_eb)

    A11 = -(1 / const1 + Lambda_eb + Lambda_ew)
    A12 = Lambda_eb
    A21 = Lambda_eb
    A22 = -(1 / const2 + Lambda_eb + Lambda_eb)

    A = np.array([[A11, A12], [A21, A22]])
    b = np.array([b1, b2])
    J = np.linalg.solve(A, b)
    J_ew = J[0]
    J_eb = J[1]

    # Only use radiation if dT > 0.1K
    if abs(T_w - T_b) > 0.1:
        h_ew_eb_r = f_ew_eb * (J_ew - J_eb) / (T_w - T_b)
    else:
        h_ew_eb_r = 0

    if abs(T_w - T_g) > 0.1:
        h_ew_g_r = f_ew_g * (J_ew - Em_g) / (T_w - T_g)
    else:
        h_ew_g_r = 0

    if abs(T_g - T_b) > 0.1:
        h_g_eb_r = f_g_eb * (Em_g - J_eb) / (T_g - T_b)
    else:
        h_g_eb_r = 0

    # calculate heat fluxes
    q_wall_bed = (h_cw_cb * Lambda_cw + h_ew_eb_r * Lambda_ew) * (T_w - T_b)
    q_wall_gas = (h_ew_g + h_ew_g_r) * Lambda_ew * (T_w - T_g)
    q_gas_bed = (h_g_eb + h_g_eb_r) * Lambda_eb * (T_g - T_b)

    q_bed = q_wall_bed + q_gas_bed
    q_gas = q_wall_gas - q_gas_bed
    q_killn = q_wall_bed + q_wall_gas
    q_loss = math.pi * 2 * R * h_ins * (T_w - T0)  # h_ins = heat transfer insulation, T0 = surroundings Temp. 298K

    # mass balance equations, energy balance equations and Seaman's model for the bed height
    f = []
    for i in range(n_solid):
        f.append((1 - epsilon_b) * S_bed * r_w[i])  # mass balance for dry wood
    for i in range(n_solid):
        f.append((1 - epsilon_b) * S_bed * r_is[i])  # mass balance for intermediate solid
    for i in range(n_solid):
        f.append((1 - epsilon_b) * S_bed * r_c[i])  # mass balance for char
    for i in range(n_solid):
        f.append((1 - epsilon_b) * S_bed * r_rt[i])  # mass balance for reactive tar
    for i in range(n_solid):
        f.append((1 - epsilon_b) * S_bed * r_H2O_l[i])  # mass balance for wet wood
    f.append((1 - epsilon_b) * S_bed * r_it)  # mass balance for inert tar (gaseous)
    f.append((1 - epsilon_b) * S_bed * r_g)  # mass balance for pyrolysis gas
    f.append((1 - epsilon_b) * S_bed * r_H2O_g)  # mass balance for H2O gas
    f.append(((1 - epsilon_b) * S_bed * G + q_bed) / FCp_b)  # heat balance for solid
    f.append(((-1) * (1 - epsilon_b) * S_bed * r_Cp_dT_gas + q_gas) / FCp_g)  # heat balance for gas
    f.append((-q_killn - q_loss) / (cpi(a_cp_fg, b_cp_fg, T_fg_in) * F_fg))  # heat balance for wall
    f.append((-a * (R ** 2 - (H - R) ** 2) ** (-3 / 2) + np.tan(beta) / np.cos(phi)))  # bed height
    f.append(q_bed) # for plotting only
    f.append(q_gas) # for plotting only
    f.append(q_loss) # for plotting only
    return f

print('\n')

ode = spi.ode(eq_system) # ínitialize equation system


#----------------------------------------------------------------------------------------------------------------------#
# solve equation system for different Peclet numbers                                                                   #
#----------------------------------------------------------------------------------------------------------------------#
for Pe_nr in range(len(Pe_list)):
    #------------------------------------------------------------------------------------------------------------------#
    # Axial Dispersion Model (ADM)
    Pe = Pe_list[Pe_nr]
    nxADM = nxADM_list[Pe_nr]
    nt = nt_list[Pe_nr]

    # stability criterion for ADM
    s = 0.45         # s <= 0.5
    ntADM_s = int(6*nxADM**2/(Pe*s))
    ntADM_c = int(6*nxADM/np.sqrt(2*s))
    C = 3 * nxADM / max(ntADM_s,ntADM_c)

    #print('2s =', 2*s, ', number of time steps =', max(ntADM_s,ntADM_c), ', C**2 =', round(C**2, 5))

    tADM, ca = dispersionModel(nxADM, 6*tau, max(ntADM_s,ntADM_c), Pe, tau)

    integral = spi.cumtrapz(ca[:,0],tADM[:,0],initial=0)[-1]
    E_ADM = ca / integral
    g = interp1d(tADM[:,0], E_ADM[:,0],fill_value="extrapolate")

    #------------------------------------------------------------------------------------------------------------------#
    # initial conditions
    # solids
    # define feed according to RTD E(t) and number of solid fractions
    F_w_wet = []
    Q_w_factor = []
    F_w = []
    F_is = []
    F_c = []
    F_rt = []
    if nt <= 0:
        print('ERROR: please enter nt > 0')
    elif nt == 1:
        print('Using 1 solid species with mean residence time according to Seamans`s Model')
        F_w_wet = [Feed]  # wood feed flow in kg/s
        Q_w_factor = [1]
        F_w = [0]
        F_is = [0]
        F_c = [0]
        F_rt = [0]
        FeedSum =1
    else:
        i = 0
        FeedSum = 0
        t_min = -1
        dt = t_max / (nt - 2)
        print('Using ', nt, ' solid species with residence time distribution according to E(t).')
        while i < 10**6:
            if_argument = Feed_factor(i*dt, (i+1)*dt)
            if if_argument > Feed_limit/100 and t_min == -1:
                t_min =i*dt
                print('Found t_min at',t_min,'s.')
            if if_argument < Feed_limit/100 and t_min > -1:
                t_max = i*dt
                print('Found t_max at', t_max, 's.')
                i = 10 ** 7
            i = i + 1

        i = 0
        dt = (t_max-t_min) / (nt - 1)
        while i < nt :
            if Feed_factor(t_min + i * dt, t_min+(i + 1) * dt) > Feed_limit / 100:
                FeedSum = FeedSum + Feed_factor(t_min + i * dt, t_min+(i + 1) * dt)
                print('Residence Time ', i+1, ':',
                      round(spi.quad(lambda t: E(t)*t, 0, math.inf)[0]*t_factor(t_min+i*dt, t_min+(i+1)*dt),2), 's',
                      'for',round(Feed_factor(t_min+i*dt, t_min+(i+1)*dt)*100,2),'% of the feed.')
                F_w_wet.append(Feed*Feed_factor(t_min+i*dt, t_min+(i+1)*dt))
                Q_w_factor.append(t_factor(t_min+i*dt, t_min+(i+1)*dt))
                F_w.append(0)
                F_is.append(0)
                F_c.append(0)
                F_rt.append(0)
            else:
                print('Residence Time ', i + 1, ':', round(Feed_factor(t_min+i * dt, t_min+(i + 1) * dt) * 100, 10),
                      '% of the feed. Thus not included in simulation!')
            i = i + 1


        FeedSum = FeedSum + Feed_factor(t_max, math.inf)
        print('Residence Time ', i+1, ':',
              round(spi.quad(lambda t: E(t)*t, 0, math.inf)[0]*t_factor(t_max, math.inf), 2), 's',
              'for', round(Feed_factor(t_max, math.inf) * 100, 4), '% of the feed.')
        F_w_wet.append(Feed*Feed_factor(t_max, math.inf))
        Q_w_factor.append(t_factor(t_max, math.inf))
        F_w.append(0)
        F_is.append(0)
        F_c.append(0)
        F_rt.append(0)

    print(round(FeedSum*100,2), ' % of Feed was inserted!')

    # gas composition
    F_it = 0
    F_g = 0
    F_H2O_g = 0
    # temperatures
    T_b = T0
    T_g = T0
    T_w = T_fg_in
    # bed height at x = 0 m
    H0 = h[0]

    n_solid = len(F_w_wet)

    # initial condition vector, last 3 values are dummy-values for  heat fluxes
    y0 = []
    for i in range(n_solid):
        y0.append(F_w[i])
    for i in range(n_solid):
        y0.append(F_is[i])
    for i in range(n_solid):
        y0.append(F_c[i])
    for i in range(n_solid):
        y0.append(F_rt[i])
    for i in range(n_solid):
        y0.append(F_w_wet[i])
    y0.append(F_it)
    y0.append(F_g)
    y0.append(F_H2O_g)
    y0.append(T_b)
    y0.append(T_g)
    y0.append(T_w)
    y0.append(h[0])
    y0.append(0)
    y0.append(0)
    y0.append(0)

    x = []          # grid

    # -----------------------------------------------------------------------------------------------------------------#
    # initializing result vectors
    # solids
    F_w = []          # dry wood flow
    F_is = []         # inermediate solid flow
    F_c = []          # char flow
    F_rt = []         # reactive tar flow
    F_w_wet = []         # wet wood flow

    # gas, temperatures, bed hight, heat fluxes
    F_it = []         # inert tar flow
    F_g = []          # pyrolysis gas flow
    F_H2O_g = []      # pyrolysis gas flow
    T_b = []          # temperature bed
    T_g = []          # temperature gas
    T_w = []          # temperature wall
    h_new = []        # bed height
    q_bed = []    # bed heat flux
    q_gas = []    # gas heat flux
    q_loss = []   # loss heat flux
    tau_l_list = [] # mean residence time

    print('\n')
    # -----------------------------------------------------------------------------------------------------------------#
    # define solver settings. BDF method suited to stiff systems of ODEs
    ode.set_integrator('vode',nsteps=1000,method='bdf')
    ode.set_initial_value(y0,0)

    s_bed_l_old = 0 # initialize bed cross section at l[i-1]
    s_bed_l = 0 # initialize bed cross section at l[i]
    volume_bed_l = 0 # initialize bed volume
    perc_old = 0

    # -----------------------------------------------------------------------------------------------------------------#
    # solve equation system, store results
    while ode.successful() and ode.t < x_max:
        perc_new = round(ode.t / x_max * 100, 2)
        if perc_new % 1 == 0 and perc_old != perc_new:
            perc_old = round(ode.t / x_max * 100, 2)
            print('Kiln model, Pe =', Pe, ',', perc_old, '%')

        x.append(ode.t) # save grid points

        F_w_n = []  # dry wood flow, containing n_solid entries for single time step
        F_is_n = []  # inermediate solid flow, containing n_solid entries for single time step
        F_c_n = []  # char flow, containing n_solid entries for single time step
        F_rt_n = []  # reactive tar flow, containing n_solid entries for single time step
        F_w_wet_n = []  # reactive tar flow, containing n_solid entries for single time step
        for i in range(n_solid):
            F_w_n.append(ode.y[i])
            F_is_n.append(ode.y[i+n_solid])
            F_c_n.append(ode.y[i+n_solid*2])
            F_rt_n.append(ode.y[i+n_solid*3])
            F_w_wet_n.append(ode.y[i+n_solid*4])
        F_w.append(F_w_n)
        F_is.append(F_is_n)
        F_c.append(F_c_n)
        F_rt.append(F_rt_n)
        F_w_wet.append(F_w_wet_n)
        F_it.append(ode.y[n_solid*5])
        F_g.append(ode.y[n_solid*5+1])
        F_H2O_g.append(ode.y[n_solid*5+2])
        T_b.append(ode.y[n_solid*5+3])
        T_g.append(ode.y[n_solid*5+4])
        T_w.append(ode.y[n_solid*5+5])
        h_new.append(ode.y[n_solid*5+6])
        q_bed.append(ode.y[n_solid*5+7])
        q_gas.append(ode.y[n_solid*5+8])
        q_loss.append(ode.y[n_solid*5+9])

        s_bed_l_old = s_bed_l
        s_bed_l = R**2*math.acos((R - h_new[-1])/R) - (R - h_new[-1])*math.sqrt(2*R*h_new[-1] - h_new[-1]**2) # bed cross section
        volume_bed_l = volume_bed_l + x_max/(nx*2)*(s_bed_l + s_bed_l_old) # bed volume
        tau_l = volume_bed_l / Q_fixed # mean residnece time
        tau_l_list.append(tau_l)

        ode.integrate(ode.t + x_max/nx)     # integrate over kiln length

    #------------------------------------------------------------------------------------------------------------------#
    # save results in log files and create plots                                                                       #
    #------------------------------------------------------------------------------------------------------------------#
    # initialize log file, log text
    logFile = open(saveDirectory+'logFiles/'+logFileName+'Pe'+str(Pe), 'w+')
    writeText = 'l,m\tTau,s\tF_w,kg/s\tF_is,kg/s\tF_c,kg/s\tF_rt,kg/s\tF_w_wet,kg/s\tF_it,kg/s\tF_g,kg/s\t'+\
                'F_H2O_g,kg/s\tT_b,K\tT_g,K\tT_w,K\tq_bed,kW/m\tq_gas,kW/m\tq_loss,kW/m\n'

    # calulate realative mass flows --> conversions
    F_w_rel=[]
    F_is_rel=[]
    F_c_rel=[]
    F_rt_rel=[]
    F_w_wet_rel=[]
    F_it_rel=[]
    F_g_rel=[]
    F_H2O_g_rel=[]

    F_w_rel2=[]
    F_is_rel2=[]
    F_c_rel2=[]
    F_rt_rel2=[]
    F_w_wet_rel2=[]
    F_it_rel2=[]
    F_g_rel2=[]
    F_H2O_g_rel2=[]

    F_w_abs=[]
    F_is_abs=[]
    F_c_abs=[]
    F_rt_abs=[]
    F_w_wet_abs=[]
    F_it_abs=[]
    F_g_abs=[]
    F_H2O_g_abs=[]

    F_w_abs2=[]
    F_is_abs2=[]
    F_c_abs2=[]
    F_rt_abs2=[]
    F_w_wet_abs2=[]
    F_it_abs2=[]
    F_g_abs2=[]
    F_H2O_g_abs2=[]

    mass_list = []

    for i in range(len(F_w)):
        mass = np.sum(F_w[i])+np.sum(F_is[i])+np.sum(F_c[i])+np.sum(F_rt[i])+np.sum(F_w_wet[i])+\
               np.sum(F_it[i])+np.sum(F_g[i])+np.sum(F_H2O_g[i])
        mass_list.append(mass)
        mass_rel = mass/Feed*100
        F_w_rel.append(np.sum(F_w[i])/mass*100)
        F_is_rel.append(np.sum(F_is[i])/mass*100)
        F_c_rel.append(np.sum(F_c[i])/mass*100)
        F_rt_rel.append(np.sum(F_rt[i])/mass*100)
        F_w_wet_rel.append(np.sum(F_w_wet[i])/mass*100)
        F_it_rel.append(np.sum(F_it[i])/mass*100)
        F_g_rel.append(np.sum(F_g[i])/mass*100)
        F_H2O_g_rel.append(np.sum(F_H2O_g[i])/mass*100)

        F_w_wet_rel2.append(F_w_wet_rel[i])
        F_w_rel2.append(F_w_wet_rel2[i]+F_w_rel[i])
        F_c_rel2.append(F_w_rel2[i]+F_c_rel[i])
        F_is_rel2.append(F_c_rel2[i]+F_is_rel[i])
        F_rt_rel2.append(F_is_rel2[i]+F_rt_rel[i])
        F_it_rel2.append(F_rt_rel2[i]+F_it_rel[i])
        F_g_rel2.append(F_it_rel2[i]+F_g_rel[i])
        F_H2O_g_rel2.append(F_g_rel2[i]+F_H2O_g_rel[i])

        F_w_abs.append(np.sum(F_w[i])*3600)
        F_is_abs.append(np.sum(F_is[i])*3600)
        F_c_abs.append(np.sum(F_c[i])*3600)
        F_rt_abs.append(np.sum(F_rt[i])*3600)
        F_w_wet_abs.append(np.sum(F_w_wet[i])*3600)
        F_it_abs.append(np.sum(F_it[i])*3600)
        F_g_abs.append(np.sum(F_g[i])*3600)
        F_H2O_g_abs.append(np.sum(F_H2O_g[i])*3600)

        F_w_wet_abs2.append(F_w_wet_abs[i])
        F_w_abs2.append(F_w_wet_abs2[i] + F_w_abs[i])
        F_c_abs2.append(F_w_abs2[i]+F_c_abs[i])
        F_is_abs2.append(F_c_abs2[i]+F_is_abs[i])
        F_rt_abs2.append(F_is_abs2[i]+F_rt_abs[i])
        F_it_abs2.append(F_is_abs2[i]+F_it_abs[i])
        F_g_abs2.append(F_it_abs2[i]+F_g_abs[i])
        F_H2O_g_abs2.append(F_g_abs2[i]+F_H2O_g_abs[i])

        if i < len(q_bed)-1:
            q_bed[i] = (q_bed[i+1]-q_bed[i])/(x[i+1]-x[i])/1000
            q_gas[i] = (q_gas[i+1]-q_gas[i])/(x[i+1]-x[i])/1000
            q_loss[i] = -(q_loss[i+1]-q_loss[i])/(x[i+1]-x[i])/1000

        for j in range(len(x_save)):
            #print(round(x[i], 3), round(x_save[j], 3))
            if round(x[i],3) == round(x_save[j],5):
                writeText = writeText+str(round(x[i],5))+'\t'+str(round(tau_l_list[i],5))+'\t'\
                            +str(round(np.sum(F_w[i]),8))+'\t'+\
                            str(round(np.sum(F_is[i]),8))+'\t'+str(round(np.sum(F_c[i]),8))+'\t'+\
                            str(round(np.sum(F_rt[i]),8))+'\t'+str(round(np.sum(F_w_wet[i]),8))+'\t'+\
                            str(round(np.sum(F_it[i]),8))+'\t'+str(round(np.sum(F_g[i]),8))+'\t'+\
                            str(round(np.sum(F_H2O_g[i]),8))+'\t'+str(round(T_b[i],8))+'\t'+str(round(T_g[i],8))+'\t'+\
                            str(round(T_w[i],8))+'\t'+str(round(q_bed[i],8))+'\t'+str(round(q_gas[i],8))+'\t'+\
                            str(round(q_loss[i],8))+'\n'

    logFile.write(writeText)
    logFile.close()
    q_bed[-1] = q_bed[-2]
    q_gas[-1] = q_gas[-2]
    q_loss[-1] = q_loss[-2]

    print('Mass Balance: delta m =', (mass_list[0]-mass_list[-1])/Feed*100, '%')

    # plotting
    xlabel = 'Kiln Length, m'
    # plotting temperatures
    T_min = 0
    T_max = 600
    standardPlot(x[::subSample], np.asarray(T_b[::subSample])-273.15, 0, x_max, T_min, T_max, xlabel = xlabel,
                 ylabel='Temperature, °C', style='-', figNr=2, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], np.asarray(T_g[::subSample])-273.15, 0, x_max, T_min, T_max, xlabel = xlabel,
                 ylabel='Temperature, °C', style='-', figNr=2, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], np.asarray(T_w[::subSample])-273.15, 0, x_max, T_min, T_max, xlabel = xlabel,
                 ylabel='Temperature, °C', style='-', figNr=2, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    legend = []
    legend.append('T_bed')
    legend.append('T_gas')
    legend.append('T_wall')
    plt.legend(legend, loc='upper right')
    plt.savefig(saveDirectory +'temperature_Pe'+str(Pe)+'.png')
    plt.close()

    # plot relative mass flows
    flow_min_perc = 0
    flow_max_perc = 105
    standardPlot(x[::subSample], F_w_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_c_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_is_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_rt_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_w_wet_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_it_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_g_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_H2O_g_rel[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=3, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    legend = []
    legend.append('Wood')
    legend.append('Char')
    legend.append('Intermed. solid')
    legend.append('React. tar')
    legend.append('Wet Wood')
    legend.append('Inert tar')
    legend.append('Gas')
    legend.append('H2O(g)')
    plt.legend(legend, loc='upper right')
    plt.savefig(saveDirectory + 'relMass_Pe' + str(Pe) + '.png')
    plt.close()

    # plot relative mass flows
    flow_min_perc2 = 0
    flow_max_perc2 = 100
    standardPlot(x[::subSample], F_w_wet_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_w_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_c_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_is_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_rt_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_it_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_g_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %',  style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')
    standardPlot(x[::subSample], F_H2O_g_rel2[::subSample], 0, x_max, flow_min_perc, flow_max_perc, xlabel = xlabel,
                 ylabel='Mass Flow, %', style='-', figNr=4, grid=True,xFormatter='%.1f', yFormatter='%.0f')

    # Fill areas
    plt.fill_between(x,F_w_wet_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_w_wet_rel2,F_w_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_w_rel2,F_c_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_c_rel2,F_is_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_is_rel2,F_rt_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_rt_rel2,F_it_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_it_rel2,F_g_rel2, step="pre", alpha=0.4)
    plt.fill_between(x,F_g_rel2,F_H2O_g_rel2, step="pre", alpha=0.4)


    legend = []
    legend.append('Wet Wood')
    legend.append('Wood')
    legend.append('Char')
    legend.append('Intermed. solid')
    legend.append('React. tar')
    legend.append('Inert tar')
    legend.append('Gas')
    legend.append('H2O(g)')
    plt.legend(legend,loc='upper right',ncol=2,frameon=False,bbox_to_anchor=(0.95, 0.95))
    plt.savefig(saveDirectory + 'relMassStacked_Pe' + str(Pe) + '.png')
    plt.close()

    # plot absolute mass flows
    flow_min_abs = 0
    flow_max_abs = Feed * 3600 * 1.1
    standardPlot(x, F_w_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_c_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_is_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_rt_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_w_wet_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_it_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_g_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    standardPlot(x, F_H2O_g_abs, 0, x_max, flow_min_abs, flow_max_abs, xlabel = xlabel, ylabel='Mass Flow, kg/h',
                    style='-', figNr=6, grid=True,xFormatter='%.1f', yFormatter='%.f')
    legend = []
    legend.append('Wood')
    legend.append('Char')
    legend.append('Intermed. solid')
    legend.append('React. tar')
    legend.append('Wet Wood')
    legend.append('Inert tar')
    legend.append('Gas')
    legend.append('H2O(g)')
    plt.legend(legend, loc='upper right')
    plt.savefig(saveDirectory + 'absMass_Pe' + str(Pe) + '.png')
    plt.close()

    # plot heat flux
    heatFlux_min = -5
    heatFLux_max = 35
    standardPlot(x[::subSample], np.asarray(q_bed[::subSample]), 0, x_max, heatFlux_min, heatFLux_max, xlabel = xlabel,
                    ylabel='Heat Flow, kW', style='-', figNr=5, grid=True,xFormatter='%.1f', yFormatter='%.0f',
                    color='darkorange')
    standardPlot(x[::subSample], np.asarray(q_bed[::subSample])+ np.asarray(q_gas[::subSample]), 0, x_max, heatFlux_min,
                    heatFLux_max, xlabel = xlabel, ylabel='Heat Flow, kW', style='-', figNr=5, grid=True,
                    xFormatter='%.1f', yFormatter='%.0f', color='k')
    standardPlot(x[::subSample], np.asarray(q_loss[::subSample]), 0, x_max, heatFlux_min, heatFLux_max, xlabel = xlabel,
                    ylabel='Heat Flow, kW', style='-', figNr=5, grid=True,xFormatter='%.1f', yFormatter='%.0f',
                    color='r')

    # Fill areas
    plt.fill_between(x,q_bed, step="pre", color='darkorange', alpha=0.4)
    plt.fill_between(x,q_bed,np.asarray(q_bed) + np.asarray(q_gas), step="pre", color='k', alpha=0.4)
    plt.fill_between(x,q_loss, facecolor='none', hatch='\ ',edgecolor='r', linewidth=0.0)

    legend = []
    legend.append('$q_{bed}$')
    legend.append('$q_{gas}$')
    legend.append('$q_{loss}$')
    plt.legend(legend, loc='upper right')
    plt.savefig(saveDirectory + 'heatFlux_Pe' + str(Pe) + '.png')
    plt.close()

print('Finished for ', len(Pe_list), 'cases!')