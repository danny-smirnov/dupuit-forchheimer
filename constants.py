from math import pi


# for alpha
# k = 176.1        # permeability coef
# drho = 814.4     # oil-gas pressure difference
# g = 9.80665      # acceleration of gravity
# mu = 18.76       # dynamic viscosity
# phi = 0.266      # effective porosity
#
# alpha = k*drho*g/(mu*phi)/1000
# # alpha = 0.1
# h_0 = 1/2*7**2         # initial height
# W = 150         # half of the drain length
# L = 1171.3      # well length
# dw = 0.089      # well diameter
# zw1 = 5.6
# zw2 = zw1-dw
# cube_to_kg = 800 # coefficient to translate m^3 to kg
# p1 = 30         # pressure at t1 point
# p2 = 10         # pressure at t3 point
#
# p0 = drho*g*h_0/10**5                    #bar
# q0 = 2*k*p0*L*h_0/mu/W/115.74
# t0 = 115.74*W**2*mu*phi/k/p0           #day
#
#
# v_well = pi*dw**2*L     # m^3
p1 = 30
p2 = 10
W = 150
cube_to_kg = 800
q0 = 60
h_0 = 7
beta = 0.191794288875527
phi = 0.27
zw1 = 6.5
dw = 0.089
zw2 = zw1-dw
k = 176
drho = 815
mu = 18.76
n_source_point = 10
Bg = 0.00935
Bo = 1.069
gamma = 100
L = 200
alpha = k*drho*9.81/(mu*phi)/1000