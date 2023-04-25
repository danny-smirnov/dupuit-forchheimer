from math import pi


# for alpha
k = 176.1       # permeability coef
drho = 814.4    # oil-gas pressure difference
g = 9.80665     # acceleration of gravity
mu = 18.76      # dynamic viscosity
phi = 0.266     # effective porosity

alpha = k*drho*g/(mu*phi)/100
# alpha = 0.1
h_0 = 7         # initial height
W = 150         # half of the drain length
L = 1171.3      # well length
dw = 0.089      # well diameter
zw1 = 0.8
zw2 = 0.75

p0 = drho*g*h_0/10**5                    #bar
q0 = 2*k*p0*L*h_0/mu/W/115.74
t0 = 115.74*W**2*mu*phi/k/p0           #day


v_well = pi*dw**2*L     # m^3
