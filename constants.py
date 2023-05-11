from math import pi


# for alpha
k = 176.1        # permeability coef
drho = 814.4     # oil-gas pressure difference
g = 9.80665      # acceleration of gravity
mu = 18.76       # dynamic viscosity
phi = 0.266      # effective porosity

alpha = k*drho*g/(mu*phi)/10000
h_0 = 1/2*7**2         # initial height
W = 150         # half of the drain length
L = 1171.3      # well length
dw = 0.089      # well diameter
zw1 = h_0*0.8
zw2 = h_0*0.75
cube_to_kg = 800  # coefficient to translate m^3 to kg
beta = 0.191794288875527




