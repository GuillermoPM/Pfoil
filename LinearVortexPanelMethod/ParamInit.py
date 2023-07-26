"""
	Thermodynamics and other parameters initialization at the required conditions. This are both for the viscous
	and inviscid case, so they'll be initialized in the first place and just modified if needed.

    @Author: Guillermo Peña Martínez
    @Date: 04/05/2023
"""
import numpy as np

def build_term(Foil):
	"""
		Thermodynamics initialization
	
	"""


	g = Foil.param.gam
	gmi = g - 1
	Foil.param.Vinf = Foil.oper.Vinf  # freestream speed
	Foil.param.muinf = Foil.oper.rho * Foil.oper.Vinf / Foil.oper.Re  # freestream dyn viscosity
	Minf = Foil.oper.Ma
	Foil.param.Minf = Minf  # freestream Mach
	if Minf > 0:
		Foil.param.KTb = np.sqrt(1 - Minf ** 2)  # Karman-Tsien beta
		Foil.param.KTl = Minf ** 2 / (1 + Foil.param.KTb) ** 2  # Karman-Tsien lambda
		Foil.param.H0 = (1 + 0.5 * gmi * Minf ** 2) * Foil.oper.Vinf ** 2 / \
			(gmi * Minf ** 2)  # stagnation enthalpy
		# freestream/stagnation temperature ratio
		Tr = 1 - 0.5 * Foil.oper.Vinf ** 2 / Foil.param.H0
		finf = Tr ** 1.5 * (1 + Foil.param.Tsrat) / \
			(Tr + Foil.param.Tsrat)  # Sutherland's ratio
		Foil.param.cps = 2 / (g * Minf ** 2) * (((1 + 0.5 *
                                         gmi * Minf ** 2) / (1 + 0.5 * gmi)) ** (g / gmi) - 1)
	else:
		finf = 1  # incompressible case
	# stag visc (Sutherland ref temp is stag)
	Foil.param.mu0 = Foil.param.muinf / finf
	Foil.param.rho0 = Foil.oper.rho * (1 + 0.5 * gmi * Minf ** 2) ** (1 / gmi)  # stag density


def station_param(M, param, i):
	"""
	Modifies parameter structure to be specific for a given station
	
	Args:
	- M: instance of a class containing data on the mesh and its properties
	- param: parameter structure to modify
	- i: station number (node index along the surface)
	
	Returns:
	- param: modified parameter structure
	"""
	param.turb = M.vsol.turb[i]  # turbulent
	param.simi = i in M.isol.Istag  # similarity
	return param


def build_param(M, is_side):
	"""
	Builds a parameter structure for a given surface side
	
	Args:
	- M: instance of a class containing data on the mesh and its properties
	- is_side: side number (1 = lower, 2 = upper, 3 = wake)
	
	Returns:
	- param: modified M.param structure with side information
	"""
	param = M.param
	param.wake = (is_side == 2)
	param.turb = param.wake  # the wake is fully turbulent
	param.simi = False  # true for similarity station
	return param
