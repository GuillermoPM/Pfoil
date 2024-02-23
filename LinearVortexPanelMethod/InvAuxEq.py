import numpy as np


def get_cp(u, param):
	"""
		Gives Cp using Karman - Tsien correction in case of compressible flow.\n

		INPUT
			u : incompresible velocity at the node.
			param : foil parameters subclass
    
    """

	Vinf = param.Vinf
	cp = 1 - (u/Vinf)**2
	cp_u = -2*u/Vinf**2
	if (param.Minf > 0):
		l = param.KTl
		b = param.KTb
		den = b + 0.5*l*(1+b)*cp
		den_cp = 0.5*l*(1+b)
		cp = cp/den
		cp_u = cp_u * (1 - cp*den_cp)/den

	return cp, cp_u


def get_uk(u, param):
	"""
		Gives corrected velocity in the compressible case and the linealization.

		INPUT
			u : incompresible velocity at the node.
			param : foil parameters subclass   
    """

	if (param.Minf > 0):
		l = param.KTl
		Vinf = param.Vinf
		den = 1 - l*(u/Vinf)**2
		den_u = -2*l*u/Vinf**2
		uk = u*(1 - l)/den
		uk_u = (1 - l)/den - (uk/den)*den_u
	else:
		uk = u
		uk_u = 1

	return uk, uk_u


def get_cf(U, param):
	"""
		Calculates skin friction coefficient from state vector.

		INPUT
			U : state vector
			param : foil parameters
		
		OUTPUT
			cf, cf_U : skin friction coefficient and linearization

	
	"""

	if param.wake:
		cf = 0
		cf_U = np.zeros(4)
		return cf, cf_U  # zero cf in wake

	# get Hk and its linearization
	Hk, Hk_U = get_Hk(U, param)

	# get Ret and its linearization
	Ret, Ret_U = get_Ret(U, param)

	# TODO: limit Hk

	if param.turb:  # turbulent cf
		# get M2 and its linearization
		M2, M2_U = get_Mach2(U, param)  # squared edge Mach number

		Fc = np.sqrt(1 + 0.5 * (param.gam - 1) * M2)
		Fc_U = 0.5 / Fc * 0.5 * (param.gam - 1) * M2_U

		aa = -1.33 * Hk
		aa_U = - 1.33 * Hk_U

		# smooth limiting of aa
		if aa < -17:
			aa = -20 + 3 * np.exp((aa + 17) / 3)
			aa_U = (aa + 20) / 3 * aa_U

		bb = np.log(Ret / Fc)
		bb_U = Ret_U / Ret - Fc_U / Fc

		if bb < 3:
			bb = 3
			bb_U = bb_U * 0

		bb = bb / np.log(10)
		bb_U = bb_U / np.log(10)

		cc = -1.74 - 0.31 * Hk
		cc_U = -0.31 * Hk_U

		dd = np.tanh(4.0 - Hk / 0.875)
		dd_U = (1 - dd ** 2) * (-Hk_U / 0.875)

		cf0 = 0.3 * np.exp(aa) * bb ** cc
		cf0_U = cf0 * aa_U + 0.3 * \
			np.exp(aa) * cc * bb ** (cc - 1) * bb_U + cf0 * np.log(bb) * cc_U

		cf = (cf0 + 1.1e-4 * (dd - 1)) / Fc
		cf_U = (cf0_U + 1.1e-4 * dd_U) / Fc - cf / Fc * Fc_U
	else:  # laminar cf
		if Hk < 5.5:
			num = 0.0727 * (5.5 - Hk) ** 3 / (Hk + 1) - 0.07
			num_Hk = 0.0727 * (3 * (5.5 - Hk) ** 2 / (Hk + 1) *
                            (-1) - (5.5-Hk) ** 3/(Hk+1) ** 2)
		else:
			num = .015*(1-1./(Hk-4.5))**2 - .07
			num_Hk = .015*2*(1-1./(Hk-4.5))/(Hk-4.5)**2
		cf = num/Ret
		cf_U = num_Hk/Ret*Hk_U - num/Ret**2*Ret_U
	return cf, cf_U


def get_rho(U, param):
	"""
	Calculates the density (useful if compressible)
	INPUT:
		U     : state vector [th; ds; sa; ue]
		param : parameter structure
	OUTPUT:
		rho, rho_U : density and linearization
	DETAILS:
		If compressible, rho is calculated from stag rho + isentropic relations
	"""
	if param.Minf > 0:
		M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
		uk, uk_u = get_uk(U[3], param)  # corrected speed
		H0 = param.H0
		gmi = param.gam - 1
		den = 1 + 0.5 * gmi * M2
		den_M2 = 0.5 * gmi
		rho = param.rho0 / den**(1/gmi)
		rho_U = (-1/gmi) * rho / den * den_M2 * M2_U
	else:
		rho = param.rho0
		rho_U = np.zeros(4)
	return rho, rho_U


def get_Hk(U, param):
	"""
	calculates Hk = kinematic shape parameter, from U
	
	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure
	OUTPUT
	  Hk, Hk_U : kinematic shape parameter and its linearization w.r.t. U (1x4)
	DETAILS
	  Hk is like H but with no density in the integrals defining th and ds
	  So it is exactly the same when density is constant (= freestream)
	  Here, it is computed from H with a correlation using the Mach number
	"""
	H, H_U = get_H(U)

	if (param.Minf > 0):
		M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
		den = (1 + 0.113 * M2)
		den_M2 = 0.113
		Hk = (H - 0.29 * M2) / den
		Hk_U = (H_U - 0.29 * M2_U) / den - Hk / den * den_M2 * M2_U
	else:
		Hk = H
		Hk_U = H_U

	return Hk, Hk_U


def get_H(U):
	"""
	calculates H = shape parameter = delta*/theta, from U
	
	INPUT
	  U     : state vector [th; ds; sa; ue]
	OUTPUT
	  H, H_U : shape parameter and its linearization w.r.t. U (1x4)
	DETAILS
	  H is the ratio of the displacement thickness to the momentum thickness
	  In U, the ds entry should be (delta*-wgap) ... i.e wake gap taken out
	  When the real H is needed with wake gap, Hw is calculated and added
	"""
	H = U[1] / U[0]
	H_U = np.array([-H / U[0], 1 / U[0], 0, 0])
	return H, H_U


def get_Mach2(U, param):

	if (param.Minf > 0):
		H0 = param.H0
		g = param.gam
		uk, uk_u = get_uk(U[3], param)
		c2 = (g-1)*(H0-0.5*uk**2)
		c2_uk = (g-1)*(-uk)  # squared speed of sound
		M2 = uk**2/c2
		M2_uk = 2*uk/c2 - M2/c2*c2_uk
		M2_U = np.array([0, 0, 0, M2_uk*uk_u])
	else:
		M2 = 0.
		M2_U = np.array([0, 0, 0, 0])

	return M2, M2_U


def get_Ret(U, param):
	"""
	calculates theta Reynolds number, Re_theta, from U
	
	INPUT
	U     : state vector [th; ds; sa; ue]
	param : parameter structure
	
	OUTPUT
	Ret, Ret_U : Reynolds number based on the momentum thickness, linearization
	
	DETAILS
	Re_theta = rho*ue*theta/mu
	If compressible, rho is calculated from stag rho + isentropic relations
	ue is the edge speed and must be comressibility corrected
	mu is the dynamic viscosity, from Sutherland's law if compressible
	"""
	if (param.Minf > 0):
		M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
		uk, uk_u = get_uk(U[3], param)  # corrected speed
		H0 = param.H0
		gmi = param.gam - 1
		Ts = param.Tsrat
		Tr = 1 - 0.5 * (uk ** 2) / H0
		Tr_uk = -uk / H0  # edge/stagnation temperature ratio
		f = Tr ** 1.5 * (1 + Ts) / (Tr + Ts)  # Sutherland's ratio
		f_Tr = 1.5 * f / Tr - f / (Tr + Ts)
		mu = param.mu0 * f  # local dynamic viscosity
		mu_uk = param.mu0 * f_Tr * Tr_uk
		den = 1 + 0.5 * gmi * M2
		den_M2 = 0.5 * gmi
		rho = param.rho0 / den ** (1 / gmi)
		rho_U = (-1 / gmi) * rho / den * den_M2 * M2_U
		Ret = rho * uk * U[0] / mu
		Ret_U = rho_U * uk * U[0] / mu + (rho * U[0] / mu - Ret / mu * mu_uk) * np.array(
			[0, 0, 0, uk_u]) + rho * uk / mu * np.array([1, 0, 0, 0])
	else:
		Ret = param.rho0 * U[0] * U[3] / param.mu0
		Ret_U = np.array([U[3], 0, 0, U[0]]) / param.mu0
	return Ret, Ret_U
