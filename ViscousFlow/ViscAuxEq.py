import numpy as np
from LinearVortexPanelMethod.InvAuxEq import *


def get_cttr(U, param):
	"""
	Calculates root of the shear stress coefficient at transition
	INPUT
		U     : state vector [th; ds; sa; ue]
		param : parameter structure
	OUTPUT
		cttr, cttr_U : sqrt(shear stress coeff) and its lin w.r.t. U (1x4)
	DETAILS
		used to initialize the first turb station after transition
	"""
	param.wake = False  # transition happens just before the wake starts
	cteq, cteq_U = get_cteq(U, param)
	Hk, Hk_U = get_Hk(U, param)
	if Hk < 1.05:
		Hk = 1.05
		Hk_U = Hk_U * 0
	C = param.CtauC
	E = param.CtauE
	c = C * np.exp(-E / (Hk - 1))
	c_U = c * E / (Hk - 1) ** 2 * Hk_U
	cttr = c * cteq
	cttr_U = c_U * cteq + c * cteq_U
	return cttr, cttr_U


def get_Hw(U, wgap):
	"""
	calculates Hw = wake gap shape parameter = wgap/theta
	
	INPUT
	  U    : state vector [th; ds; sa; ue]
	  wgap : wake gap
	OUTPUT
	  Hw, Hw_U : wake gap shape parameter and its linearization w.r.t. U (1x4)
	DETAILS
	  Hw is the ratio of the wake gap to the momentum thickness
	  The wake gap is the TE gap extrapolated into the wake (dead air region)
	"""
	Hw = wgap / U[0]  # wgap/th
	Hw_U = np.array([-Hw / U[0], 0, 0, 0])
	return Hw, Hw_U


def get_Hss(U, param):
	"""
	Calculates Hss = density shape parameter, from U
	INPUT:
		U     : state vector [th; ds; sa; ue]
		param : parameter structure
	OUTPUT:
		Hss, Hss_U : density shape parameter and its linearization w.r.t. U (1x4)
	"""
	M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
	Hk, Hk_U = get_Hk(U, param)
	num, num_U = 0.064 / (Hk - 0.8) + 0.251, -0.064 / (Hk - 0.8)**2 * Hk_U
	Hss = M2 * num
	Hss_U = M2_U * num + M2 * num_U
	return Hss, Hss_U


def get_de(U, param):
	"""
	Calculates simplified BL thickness measure
	INPUT:
		U     : state vector [th; ds; sa; ue]
		param : parameter structure
	OUTPUT:
		de, de_U : BL thickness "delta" and its linearization w.r.t. U (1x4)
	DETAILS:
		delta is delta* incremented with a weighted momentum thickness, theta
		The weight on theta depends on Hk, and there is an overall cap
	"""
	Hk, Hk_U = get_Hk(U, param)
	aa = 3.15 + 1.72 / (Hk - 1)
	aa_U = -1.72 / (Hk - 1)**2 * Hk_U
	de = U[0] * aa + U[1]
	de_U = np.array([aa, 1, 0, 0]) + U[0] * aa_U
	dmx = 12.0
	if de > dmx * U[0]:
		de = dmx * U[0]
		de_U = np.array([dmx, 0, 0, 0])
	return de, de_U


def get_cfxt(U, x, param):
	"""	
	calculates cf*x/theta from the state
	INPUT
	  U     : state vector [th; ds; sa; ue]
	  x     : distance along wall (xi)
	  param : parameter structure
	OUTPUT
	  cfxt,  : the combination cf*x/theta (calls cf function)
	  cfxt_U : linearization w.r.t. U (1x4)
	  cfxt_x : linearization w.r.t x (scalar)
	DETAILS
	  This combination appears in the momentum and shape parameter equations

	"""

	cf, cf_U = get_cf(U, param)
	cfxt = cf * x / U[0]
	cfxt_U = cf_U * x / U[0]
	cfxt_U[0] = cfxt_U[0] - cfxt / U[0]
	cfxt_x = cf / U[0]

	return cfxt, cfxt_U, cfxt_x


def get_cdutstag(U, param):
	"""
	Calculates cDi*ue*theta, used in stagnation station calculations
	
	INPUT----
		U : state vector [th; ds; sa; ue]
		param : parameter structure
		
	OUTPUT----
		D, D_U : value and linearization of cDi*ue*theta
	
	DETAILS----
		Only for stagnation and laminar
	"""
	U[3] = 0
	Hk, Hk_U = get_Hk(U, param)

	if Hk < 4:
		num = 0.00205 * (4 - Hk) ** 5.5 + 0.207
		num_Hk = 0.00205 * 5.5 * (4 - Hk) ** 4.5 * (-1)
	else:
		Hk1 = Hk - 4
		num = -0.0016 * Hk1 ** 2 / (1 + 0.02 * Hk1 ** 2) + 0.207
		num_Hk = (-0.0016 * (2 * Hk1 / (1 + 0.02 * Hk1 ** 2) - Hk1 ** 2 / (1 + 0.02 * Hk1 ** 2) ** 2 * 0.02 * 2 * Hk1))

	nu = param.mu0 / param.rho0
	D = nu * num
	D_U = nu * num_Hk * Hk_U

	return D, D_U


def get_cDixt(U, x, param):
	"""
	Calculates cDi*x/theta from the state
	
	INPUT----
		U : state vector [th; ds; sa; ue]
		x : distance along wall (xi)
		param : parameter structure
		
	OUTPUT----
		cDixt   : the combination cDi*x/theta (calls cDi function)
		cDixt_U : linearization w.r.t. U (1x4)
		cDixt_x : linearization w.r.t x (scalar)
		
	DETAILS----
		cDi is the dissipation function
	"""
	cDi, cDi_U = get_cDi(U, param)
	cDixt = cDi * x / U[0]
	cDixt_U = cDi_U * x / U[0]
	cDixt_U[0] = cDixt_U[0] - cDixt / U[0]
	cDixt_x = cDi / U[0]

	return cDixt, cDixt_U, cDixt_x


def get_cDi(U, param):
	"""
	calculates cDi = dissipation function = 2*cD/H*, from the state
	
	INPUT----
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure
	
	OUTPUT----
	  cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
	
	DETAILS----
	  cD is the dissipation coefficient, int(tau*du/dn*dn)/(rho*ue^3)
	  The combination with H* appears in the shape parameter equation
	"""

	if param.turb:  # turbulent includes wake

		# initialize to 0; will add components that are needed
		cDi = 0
		cDi_U = np.zeros(4)

		if not param.wake:
			# turbulent wall contribution (0 in the wake)
			cDi0, cDi0_U = get_cDi_turbwall(U, param)
			cDi += cDi0
			cDi_U += cDi0_U
			cDil, cDil_U = get_cDi_lam(U, param)  # for max check
		else:
			cDil, cDil_U = get_cDi_lamwake(U, param)  # for max check

		# outer layer contribution
		cDi0, cDi0_U = get_cDi_outer(U, param)
		cDi += cDi0
		cDi_U += cDi0_U

		# laminar stress contribution
		cDi0, cDi0_U = get_cDi_lamstress(U, param)
		cDi += cDi0
		cDi_U += cDi0_U

		# maximum check
		if cDil > cDi:
			cDi = cDil
			cDi_U = cDil_U

		# double dissipation in the wake
		if param.wake:
			cDi *= 2
			cDi_U *= 2
	else:
		# just laminar dissipation
		cDi, cDi_U = get_cDi_lam(U, param)

	return cDi, cDi_U


def get_cDi_turbwall(U, param):
	# initializing variables
	cDi = 0
	cDi_U = np.zeros(4)

	# check for wake condition
	if (param.wake):
		return cDi, cDi_U

	# get cf, Hk, Hs, Us, Ret
	cf, cf_U = get_cf(U, param)
	Hk, Hk_U = get_Hk(U, param)
	Hs, Hs_U = get_Hs(U, param)
	Us, Us_U = get_Us(U, param)
	Ret, Ret_U = get_Ret(U, param)

	# calculate required variables for cDi
	lr = np.log(Ret)
	lr_U = Ret_U/Ret
	Hmin = 1 + 2.1/lr
	Hmin_U = -2.1/lr**2*lr_U
	aa = np.tanh((Hk-1)/(Hmin-1))
	fac = 0.5 + 0.5*aa
	fac_U = 0.5*(1-aa**2)*(Hk_U/(Hmin-1)-(Hk-1)/(Hmin-1)**2*Hmin_U)

	# calculate cDi and its linearization w.r.t. U
	cDi = 0.5*cf*Us*(2/Hs)*fac
	cDi_U = cf_U*Us/Hs*fac + cf*Us_U/Hs*fac - cDi/Hs*Hs_U + cf*Us/Hs*fac_U

	return cDi, cDi_U


def get_cDi_lam(U, param):
	# get Hk and Ret
	Hk, Hk_U = get_Hk(U, param)
	Ret, Ret_U = get_Ret(U, param)

	# calculate cDi and its linearization w.r.t. U based on Hk
	if (Hk < 4):
		num = .00205*(4-Hk)**5.5 + .207
		num_Hk = .00205*5.5*(4-Hk)**4.5*(-1)
	else:
		Hk1 = Hk-4
		num = -.0016*Hk1**2/(1+.02*Hk1**2) + .207
		num_Hk = -.0016*(2*Hk1/(1+.02*Hk1**2) - Hk1 **
                   2/(1+.02*Hk1**2)**2*.02*2*Hk1)

	# calculate cDi and its linearization w.r.t. U
	cDi = num/Ret
	cDi_U = num_Hk/Ret*Hk_U - num/Ret**2*Ret_U

	return cDi, cDi_U


def get_cDi_lamwake(U, param):
	"""
	Laminar wake dissipation function cDi

	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure

	OUTPUT
	  cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)

	DETAILS
	  This is one contribution to the dissipation function cDi = 2*cD/H*
	"""

	tb = param.turb

	param.turb = False  # force laminar

	# dependencies
	Hk, Hk_U = get_Hk(U, param)
	Hs, Hs_U = get_Hs(U, param)
	Ret, Ret_U = get_Ret(U, param)
	HsRet = Hs*Ret
	HsRet_U = Hs_U*Ret + Hs*Ret_U

	num = 2*1.1*(1-1/Hk)**2*(1/Hk)
	num_Hk = 2*1.1*(2*(1-1/Hk)*(1/Hk**2)*(1/Hk)+(1-1/Hk)**2*(-1/Hk**2))
	cDi = num/HsRet
	cDi_U = num_Hk*Hk_U/HsRet - num/HsRet**2*HsRet_U

	param.turb = tb

	return cDi, cDi_U


def get_cDi_outer(U, param):
	"""
	Turbulent outer layer contribution to dissipation function cDi

	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure

	OUTPUT
	  cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
	  
	DETAILS
	  This is one contribution to the dissipation function cDi = 2*cD/H*
	"""

	if (not param.turb):
		cDi = 0
		cDi_U = np.zeros(4)
		return cDi, cDi_U  # for pinging

	# first get Hs, Us
	Hs, Hs_U = get_Hs(U, param)
	Us, Us_U = get_Us(U, param)

	# shear stress: note, state stores ct^.5
	ct = U[2]**2
	ct_U = np.array([0, 0, 2*U[2], 0])

	cDi = ct*(0.995-Us)*2/Hs
	cDi_U = ct_U*(0.995-Us)*2/Hs + ct*(-Us_U)*2/Hs - ct*(0.995-Us)*2/Hs**2*Hs_U

	return cDi, cDi_U


def get_cDi_lamstress(U, param):

	"""
	Laminar stress contribution to dissipation function cDi

	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure

	OUTPUT
	  cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)

	DETAILS
	  This is one contribution to the dissipation function cDi = 2*cD/H*
	"""
	  
	# first get Hs, Us, and Ret
	Hs, Hs_U = get_Hs(U, param)
	Us, Us_U = get_Us(U, param)
	Ret, Ret_U = get_Ret(U, param)
	HsRet = Hs*Ret
	HsRet_U = Hs_U*Ret + Hs*Ret_U

	num = 0.15*(0.995-Us)**2*2
	num_Us = 0.15*2*(0.995-Us)*(-1)*2
	cDi = num/HsRet
	cDi_U = num_Us*Us_U/HsRet - num/HsRet**2*HsRet_U

	return cDi, cDi_U


def get_uq(ds, ds_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param):
	"""
	Calculates the equilibrium 1/ue*due/dx

	INPUT
	  ds, ds_U   : delta star and linearization (1x4)
	  cf, cf_U   : skin friction and linearization (1x4)
	  Hk, Hk_U   : kinematic shape parameter and linearization (1x4)
	  Ret, Ret_U : theta Reynolds number and linearization (1x4)
	  param      : parameter structure

	OUTPUT
	  uq, uq_U   : equilibrium 1/ue*due/dx and linearization w.r.t. state (1x4)
	"""
	  
	beta = param.GB
	A = param.GA
	C = param.GC

	if param.wake:
		A = A * param.Dlr
		C = 0

	# limit Hk (TODO smooth/eliminate)
	if param.wake and Hk < 1.00005:
		Hk = 1.00005
		Hk_U = Hk_U * 0
	elif not param.wake and Hk < 1.05:
		Hk = 1.05
		Hk_U = Hk_U * 0

	Hkc = Hk - 1 - C / Ret
	Hkc_U = Hk_U + C / Ret**2 * Ret_U

	# [Hkc, rd] = slimit_Hkc(Hkc); Hkc_U = rd*Hkc_U; % smooth limiting of Hkc
	if Hkc < 0.01:
		Hkc = 0.01
		Hkc_U = Hkc_U * 0

	ut = 0.5 * cf - (Hkc / (A * Hk))**2
	ut_U = 0.5 * cf_U - 2 * (Hkc / (A * Hk)) * \
		(Hkc_U / (A * Hk) - Hkc / (A * Hk**2) * Hk_U)
	uq = ut / (beta * ds)
	uq_U = ut_U / (beta * ds) - uq / ds * ds_U

	return uq, uq_U


def get_cteq(U, param):
	"""
	Calculates root of the equilibrium shear stress coefficient: sqrt(ctau_eq)

	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure

	OUTPUT
	  cteq, cteq_U : sqrt(equilibrium shear stress) and its lin w.r.t. U (1x4)

	DETAILS
	  uses equilibrium shear stress correlations
	"""
	CC = 0.5 / (param.GA ** 2 * param.GB)
	C = param.GC
	Hk, Hk_U = get_Hk(U, param)
	Hs, Hs_U = get_Hs(U, param)
	H, H_U = get_H(U)
	Ret, Ret_U = get_Ret(U, param)
	Us, Us_U = get_Us(U, param)

	if param.wake:
		if Hk < 1.00005:
			Hk = 1.00005
			Hk_U *= 0
		Hkc = Hk - 1
		Hkc_U = Hk_U
	else:
		if Hk < 1.05:
			Hk = 1.05
			Hk_U *= 0
		Hkc = Hk - 1 - C / Ret
		Hkc_U = Hk_U + C / Ret ** 2 * Ret_U

		# [Hkc, rd] = slimit_Hkc(Hkc); Hkc_U = rd*Hkc_U; % smooth limiting of Hkc
		if Hkc < 0.01:
			Hkc = 0.01
			Hkc_U *= 0

	num = CC * Hs * (Hk - 1) * Hkc ** 2
	num_U = CC * (Hs_U * (Hk - 1) * Hkc ** 2 + Hs * Hk_U *
               Hkc ** 2 + Hs * (Hk - 1) * 2 * Hkc * Hkc_U)
	den = (1 - Us) * H * Hk ** 2
	den_U = (-Us_U) * H * Hk ** 2 + (1 - Us) * H_U * \
		Hk ** 2 + (1 - Us) * H * 2 * Hk * Hk_U
	cteq = np.sqrt(num / den)
	cteq_U = 0.5 / cteq * (num_U / den - num / den ** 2 * den_U)

	return cteq, cteq_U


def get_upw(U1, U2, param):
	"""
	Calculates a local upwind factor (0.5 = trap; 1 = BE) based on two states

	INPUT
	  U1,U2 : first/upwind and second/downwind states (4x1 each)
	  param : parameter structure

	OUTPUT
	  upw   : scalar upwind factor
	  upw_U : 1x8 linearization vector, [upw_U1, upw_U2]

	DETAILS
	  Used to ensure a stable viscous discretization
	  Decision to upwind is made based on the shape factor change
	"""

	Hk1, Hk1_U1 = get_Hk(U1, param)
	Hk2, Hk2_U2 = get_Hk(U2, param)
	Z = np.zeros_like(Hk1_U1)
	Hut = 1.0  # triggering constant for upwinding
	C = 5.0 if param.wake else 1.0
	Huc = C * Hut / (Hk2 ** 2)  # only depends on U2
	Huc_U = np.hstack((Z, -2 * Huc / Hk2 * Hk2_U2))
	aa = (Hk2 - 1) / (Hk1 - 1)
	sga = np.sign(aa)
	la = np.log(sga * aa)
	la_U = np.hstack((-1 / (Hk1 - 1) * Hk1_U1, 1 / (Hk2 - 1) * Hk2_U2))
	Hls = la ** 2
	Hls_U = 2 * la * la_U
	if Hls > 15:
		Hls = 15
		Hls_U = Hls_U * 0
	upw = 1 - 0.5 * np.exp(-Hls * Huc)
	upw_U = -0.5 * np.exp(-Hls * Huc) * (-Hls_U * Huc - Hls * Huc_U)

	return upw, upw_U


def upwind(upw, upw_U, f1, f1_U1, f2, f2_U2):
	"""
	Calculates an upwind average (and derivatives) of two scalars

	INPUT
	  upw, upw_U : upwind scalar and its linearization w.r.t. U1,U2
	  f1, f1_U   : first scalar and its linearization w.r.t. U1
	  f2, f2_U   : second scalar and its linearization w.r.t. U2

	OUTPUT
	  f    : averaged scalar
	  f_U  : linearization of f w.r.t. both states, [f_U1, f_U2]
	"""

	f = (1 - upw) * f1 + upw * f2
	f_U = (-upw_U) * f1 + upw_U * f2 + \
            np.concatenate([(1 - upw) * f1_U1, upw * f2_U2])

	return f, f_U


def get_Us(U, param):
	"""
	Calculates the normalized wall slip velocity Us

	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure

	OUTPUT
	  Us, Us_U : normalized wall slip velocity and its linearization w.r.t. U (1x4)
	
	"""

	Hs, Hs_U = get_Hs(U, param)
	Hk, Hk_U = get_Hk(U, param)
	H, H_U = get_H(U)

	# limit Hk (TODO smooth/eliminate)
	if (param.wake) and (Hk < 1.00005):
		Hk = 1.00005
		Hk_U = Hk_U*0
	if (not param.wake) and (Hk < 1.05):
		Hk = 1.05
		Hk_U = Hk_U*0

	beta = param.GB
	bi = 1./beta
	Us = 0.5*Hs*(1-bi*(Hk-1)/H)
	Us_U = 0.5*Hs_U*(1-bi*(Hk-1)/H) + 0.5*Hs * \
		(-bi*(Hk_U)/H + bi*(Hk-1)/H**2*H_U)
	# limits
	if (not param.wake) and (Us > 0.95):
		Us = 0.98
		Us_U = Us_U*0
	if (param.wake) and (Us > 0.99995):
		Us = 0.99995
		Us_U = Us_U*0

	return Us, Us_U


def get_Hs(U, param):
	"""
	Calculates Hs = Hstar = K.E. shape parameter, from U

	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure

	OUTPUT
	  Hs, Hs_U : Hstar and its lin w.r.t. U (1x4)

	DETAILS
	  Hstar is the ratio theta*/theta, where theta* is the KE thickness
	"""

	Hk, Hk_U = get_Hk(U, param)

	# limit Hk (TODO smooth/eliminate)
	if (param.wake) and (Hk < 1.00005):
		Hk = 1.00005
		Hk_U = np.zeros(4)
	if (not param.wake) and (Hk < 1.05):
		Hk = 1.05
		Hk_U = np.zeros(4)

	if (param.turb):  # turbulent
		Hsmin = 1.5
		dHsinf = 0.015
		Ret, Ret_U = get_Ret(U, param)

		# limit Re_theta and dependence
		Ho = 4
		Ho_U = 0
		if (Ret > 400):
			Ho = 3 + 400/Ret
			Ho_U = -400/Ret**2 * Ret_U
		Reb = Ret
		Reb_U = Ret_U
		if (Ret < 200):
			Reb = 200
			Reb_U = np.zeros(4)
		if (Hk < Ho):  # attached branch
			Hr = (Ho-Hk)/(Ho-1)
			Hr_U = (Ho_U - Hk_U)/(Ho-1) - (Ho-Hk)/(Ho-1)**2 * Ho_U
			aa = (2-Hsmin-4/Reb)*Hr**2
			aa_U = (4/Reb**2 * Reb_U) * Hr**2 + (2-Hsmin-4/Reb) * 2 * Hr * Hr_U
			Hs = Hsmin + 4/Reb + aa * 1.5/(Hk+.5)
			Hs_U = -4/Reb**2*Reb_U + aa_U*1.5/(Hk+.5) - aa*1.5/(Hk+.5)**2 * Hk_U
		else:  # separated branch
			lrb = np.log(Reb)
			lrb_U = 1/Reb * Reb_U
			aa = Hk - Ho + 4/lrb
			aa_U = Hk_U - Ho_U - 4/lrb**2 * lrb_U
			bb = 0.007*lrb/aa**2 + dHsinf/Hk
			bb_U = 0.007 * (lrb_U/aa**2 - 2*lrb/aa**3 * aa_U) - dHsinf/Hk**2 * Hk_U
			Hs = Hsmin + 4/Reb + (Hk-Ho)**2 * bb
			Hs_U = -4/Reb**2 * Reb_U + 2 * \
				(Hk-Ho) * (Hk_U-Ho_U) * bb + (Hk-Ho)**2 * bb_U

		# turbulent
		M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
		den = 1 + 0.014 * M2
		den_M2 = 0.014
		Hs = (Hs + 0.028 * M2) / den
		Hs_U = (Hs_U + 0.028 * M2_U) / den - Hs / den * den_M2 * M2_U

	else:
		# laminar
		a = Hk - 4.35
		if (Hk < 4.35):
			num = 0.0111 * a**2 - 0.0278 * a**3
			Hs = num / (Hk + 1) + 1.528 - 0.0002 * (a * Hk)**2
			Hs_Hk = (0.0111 * 2 * a - 0.0278 * 3 * a**2) / (Hk + 1) - \
				num / (Hk + 1)**2 - 0.0002 * 2 * a * Hk * (Hk + a)
		else:
			Hs = 0.015 * a**2 / Hk + 1.528
			Hs_Hk = 0.015 * 2 * a / Hk - 0.015 * a**2 / Hk**2
		Hs_U = Hs_Hk * Hk_U
	return Hs, Hs_U


def get_damp(U, param):
	"""
	Calculates the amplification rate, dn/dx, used in predicting transition
	
	INPUT
	  U     : state vector [th; ds; sa; ue]
	  param : parameter structure
	
	OUTPUT
	  damp, damp_U : amplification rate and its linearization w.r.t. U (1x4)
	
	DETAILS
	  damp = dn/dx is used in the amplification equation, prior to transition
	"""

	Hk, Hk_U = get_Hk(U, param)
	Ret, Ret_U = get_Ret(U, param)
	th = U[0]

	# limit Hk (TODO smooth/eliminate)
	if Hk < 1.05:
		Hk = 1.05
		Hk_U = Hk_U * 0

	Hmi = 1 / (Hk - 1)
	Hmi_U = -Hmi**2 * Hk_U
	aa = 2.492 * Hmi**0.43
	aa_U = 0.43 * aa / Hmi * Hmi_U
	bb = np.tanh(14 * Hmi - 9.24)
	bb_U = (1 - bb**2) * 14 * Hmi_U
	lrc = aa + 0.7 * (bb + 1)
	lrc_U = aa_U + 0.7 * bb_U
	lten = np.log(10)
	lr = np.log(Ret) / lten
	lr_U = (1 / Ret) * Ret_U / lten
	dl = 0.1  # changed from .08 to make smoother
	damp = 0
	damp_U = np.zeros(U.shape)  # default no amplification
	if lr >= lrc - dl:
		rn = (lr - (lrc - dl)) / (2 * dl)
		rn_U = (lr_U - lrc_U) / (2 * dl)
		if rn >= 1:
			rf = 1
			rf_U = np.zeros(U.shape)
		else:
			rf = 3 * rn**2 - 2 * rn**3
			rf_U = (6 * rn - 6 * rn**2) * rn_U
		ar = 3.87 * Hmi - 2.52
		ar_U = 3.87 * Hmi_U
		ex = np.exp(-ar**2)
		ex_U = ex * (-2 * ar * ar_U)
		da = 0.028 * (Hk - 1) - 0.0345 * ex
		da_U = 0.028 * Hk_U - 0.0345 * ex_U
		af = -0.05 + 2.7 * Hmi - 5.5 * Hmi**2 + \
			3 * Hmi**3 + 0.1 * np.exp(-20 * Hmi)
		af_U = (2.7 - 11 * Hmi + 9 * Hmi**2 - 1 * np.exp(-20 * Hmi)) * Hmi_U
		damp = rf * af * da / th
		damp_U = (rf_U * af * da + rf * af_U * da + rf * af *
                    da_U) / th - damp / th * np.array([1, 0, 0, 0])

	# extra amplification to ensure dn/dx > 0 near ncrit
	ncrit = param.ncrit

	Cea = 5
	nx = Cea * (U[2] - ncrit)
	nx_U = Cea * np.array([0, 0, 1, 0])

	eex = 1 + np.tanh(nx)
	eex_U = (1 - np.tanh(nx)**2) * nx_U

	ed = eex * 0.001 / U[0]
	ed_U = eex_U * 0.001 / U[0] - ed / U[0] * np.array([1, 0, 0, 0])

	damp = damp + ed
	damp_U = damp_U + ed_U
	return damp, damp_U
