import numpy as np
from LinearVortexPanelMethod.ParamInit import *
from ViscousFlow.ViscAuxEq import *


def residual_transition(M, param, x, U, Aux):
	"""
	Calculates the combined lam + turb residual for a transition station
	
	INPUT
	  param : parameter structure
	  x     : 2x1 vector, [x1, x2], containing xi values at the points
	  U     : 4x2 matrix, [U1, U2], containing the states at the points
	  Aux   : ()x2 matrix, [Aux1, Aux2] of auxiliary data at the points
	OUTPUT
	  R     : 3x1 transition residual vector
	  R_U   : 3x8 residual Jacobian, [R_U1, R_U2]
	  R_x   : 3x2 residual linearization w.r.t. x, [R_x1, R_x2]
	DETAILS
	  The state U1 should be laminar; U2 should be turbulent
	  Calculates and linearizes the transition location in the process
	  Assumes linear variation of th and ds from U1 to U2
	"""

	# states
	U1 = U[:, 0]
	U2 = U[:, 1]
	sa = U[2, :]
	I1 = np.arange(4)
	I2 = np.arange(4, 8)
	Z = np.zeros((1, 4))

	# interval
	x1 = x[0]
	x2 = x[1]
	dx = x2-x1

	# determine transition location (xt) using amplification equation
	xt = x1 + 0.5*dx  # guess
	ncrit = param.ncrit  # critical amp factor
	nNewton = 20
	print(param, 3, f'  Transition interval = [{x1:.5e}, {x2:.5e}]')
	#  U1, U2
	w1 = w2 = Rxt_x1 = Rxt_x2 = damp1 = Rxt_xt = dampt_Ut = dampt = damp1_U1 = 0
	Ut_xt = Ut = np.array([])

	iNewton = 0
	for iNewton in range(1, nNewton+1):
		w2 = (xt-x1)/dx
		w1 = 1-w2
		# weights
		Ut = w1*U1 + w2*U2
		Ut_xt = (U2-U1)/dx
		# state at xt
		Ut[2] = ncrit
		Ut_xt[2] = 0.
		# amplification at transition
		damp1, damp1_U1 = get_damp(U1, param)
		dampt, dampt_Ut = get_damp(Ut, param)
		dampt_Ut[2] = 0.
		Rxt = ncrit - sa[0] - 0.5*(xt-x1)*(damp1 + dampt)
		Rxt_xt = -0.5*(damp1+dampt) - 0.5*(xt-x1)*(np.dot(dampt_Ut, Ut_xt))
		dxt = -Rxt/Rxt_xt
		print(
			param, 4, f'   Transition: iNewton,Rxt,xt = {iNewton},{Rxt:.5e},{xt:.5e}')
		dmax = 0.2*dx*(1.1-iNewton/nNewton)
		if abs(dxt) > dmax:
			dxt = dxt*dmax/abs(dxt)
		if abs(Rxt) < 1e-10:
			break
		if iNewton < nNewton:
			xt = xt + dxt

	if iNewton >= nNewton:
		print('Transition location calculation failed.')
	M.vsol.xt = xt  # save transition location

	# prepare for xt linearizations
	Rxt_U = -0.5*(xt-x1)*np.concatenate((damp1_U1 + dampt_Ut*w1, dampt_Ut*w2))
	Rxt_U[2] -= 1
	Ut_x1 = (U2-U1)*(w2-1)/dx
	Ut_x2 = (U2-U1)*(-w2)/dx
	Ut_x1[2] = 0
	Ut_x2[2] = 0
	Rxt_x1 = 0.5*(damp1+dampt) - 0.5*(xt-x1)*np.dot(dampt_Ut, Ut_x1)
	Rxt_x2 = -0.5*(xt-x1)*np.dot(dampt_Ut, Ut_x2)

	# sensitivity of xt w.r.t. U,x from Rxt(xt,U,x) = 0 constraint
	xt_U = -Rxt_U/Rxt_xt
	xt_U1 = xt_U[I1]
	xt_U2 = xt_U[I2]
	xt_x1 = -Rxt_x1/Rxt_xt
	xt_x2 = -Rxt_x2/Rxt_xt

	# include derivatives w.r.t. xt in Ut_x1 and Ut_x2
	Utl_x1 = Ut_x1 = Ut_x1 + Ut_xt*xt_x1
	Utl_x2 = Ut_x2 = Ut_x2 + Ut_xt*xt_x2

	# sensitivity of Ut w.r.t. U1 and U2
	Utl_U1 = Ut_U1 = w1*np.eye(4) + np.dot((U2-U1).reshape(-1, 1), xt_U1.reshape(1, -1)/dx)
	Utl_U2 = Ut_U2 = w2*np.eye(4) + np.dot((U2-U1).reshape(-1, 1), xt_U2.reshape(1, -1)/dx)

	# laminar and turbulent states at transition
	Utl = Ut.copy()
	Utl[2] = ncrit
	Utl_U1[2, :] = Z
	Utl_U2[2, :] = Z
	Utl_x1[2] = 0
	Utl_x2[2] = 0
	Utt = Ut.copy()
	Utt_U1 = Ut_U1.copy()
	Utt_U2 = Ut_U2.copy()
	Utt_x1 = Ut_x1.copy()
	Utt_x2 = Ut_x2.copy()

	# parameter structure
	par = build_param(M, 0)

	# set turbulent shear coefficient, sa, in Utt
	par.turb = True
	cttr, cttr_Ut = get_cttr(Ut, par)
	Utt[2] = cttr
	Utt_U1[2, :] = np.dot(cttr_Ut, Ut_U1)
	Utt_U2[2, :] = np.dot(cttr_Ut, Ut_U2)
	Utt_x1[2] = np.dot(cttr_Ut, Ut_x1)
	Utt_x2[2] = np.dot(cttr_Ut, Ut_x2)

	# laminar/turbulent residuals and linearizations
	par.turb = False
	Rl, Rl_U, Rl_x = residual_station(
		par, np.array([x1, xt]), np.array([U1, Utl]).T, Aux)
	Rl_U1 = Rl_U[:, I1]
	Rl_Utl = Rl_U[:, I2]
	par.turb = True
	Rt, Rt_U, Rt_x = residual_station(
		par, np.array([xt, x2]), np.array([Utt, U2]).T, Aux)
	Rt_Utt = Rt_U[:, I1]
	Rt_U2 = Rt_U[:, I2]

	# combined residual and linearization
	R = Rl + Rt
	if any(R.imag):
		raise ValueError('imaginary transition residual')

	R_U1 = Rl_U1 + Rl_Utl @ Utl_U1 + \
            np.dot(Rl_x[:, 1].reshape(-1, 1), xt_U1.reshape(1, -1)) + \
            Rt_Utt @ Utt_U1 + \
            np.dot(Rt_x[:, 0].reshape(-1, 1), xt_U1.reshape(1, -1))
	R_U2 = Rl_Utl @ Utl_U2 + np.dot(Rl_x[:, 1].reshape(-1, 1), xt_U2.reshape(1, -1)) + \
            Rt_Utt @ Utt_U2 + Rt_U2 + \
            np.dot(Rt_x[:, 0].reshape(-1, 1), xt_U2.reshape(1, -1))
	R_U = np.column_stack((R_U1, R_U2))
	R_x = np.column_stack((
            Rl_x[:, 0] + Rl_x[:, 1] * xt_x1 + Rt_x[:, 0] *
           	xt_x1 + Rl_Utl @ Utl_x1 + Rt_Utt @ Utt_x1,
            Rt_x[:, 1] + Rl_x[:, 1] * xt_x2 + Rt_x[:, 0] * xt_x2 + Rl_Utl @ Utl_x2 + Rt_Utt @ Utt_x2))
	par.turb = False
	return R, R_U, R_x


def residual_station(param, x, U, Aux):
	"""
	calculates the viscous residual at one non-transition station
	INPUT
	  param : parameter structure
	  x     : 2x1 vector, [x1, x2], containing xi values at the points
	  U     : 4x2 matrix, [U1, U2], containing the states at the points
	  Aux   : ()x2 matrix, [Aux1, Aux2] of auxiliary data at the points
	OUTPUT
	  R     : 3x1 residual vector (mom, shape-param, amp/lag)
	  R_U   : 3x8 residual Jacobian, [R_U1, R_U2]
	  R_x   : 3x2 residual linearization w.r.t. x, [R_x1, R_x2]
	DETAILS
	  The input states are U = [U1, U2], each with th,ds,sa,ue

	modify ds to take out wake gap (in Aux) for all calculations below
	
	"""

	U[1, :] -= Aux[0, :]

	# states
	U1 = U[:, 0]
	U2 = U[:, 1]
	Um = 0.5 * (U1 + U2)
	th = U[0, :]
	ds = U[1, :]
	sa = U[2, :]

	# speed needs compressibility correction
	uk1, uk1_u = get_uk(U1[3], param)
	uk2, uk2_u = get_uk(U2[3], param)

	# log changes
	thlog = np.log(th[1] / th[0])
	thlog_U = np.array([-1 / th[0], 0, 0, 0, 1 / th[1], 0, 0, 0])
	uelog = np.log(uk2 / uk1)
	uelog_U = np.array([0, 0, 0, -uk1_u / uk1, 0, 0, 0, uk2_u / uk2])
	xlog = np.log(x[1] / x[0])
	xlog_x = np.array([-1 / x[0], 1 / x[1]])
	dx = x[1] - x[0]
	dx_x = np.array([-1, 1])

	# upwinding factor
	upw, upw_U = get_upw(U1, U2, param)

	# shape parameter
	H1, H1_U1 = get_H(U[:, 0])
	H2, H2_U2 = get_H(U[:, 1])
	H = 0.5 * (H1 + H2)
	H_U = 0.5 * np.concatenate([H1_U1, H2_U2])

	# Hstar = KE shape parameter, averaged
	Hs1, Hs1_U1 = get_Hs(U1, param)
	Hs2, Hs2_U2 = get_Hs(U2, param)
	Hs, Hs_U = upwind(0.5, 0, Hs1, Hs1_U1, Hs2, Hs2_U2)

	# log change in Hstar
	Hslog = np.log(Hs2/Hs1)
	Hslog_U = np.concatenate([-1/Hs1*Hs1_U1, 1/Hs2*Hs2_U2])

	# similarity station is special: U1 = U2, x1 = x2
	if (param.simi):
		thlog = 0
		thlog_U = thlog_U*0
		Hslog = 0
		Hslog_U = np.zeros(8)
		uelog = 1
		uelog_U = np.zeros(8)
		xlog = 1
		xlog_x = np.array([0, 0])
		dx = 0.5*(x[0]+x[1])
		dx_x = [0.5, 0.5]

	# Hw = wake shape parameter
	Hw1, Hw1_U1 = get_Hw(U[:, 0], Aux[0, 0])
	Hw2, Hw2_U2 = get_Hw(U[:, 1], Aux[0, 1])
	Hw = 0.5*(Hw1 + Hw2)
	Hw_U = 0.5*np.concatenate([Hw1_U1, Hw2_U2])

	if param.turb:
		# log change of root shear stress coeff
		salog = np.log(sa[1]/sa[0])
		salog_U = np.array([0, 0, -1/sa[0], 0, 0, 0, 1/sa[1], 0])

		# BL thickness measure, averaged
		de1, de1_U1 = get_de(U1, param)
		de2, de2_U2 = get_de(U2, param)
		de, de_U = upwind(0.5, 0, de1, de1_U1, de2, de2_U2)

		# normalized slip velocity, averaged
		Us1, Us1_U1 = get_Us(U1, param)
		Us2, Us2_U2 = get_Us(U2, param)
		Us, Us_U = upwind(0.5, 0, Us1, Us1_U1, Us2, Us2_U2)

		# Hk, upwinded
		Hk1, Hk1_U1 = get_Hk(U1, param)
		Hk2, Hk2_U2 = get_Hk(U2, param)
		Hk, Hk_U = upwind(upw, upw_U, Hk1, Hk1_U1, Hk2, Hk2_U2)

		# Re_theta, averaged
		Ret1, Ret1_U1 = get_Ret(U1, param)
		Ret2, Ret2_U2 = get_Ret(U2, param)
		Ret, Ret_U = upwind(0.5, 0, Ret1, Ret1_U1, Ret2, Ret2_U2)

		# skin friction, upwinded
		cf1, cf1_U1 = get_cf(U1, param)
		cf2, cf2_U2 = get_cf(U2, param)
		cf, cf_U = upwind(upw, upw_U, cf1, cf1_U1, cf2, cf2_U2)

		# displacement thickness, averaged
		dsa = 0.5*(ds[0] + ds[1])
		dsa_U = 0.5*np.array([0, 1, 0, 0, 0, 1, 0, 0])

		# uq = equilibrium 1/ue * due/dx
		uq, uq_U = get_uq(dsa, dsa_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param)

		# cteq = root equilibrium wake layer shear coeficient: (ctau eq)^.5
		cteq1, cteq1_U1 = get_cteq(U1, param)
		cteq2, cteq2_U2 = get_cteq(U2, param)
		cteq, cteq_U = upwind(upw, upw_U, cteq1, cteq1_U1, cteq2, cteq2_U2)

		# root of shear coefficient (a state), upwinded
		saa, saa_U = upwind(upw, upw_U, sa[0], np.array(
			[0, 0, 1, 0]), sa[1], np.array([0, 0, 1, 0]))

		# lag coefficient
		Klag = param.SlagK
		beta = param.GB
		Clag = Klag / beta * 1 / (1 + Us)
		Clag_U = -Clag / (1 + Us) * Us_U

		# extra dissipation in wake
		ald = 1.0
		if (param.wake):
			ald = param.Dlr

		# shear lag equation
		Rlag = Clag * (cteq - ald * saa) * dx - 2 * de * salog + \
                    2 * de * (uq * dx - uelog) * param.Cuq
		Rlag_U = Clag_U * (cteq - ald * saa) * dx + Clag * (cteq_U - ald * saa_U) * dx - 2 * de_U * salog - 2 * \
                    de * salog_U + 2 * de_U * \
                    (uq * dx - uelog) * param.Cuq + 2 * \
                    de * (uq_U * dx - uelog_U) * param.Cuq
		Rlag_x = Clag * (cteq - ald * saa) * dx_x + 2 * de * uq * dx_x
	else:
		# laminar, amplification factor equation
		if param.simi:
			# similarity station
			Rlag = sa[0] + sa[1]  # no amplification
			Rlag_U = np.array([0, 0, 1, 0, 0, 0, 1, 0])
			Rlag_x = np.array([0, 0])
		else:
			# amplification factor equation in Rlag

			# amplification rate, averaged
			damp1, damp1_U1 = get_damp(U1, param)
			damp2, damp2_U2 = get_damp(U2, param)
			damp, damp_U = upwind(0.5, 0, damp1, damp1_U1, damp2, damp2_U2)

			Rlag = sa[1] - sa[0] - damp*dx
			Rlag_U = [0, 0, -1, 0, 0, 0, 1, 0] - damp_U*dx
			Rlag_x = -damp*dx_x

	Ms1, Ms1_U1 = get_Mach2(U1, param)
	Ms2, Ms2_U2 = get_Mach2(U2, param)
	Ms, Ms_U = upwind(0.5, 0, Ms1, Ms1_U1, Ms2, Ms2_U2)

	cfxt1, cfxt1_U1, cfxt1_x1 = get_cfxt(U1, x[0], param)
	cfxt2, cfxt2_U2, cfxt2_x2 = get_cfxt(U2, x[1], param)
	cfxtm, cfxtm_Um, cfxtm_xm = get_cfxt(Um, 0.5*(x[0]+x[1]), param)
	cfxt = 0.25*cfxt1 + 0.5*cfxtm + 0.25*cfxt2
	cfxt_U = np.concatenate([0.25*(cfxt1_U1+cfxtm_Um), 0.25*(cfxtm_Um+cfxt2_U2)])
	cfxt_x = np.array([0.25*(cfxt1_x1+cfxtm_xm), 0.25*(cfxtm_xm+cfxt2_x2)])

	Rmom = thlog + (2+H+Hw-Ms)*uelog - 0.5*xlog*cfxt
	Rmom_U = thlog_U + (H_U+Hw_U-Ms_U)*uelog + (2+H+Hw-Ms) * \
            uelog_U - 0.5*xlog*cfxt_U
	Rmom_x = -0.5*xlog_x*cfxt - 0.5*xlog*cfxt_x

	cDixt1, cDixt1_U1, cDixt1_x1 = get_cDixt(U1, x[0], param)
	cDixt2, cDixt2_U2, cDixt2_x2 = get_cDixt(U2, x[1], param)
	cDixt, cDixt_U = upwind(upw, upw_U, cDixt1, cDixt1_U1, cDixt2, cDixt2_U2)
	cDixt_x = np.array([(1-upw)*cDixt1_x1, upw*cDixt2_x2])

	cfxtu, cfxtu_U = upwind(upw, upw_U, cfxt1, cfxt1_U1, cfxt2, cfxt2_U2)
	cfxtu_x = np.array([(1-upw)*cfxt1_x1, upw*cfxt2_x2])

	Hss1, Hss1_U1 = get_Hss(U1, param)
	Hss2, Hss2_U2 = get_Hss(U2, param)
	Hss, Hss_U = upwind(0.5, 0, Hss1, Hss1_U1, Hss2, Hss2_U2)

	Rshape = Hslog + (2*Hss/Hs + 1-H-Hw)*uelog + xlog*(0.5*cfxtu - cDixt)
	Rshape_U = Hslog_U + (2*Hss_U/Hs - 2*Hss/(Hs**2)*Hs_U - H_U - Hw_U) * \
            uelog + (2*Hss/Hs + 1-H-Hw)*uelog_U + xlog*(0.5*cfxtu_U - cDixt_U)
	Rshape_x = xlog_x*(0.5*cfxtu - cDixt) + xlog*(0.5*cfxtu_x - cDixt_x)

	# put everything together
	R = np.array([Rmom, Rshape, Rlag])
	R_U = np.vstack((Rmom_U, Rshape_U, Rlag_U))
	R_x = np.vstack([Rmom_x, Rshape_x, Rlag_x])

	return R, R_U, R_x
