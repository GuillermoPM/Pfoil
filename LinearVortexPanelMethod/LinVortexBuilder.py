"""
	Inviscid solution functions and base functions for viscous case.
    
    @Author: Guillermo Peña Martínez
    @Date: 04/05/2023
"""
import numpy as np

from LinearVortexPanelMethod.InvAuxEq import *
from LinearVortexPanelMethod.StreamFuncEq import *


from Geometry import *
from LinearVortexPanelMethod.ParamInit import *


def LVSolver(Foil):
	"""
	Linear Vortex Method solver
	
	"""

	Foil.oper.viscous = False
	# do not distinguish sign of ue if inviscid
	Foil.isol.sgnue = np.ones(Foil.N+1)
	vortex_builder(Foil, Foil.oper.alpha)
	calc_force(Foil)
	Foil.glob.conv = True  # no coupled system ... convergence is guaranteed


def vortex_builder(foil, alpha):
	"""
		Distributes the vortices around the airfoil and calculates the circulation that is generated.

		Input:
			- foil : airfoil class
			- alpha : angle of attack
	"""
	panels = foil.geom.panels
	N = foil.N   
	coord = foil.geom.coord

	A = np.zeros((N+2, N+2))  # influence matrix
	rhs = np.zeros((N+2, 2))  # right-hand sides for 0,90
	_, hTE, _, tcp, tdp = trailing_specs(foil)  # trailing-edge info
	nogap = (abs(hTE) < 1e-10*foil.geom.chord)  # indicates no TE gap

	# Influence matrix and rhs of equation
	for i, panel_i in enumerate(panels):
		xi = coord[i, :]  # node coordinates
		for j, panel_j in enumerate(panels[:-1]):  # TE panel is excluded
			aij, bij = panel_linvortex_stream(panel_i=panel_i, panel_j=panel_j)
			A[i, j] = A[i, j] + aij
			A[i, j+1] = A[i, j+1] + bij
			A[i, -1] = -1  # Last element is streamline
		# right-hand sides
		rhs[i, :] = [-xi[1], xi[0]]
		# TE source influence
		a = panel_constsource_stream(panel_i=panel_i, panel_j=panels[-1])

		A[i, 0] = A[i, 0] - a*(0.5*tcp)
		A[i, N] = A[i, N] + a*(0.5*tcp)
		# TE influencia vórtice
		a, b = panel_linvortex_stream(panel_i=panel_i, panel_j=panels[-1])
		A[i, 0] = A[i, 0] - (a+b)*(-0.5*tdp)
		A[i, N] = A[i, N] + (a+b)*(-0.5*tdp)

	# No TE gap
	if nogap:
		A[N, :] = 0
		A[N, [0, 1, 2, N-2, N-1, N]] = [1, -2, 1, -1, 2, -1]

	# Kutta condition
	A[N+1, 0] = 1
	A[N+1, N] = 1

	# Solving the system
	foil.isol.vMatrix = A
	g = np.linalg.solve(foil.isol.vMatrix, rhs)

	foil.isol.gamref = np.array(g[:-1, :])
	foil.isol.gam = foil.isol.gamref[:, 0] * np.cos(np.deg2rad(
		alpha)) + foil.isol.gamref[:, 1] * np.sin(np.deg2rad(alpha))


def inviscid_velocity(Foil, gamma, Vinf, alpha, x):

	panels = Foil.geom.panels
	X = Foil.geom.coord

	N = len(X)  # (Npanels +1 )
	V = np.zeros(2)  # velocity

	V_G = type(object)

	V_G = np.zeros((2, N))
	_, _, _, tcp, tdp = trailing_specs(Foil)  # trailing-edge info

	for j, panel in enumerate(panels[:-1]):  # loop over panels
		a, b = panel_linvortex_velocity(xi=x, panel=panel, vdir=None, midpt=False)
		V += a*gamma[j] + b*gamma[j+1]

		V_G[:, j] += a
		V_G[:, j+1] += b
	# TE source influence
	a = panel_constsource_velocity(xi=x, panel=panels[-1], vdir=None)
	f1 = a*(-0.5*tcp)
	f2 = a*0.5*tcp
	V += f1*gamma[0] + f2*gamma[-1]

	V_G[:, 0] += f1
	V_G[:, N-1] += f2
	# TE vortex influence
	a, b = panel_linvortex_velocity(
		panel=panels[-1], xi=x, vdir=None, midpt=False)
	f1 = (a+b)*(0.5*tdp)
	f2 = (a+b)*(-0.5*tdp)
	V += f1*gamma[0] + f2*gamma[-1]

	V_G[:, 0] += f1
	V_G[:, N-1] += f2
	# freestream influence
	V += Vinf*np.array([np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])

	return V, V_G


def calc_force(foil):
	"""
	
	
	
	"""
	N = foil.N + 1  # foil points
	coord = foil.geom.coord
	xref = foil.geom.xref
	panels = foil.geom.panels
	Vinf = foil.param.Vinf
	rho = foil.oper.rho
	alpha = foil.oper.alpha
	alpha_vector = np.array(
		[np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])
	qinf = 0.5 * rho * Vinf ** 2  # dynamic pressure

	# calculate the pressure coefficient at each node
	if foil.oper.viscous:
		ue = foil.glob.U[3, :]
	else:
		ue = get_ueinv(foil).T

	cp, cp_ue = get_cp(ue, foil.param)
	foil.post.cp = cp
	foil.post.cpi = get_cp(get_ueinv(foil).T, foil.param)[0]  # inviscid cp
	foil.post.ue = ue

	# Inicialización de los coeficientes:
	cl = cl_alpha = cm = cdpi = 0
	cl_ue = np.zeros(N)

	for i, panel in enumerate(panels[:-1]):
		x1 = panel.leftcoord
		x2 = panel.rightcoord
		panel_vector = panel.t*panel.len
		lh_vector = x1 - xref
		rh_vector = x2 - xref
		dx1nds = np.dot(panel_vector, lh_vector)
		dx2nds = np.dot(panel_vector, rh_vector)
		dx = -1*np.dot(panel_vector, alpha_vector)			# Proyección eje x
		dz = float(np.cross(alpha_vector, panel_vector))  # Proyección eje y

		cp1 = cp[i]
		cp2 = cp[i+1]
		panel.cpi = 0.5*(cp1 + cp2)
		cl += dx*panel.cpi

		I = [i, i+1]
		cl_ue[I] += dx*0.5*cp_ue[I]
		cl_alpha += panel.cpi*np.cross(panel_vector, alpha_vector)*np.pi/180
		cm += cp1 * dx1nds / 3 + cp1 * dx2nds / 6 + cp2 * dx1nds / 6 + cp2 * dx2nds / 3
		cdpi += dz * panel.cpi

	foil.post.cl = cl

	foil.post.cl_ue = cl_ue
	foil.post.cl_alpha = cl_alpha
	foil.post.cm = cm
	foil.post.cdpi = cdpi

	# viscous contributions
	cd = 0
	cdf = 0
	if foil.oper.viscous:

		# Squire-Young relation for total drag (extrapolates theta from end of wake)
		iw = foil.vsol.Is[2][-1]  # station at the end of the wake
		U = foil.glob.U[:, iw]
		H = U[1] / U[0]
		ue = get_uk(U[3], foil.param)[0]  # state
		cd = 2.0 * U[0] * (ue / Vinf) ** ((5 + H) / 2.0)

		# skin friction drag
		Df = 0.0
		ue1 = ue2 = cf1 = cf2 = rho1 = rho2 = 0
		for isurf in range(2):
			Is = foil.vsol.Is[isurf]
			param = build_param(foil, isurf)
			param = station_param(foil, param, Is[0])
			cf1 = 0.0  # get_cf(M.glob.U[:,Is[0]], param) # first cf value
			ue1 = 0.0  # get_uk(M.glob.U[3,Is[0]], param)
			rho1 = rho
			x1 = foil.isol.xstag
			for i in range(len(Is)):
				param = station_param(foil, param, Is[i])
				cf2 = get_cf(foil.glob.U[:, Is[i]], param)[0]  # get cf value
				ue2 = get_uk(foil.glob.U[3, Is[i]], param)[0]
				rho2 = get_rho(foil.glob.U[:, Is[i]], param)[0]
				x2 = coord[Is[i], :]
				dxv = x2 - x1
				dx = dxv[0] * np.cos(alpha) + dxv[1] * np.sin(alpha)
				Df += 0.25 * (rho1 * cf1 * ue1**2 + rho2 * cf2 * ue2**2) * dx
				cf1 = cf2
				ue1 = ue2
				x1 = x2
				rho1 = rho2
		cdf = Df / qinf

	# store results
	foil.post.cd = cd
	foil.post.cdf = abs(cdf)
	foil.post.cdp = cd - abs(cdf)

	if foil.oper.viscous:
		if foil.vsol.Xt[0, 1] ==0:
			foil.vsol.Xt[0, 1] = 1
		print("Viscous results for alpha = ", foil.oper.alpha, ": \n", "cl = ", foil.post.cl, "\n cd = ",
				foil.post.cd, "\n cdpi = ", foil.post.cdpi, "\n cdf = ", foil.post.cdf, "\n cdp = ", foil.post.cdp, "\n cm = ", foil.post.cm, "\n Xt lower = ", foil.vsol.Xt[0, 1], "\n Xt upper = ", foil.vsol.Xt[1, 1])

	else:
		print("Inviscid results for alpha = ", foil.oper.alpha, ": \n",
		      "cl = ", foil.post.cl, "\n cdp = ", foil.post.cdpi, "\n cm = ", foil.post.cm)


def get_ueinv(M):
	"""
	Computes invicid tangential velocity at every node
	INPUT
		M : mfoil structure
	OUTPUT
		ueinv : inviscid velocity at airfoil and wake (if exists) points
	DETAILS
		The airfoil velocity is computed directly from gamma
		The tangential velocity is measured + in the streamwise direction
	"""

	alpha = M.oper.alpha
	cs = np.array([np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])
	uea = M.isol.sgnue.T * np.dot(M.isol.gamref, cs)  # airfoil
	if M.oper.viscous and M.wake.N > 0:
		uew = np.dot(M.isol.uewiref, cs)  # wake
		uew[0] = uea[-1]  # ensures continuity of upper surface and wake ue
	else:
		uew = []
	ueinv = np.concatenate((uea, uew))  # airfoil/wake edge velocity
	return ueinv


def get_ueinvref(foil):
	"""
	Computes 0, 90-degree inviscid tangential velocities at every node. Uses gamref for the airfoil, uewiref for the wake (if exists)

	Input:
	- foil: airfoil class

	Output:
	- ueinvref: 0,90 inviscid tangential velocity at all points (N+Nw)x2
	
	"""
	assert foil.isol.gam, 'No inviscid solution'
	uearef = foil.isol.sgnue.T * foil.isol.gamref  # airfoil
	uewref = []
	if foil.oper.viscous and foil.wake.N > 0:
		uewref = foil.isol.uewiref  # wake
		uewref[0, :] = uearef[-1, :]  # continuity of upper surface and wake
	ueinvref = np.vstack((uearef, uewref))
	return ueinvref
