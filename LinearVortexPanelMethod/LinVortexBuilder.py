"""
    En este archivo se establecen las diferentes funciones requeridas para el cálculo de la solución no viscosa, así como
    otras que son comunes con el caso viscoso.
    
    @Author: Guillermo Peña Martínez
    @Date: 04/05/2023
"""
import numpy as np
import sys

from LinearVortexPanelMethod.InvAuxEq import *
from LinearVortexPanelMethod.StreamFuncEq import *


from Geometry import *
from LinearVortexPanelMethod.ParamInit import *


def LVSolver(Foil):

	Foil.oper.viscous = False
	# do not distinguish sign of ue if inviscid
	Foil.isol.sgnue = np.ones(Foil.N+1)
	vortex_builder(Foil, Foil.oper.alpha)
	if (Foil.oper.givencl):
		cltrim_inviscid(Foil)
	calc_force(Foil)
	Foil.glob.conv = True  # no coupled system ... convergence is guaranteed


def vortex_builder(Foil, alpha):
	"""
		Resolución del sistema de ecuaciones matricial que permite calcular la matriz que mapea los valores de las
		intensidades de los vórtices con los distintos puntos del perfil.

		INPUT
		Foil : perfil
		alpha : ángulo de ataque

		OUTPUT
		Foil.isol.vMatrix : matriz de mapeado de las intensidades de los vórtices
		Foil.isol.gamref : distribución de intensidades de vórtice para los ángulos de 0 a 90
		Foil.isol.gam : distribución de intensidades para el alpha indicado
	"""
	paneles = Foil.geom.paneles
	N = Foil.N         # number of points
	coord = Foil.geom.coord

	A = np.zeros((N+2, N+2))  # influence matrix
	rhs = np.zeros((N+2, 2))  # right-hand sides for 0,90
	_, hTE, _, tcp, tdp = TE_info(Foil)  # trailing-edge info
	nogap = (abs(hTE) < 1e-10*Foil.geom.chord)  # indicates no TE gap

	# Matriz de influencia y rhs
	for i, panel_i in enumerate(paneles):
		xi = coord[i, :]  # coordenadas de cada nodo
		for j, panel_j in enumerate(paneles[:-1]):  # Se excluye el panel del TE
			aij, bij = panel_linvortex_stream(panel_i=panel_i, panel_j=panel_j)
			A[i, j] = A[i, j] + aij
			A[i, j+1] = A[i, j+1] + bij
			A[i, -1] = -1  # El último elemento es la línea de corriente en el perfil
		# right-hand sides
		rhs[i, :] = [-xi[1], xi[0]]
		# TE influencia del source
		a = panel_constsource_stream(panel_i=panel_i, panel_j=paneles[-1])

		A[i, 0] = A[i, 0] - a*(0.5*tcp)
		A[i, N] = A[i, N] + a*(0.5*tcp)
		# TE influencia vórtice
		a, b = panel_linvortex_stream(panel_i=panel_i, panel_j=paneles[-1])
		A[i, 0] = A[i, 0] - (a+b)*(-0.5*tdp)
		A[i, N] = A[i, N] + (a+b)*(-0.5*tdp)

	# Ecuación especial en caso de no haber hueco en el TE
	if nogap:
		A[N, :] = 0
		A[N, [0, 1, 2, N-2, N-1, N]] = [1, -2, 1, -1, 2, -1]

	# Kutta condition
	A[N+1, 0] = 1
	A[N+1, N] = 1

	# Resolución del sistema
	Foil.isol.vMatrix = A
	g = np.linalg.solve(Foil.isol.vMatrix, rhs)
	# Se elimina el valor de la línea de corriente en la superficie
	Foil.isol.gamref = np.array(g[:-1, :])
	Foil.isol.gam = Foil.isol.gamref[:, 0] * np.cos(np.deg2rad(
		alpha)) + Foil.isol.gamref[:, 1] * np.sin(np.deg2rad(alpha))


def inviscid_velocity(Foil, gamma, Vinf, alpha, x):
	# returns inviscid velocity at x due to gamma (G) on panels X, and Vinf
	# INPUT
	#   X     : coordinates of N panel nodes (N-1 panels) (Nx2)
	#   G     : vector of gamma values at each airfoil node (Nx1)
	#   Vinf  : freestream speed magnitude
	#   alpha : angle of attack (degrees)
	#   x     : location of point at which velocity vector is desired
	# OUTPUT
	#   V    : velocity at the desired point (2x1)
	#   V_G  : (optional) linearization of V w.r.t. G, (2xN)
	# DETAILS
	#   Uses linear vortex panels on the airfoil
	#   Accounts for TE const source/vortex panel
	#   Includes the freestream contribution

	paneles = Foil.geom.paneles
	X = Foil.geom.coord

	N = len(X)  # número de puntos (Npaneles +1 )
	V = np.zeros(2)  # velocity

	V_G = type(object)

	V_G = np.zeros((2, N))
	_, _, _, tcp, tdp = TE_info(Foil)  # trailing-edge info
	# assume x is not a midpoint of a panel (can check for this)
	for j, panel in enumerate(paneles[:-1]):  # loop over panels
		a, b = panel_linvortex_velocity(xi=x, panel=panel, vdir=None, midpt=False)
		V += a*gamma[j] + b*gamma[j+1]

		V_G[:, j] += a
		V_G[:, j+1] += b
	# TE source influence
	a = panel_constsource_velocity(xi=x, panel=paneles[-1], vdir=None)
	f1 = a*(-0.5*tcp)
	f2 = a*0.5*tcp
	V += f1*gamma[0] + f2*gamma[-1]

	V_G[:, 0] += f1
	V_G[:, N-1] += f2
	# TE vortex influence
	a, b = panel_linvortex_velocity(
		panel=paneles[-1], xi=x, vdir=None, midpt=False)
	f1 = (a+b)*(0.5*tdp)
	f2 = (a+b)*(-0.5*tdp)
	V += f1*gamma[0] + f2*gamma[-1]

	V_G[:, 0] += f1
	V_G[:, N-1] += f2
	# freestream influence
	V += Vinf*np.array([np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])

	return V, V_G


def calc_force(Foil):
	"""
	
	
	
	"""
	N = Foil.N + 1  # Número de puntos en el perfil
	coord = Foil.geom.coord
	xref = Foil.geom.xref
	paneles = Foil.geom.paneles
	Vinf = Foil.param.Vinf
	rho = Foil.oper.rho
	alpha = Foil.oper.alpha
	alpha_vector = np.array(
		[np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])
	qinf = 0.5 * rho * Vinf ** 2  # dynamic pressure

	# calculate the pressure coefficient at each node
	if Foil.oper.viscous:
		ue = Foil.glob.U[3, :]
	else:
		ue = get_ueinv(Foil).T

	cp, cp_ue = get_cp(ue, Foil.param)
	Foil.post.cp = cp
	Foil.post.cpi = get_cp(get_ueinv(Foil).T, Foil.param)[0]  # inviscid cp
	Foil.post.ue = ue

	# Inicialización de los coeficientes:
	cl = cl_alpha = cm = cdpi = 0
	cl_ue = np.zeros(N)

	for i, panel in enumerate(paneles[:-1]):
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

	Foil.post.cl = cl

	Foil.post.cl_ue = cl_ue
	Foil.post.cl_alpha = cl_alpha
	Foil.post.cm = cm
	Foil.post.cdpi = cdpi

	# viscous contributions
	cd = 0
	cdf = 0
	if Foil.oper.viscous:

		# Squire-Young relation for total drag (extrapolates theta from end of wake)
		iw = Foil.vsol.Is[2][-1]  # station at the end of the wake
		U = Foil.glob.U[:, iw]
		H = U[1] / U[0]
		ue = get_uk(U[3], Foil.param)[0]  # state
		cd = 2.0 * U[0] * (ue / Vinf) ** ((5 + H) / 2.0)

		# skin friction drag
		Df = 0.0
		ue1 = ue2 = cf1 = cf2 = rho1 = rho2 = 0
		for isurf in range(2):
			Is = Foil.vsol.Is[isurf]
			param = build_param(Foil, isurf)
			param = station_param(Foil, param, Is[0])
			cf1 = 0.0  # get_cf(M.glob.U[:,Is[0]], param) # first cf value
			ue1 = 0.0  # get_uk(M.glob.U[3,Is[0]], param)
			rho1 = rho
			x1 = Foil.isol.xstag
			for i in range(len(Is)):
				param = station_param(Foil, param, Is[i])
				cf2 = get_cf(Foil.glob.U[:, Is[i]], param)[0]  # get cf value
				ue2 = get_uk(Foil.glob.U[3, Is[i]], param)[0]
				rho2 = get_rho(Foil.glob.U[:, Is[i]], param)[0]
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
	Foil.post.cd = cd
	Foil.post.cdf = abs(cdf)
	Foil.post.cdp = cd - abs(cdf)

	if Foil.oper.viscous:
		if Foil.vsol.Xt[0, 1] ==0:
			Foil.vsol.Xt[0, 1] = 1
		print("Resultados viscosos para alpha = ", Foil.oper.alpha, ": \n", "cl = ", Foil.post.cl, "\n cd = ",
				Foil.post.cd, "\n cdpi = ", Foil.post.cdpi, "\n cdf = ", Foil.post.cdf, "\n cdp = ", Foil.post.cdp, "\n cm = ", Foil.post.cm, "\n Xt intrados = ", Foil.vsol.Xt[0, 1], "\n Xt extrados = ", Foil.vsol.Xt[1, 1])

	else:
		print("Resultados no viscosos para alpha = ", Foil.oper.alpha, ": \n",
		      "cl = ", Foil.post.cl, "\n cdp = ", Foil.post.cdpi, "\n cm = ", Foil.post.cm)


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
	# assert M.isol.gam.size > 0, 'No inviscid solution'
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


def cltrim_inviscid(M):
	"""
	Trims inviscid solution to prescribed target cl, using alpha
	INPUT
		M : mfoil structure
	OUTPUT
		inviscid vorticity distribution is computed for a given cl
	DETAILS
		Iterates using cl_alpha computed in post-processing
		Accounts for cl_ue in total derivative
	"""
	for i in range(15):  # trim iterations
		alpha = M.oper.alpha  # current angle of attack
		calc_force(M)  # calculate current cl and linearization
		R = M.post.cl - M.oper.cltgt
		if np.linalg.norm(R) < 1e-10:
			break
		sc = np.array([-np.sin(np.deg2rad(alpha)),
                 np.cos(np.deg2rad(alpha))]) * np.pi/180
		cl_a = M.post.cl_alpha + M.post.cl_ue*(M.isol.gamref*sc)  # total deriv
		dalpha = -R/cl_a
		M.oper.alpha = alpha + min(max(dalpha, -2), 2)
		if i >= 14:
			print('** inviscid cl trim not converged **')
	M.isol.gam = M.isol.gamref * \
            np.array([np.cos(np.deg2rad(M.oper.alpha)),
                      np.sin(np.deg2rad(M.oper.alpha))])


def get_ueinvref(M):
	"""
	Computes 0, 90-degree inviscid tangential velocities at every node.

	Args:
	- M: mfoil structure

	Returns:
	- ueinvref: 0,90 inviscid tangential velocity at all points (N+Nw)x2

	Details:
	- Uses gamref for the airfoil, uewiref for the wake (if exists)
	"""
	assert M.isol.gam, 'No inviscid solution'
	uearef = M.isol.sgnue.T * M.isol.gamref  # airfoil
	uewref = []
	if M.oper.viscous and M.wake.N > 0:
		uewref = M.isol.uewiref  # wake
		uewref[0, :] = uearef[-1, :]  # continuity of upper surface and wake
	ueinvref = np.vstack((uearef, uewref))
	return ueinvref
