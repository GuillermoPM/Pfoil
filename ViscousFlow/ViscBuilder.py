"""
## Viscous builder
This code was mainly taken from the MFOIL and adapted to Python and modified to fullfill the requirements.
"""

import numpy as np
import scipy as sp
from Geometry import *
from ViscousFlow.ResidualsBuilder import *
from ViscousFlow.GlobalSolution import *


def get_distributions(Foil):
	"""
	Computes the various distributions and stores them in the Foil class.
	"""

	# Quantities already in the global state
	Foil.post.th = Foil.glob.U[0, :]  # theta
	Foil.post.ds = Foil.glob.U[1, :]  # delta*
	Foil.post.sa = Foil.glob.U[2, :]  # amp or ctau
	# compressible edge velocity
	Foil.post.ue = get_uk(Foil.glob.U[3, :], Foil.param)[0]
	Foil.post.uei = get_ueinv(Foil)  # compressible inviscid edge velocity

	# Derived viscous quantities
	N = Foil.glob.Nsys
	cf = np.zeros(N)
	Ret = np.zeros(N)
	Hk = np.zeros(N)
	for isurf in range(3):   # Loop over surfaces
		Is = Foil.vsol.Is[isurf]  # Surface point indices
		param = build_param(Foil, isurf)  # Get parameter structure
		for i in range(len(Is)):  # Loop over points
			j = Is[i]
			Uj = Foil.glob.U[:, j]
			param = station_param(Foil, param, j)
			uk = get_uk(Uj[3], param)[0]  # Corrected edge speed
			cfloc = get_cf(Uj, param)[0]  # Local skin friction coefficient
			cf[j] = cfloc * uk**2/param.Vinf**2  # Free-stream-based cf
			Ret[j] = get_Ret(Uj, param)[0]  # Re_theta
			Hk[j] = get_Hk(Uj, param)[0]  # Kinematic shape factor
	Foil.post.cf = cf
	Foil.post.Ret = Ret
	Foil.post.Hk = Hk


def build_wake(Foil):
	"""
	Builds wake panels from the inviscid solution

	INPUTS:
		Foil: mfoil class with a valid inviscid solution (gam)

	OUTPUTS:
		Foil.wake.N: Nw, the number of wake points
		Foil.wake.x: coordinates of the wake points (2xNw)
		Foil.wake.s: s-values of wake points (continuation of airfoil) (1xNw)
		Foil.wake.t: tangent vectors at wake points (2xNw)

	DETAILS:
		Constructs the wake path through repeated calls to inviscid_velocity
		Uses a predictor-corrector method
		Point spacing is geometric; prescribed wake length and number of points
	"""

	assert Foil.isol.gam is not None, "No inviscid solution"
	N = Foil.N + 1  # number of points on the airfoil
	Vinf = Foil.oper.Vinf  # freestream speed
	Nw = int(np.ceil(N/10 + 10*Foil.geom.wakelen))  # number of points on wake
	S = Foil.geom.s  # airfoil S values

	ds1 = 0.5*(S[1]-S[0] + S[-1]-S[-2])  # first nominal wake panel size
	sv = space_geom(ds1, Foil.geom.wakelen, Nw)  # geometrically-spaced points
	xyw = np.zeros((2, Nw))  # arrays of x,y points and tangents on wake
	tw = xyw.copy()
	xy1 = Foil.geom.coord[0, :]
	xyN = Foil.geom.coord[-1, :]  # airfoil TE points

	TE_panel = Foil.geom.paneles[-1]
	t = np.flip(-1*TE_panel.t*TE_panel.len, axis = 0)
	n = np.flip(TE_panel.n*TE_panel.len, axis = 0)

	xyte = TE_panel.midpt  # TE midpoint

	assert t[0] > 0, "Wrong wake direction; ensure airfoil points are CCW"
	xyw[:, 0] = xyte + 1e-5*t*1  # first wake point, just behind TE
	sw = S[-1] + sv  # s-values on wake, measured as continuation of the airfoil

	v1 = v2 = np.ones(2, dtype=object)
	# loop over rest of wake
	for i in range(Nw-1):
		v1 = inviscid_velocity(Foil, Foil.isol.gam, Vinf, Foil.oper.alpha, xyw[:, i])[0]
		v1 = v1/np.linalg.norm(v1)  # normalized
		tw[:, i] = v1
		# forward Euler (predictor) step
		xyw[:, i+1] = xyw[:, i] + (sv[i+1]-sv[i])*v1
		v2 = inviscid_velocity(Foil, Foil.isol.gam, Vinf, Foil.oper.alpha, xyw[:, i+1])[0]
		v2 = v2/np.linalg.norm(v2)  # normalized
		tw[:, i+1] = v2
		xyw[:, i+1] = xyw[:, i] + (sv[i+1]-sv[i])*0.5*(v1+v2)  # corrector step

	# determine inviscid ue in the wake, and 0,90
	uewi = np.zeros(Nw)
	uewiref = np.zeros((Nw, 2))

	v = np.empty((2, 1))
	for i in range(Nw):
		v = inviscid_velocity(Foil, Foil.isol.gam, Vinf, Foil.oper.alpha, xyw[:, i])[0]
		uewi[i] = np.dot(v.T, tw[:, i])
		v = inviscid_velocity(Foil,  Foil.isol.gamref[:, 0], Vinf, 0, xyw[:, i])[0]
		uewiref[i, 0] = np.dot(v.T, tw[:, i])
		v = inviscid_velocity(Foil, Foil.isol.gamref[:, 1], Vinf, 90, xyw[:, i])[0]
		uewiref[i, 1] = np.dot(v.T, tw[:, i])

	wakepanels = np.empty(Nw-1, dtype=object)
	for i in range(Nw-1):
		wakepanels[i] = Wakepanel(coordmin=xyw[:, i], coordmax=xyw[:, i+1], i=i+1)

	# set values
	Foil.wake.N = Nw
	Foil.wake.wpaneles = wakepanels
	Foil.wake.x = xyw
	Foil.wake.s = sw
	Foil.wake.t = tw
	Foil.isol.uewi = uewi
	Foil.isol.uewiref = uewiref


def wake_init(Foil, ue):
	"""
	Initializes the first point of the wake, using data in M.glob.U
	
	Input:
	- Foil: Foil class with inviscid solution
	- ue: edge velocity at the wake point
	
	Returns:
	- Uw: 4x1 state vector at the wake point
	"""
	iw = Foil.vsol.Is[2][0]  # first wake index
	Uw = Foil.glob.U[:, iw]  # state vector at the first wake index
	R = wake_sys(Foil, Foil.param)[0]  # construct the wake system
	Uw[0:3] = Uw[0:3] - R  # solve the wake system
	Uw[3] = ue  # assign the edge velocity to the wake state vector
	return Uw


def wake_sys(Foil, param):
	"""
	Builds residual system corresponding to wake initialization

	INPUT
	  param  : parameters

	OUTPUT
	  R   : 3x1 residual vector for th, ds, sa
	  R_U : 3x12 residual linearization, as three 3x4 blocks
	  J   : indices of the blocks of U in R_U (lower, upper, wake)
	"""

	il = Foil.vsol.Is[0][-1]
	Ul = Foil.glob.U[:, il]  # lower surface TE index, state
	iu = Foil.vsol.Is[1][-1]
	Uu = Foil.glob.U[:, iu]  # upper surface TE index, state
	iw = Foil.vsol.Is[2][0]
	Uw = Foil.glob.U[:, iw]  # first wake index, state
	hTE = trailing_specs(Foil)[1]  # trailing-edge gap is hTE

	# Obtain wake shear stress from upper/lower; transition if not turb
	param.turb = True
	param.wake = False  # calculating turbulent quantities right before wake
	if Foil.vsol.turb[il]:
		ctl = Ul[2]
		ctl_Ul = np.array([0, 0, 1, 0])  # already turb; use state
	else:
		ctl, ctl_Ul = get_cttr(Ul, param)  # transition shear stress, lower
	if Foil.vsol.turb[iu]:
		ctu = Uu[2]
		ctu_Uu = np.array([0, 0, 1, 0])  # already turb; use state
	else:
		ctu, ctu_Uu = get_cttr(Uu, param)  # transition shear stress, upper
	thsum = Ul[0] + Uu[0]  # sum of thetas
	ctw = (ctl*Ul[0] + ctu*Uu[0])/thsum  # theta-average
	ctw_Ul = (ctl_Ul*Ul[0] + (ctl - ctw)*np.array([1, 0, 0, 0]))/thsum
	ctw_Uu = (ctu_Uu*Uu[0] + (ctu - ctw)*np.array([1, 0, 0, 0]))/thsum

	# residual; note, delta star in wake includes the TE gap, hTE
	R = np.array([Uw[0]-(Ul[0]+Uu[0]), Uw[1]-(Ul[1]+Uu[1]+hTE), Uw[2]-ctw])
	J = np.array([il, iu, iw])  # R depends on states at these nodes
	R_Ul = np.vstack((-np.eye(2, 4), -ctw_Ul))
	R_Uu = np.vstack((-np.eye(2, 4), -ctw_Uu))
	R_U = np.hstack((R_Ul, R_Uu, np.eye(3, 4)))

	param.wake = True
	return R, R_U, J


def stagpoint_find(Foil):
	"""
	finds the LE stagnation point on the airfoil (using inviscid solution)

	INPUTS
	  Foil  : mfoil class with inviscid solution, gam

	OUTPUTS
	  Foil.isol.sstag   : scalar containing s value of stagnation point
	  Foil.isol.sstag_g : linearization of sstag w.r.t gamma (1xN)
	  Foil.isol.Istag   : [i,i+1] node indices before/after stagnation (1x2)
	  Foil.isol.sgnue   : sign conversion from CW to tangential velocity (1xN)
	  Foil.isol.xi      : distance from stagnation point at each node (1xN)
	
	"""

	N = Foil.N  # number of points on the airfoil
	J = np.where(Foil.isol.gam > 0)[0]
	# assert (J.size != 0, 'no stagnation point')
	I = np.array([J[0]-1, J[0]])
	G = Foil.isol.gam[I]
	S = Foil.geom.s[I]
	Foil.isol.Istag = I  # indices of neighboring gammas

	ue2, ue1 = G[1], -1*G[0]
	s2, s1 = S[1], S[0]
	s_stag = (ue1 * s2 + ue2 * s1)/(ue1 + ue2) # 

	den = (ue2 + ue1)
	w1 = ue2/den; w2 = ue1/den


	Foil.isol.sstag = s_stag
	Foil.isol.xstag = np.dot(Foil.geom.coord[I, :].T, np.array([w1, w2]))  # x location
	st_g1 = G[1]*(S[0]-S[1])/(den*den)
	Foil.isol.sstag_g = np.array([st_g1, -st_g1])
	sgnue = -1*np.ones(N+1)
	sgnue[J] = 1  # upper/lower surface sign
	Foil.isol.sgnue = sgnue
	Foil.isol.xi = np.concatenate((abs(Foil.geom.s-Foil.isol.sstag), Foil.wake.s-Foil.isol.sstag))


def identify_surfaces(Foil):
	"""
	Identifies lower/upper/wake surfaces.

	Parameters:
	M (mfoil class): mfoil class with stagnation point found

	Returns:
	None. The function updates M.vsol.Is with a cell array of node indices
	for lower(1), upper(2), wake(3).
	"""
	Foil.vsol.Is = [list(range(Foil.isol.Istag[0], -1, -1)), list(range(
		Foil.isol.Istag[1], Foil.N+1)), list(range(Foil.N+1, Foil.N+Foil.wake.N+1))]


def space_geom(dx0, L, Np):

	N = Np - 1  # number of intervals

	d = L /( dx0*10)
	a = N * (N - 1.) * (N - 2.) / 6.
	b = N * (N - 1.) / 2.
	c = N - d

	disc = max(b * b - 4. * a * c, 0.)

	r = 1. + (-b + np.sqrt(disc)) / (2. * a)
	for k in range(100):
		R = r**N - 1. - d * (r - 1.)
		R_r = N * r**(N - 1) - d
		dr = -R / R_r
		if abs(dr) < 1e-6:
			break
		r = r + dr
	cumsum = np.cumsum(dx0 * np.power(r, np.arange(N))*10)
	x = np.insert(cumsum, 0, 0.0)


	return x


def set_wake_gap(Foil):
	"""
	Sets height (delta*) of dead air in wake.

	Parameters:
	Foil: Foil class with wake built and stagnation point found

	Details:
	Uses cubic function to extrapolate the TE gap into the wake.
	See Drela, IBL for Blunt Trailing Edges, 1989, 89-2166-CP.
	"""
	_, hTE, dtdx, _, _ = trailing_specs(Foil)
	flen = 2.5  # length-scale factor
	dtdx = min(max(dtdx, -3./flen), 3./flen)  # clip TE thickness slope
	Lw = flen * hTE
	wgap = np.zeros(Foil.wake.N)
	for i in range(Foil.wake.N):
		xib = (Foil.isol.xi[Foil.N+i] - Foil.isol.xi[Foil.N]) / Lw
		if xib <= 1:
			wgap[i] = hTE * (1 + (2 + flen*dtdx) * xib) * (1 - xib)**2
	Foil.vsol.wgap = wgap


def calc_ue_m(M):


	assert M.isol.gam.size != 0, 'No inviscid solution'
	N, Nw = M.N + 1, M.wake.N  # number of points on the airfoil/wake
	paneles = M.geom.paneles
	wakepanels = M.wake.wpaneles
	assert Nw > 0, 'No wake'

	# Cgam = d(wake uei)/d(gamma)   [Nw x N]   (not sparse)
	Cgam = np.zeros((Nw, N))

	for i, w_panel in enumerate(wakepanels):
		v_G = inviscid_velocity(M, M.isol.gam, 0, 0, w_panel.leftcoord)[1]
		Cgam[i, :] = v_G[0, :] * w_panel.t[0] + v_G[1, :] * w_panel.t[1]

	B = np.zeros((N+1, N+Nw-2))  # note, N+Nw-2 = # of panels
	# loop over points on the airfoil
	for i, panel_i in enumerate(paneles):
		for j, panel_j in enumerate(paneles[:-1]):  # loop over airfoil panels
			B[i, j] = panel_constsource_stream(panel_j=panel_j, panel_i = panel_i)
		for j, panel_j in enumerate(wakepanels):  # loop over wake panels

			Xj = np.column_stack([panel_j.leftcoord, panel_j.midpt, panel_j.rightcoord])
			lefthalf = Wakepanel(panel_j.leftcoord, panel_j.midpt, j)
			righthalf = Wakepanel(panel_j.midpt, panel_j.rightcoord, j)
			if j == (Nw-2):
				# ghost extension at last point
				Xj[:, 2] = 2 * Xj[:, 2] - Xj[:, 1]
			a, b = panel_linsource_stream(panel_j = lefthalf, panel_i=panel_i)  # left half panel
			if j > 0:
				B[i, N-1+j] += 0.5 * a + b
				B[i, N-1+j-1] += 0.5 * a
			else:
				B[i, N-1+j] += b
			a, b = panel_linsource_stream(panel_j = righthalf, panel_i=panel_i)  # right half panel
			B[i, N-1 + j] += a + 0.5*b
			if j < Nw - 2:
				B[i, N-1 + j+1] += 0.5*b
			else:
				B[i, N - 1 + j] += 0.5*b

	# this has N+1 rows, but the last one is zero
	Bp = -1*np.linalg.solve(M.isol.vMatrix, B)
	Bp = Bp[:-1, :]  # trim the last row

	# Csig = d(wake uei)/d(source) [Nw x (N+Nw-2)]  (not sparse)
	Csig = np.zeros((Nw, N+Nw-2))
	for i,w_panel in enumerate(wakepanels):
		xi = M.wake.x[:, i]
		ti = M.wake.t[:, i]  # point, tangent on wake

		# constant sources on airfoil panels
		for j, panel_j in enumerate(paneles[:-1]):
			Csig[i, j] = panel_constsource_velocity(xi, panel_j, ti)

		# piecewise linear sources across wake panel halves (else singular)
		for j in range(Nw):  # loop over wake points
			I = [max(j-1, 0), j, min(j+1, Nw-1)]  # left, self, right
			Xj = M.wake.x[:, I]  # point coordinates
			Xj[:, 0] = 0.5 * (Xj[:, 0] + Xj[:, 1])  # left midpoint
			Xj[:, 2] = 0.5 * (Xj[:, 1] + Xj[:, 2])  # right midpoint
			if j == Nw-1:
				Xj[:, 2] = 2*Xj[:, 1] - Xj[:, 0]  # ghost extension at last point
			d1 = np.linalg.norm(Xj[:, 1] - Xj[:, 0])  # left half-panel length
			d2 = np.linalg.norm(Xj[:, 2] - Xj[:, 1])  # right half-panel length
			if i == j:
				if j == 0:  # first point: special TE system (three panels meet)
					# lower surface panel length
					dl = paneles[0].len
					# upper surface panel length
					du = paneles[-2].len
					# lower panel effect
					Csig[i, 0] += (0.5/np.pi) * (np.log(dl/d2) + 1)
					Csig[i, N-1] += (0.5/np.pi) * (np.log(du/d2) + 1)  # upper panel effect
					Csig[i, N-1+1] -= 0.5/np.pi  # self effect
				# last point: no self effect of last pan (ghost extension)
				elif j == Nw-1:
					Csig[i, N-1+j-1] += 0  # hence the 0
				else:  # all other points
					aa = (0.25/np.pi) * np.log(d1/d2)
					Csig[i, N-1+j-1] += aa + 0.5/np.pi
					Csig[i, N-1+j] += aa - 0.5/np.pi
			else:
				if j == 0:
					# First point only has a half panel on the right
					a, b = panel_linsource_velocity(Xj[:, [1, 2]], xi, ti)
					Csig[i, N-1+1] += b  # right half panel effect
					Csig[i, 0] += a      # lower airfoil panel effect
					Csig[i, N-1] += a    # upper airfoil panel effect
				elif j == Nw-1:
					# Last point has a constant source ghost extension
					a = panel_constsource_velocity2(Xj[:, [0, 2]], xi, ti)
					Csig[i, -1] += a  # full const source panel effect
				else:
					# All other points have a half panel on left and right
					a1, b1 = panel_linsource_velocity(
						Xj[:, [0, 1]], xi, ti)  # left half-panel ue contrib
					a2, b2 = panel_linsource_velocity(
						Xj[:, [1, 2]], xi, ti)  # right half-panel ue contrib
					Csig[i, N-1+j-1] += a1 + 0.5*b1 # type: ignore 
					Csig[i, N-1+j] += 0.5*a2 + b2  # type: ignore
	Dw = np.dot(Cgam, Bp) + Csig
	Dw[0, :] = Bp[-1, :]  # ensure first wake point has same ue as TE
	M.vsol.ue_sigma = np.concatenate((Bp, Dw))  # store combined matrix

	# build ue_m from ue_sigma, using sgnue
	rebuild_ue_m(M)


def rebuild_ue_m(Foil):
	"""
	Rebuilds ue_m matrix after stagnation panel change (new sgnue)
	
	INPUT
		M : mfoil class with calc_ue_m already called once
		
	OUTPUT
		M.vsol.sigma_m : d(source)/d(mass) matrix, for computing source strengths
		M.vsol.ue_m    : d(ue)/d(mass) matrix, for computing tangential velocity
	
	DETAILS
		"mass" flow refers to area flow (we exclude density)
		sigma_m and ue_m return values at each node (airfoil and wake)
		airfoil panel sources are constant strength
		wake panel sources are two-piece linear
	"""
	assert Foil.vsol.ue_sigma is not None, "Need ue_sigma to build ue_m"

	# Dp = d(source)/d(mass)  [(N+Nw-2) x (N+Nw)]  (sparse)
	N, Nw = Foil.N+1, Foil.wake.N  # number of points on the airfoil/wake
	# Dp = sp.sparse.csr_matrix((2 * (N + Nw - 1), N-1 + Nw))
	Dp = sp.sparse.lil_matrix((N + Nw - 2, N + Nw), dtype=float)
	# Dp = np.array((N+Nw-1,N+Nw+1),dtype=object)
	for i in range(N-1):
		ds = Foil.geom.s[i + 1] - Foil.geom.s[i]
		# Note, at stagnation: ue = K*s, dstar = const, m = K*s*dstar
		# sigma = dm/ds = K*dstar = m/s (separate for each side, +/-)
		Dp[i, [i, i + 1]] = Foil.isol.sgnue[[i, i + 1]] * np.array([-1, 1]) / ds
	for i in range(Nw - 1):
		ds = Foil.wake.s[i + 1] - Foil.wake.s[i]
		Dp[N - 1 + i, [N + i, N + i + 1]] = np.array([-1, 1]) / ds

	Foil.vsol.sigma_m = Dp

	# sign of ue at all points (wake too)
	sgue = np.hstack([Foil.isol.sgnue, np.ones(Nw)])

	# ue_m = ue_sigma * sigma_m [(N+Nw) x (N+Nw)] (not sparse)
	Foil.vsol.ue_m = sp.sparse.diags(
		sgue, 0, (N + Nw, N + Nw)) @ Foil.vsol.ue_sigma @ Foil.vsol.sigma_m


def init_boundary_layer(M):
	"""
	Initializes the boundary layer both on foil and wake using the given edge velocity.

	INPUT
	  The edge velocity field ue must be filled in on the airfoil and wake

	OUTPUT
	  The state vector U is filled in Foil.glob.U
	"""

	Hmaxl = 3.8  # above this shape param value, laminar separation occurs
	Hmaxt = 2.5  # above this shape param value, turbulent separation occurs

	ueinv = get_ueinv(M)  # get inviscid velocity
	M.glob.Nsys = M.N + 1 + M.wake.N
	M.glob.U = np.zeros((4, M.glob.Nsys))
	M.vsol.turb = np.zeros((M.glob.Nsys, 1))


	for side in range(3):  # loop over surfaces

		print(f'\nSide is = {side}:\n')

		Is = M.vsol.Is[side]  # surface point indices
		xi = M.isol.xi[Is]  # distance from LE stag point
		ue = ueinv[Is]  # edge velocities
		N = len(Is)  # number of points
		U = np.zeros((4, N))  # states at all points: [th, ds, sa, ue]
		Aux = np.zeros((1, N))  # auxiliary data at all points: [wgap]

		# ensure edge velocities are not tiny
		uemax = np.max(np.abs(ue))
		ue = np.maximum(ue, 1e-8 * uemax)

		# get parameter structure
		param = build_param(M, side)

		# # set auxiliary data
		if side == 2:
			Aux[0, :] = M.vsol.wgap

		# initialize state at first point
		i0 = 0
		if side < 2:
			# Solve for the stagnation state (Thwaites initialization + Newton)
			if xi[0] < 1e-8 * xi[-1]:
				K = ue[1] / xi[1]
				hitstag = True
			else:
				K = ue[0] / xi[0]
				hitstag = False

			th, ds = thwaites_init(K, param.mu0 / param.rho0)
			xst = 1e-6  # small but nonzero
			Ust = np.array([th, ds, 0, K * xst])

			nNewton = 20
			for iNewton in range(nNewton):
				# call residual at stagnation
				param.turb = False
				param.simi = True  # similarity station flag
				R, R_U, _ = residual_station(param, np.array(
					[xst, xst]), np.array([Ust, Ust]).T, np.zeros((1, 2)))
				param.simi = False
				if np.linalg.norm(R) < 1e-10:
					break
				ID = np.array([0, 1, 2])
				A = R_U[:, ID + 4] + R_U[:, ID]
				b = -1*R # type: ignore 
				dU = np.concatenate([np.linalg.solve(A, b), [0]])

				# under-relaxation
				dm = max([abs(dU[0]/Ust[0]), abs(dU[1]/Ust[1])])
				omega = 1
				if dm > 0.2:
					omega = 0.2 / dm
				dU *= omega
				Ust += dU
			if hitstag:
				U[:, 0] = Ust
				U[3, 0] = ue[0]
				i0 = 1
			U[:, i0] = Ust
			U[3, i0] = ue[i0]
		else:  # wake
			Aux[0, :] = M.vsol.wgap
			U[:, 0] = wake_init(M, ue[0])  # initialize wake state properly
			param.turb = True  # force turbulent in wake if still laminar
			param.wake = True
			M.vsol.turb[Is[0]] = True  # wake starts turbulent

		tran = False
		i = i0 + 1
		ct = Hktgt = 0
		while i < N:
			Ip = [i-1, i]  # two points involved in the calculation
			U[:, i] = U[:, i-1]
			U[3, i] = ue[i]  # guess = same state, new ue

			if tran:  # set shear stress at transition interval
				ct = get_cttr(U[:, i], param)[0]
				U[2, i] = ct

			M.vsol.turb[Is[i]] = tran or param.turb  # flag node i as turbulent
			direct = True  # default is direct mode
			nNewton = 30
			iNswitch = 12
			iNewton = 1

			R_U = np.array([[], []])
			R = np.array([])

			for iNewton in range(1, nNewton+1):
				# call residual at this station
				if tran:  # we are at transition
					print(param, 4, f"i={i}, residual_transition (iNewton={iNewton})\n")
					try:
						R, R_U, _ = residual_transition(M, param, xi[Ip], U[:, Ip], Aux[:, Ip])

					except:
						print("Transition calculation failed in BL init. Continuing.")
						M.vsol.xt = 0.5 * sum(xi[Ip])

						U[:, i] = U[:, i-1]
						U[2, i] = ct
						U[3, i] = ue[i]
						R = 0  # so we move on
				else:
					print(param, 4, f"i={i}, residual_station (iNewton={iNewton})\n")
					R, R_U, _ = residual_station(param, xi[Ip], U[:, Ip], Aux[:, Ip])

				if np.linalg.norm(R) < 1e-10:
					break

				if direct:  # direct mode => ue is prescribed => solve for th, ds, sa
					ID = np.array([0, 1, 2])
					A = R_U[:, ID+4]
					b = -1*R  # type: ignore
					dU = np.concatenate((np.linalg.solve(A, b), [0]))
				else:  # inverse mode => Hk is prescribed
					Hk, Hk_U = get_Hk(U[:, i], param)
					A = np.vstack((R_U[:, 4:], Hk_U))
					b = np.concatenate((-1*R, [Hktgt-Hk])) # type: ignore 
					dU = np.linalg.solve(A, b)

				# under-relaxation
				dm = max(np.abs([dU[0]/U[0, i-1], dU[1]/U[1, i-1]]))
				if not direct:
					dm = max(dm, np.abs(dU[3]/U[3, i-1]))
				if param.turb:
					dm = max(dm, np.abs(dU[2]/U[2, i-1]))
				elif direct:
					dm = max(dm, np.abs(dU[2]/10))

				omega = 1
				if dm > 0.3:
					omega = 0.3/dm
				dU *= omega

				# trial update
				Ui = U[:, i] + dU

				# clip extreme values
				if param.turb:
					Ui[2] = max(min(Ui[2], 0.3), 1e-7)

				# check if about to separate
				Hmax = Hmaxl
				if param.turb:
					Hmax = Hmaxt
				Hk, _ = get_Hk(Ui, param)

				if direct and (Hk > Hmax or iNewton > iNswitch):
					# no update; need to switch to inverse mode: prescribe Hk
					direct = False
					print('** switching to inverse: i={}, iNewton={}'.format(i, iNewton))
					Hk, _ = get_Hk(U[:, i-1], param)
					Hkr = (xi[i] - xi[i-1]) / U[0, i-1]

					if param.wake:
						H2 = Hk
						for k in range(6):
							H2 = H2 - (H2 + 0.03 * Hkr * (H2 - 1) ** 3 - Hk) / \
								(1 + 0.09 * Hkr * (H2 - 1) ** 2)
						Hktgt = max(H2, 1.01)
					elif param.turb:
						Hktgt = Hk - 0.15 * Hkr  # turb: decrease in Hk
					else:
						Hktgt = Hk + 0.03 * Hkr  # lam: increase in Hk

					if not param.wake:
						Hktgt = max(Hktgt, Hmax)
					if iNewton > iNswitch:
						U[:, i] = U[:, i-1]
						U[3, i] = ue[i]  # reinit
				else:
					U[:, i] = Ui  # take the update

			if (iNewton >= nNewton):
				print(param, 1, '** BL init not converged: is=%d, i=%d **\n', side, i)
				# extrapolate values
				U[:, i] = U[:, i-1]
				if tran:
					U[2, i] = ct
				U[3, i] = ue[i]
				if (side < 2):
					U[0, i] = U[0, i-1]*(xi[i]/xi[i-1])**0.5
					U[1, i] = U[1, i-1]*(xi[i]/xi[i-1])**0.5
				else:
					rlen = (xi[i]-xi[i-1])/(10*U[2, i-1])
					U[1, i] = (U[1, i-1] + U[0, i-1]*rlen)/(1+rlen)

				# check for transition
			if (not param.turb) and (not tran) and (U[2, i] > param.ncrit):
				print(param, 2, 'Identified transition at (is=%d, i=%d): n=%.5f, ncrit=%.5f\n',
					  side, i, U[2, i], param.ncrit)
				tran = True
				continue  # redo station with transition

			if (tran):
				store_transition(M, side, i)  # store transition location
				param.turb = True
				tran = False
			i += 1  # next point
		M.glob.U[:, Is] = U


def thwaites_init(K, nu):
	"""
	Uses Thwaites correlation to initialize first node in stagnation point flow
	
	Args:
	K (float): Stagnation point constant
	nu (float): Kinematic viscosity
	
	Returns:
	th (float): Momentum thickness
	ds (float): Displacement thickness
	"""

	# Compute momentum thickness using Thwaites correlation
	th = np.sqrt(0.45 * nu / (6.0 * K))

	# Compute displacement thickness
	ds = 2.2 * th

	return th, ds


def store_transition(M, side, i):
	"""
	stores xi and x transition locations using current M.vsol.xt
	INPUT
	  is,i : side,station number
	OUTPUT
	  M.vsol.Xt stores the transition location s and x values
	"""
	xt = M.vsol.xt
	i0 = M.vsol.Is[side][i-1]
	i1 = M.vsol.Is[side][i]  # pre/post transition nodes
	xi0 = M.isol.xi[i0]
	xi1 = M.isol.xi[i1]  # xi (s) locations at nodes
	assert (i0 <= M.N) and (
		i1 <= M.N), 'Can only store transition on airfoil'
	x0 = M.geom.coord[i0, 0]
	x1 = M.geom.coord[i1, 0]  # x locations at nodes
	if (xt < xi0) or (xt > xi1):
		print('Warning: transition ({:.3f}) off interval ({:.3f},{:.3f})!'.format(
			xt, xi0, xi1))
	M.vsol.Xt[side, 0] = xt  # xi location
	M.vsol.Xt[side, 1] = x0 + (xt-xi0)/(xi1-xi0)*(x1-x0)  # x location
	slu = ['lower', 'upper']
	print('  transition on {} side at x={:.5f}'.format(
		slu[side], M.vsol.Xt[side, 1]))


def stagpoint_move(M):
	"""
	moves the LE stagnation point on the airfoil using the global solution ue
	INPUT
	  M  : mfoil class with a valid solution in M.glob.U
	OUTPUT
	  New sstag, sstag_ue, xi in M.isol
	  Possibly new stagnation panel, Istag, and hence new surfaces and matrices
	"""

	N = M.N  # number of points on the airfoil
	I = M.isol.Istag  # current adjacent node indices
	ue = M.glob.U[3, :].reshape(-1, 1)  # edge velocity
	sstag0 = M.isol.sstag  # original stag point location
	newpanel = True  # are we moving to a new panel?

	if ue[I[1]] < 0:
		# move stagnation point up (larger s, new panel)
		print('  Moving stagnation point up')
		J = np.where(ue[I[1]:] > 0)[0]
		I2 = J[0] + I[1]
		for j in range(I[1], I2):
			ue[j] = -ue[j]
		I = [I2 - 1, I2]  # new panel
	elif ue[I[0]] < 0:
		# move stagnation point down (smaller s, new panel)
		print('  Moving stagnation point down')
		J = np.where(ue[I[0]::-1] > 0)[0]
		I1 = I[0] - J[0]
		for j in range(I1 + 1, I[0] + 1):
			ue[j] = -ue[j]
		I = [I1, I1 + 1]  # new panel
	else:
		newpanel = False  # staying on the current panel

	# move point along panel
	ues = ue[I]
	S = M.geom.s[I]
	assert (ues[0] > 0 and ues[1] > 0), 'stagpoint_move: velocity error'
	den = ues[0] + ues[1]
	w1 = ues[1] / den
	w2 = ues[0] / den
	M.isol.sstag = w1 * S[0] + w2 * S[1]  # s location
	M.isol.xstag = np.dot(M.geom.coord[I, :].T, np.array([w1, w2]))  # x location
	M.isol.sstag_ue = np.array([ues[1], -ues[0]]) * (S[1] - S[0]) / (den * den)
	print(f'  Moving stagnation point: s={sstag0} -> s={M.isol.sstag}')

	# set new xi coordinates for every point
	M.isol.xi = np.concatenate(
		(np.abs(M.geom.s - M.isol.sstag), M.wake.s - M.isol.sstag))

	# matrices need to be recalculated if on a new panel
	if newpanel:
		print(f'  New stagnation panel = {I[0]} {I[1]}')
		M.isol.Istag = I  # new panel indices
		sgnue = np.ones(N+1)
		sgnue[:I[0] + 1] = -1
		M.isol.sgnue = sgnue
		identify_surfaces(M)
		# Assumes that the index for the fourth row is actually 3
		ue = ue.T.reshape(M.glob.Nsys)
		M.glob.U[3, :] = ue
		ue = ue.reshape(-1, 1)
		rebuild_ue_m(M)


def solve_coupled(Foil):
	"""
	Solves the coupled inviscid and viscous system
	
	Input:
	- Foil: foil class with an inviscid solution and a generated boundary layer
	
	Output:
	- Foil.glob.U: global coupled solution
	
	Details:
	- Inviscid solution should exist, and BL variables should be initialized
	- The global variables are [th, ds, sa, ue] at every node
	- th = momentum thickness; ds = displacement thickness
	- sa = amplification factor or sqrt(ctau); ue = edge velocity
	- Nsys = N + Nw = total number of unknowns
	- ue is treated as a separate variable for improved solver robustness
	- The alternative is to eliminate ue, ds and use mass flow (not done here):
		- Starting point: ue = uinv + D*m -> ue_m = D
		- Since m = ue*ds, we have ds = m/ue = m/(uinv + D*m)
		- So, ds_m = diag(1/ue) - diag(ds/ue)*D
		- The residual linearization is then: R_m = R_ue*ue_m + R_ds*ds_m
	
	Newton loop
	"""
	nNewton = Foil.param.niglob  # number of iterations
	Foil.glob.conv = False
	print(Foil.param, 1, '\n <<< Beginning coupled solver iterations >>> \n')
	for iNewton in range(1, nNewton+1):
		# set up the global system
		build_glob_sys(Foil)

		# compute forces
		calc_force(Foil)

		# convergence check
		print("I newton : ", iNewton)
		Rnorm = np.linalg.norm(Foil.glob.R, 2)
		print(Foil.param, 1, '\nNewton iteration %d, Rnorm = %.5e\n' %
					(iNewton, Rnorm))
		if Rnorm < Foil.param.rtol:
			Foil.glob.conv = True
			break

		# solve global system
		solve_glob(Foil)

		# update the state
		update_state(Foil)

		# update stagnation point; Newton still OK; had R_x effects in R_U
		stagpoint_move(Foil)

		# update transition
		update_transition(Foil)

	if not Foil.glob.conv:
		print(Foil.param, 1, '\n** Global Newton NOT CONVERGED **\n')


def update_state(Foil):
	"""
	Updates state, taking into account physical constraints.
	
	Args:
		M (object): mfoil class with a valid solution (U) and proposed update (dU)
	
	Returns:
		None
	
	Details:
		U = U + omega * dU; omega = under-relaxation factor
		Calculates omega to prevent big changes in the state or negative values
	"""

	# max ctau
	It = Foil.vsol.turb.nonzero()[0]
	ctmax = Foil.glob.U[2, It].max()

	# starting under-relaxation factor
	omega = 1.0

	# first limit theta and delta*
	for k in range(2):
		Uk, dUk = Foil.glob.U[k, :], Foil.glob.dU[k, :]
		# prevent big decreases in th, ds
		fmin = (dUk / Uk).min()
		#  most negative ratio
		if fmin < -0.5:
			om = abs(0.5 / fmin)
		else:
			om = 1
		if om < omega:
			omega = om
			print(f'  th/ds decrease: omega = {omega:.5f}')

	# limit negative amp/ctau
	Uk, dUk = Foil.glob.U[2, :], Foil.glob.dU[2, :]
	for i, (uk, duk) in enumerate(zip(Uk, dUk)):
		if not Foil.vsol.turb[i] and uk < 0.2:
			continue  # do not limit very small amp (too restrictive)
		if Foil.vsol.turb[i] and uk < 0.1 * ctmax:
			continue  # do not limit small ctau
		if uk == 0. or duk == 0.:
			continue
		if uk + duk < 0:
			om = 0.8 * abs(uk / duk)
			if om < omega:
				omega = om
				print(f'  neg sa: omega = {omega:.5f}')

	# prevent big changes in amp
	I = np.where(Foil.vsol.turb == 0)[0]
	if any(Foil.glob.U[2, I].imag):
		raise ValueError('imaginary amplification')
	dumax = abs(dUk[I]).max()
	if dumax > 0:
		om = abs(2 / dumax)
	else:
		om = 1
	if om < omega:
		omega = om
		print(f'  amp: omega = {omega:.5f}')

	# prevent big changes in ctau
	I = It
	dumax = abs(dUk[I]).max()
	if dumax > 0:
		om = abs(0.05 / dumax)
	else:
		om = 1
	if om < omega:
		omega = om
		print(f'  ctau: omega = {omega:.5f}')

	# prevent large ue changes
	dUk = Foil.glob.dU[3, :]
	fmax = abs(dUk / Foil.oper.Vinf).max()
	if fmax > 0:
		om = 0.2 / fmax
	else:
		om = 1

	if om < omega:
		omega = om
		print(f'  ue: omega = {omega:.5f}\n')

	if abs(Foil.glob.dalpha) > 2:
		omega = min(omega, abs(2/Foil.glob.dalpha))

	print(f'  state update: under-relaxation = {omega:.5f}\n')
	Foil.glob.U = Foil.glob.U + omega * Foil.glob.dU
	Foil.oper.alpha = Foil.oper.alpha + omega * Foil.glob.dalpha

	Uj = np.zeros(4)
	for i_s in range(3):
		if i_s == 2:
			Hkmin = 1.00005
		else:
			Hkmin = 1.02
		Is = Foil.vsol.Is[i_s]
		param = build_param(Foil, i_s)
		for i in range(len(Is)):
			j = Is[i]
			Uj = Foil.glob.U[:, j]
			param = station_param(Foil, param, j)
			Hk, _ = get_Hk(Uj, param)
			if Hk < Hkmin:
				Foil.glob.U[1, j] += 2 * (Hkmin - Hk) * Foil.glob.U[0, j]

	for ii in range(len(I)):
		i = It[ii]
		if Foil.glob.U[2, i] < 0:
			Foil.glob.U[2, i] = 0.1 * ctmax

	if abs(omega * Foil.glob.dalpha) > 1e-10:
		rebuild_isol(Foil)


def build_glob_sys(M):
	"""Builds the primary variable global residual system for the coupled problem.

	Args:
		M: mfoil class with a valid solution in M.glob.U

	Outputs:
		M.glob.R: global residual vector (3 * Nsys x 1)
		M.glob.R_U: residual Jacobian matrix (3 * Nsys x 4 * Nsys, sparse)
		M.glob.R_x: residual linearization w.r.t. x (3 * Nsys x Nsys, sparse)

	Details:
		Loops over nodes/stations to assemble residual and Jacobian.
		Transition dictated by M.vsol.turb, which should be consistent with the state.
		Accounts for wake initialization and first-point similarity solutions.
		Also handles stagnation point on node via simple extrapolation.
	"""
	Nsys = M.glob.Nsys
	M.glob.R = np.zeros((3 * Nsys, 1))
	M.glob.R_U = sp.sparse.lil_matrix((3 * Nsys, 4 * Nsys))
	M.glob.R_x = sp.sparse.lil_matrix((3 * Nsys, Nsys))

	for side in range(3):  # loop over surfaces
		Is = np.array(M.vsol.Is[side])  # surface point indices
		xi = M.isol.xi[Is]  # distance from LE stag point
		N = len(Is)  # number of points on this surface
		# [th, ds, sa, ue] states at all points on this surface
		U = M.glob.U[:, Is]
		Aux = np.zeros((1, N))  # auxiliary data at all points: [wgap]

		# get parameter structure
		param = build_param(M, side)

		# set auxiliary data
		if side == 2:
			Aux[0, :] = M.vsol.wgap

		# special case of tiny first xi -- will set to stagnation state later
		if side < 2 and xi[0] < 1e-8 * xi[-1]:
			i0 = 1
		else:
			i0 = 0  # i0 indicates the "first" point station

		# first point system
		if side < 2:
			# calculate the stagnation state, a function of U1 and U2
			Ip = [i0, i0 + 1]
			Ust, Ust_U, Ust_x, xst = stagnation_state(U[:, Ip], xi[Ip])  # stag state
			param.turb, param.simi = False, True  # similarity station flag
			R1, R1_Ut, _ = residual_station(param, np.array(
				[xst, xst]), np.array([Ust, Ust]).T, Aux[:, [i0, i0]])
			param.simi = False
			R1_Ust = R1_Ut[:, :4] + R1_Ut[:, 4:]
			R1_U = R1_Ust @ Ust_U
			R1_x = R1_Ust @ Ust_x
			J = [Is[i0], Is[i0+1]]

			if i0 == 1:
				# i0=1 point landed right on stagnation: set value to Ust
				print(param, 2, 'hit stagnation!\n')
				Ig = 3*Is[0] + np.arange(-2, 1)
				Jg = 4*Is[0] + np.arange(-3, 1)
				M.glob.R[Ig] = U[0:3, 0] - Ust[0:3]
				M.glob.R_U[Ig, Jg] = M.glob.R_U[Ig, Jg] + np.eye(3, 4)
				Jg = np.concatenate([4*J[0] + np.arange(-3, 1), 4*J[1] + np.arange(-3, 1)])
				M.glob.R_U[Ig, Jg] = M.glob.R_U[Ig, Jg] - Ust_U[:3, :]
				M.glob.R_x[Ig, J] = -Ust_x[:3, :]
		else:
			# wake initialization
			param.turb = True  # force turbulent in wake if still laminar
			param.wake = True
			R1, R1_U, J = wake_sys(M, param)
			R1_x = np.array([])  # no xi dependence of first wake residual

		# store first point system in global residual, Jacobian
		Ig = 3*(Is[i0]+1)-1+np.array([-2, -1, 0])
		R1 = R1.reshape(-1, 1)
		M.glob.R[Ig] = R1
		for j in range(len(J)):
			Jg = 4*(J[j]+1)-1 + np.array([-3, -2, -1, 0])
			M.glob.R_U[Ig.reshape(-1, 1), Jg] += R1_U[:, 4 *
                                             (j+1)-1+np.array([-3, -2, -1, 0])]
			if not R1_x.size == 0:
				M.glob.R_x[Ig.reshape(-1, 1), J[j]] += R1_x[:, j].reshape(-1, 1)

		# march over rest of points
		for i in range(i0+1, N):
			Ip = [i-1, i]  # two points involved in the calculation

			tran = M.vsol.turb[Is[i-1]] != M.vsol.turb[Is[i]]  # transition flag

			# residual, Jacobian for point i
			if tran:
				Ri, Ri_U, Ri_x = residual_transition(
					M, param, xi[Ip], U[:, Ip], Aux[:, Ip])
				store_transition(M, side, i)
			else:
				Ri, Ri_U, Ri_x = residual_station(param, xi[Ip], U[:, Ip], Aux[:, Ip])

			# store point i contribution in global residual, Jacobian
			Ig = 3*(Is[i]+1)-1 + np.array([-2, -1, 0])
			Jg = np.concatenate(
				[4*(Is[i-1]+1)-1+np.array([-3, -2, -1, 0]), 4*(Is[i]+1)-1+np.array([-3, -2, -1, 0])])
			M.glob.R[Ig] += Ri.reshape(-1, 1)
			M.glob.R_U[Ig.reshape(-1, 1), Jg.reshape(1, -1)] += Ri_U
			M.glob.R_x[Ig.reshape(-1, 1), Is[Ip].reshape(1, -1)] += Ri_x

			# following transition, all stations will be turbulent
			if tran:
				param.turb = True
def rebuild_isol(M):
	# rebuilds inviscid solution, after an angle of attack change
	# INPUT
	#   M     : mfoil class with inviscid reference solution and angle of attack
	# OUTPUT
	#   M.isol.gam : correct combination of reference gammas
	#   New stagnation point location if inviscid
	#   New wake and source influence matrix if viscous

	print('\n  Rebuilding the inviscid solution.\n')
	alpha = M.oper.alpha
	M.isol.gam = M.isol.gamref[:, 0]*np.cos(np.radians(
		alpha)) + M.isol.gamref[:, 1]*np.sin(np.radians(alpha))

	if not M.oper.viscous:
		# viscous stagnation point movement is handled separately
		stagpoint_find(M)
	elif M.oper.redowake:
		build_wake(M)
		identify_surfaces(M)
		calc_ue_m(M)  # rebuild matrices due to changed wake geometry


def update_transition(M):
	"""
	updates transition location using current state

	INPUT:
		a valid state in M.glob.U
	OUTPUT:
		M.vsol.turb : updated with latest lam/turb flags for each node
		M.glob.U    : updated with amp factor or shear stress as needed at each node
	"""
	for is_ in range(2):  # loop over lower/upper surfaces
		Is = M.vsol.Is[is_]  # surface point indices
		N = len(Is)  # number of points

		# get parameter structure
		param = build_param(M, is_)

		# current last laminar station
		I = np.where(M.vsol.turb[Is] == 0)[0]
		ilam0 = I[-1]

		# current amp/ctau solution (so we do not change it unnecessarily)
		sa = M.glob.U[2, Is]

		# march amplification equation to get new last laminar station
		ilam = march_amplification(M, is_)

		if ilam == ilam0:
			M.glob.U[2, Is] = sa
			continue  # no change

		print(f"Update transition: last lam [{ilam0}]->[{ilam}]")

		if ilam < ilam0 -1:
			# transition is now earlier: fill in turb between [ilam+1, ilam0]
			param.turb = True
			sa0, _ = get_cttr(M.glob.U[:, Is[ilam+1]], param)
			sa1 = sa0
			if ilam0 < N -1:
				sa1 = M.glob.U[2, Is[ilam0+1]]
			xi = M.isol.xi[Is]
			dx = xi[min(ilam0+1, N-1)] - xi[ilam+1]
			for i in range(ilam+1, ilam0+1):
				if dx == 0 or i == ilam+1:
					f = 0
				else:
					f = (xi[i] - xi[ilam+1]) / dx
				if ilam+1 == ilam0:
					f = 1
				M.glob.U[2, Is[i]] = sa0 + f*(sa1 - sa0)
				assert M.glob.U[2, Is[i]
                    ] > 0, "negative ctau in update_transition"
				M.vsol.turb[Is[i]] = 1

		elif ilam > ilam0 -1:
			# transition is now later: lam already filled in; leave turb alone
			for i in range(ilam0, ilam+1):
				M.vsol.turb[Is[i]] = 0



def march_amplification(Foil, is_):
	"""
	Marches amplification equation on surface is
	:param M: input matrix M
	:param is: surface number index
	:return: ilam - index of last laminar station before transition
			 M.glob.U - updated with amp factor at each (new) laminar station
	"""
	Is = Foil.vsol.Is[is_]  # surface point indices
	N = len(Is)  # number of points
	param = build_param(Foil, is_)  # get parameter structure
	U = Foil.glob.U[:, Is]  # states
	turb = Foil.vsol.turb[Is]  # turbulent station flag

	# loop over stations, calculate amplification
	U[2, 0] = 0.0  # no amplification at first station
	param.turb = False
	param.wake = False
	i = 1
	while i < N:
		U1 = U[:, i - 1]
		U2 = U[:, i]  # states
		if turb[i]:
			U2[2] = U1[2] * 1.01  # initialize amp if turb
		dx = Foil.isol.xi[Is[i]] - Foil.isol.xi[Is[i - 1]]  # interval length

		# Newton iterations, only needed if adding extra amplification in damp
		nNewton = 20
		iNewton = 0
		for iNewton in range(nNewton):
			# amplification rate, averaged
			damp1, damp1_U1 = get_damp(U1, param)
			damp2, damp2_U2 = get_damp(U2, param)
			damp, damp_U = upwind(0.5, 0, damp1, damp1_U1, damp2, damp2_U2)

			Ramp = U2[2] - U1[2] - damp * dx

			if iNewton > 12:
				print(
					param, 3, f"i={i}, iNewton={iNewton}, sa = [{U1[2]:.5e}, {U2[2]:.5e}], damp = {damp:.5e}, Ramp = {Ramp:.5e}\n")

			if abs(Ramp) < 1e-12:
				break  # converged
			Ramp_U = [0, 0, -1, 0, 0, 0, 1, 0] - damp_U * dx
			dU = -Ramp / Ramp_U[6]
			omega = 1
			dmax = 0.5 * (1.01 - iNewton / nNewton)
			if abs(dU) > dmax:
				omega = dmax / abs(dU)
			U2[2] = U2[2] + omega * dU

			iNewton += 1

		if iNewton >= nNewton:
			print(param, 1, "march amp Newton unconverged!\n")

		# check for transition
		if U2[2] > param.ncrit:
			print(
				param, 2, f"  march_amplification (is,i={is_},{i}): {U2[2]:.5e} is above critical.\n")
			break
		else:
			Foil.glob.U[2, Is[i]] = U2[2]  # store amplification in M.glob.U
			U[2, i] = U2[2]  # also store in local copy!
			if abs(U[2, i].imag) > 0:
				raise ValueError('imaginary amp during march')
		i += 1
	ilam = i-1
	return ilam


def stagnation_state(U, x):
	"""
	INPUT
	  U  : [U1,U2] = states at first two nodes (4x2)
	  x  : [x1,x2] = x-locations of first two nodes (2x1)
	OUTPUT
	  Ust    : stagnation state (4x1)
	  Ust_U  : linearization of Ust w.r.t. U1 and U2 (4x8)
	  Ust_x  : linearization of Ust w.r.t. x1 and x2 (4x2)
	  xst    : stagnation point location ... close to 0
	DETAILS
	  fits a quadratic to the edge velocity: 0 at x=0, then through two states
	  linearly extrapolates other states in U to x=0, from U1 and U2
	"""

	# pull off states
	U1 = U[:, 0]
	U2 = U[:, 1]

	x1 = x[0]
	x2 = x[1]
	dx = x2 - x1
	dx_x = np.array([-1, 1])
	rx = x2 / x1
	rx_x = np.array([-rx, 1]) / x1

	# linear extrapolation weights and stagnation state
	w1 = x2 / dx
	w1_x = -w1 / dx * dx_x + np.array([0, 1]) / dx
	w1_x = w1_x.reshape(1, -1)
	w2 = -x1 / dx
	w2_x = -w2 / dx * dx_x + np.array([-1, 0]) / dx
	w2_x = w2_x.reshape(1, -1)
	Ust = U1 * w1 + U2 * w2

	# quadratic extrapolation of the edge velocity for better slope, ue=K*x
	wk1 = rx / dx
	wk1_x = rx_x / dx - wk1 / dx * dx_x
	wk2 = -1 / (rx * dx)
	wk2_x = -wk2 * (rx_x / rx + dx_x / dx)
	K = wk1 * U1[3] + wk2 * U2[3]
	K_U = np.array([0, 0, 0, wk1, 0, 0, 0, wk2])
	K_x = U1[3] * wk1_x + U2[3] * wk2_x

	U1 = U1.reshape(-1, 1)
	U2 = U2.reshape(-1, 1)

	# stagnation coord cannot be zero, but must be small
	xst = 1e-6
	Ust[3] = K * xst  # linear dep of ue on x near stagnation
	Ust_U = np.vstack(
		[np.hstack([w1 * np.eye(3, 4), w2 * np.eye(3, 4)]), K_U * xst])
	Ust_x = np.vstack([U1[0:3] * w1_x + U2[0:3] * w2_x, K_x * xst])

	return Ust, Ust_U, Ust_x, xst


def ViscousSolver(Foil):
	"""
	Solves the viscous system (BL + outer flow concurrently)
	
	Input:
	Foil : foil class

	Output:
	Viscous Solver
	"""
	
	LVSolver(Foil)
	Foil.oper.viscous = True
	build_term(Foil)			# thermodynamics
	build_wake(Foil)
	stagpoint_find(Foil)		# from the inviscid solution
	identify_surfaces(Foil)
	set_wake_gap(Foil)			# blunt TE dead air extent in wake
	calc_ue_m(Foil)
	init_boundary_layer(Foil)	# initialize boundary layer from ue
	stagpoint_move(Foil)		# move stag point, using viscous solution
	solve_coupled(Foil)			# solve coupled system
	calc_force(Foil)
	get_distributions(Foil)