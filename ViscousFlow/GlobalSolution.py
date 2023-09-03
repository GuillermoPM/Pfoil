import numpy as np
from LinearVortexPanelMethod.LinVortexBuilder import *
import scipy as sp
from scipy.sparse.linalg import spsolve


def solve_glob(M):
	"""
	Solves the global system
	"""
	
	Nsys = M.glob.Nsys  # number of dofs
	docl = M.oper.givencl  # 1 if in cl-constrained mode

	# get edge velocity and displacement thickness
	ue = M.glob.U[3, :].reshape(-1, 1)
	ds = M.glob.U[1, :].reshape(-1, 1)
	unresh_ds = M.glob.U[1, :]
	uemax = np.max(np.abs(ue))
	ue = np.maximum(ue, 1e-10 * uemax)  # avoid 0/negative ue

	# use augmented system: variables = th, ds, sa, ue

	# inviscid edge velocity on the airfoil and wake
	ueinv = get_ueinv(M)

	R_V = sp.sparse.lil_matrix((4 * Nsys + docl, 4 * Nsys + docl))

	# state indices in the global system
	Ids = np.arange(1, 4 * Nsys + 1, 4)  # delta star indices
	Iue = np.arange(3, 4 * Nsys + 1, 4)  # ue indices

	# include effects of R_x into R_U: R_ue += R_x*x_st*st_ue
	jacobian_add_Rx(M)

	# assemble the residual
	R = np.vstack((M.glob.R, ue - (ueinv.reshape(-1, 1) + np.dot(M.vsol.ue_m, (ds * ue)))))

	# assemble the Jacobian
	R_V[0:3 * Nsys, 0:4 * Nsys] = M.glob.R_U
	I = np.arange(3 * Nsys, 4 * Nsys)
	R_V[I.reshape(-1, 1), Iue] = np.identity(Nsys, dtype=float) - np.matmul(M.vsol.ue_m, np.diag(unresh_ds))
	R_V[I.reshape(-1, 1), Ids] = -M.vsol.ue_m @ sp.sparse.diags(np.squeeze(ue))

	if docl:
		# include cl-alpha residual and Jacobian
		Rcla, Ru_alpha, Rcla_U = clalpha_residual(M)
		R = np.vstack((R, Rcla))
		R_V[I, 4 * Nsys] = Ru_alpha
		R_V[4 * Nsys, :] = Rcla_U

	# solve system for dU, dalpha
	dV = -1 * spsolve(R_V, R)

	# store dU, reshaped, in M
	M.glob.dU = np.reshape(dV[:4 * Nsys], (4, Nsys), order="F")
	if docl:
		M.glob.dalpha = dV[-1]


def jacobian_add_Rx(M):
	"""
	Include effects of R_x into R_U: R_ue += R_x*x_st*st_ue

	INPUT
	  M  : mfoil class with residual Jacobian calculated
	  
	OUTPUT
	  M.glob.R_U : ue linearization updated with R_x

	DETAILS
	  The global residual Jacobian has a column for ue sensitivity
	  ue, the edge velocity, also affects the location of the stagnation point
	  The location of the stagnation point (st) dictates the x value at each node
	  The residual also depends on the x value at each node (R_x)
	  We use the chain rule (formula above) to account for this
	"""
	Nsys = M.glob.Nsys  # number of dofs
	Iue = np.arange(3, 4 * Nsys + 1, 4)   # ue indices in U
	x_st = -M.isol.sgnue.reshape(-1, 1)  # st = stag point [Nsys x 1]
	# wake same sens as upper surface
	x_st = np.vstack((x_st, -np.ones((M.wake.N, 1))))
	R_st = M.glob.R_x @ x_st  # [3*Nsys x 1]
	Ist = M.isol.Istag
	st_ue = M.isol.sstag_ue  # stag points, sens
	M.glob.R_U[:, Iue[Ist]] += R_st @ st_ue.T


def clalpha_residual(M):
	"""
	computes cl constraint (or just prescribed alpha) residual and Jacobian
	
	INPUT
	M  : mfoil class with inviscid solution and post-processed cl_alpha, cl_ue
	
	OUTPUT
	Rcla     : cl constraint residual = cl - cltgt (scalar)
	Ru_alpha : lin of ue residual w.r.t. alpha (Nsys x 1)
	Rcla_U   : lin of cl residual w.r.t state (1 x 4*Nsys)
	
	DETAILS
	Used for cl-constrained mode, with alpha as the extra variable
	Should be called with up-to-date cl and cl linearizations
	"""

	Nsys = M.glob.Nsys   # number of dofs
	alpha = M.oper.alpha  # angle of attack (deg)

	if (M.oper.givencl):  # cl is prescribed, need to trim alpha
		Rcla = M.post.cl - M.oper.cltgt   # cl constraint residual
		Rcla_U = sp.sparse.lil_matrix((1, 4*Nsys))
		Rcla_U[0, 3*Nsys+1:4*Nsys:4] = M.post.cl_alpha
		# only airfoil nodes affected
		Rcla_U[0, 3*Nsys+4:4*Nsys:4] = M.post.cl_ue

		# Ru = ue - [uinv + ue_m*(ds.*ue)], uinv = uinvref*[cos(alpha);sin(alpha)]
		Ru_alpha = -get_ueinvref(M) @ np.array(
			[-np.sin(np.deg2rad(alpha)), np.cos(np.deg2rad(alpha))]) * np.pi / 180

	else:  # alpha is prescribed, easy
		Rcla = 0  # no residual
		Ru_alpha = np.zeros((Nsys, 1))  # not really, but alpha is not changing
		Rcla_U = sp.sparse.lil_matrix((1, 4*Nsys))
		Rcla_U[0, 4*Nsys] = 1

	return Rcla, Ru_alpha, Rcla_U.toarray()
