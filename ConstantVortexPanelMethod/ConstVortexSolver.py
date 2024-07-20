import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from ConstantVortexPanelMethod.SourceMatrix import *
from ConstantVortexPanelMethod.VortexMatrix import *
from ConstantVortexPanelMethod.CloseCond import *

"""
CVPM method solver wrapper

@Author: Guillermo Peña Martínez
@Date: 06/11/2022
"""

import matplotlib.pyplot as plt



def CVPM_solver(Foil):
	"""
	Solves the CVPM given the source, vortex, and Kutta condition matrices.
	"""
	N = Foil.N
	AOA = Foil.oper.alpha
	Vinf = Foil.oper.Vinf
	panels = Foil.geom.panels

	SourceMatrix(Foil)
	VortexMatrix(Foil)
	KuttaMatrix(Foil)

	SystemMatrix = np.empty((N+1, N+1), float)
	SystemMatrix[0:N, 0:N] = Foil.cvortm.sourceMatrix[:, :]
	SystemMatrix[0:N, N] = Foil.cvortm.vortexMatrix[:]
	SystemMatrix[N, :] = Foil.cvortm.kuttaCond[:]

	b = np.empty(N+1, float)
	for i, panel in enumerate(panels):
		b[i] = -Vinf * np.cos(np.radians(AOA) - panel.beta)

	b[N] = -Vinf * (np.sin(np.radians(AOA) - panels[0].beta) +
					np.sin(np.radians(AOA) - panels[N-1].beta))
	Sol = np.linalg.solve(SystemMatrix, b)
	for i, panel in enumerate(panels):
		panel.intens = Sol[i]

	intens = Sol[-1]
	cl = intens * sum(panel.len for panel in panels) / (0.5 * Vinf)

	Vt_Matrix = TangVel(Foil)
	Vt_b = Vinf * np.sin([np.radians(AOA) - panel.beta for panel in panels])

	Vt = abs(np.dot(Vt_Matrix, Sol) + Vt_b)
	for i, panel in enumerate(panels):
		panel.Vi = Vt[i]

	Cpi = np.zeros(N)
	for i, panel in enumerate(panels):
		Cpi[i] = 1 - (Vt[i] / Vinf) ** 2
		panel.cpi = Cpi[i]

	Foil.cvortm.Cpi = Cpi
	Foil.cvortm.KJcl = cl

	print("Inviscid results for alpha =", Foil.oper.alpha, ":\n",
		  "cl =", Foil.cvortm.cl, "\ncl KJ =", Foil.cvortm.KJcl)


def cl_calc(Foil):
	"""
	Returns the lift coefficient given the angle of attack and Cp distribution
	"""
	N = Foil.N
	index = int(N / 2)

	s = np.zeros(N, dtype=float)
	beta = np.zeros(N, dtype=float)
	Cp = np.zeros(N, dtype=float)
	Normal_coeff = np.zeros(N, dtype=float)
	Axial_coeff = np.zeros(N, dtype=float)

	for i, panel in enumerate(Foil.geom.panels[0:N]):
		s[i] = panel.len
		beta[i] = panel.beta
		Cp[i] = panel.cpi
		Normal_coeff[i] = -Cp[i] * s[i] * np.sin(beta[i])
		Axial_coeff[i] = -Cp[i] * s[i] * np.cos(beta[i])

	Cpu = Cp[:index]
	Cpl = Cp[index:]
	Cpu[::-1]

	panelinv = Foil.geom.panels[:index]

	upper = [panel.midx for panel in panelinv[::-1]]
	lower = [panel.midx for panel in Foil.geom.panels[index:]]

	Cpi_u = CubicSpline(upper, Cpu)
	Cpi_l = CubicSpline(lower, Cpl)
	Cl = quad(lambda s: Cpi_l(s) - Cpi_u(s), 0, 1)[0]

	Foil.cvortm.cl = round(Cl, 3)
