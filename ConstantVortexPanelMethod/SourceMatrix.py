"""
	## Source mapping matrix

"""

import numpy as np
import scipy.integrate as integrate

def SourceMatrix(Foil):
	"""
	Gives the source matrix solving the integral Iij (Anderson) that gives the influence coefficient for each panel. Integration made with scipy
	instead of using the analytical result of the integral. The source matrix is N x N with delta(ij) = 0.5 being delta the Kronecker delta.

	"""
	N = Foil.N
	paneles = Foil.geom.paneles
	
	SourceM = np.empty((N,N),dtype=float)
	np.fill_diagonal(SourceM, 0.5)
	for i, panel_i in enumerate(paneles):
		for j, panel_j in enumerate(paneles):
			if j != i:
				SourceM[i, j] = 0.5/np.pi*integrate.quad(lambda s: ((panel_i.midx-(panel_j.xmin-s*np.sin(panel_j.beta)))*(np.cos(panel_i.beta))
											+ (panel_i.midy-(panel_j.ymin+s*np.cos(panel_j.beta)))*(np.sin(panel_i.beta))) /
											((panel_i.midx-(panel_j.xmin-s*np.sin(panel_j.beta)))**2 +
											(panel_i.midy-(panel_j.ymin+s*np.cos(panel_j.beta)))**2), 0, panel_j.len)[0]
	Foil.cvortm.sourceMatrix = SourceM


