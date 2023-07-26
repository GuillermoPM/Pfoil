"""
	## Vortex mapping matrix
	
"""

import numpy as np
import scipy.integrate as integrate


def VortexMatrix(Foil):
	"""
	Gives the column array that maps the vortex intensity with the pannels. The intensity is constant and its influence is calculated.

	"""
	panels = Foil.geom.panels
	N = Foil.N
	
	VortexM = np.zeros(N, float)
	for i, panel_i in enumerate(panels):
		for j, panel_j in enumerate(panels):
			if j != i:
				VortexM[i] -= 0.5/np.pi*integrate.quad(lambda s: ((panel_i.midx-(panel_j.xmin-s*np.sin(panel_j.beta)))*(np.sin(panel_i.beta))
																  + (panel_i.midy-(panel_j.ymin+s*np.cos(panel_j.beta)))*(-np.cos(panel_i.beta))) /
													   ((panel_i.midx-(panel_j.xmin-s*np.sin(panel_j.beta)))**2 +
														   (panel_i.midy-(panel_j.ymin+s*np.cos(panel_j.beta)))**2), 0, panel_j.len)[0]
	Foil.cvortm.vortexMatrix = VortexM
