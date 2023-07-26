import numpy as np
import scipy.integrate as integrate


def VortexMatrix(Foil):
	"""
	Gives the column array that maps the vortex intensity with the pannels. The intensity is constant and its influence is calculated.

	"""
	paneles = Foil.geom.paneles
	N = Foil.N
	
	VortexM = np.zeros(N, float)
	for i, pannel_i in enumerate(paneles):
		for j, pannel_j in enumerate(paneles):
			if j != i:
				VortexM[i] -= 0.5/np.pi*integrate.quad(lambda s: ((pannel_i.midx-(pannel_j.xmin-s*np.sin(pannel_j.beta)))*(np.sin(pannel_i.beta))
																  + (pannel_i.midy-(pannel_j.ymin+s*np.cos(pannel_j.beta)))*(-np.cos(pannel_i.beta))) /
													   ((pannel_i.midx-(pannel_j.xmin-s*np.sin(pannel_j.beta)))**2 +
														   (pannel_i.midy-(pannel_j.ymin+s*np.cos(pannel_j.beta)))**2), 0, pannel_j.len)[0]
	Foil.cvortm.vortexMatrix = VortexM
