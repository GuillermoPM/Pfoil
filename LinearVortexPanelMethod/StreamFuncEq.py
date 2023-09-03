"""
	Fundamental potential flow functions that are used to calculate the mapping matrix for both sources and vortex.
"""

import numpy as np

def panel_linvortex_stream(panel_i, panel_j):
	"""
		Linear vortex distribution effect over the streamline at a control point.

		INPUT:
		panel_i : panel where the effect is calculated
		panel_j : panel whose effect is calculated

		OUTPUT:
		coef1, coef2 : weighted coefficients for the calculation
    """

	d = np.array([panel_i.xmin - panel_j.xmin, panel_i.ymin - panel_j.ymin])
	x1 = np.dot(d, panel_j.t)
	y = np.dot(d, panel_j.n)

	r1 = np.linalg.norm([x1, y])
	r2 = np.linalg.norm([x1-panel_j.len, y])
	theta1 = np.arctan2(y, x1)
	theta2 = np.arctan2(y, x1-panel_j.len)

	if r1 < 10**(-10):
		logr1 = 0
	else:
		logr1 = np.log(r1)

	if r2 < 10**(-10):
		logr2 = 0
	else:
		logr2 = np.log(r2)

	# print("r2,r1",r2,r1)

	S1 = 0.5/np.pi * (y*(theta2 - theta1) - panel_j.len + x1*logr1 - (x1-panel_j.len)*logr2)
	S2 = x1/panel_j.len * S1 + 0.25/(np.pi*panel_j.len)*(r2**2*logr2 - r1**2 *
                                  logr1 - 0.5*r2**2 + 0.5*r1**2)

	coef1 = S1 - S2
	coef2 = S2
	return coef1, coef2

def panel_linvortex_velocity(xi, panel, midpt, vdir=None):
	"""
		Lineal vortex effect over the velocity at a certain panel point

		INPUT:
        xi : calculation point
        panel : panel studied
        midpt : flag that check if the point is at the middle of the panel
        vdir : velocity vector direction (if specified)

		OUTPUT:
		coef1, coef2 : weighted coefficients for calculation

    """
    
	d = np.array([xi[0] - panel.xmin, xi[1] - panel.ymin])
	x1 = np.dot(d, panel.t)
	y = np.dot(d, panel.n)

	r1 = np.linalg.norm([x1, y])
	r2 = np.linalg.norm([x1-panel.len, y])
	theta1 = np.arctan2(y, x1)
	theta2 = np.arctan2(y, x1-panel.len)

	if midpt:
		ug1 = ug2 = 0.25
		wg1 = -0.5/np.pi
		wg2 = -1*wg1
	else:
		ug1 = 0.5/np.pi*((theta2 - theta1) - (y*np.log(r2/r1) + x1*(theta2-theta1))/panel.len)
		ug2 = 0.5/np.pi*(y*np.log(r2/r1) + x1*(theta2-theta1))/panel.len
		wg1 = np.log(r2/r1)*0.5/np.pi - (x1*np.log(r2/r1) + panel.len - y*(theta2-theta1))*0.5/(np.pi*panel.len)
		wg2 = (x1*np.log(r2/r1) + panel.len - y*(theta2-theta1))*0.5/(np.pi*panel.len)

	# Conversión al sistema de referencia original
	coef1 = np.array([ug1*panel.t[0] + wg1*panel.n[0],
	                 ug1*panel.t[1] + wg1*panel.n[1]])
	coef2 = np.array([ug2*panel.t[0] + wg2*panel.n[0],
		             ug2*panel.t[1] + wg2*panel.n[1]])

	if vdir is not None:
		coef1 = np.dot(coef1, vdir)
		coef2 = np.dot(coef2, vdir)
	return coef1, coef2

def panel_constsource_stream(panel_i, panel_j):
	"""
		Constant source effect over the streamline at a point.

		INPUT:
		panel_i : panel where the effect is calculated
		panel_j : panel whose effect is calculated

		OUTPUT:
		coef1, coef2 : weighted coefficients for the calculation
    """

	xz = np.array([panel_i.xmin-panel_j.xmin, panel_i.ymin-panel_j.ymin])
	x = np.dot(xz, panel_j.t)
	z = np.dot(xz, panel_j.n)

	r1 = np.linalg.norm(np.array([x, z]))
	r2 = np.linalg.norm(np.array([x-panel_j.len, z]))
	theta1 = np.arctan2(z, x)
	theta2 = np.arctan2(z, x-panel_j.len)

	ep = 1e-9
	if r1 < ep:
		logr1 = 0
		theta1 = np.pi
		theta2 = np.pi
	else:
		logr1 = np.log(r1)
	if r2 < ep:
		logr2 = 0
		theta1 = 0
		theta2 = 0
	else:
		logr2 = np.log(r2)
	P = (x*(theta1-theta2) + panel_j.len*theta2 + z*logr1 - z*logr2) / (2*np.pi)

	if (theta1+theta2) > np.pi:
		P = P - 0.25*panel_j.len
	else:
		P = P + 0.75*panel_j.len

	return P

def panel_constsource_velocity(xi, panel, vdir=None):
	"""
		Lineal vortex effect over the velocity at a certain panel point

		INPUT:
        xi : calculation point
        panel : panel studied
        midpt : flag that check if the point is at the middle of the panel
        vdir : velocity vector direction (if specified)

		OUTPUT:
		coef1, coef2 : weighted coefficients for calculation

    """

	xz = xi - np.array([panel.xmin, panel.ymin])
	x = np.dot(xz, panel.t)
	z = np.dot(xz, panel.n)

	r1 = np.linalg.norm([x, z])
	r2 = np.linalg.norm([x - panel.len, z])
	theta1 = np.arctan2(z, x)
	theta2 = np.arctan2(z, x - panel.len)

	ep = 1e-9
	if r1 < ep:
		logr1 = 0
		theta1 = np.pi
		theta2 = np.pi
	else:
		logr1 = np.log(r1)
	if r2 < ep:
		logr2 = 0
		theta1 = 0
		theta2 = 0
	else:
		logr2 = np.log(r2)

	u = (0.5 / np.pi) * (logr1 - logr2)
	w = (0.5 / np.pi) * (theta2 - theta1)

	a = np.array([u * panel.t[0] + w * panel.n[0], u * panel.t[1] + w * panel.n[1]])
	if vdir is not None:
		a = np.dot(a, vdir)
	return a

def panel_linsource_stream(panel_j, panel_i):
	"""
		Linear source distribution effect over the streamline at a control point.

		INPUT:
		panel_i : panel where the effect is calculated
		panel_j : panel whose effect is calculated

		OUTPUT:
		coef1, coef2 : weighted coefficients for the calculation
    """
	xz = np.array([panel_i.xmin-panel_j.xmin, panel_i.ymin-panel_j.ymin])
	x = np.dot(xz, panel_j.t)
	z = np.dot(xz, panel_j.n)

	r1 = np.linalg.norm(np.array([x, z]))
	r2 = np.linalg.norm(np.array([x-panel_j.len, z]))
	theta1 = np.arctan2(z, x)
	theta2 = np.arctan2(z, x-panel_j.len)

	# make branch cut at theta = 0
	if (theta1 < 0):
		theta1 = theta1 + 2*np.pi
	if (theta2 < 0):
		theta2 = theta2 + 2*np.pi

	# check for r1, r2 zero
	ep = 1e-9
	if (r1 < ep):
		logr1 = 0
		theta1 = np.pi
		theta2 = np.pi
	else:
		logr1 = np.log(r1)
	if (r2 < ep):
		logr2 = 0
		theta1 = 0
		theta2 = 0
	else:
		logr2 = np.log(r2)

	# streamfunction components
	P1 = (0.5/np.pi)*(x*(theta1-theta2) + theta2*panel_j.len + z*logr1 - z*logr2)
	P2 = x*P1 + (0.5/np.pi)*(0.5*r2**2*theta2 - 0.5 * r1**2*theta1 - 0.5*z*panel_j.len)

	# influence coefficients
	a = P1 - P2/panel_j.len
	b = P2/panel_j.len

	return a, b


def panel_constsource_velocity2(Xj, xi, vdir=None):
	# panel coordinates
	xj1, zj1 = Xj[:, 0]
	xj2, zj2 = Xj[:, 1]

	# panel-aligned tangent and normal vectors
	t = np.array([xj2 - xj1, zj2 - zj1]) / np.linalg.norm([xj2 - xj1, zj2 - zj1])
	n = np.array([-t[1], t[0]])

	# control point relative to (xj1, zj1)
	xz = xi - np.array([xj1, zj1])
	x = np.dot(xz, t)
	z = np.dot(xz, n)

	# distances and angles
	d = np.linalg.norm([xj2 - xj1, zj2 - zj1])  # panel length
	r1 = np.linalg.norm([x, z])  # left edge to control point
	r2 = np.linalg.norm([x - d, z])  # right edge to control point
	theta1 = np.arctan2(z, x)  # left angle
	theta2 = np.arctan2(z, x - d)  # right angle

	ep = 1e-9
	if r1 < ep:
		logr1 = 0
		theta1 = np.pi
		theta2 = np.pi
	else:
		logr1 = np.log(r1)
	if r2 < ep:
		logr2 = 0
		theta1 = 0
		theta2 = 0
	else:
		logr2 = np.log(r2)

	# velocity in panel-aligned coord system
	u = (0.5 / np.pi) * (logr1 - logr2)
	w = (0.5 / np.pi) * (theta2 - theta1)

	# velocity in original coord system dotted with given vector
	a = np.array([u * t[0] + w * n[0], u * t[1] + w * n[1]])
	if vdir is not None:
		a = np.dot(a, vdir)

	return a


def panel_linsource_velocity(Xj, xi, vdir=None):
	# panel coordinates
	xj1, zj1 = Xj[:, 0]
	xj2, zj2 = Xj[:, 1]

	# panel-aligned tangent and normal vectors
	t = np.array([xj2-xj1, zj2-zj1])
	t /= np.linalg.norm(t)
	n = np.array([-t[1], t[0]])

	# control point relative to (xj1,zj1)
	xz = np.array([xi[0]-xj1, xi[1]-zj1])
	x = np.dot(xz, t)  # in panel-aligned coord system
	z = np.dot(xz, n)  # in panel-aligned coord system

	# distances and angles
	d = np.linalg.norm([xj2-xj1, zj2-zj1])  # panel length
	r1 = np.linalg.norm([x, z])  # left edge to control point
	r2 = np.linalg.norm([x-d, z])  # right edge to control point
	theta1 = np.arctan2(z, x)  # left angle
	theta2 = np.arctan2(z, x-d)  # right angle

	# velocity in panel-aligned coord system
	temp1 = np.log(r1/r2)/(2*np.pi)
	temp2 = (x*np.log(r1/r2) - d + z*(theta2-theta1))/(2*np.pi*d)
	ug1 = temp1 - temp2
	ug2 = temp2
	temp1 = (theta2-theta1)/(2*np.pi)
	temp2 = (-z*np.log(r1/r2) + x*(theta2-theta1))/(2*np.pi*d)
	wg1 = temp1 - temp2
	wg2 = temp2

	# velocity influence in original coord system
	a = np.array([ug1*t[0]+wg1*n[0], ug1*t[1]+wg1*n[1]])  # point 1
	b = np.array([ug2*t[0]+wg2*n[0], ug2*t[1]+wg2*n[1]])  # point 2
	if vdir is not None:
		a = np.dot(a, vdir)
		b = np.dot(b, vdir)
	return a, b
