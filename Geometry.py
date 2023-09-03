"""
	## Geometry definition functions
	>>> pannel \n
	>>> Wakepanel \n
	>>> Segment \n
	>>> SplineGeom \n
	>>> SplineLen \n
	>>> pannel_Division \n
	>>> NacaFoil \n
	>>> Wake_Division \n
	>>> xPond \n
	>>> Sval \n
"""

import scipy.interpolate as interp
from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt

class Pannel():
	"""
	A pannel is deffined from it's extreme points and the angle between the normal and the horizontal.

	>>> Panel(coordmin, coordmax, identnumber)

	### Args:

	(xmin, ymin) -> First pannel point
	(xmax, ymax) -> Second pannel point
	i -> Identification number
	
	### Params:

	beta : Angle between the normal and the horizontal \n
	lug : Flag for the position (upper or lower) \n
	intens : pannel intensity \n
	cpi : Pressure coefficient induced by the pannel \n
	Vt : Tangential velocity \n
	(midx, midy) : Midpoint \n


	"""

	def __init__(self, coordmin, coordmax, i):

		self.xmin, self.ymin = coordmin[0], coordmin[1]		# (xmin, ymin) point
		self.xmax, self.ymax = coordmax[0], coordmax[1]		# (xmax, ymax) point
		
		self.len = ((self.xmax-self.xmin)**2+(self.ymax-self.ymin)**2)**0.5  # pannel length
		
		self.ident = i  # pannel identification

		# Parameter initialization
		self.Vi = 0.		# induced velocity
		self.Vi_visc = 0	# viscous induced velocity
		self.intens = 0.	# intensity
		self.cpi = 0.		# pressure coefficient
		self.Cpi_visc = 0.	# viscous pressure coefficient

		self.midx, self.midy = (self.xmax + self.xmin)/2, (self.ymax+self.ymin)/2  
		self.midpt = np.array([self.midx, self.midy])  # pannel midpoint

		self.t = np.array([self.xmax - self.xmin,self.ymax - self.ymin])	# tangential vector
		self.t = self.t/np.linalg.norm(self.t)								# type: ignore # tangential vector normalized
		self.n = np.array([-self.t[1], self.t[0]])							# normal vector

		self.leftcoord = coordmin			# min point coordinates
		self.rightcoord = coordmax			# max point coordinates

	

		# Beta angle
		if self.xmax-self.xmin <= 0.0:
			self.beta = np.arccos((self.ymax-self.ymin)/self.len)
		elif self.xmax-self.xmin > 0.0:
			self.beta = np.pi + np.arccos(-(self.ymax-self.ymin)/self.len)
		self.phi = self.beta - np.pi/2
		if self.phi > 2*np.pi:
			self.phi -= 2*np.pi

		# Pannel location flag
		if self.beta <= np.pi:
			self.lug = 'upper'
		elif self.beta > np.pi:
			self.lug = 'lower'

	def __repr__(self):
		return 'pannel ' + str(self.ident) + ' : (' + str(self.xmin)+',' +\
				str(self.ymin)+') ' + '(' + str(self.xmax)+',' + str(self.ymax) + \
				') ' + ' ' + 'Lugar: ' + \
				str(self.lug) + '\n' + 'V : ' + str(self.Vi)

class Wakepanel(Pannel):
	"""
		A wake pannel is a pannel located at the wake
	"""

	def __init__(self, coordmin, coordmax, i):
		super().__init__(coordmin, coordmax, i)
		self.lug = "wake"

	def __repr__(self):
		return super().__repr__()

def Segment(pt1, pt2):
	"""
		Segment that joins two points (pt1, pt2)
	"""

	return ((pt2[1] - pt1[1])**2+(pt2[0] - pt1[0])**2)**0.5

def SplineGeom(coord, foil_name):
	"""
	Gives the splines that interpolate the foil coordinates. The foil is divided in upper and lower and the splines are stored in
	variables and are introduced in the pannel division function

	INPUT:
	coord : foil coordinate array (2 x N)
	foil_name : foil name

	OUTPUT:
	spline_sup, spline_inf : geometry interpolating splines
	"""
	xsup, ysup, xinf, yinf = coord_adjustment(coord)
		
	# Spline generation
	if foil_name[0] == '0':
		spline_sup = interp.CubicSpline	(xsup, ysup)
		spline_inf = interp.CubicSpline(xinf, yinf)
	else:
		spline_sup = interp.PchipInterpolator(xsup, ysup)
		spline_inf = interp.PchipInterpolator(xinf, yinf)

	return spline_sup, spline_inf

def SplineLen(spline, xlimit):
	"""
	Calculates the spline length up to the specified point
	
	INPUT:
	spline : geometry interpolating spline
	xlimit : limit point

	OUPUT:
	s : spline lenght

	"""
	Ndiv = 100000  # Número de divisiones del spline
	xValue = np.linspace(0, xlimit, Ndiv)
	yValue = spline(xValue)
	s = 0

	for i in range(Ndiv-2):
		s += Segment([xValue[i+1], yValue[i+1]], [xValue[i], yValue[i]])
		
	return s

def panel_division(coord, N, foil_name, presc):
	"""
	Discretizes the geometry in the number of panels indicated

	INPUT:
	coord : foil coordinate array (2 x N)
	N : number of panels
	foil_name: foil name

	OUTPUT:
	panels : array of panels
	coords : array of nodes that define the panels

	"""
	
	spline_sup, spline_inf = SplineGeom(coord, foil_name)


	# x coordinate generation
	if N%2 == 0:
		xpa = 0.5 - 0.5*np.cos(np.linspace(np.pi, 0, int(N/2)+2))
		xpb = 0.5 - 0.5*np.cos(np.linspace(0, np.pi,  int(N/2)+1))
		xpb = xpb[1:]
		xpa = xpa[:-1]

	else:
		xpa = 0.5 - 0.5*np.cos(np.linspace(np.pi, 0, int((N+1)/2)+1))
		xpb = 0.5 - 0.5*np.cos(np.linspace(0, np.pi,  int((N+1)/2)+1))
		xpa = xpa[:-1]
		xpb = xpb[1:]

	yp = []
	ypi = []
	xp = np.concatenate((xpa, xpb))
	for coord in xpa:
		yp.append(spline_inf(coord))
	for coord in xpb:
		ypi.append(spline_sup(coord))
	for item in ypi:
		yp.append(item)

	# pannel generation
	panels = np.empty(N+1, dtype=object)
	coords = np.array(list(zip(xp,yp)))
	for i in range(N):
		panels[i] = Pannel(coords[i],coords[i+1], i+1)

	panels[-1] = Pannel(coords[-1], coords[0], 999)

	return panels, coords


def coord_adjustment(coord):
	"""
		Adjusts the coordinates to fit the correct order.
	"""
	x, y = coord[:, 0], coord[:, 1]
	if len(x) %2 ==0:
		index = int(len(x)/2-1)
	else:
		index = int((len(x)+1)/2)
	

	xsup = np.empty(index, float)
	ysup = np.empty(index, float)
	xinf = np.empty_like(xsup)
	yinf = np.empty_like(xsup)

	if x[0] != 0.000000:  # Counterclockwise

		while x[index-1] < x[index]:
			index -= 1

		xinf = x[:index+1]
		yinf = y[:index+1]
		xinf = np.flip(xinf, axis=0)
		yinf = np.flip(yinf, axis=0)
	
		xsup = x[index+1:]
		ysup = y[index+1:]

	else:  # Clockwise
		for i in range(index):
			xsup[i] = x[i]
			ysup[i] = y[i]
			xinf[i] = x[i+index+1]
			yinf[i] = y[i+index+1]
	
	return xsup, ysup, xinf, yinf
	
def NacaFoil(foil_name,model):
	"""
		NACA foil generator using AirfoilTools function for 4 digit airfoils
	"""

	m = int(foil_name[0])/100
	p = int(foil_name[1])/10
	tk = int(foil_name[2:])/100
	print("\n")
	print("{:^30}".format("NACA FOIL GENERATOR"))
	print("{:^30}".format("------------------"))
	print("{:^30}".format("Airfoil : NACA {}".format(foil_name)))
	print("{:^30}".format("Max camber : {}".format(m)))
	print("{:^30}".format("Max camber xpos : {}".format(p)))
	print("{:^30}".format("Thickness : {}%".format(tk * 100)))
	N = 20000

	points = int((N)/2)+1  # Number of points
	beta = np.linspace(0, np.pi, points)

	# More points density in the leading and trailing edge.
	x = (1-np.cos(beta))/2
	
	if p == 0:  # Symmetric foil
		yc = np.zeros_like(x)
		dydx = np.zeros_like(x)
	else:
		yc = np.where(x < p, m/p**2*(2*p*x - x**2), m /
				(1-p)**2*((1-2*p) + 2*p*x - x**2))
		dydx = np.where(x < p, 2*m/p**2*(p-x), 2*m/(1-p)**2*(p-x))


	a = 0.1015
	yt = 5*tk*(0.2969*np.sqrt(x) - 0.126*x - 0.3516 * x**2 +
			0.2843*x**3 - a*x**4)  # Distribución del espesor

	xu = x  		# x coord sup 0 -> 1
	yu = yc + yt  	# y coord sup
	xl = x  		# x coord inf 0 -> 1
	yl = yc - yt  	# y coord inf


	xp_sup = np.array(xu)
	yp_sup = np.array(yu)
	xp_inf = np.array(xl)
	yp_inf = np.array(yl)

	if model == 0:
		xp_sup = np.delete(xp_sup, 0)
		yp_sup = np.delete(yp_sup, 0)

	else:
		xp_sup, xp_inf = np.delete(xp_sup, 0), np.delete(xp_inf, 0)
		yp_sup, yp_inf = np.delete(yp_sup, 0), np.delete(yp_inf, 0)

	xp_sup = np.flip(xp_sup, axis=0)
	yp_sup = np.flip(yp_sup, axis=0)


	sup_points = [xp_sup, yp_sup]
	inf_points = [xp_inf, yp_inf]

	xPoints = np.append(xp_sup, xp_inf)
	yPoints = np.append(yp_sup, yp_inf)

	coord = np.stack((xPoints, yPoints), axis=1)
	coord = np.flip(coord, axis=0)

	return sup_points, inf_points, coord

def xPond(refpoint, calcpoints):
	"""
	Returns the distance in x between the reference point and the target
	
	INPUT:
	refpoint : distancia de referencia x
	calcpoints : array de distancias de cálculo

	OUTPUT:
	dist : array distancias valor absoluto
	"""

	dist = abs(calcpoints - refpoint)
	return dist

def Sval(Foil):
	"""
		Spline length from the lower surface trailing edge to each node.

		INPUT:
		Foil : airfoil

		OUTPUT:
		Foil.geom.s : spline that defines the full geometry.
		Foil.isol.xi : distance from any point in the foil to the stagnation point.
		Foil.isol.chorddist : distance in chord percentaje to the stagnation point.

	"""
	# Variable definitions
	geom_sup = Foil.geom.spline_sup 		# lower spline
	geom_inf = Foil.geom.spline_inf 		# upper spline
	panneles = Foil.geom.panneles 			# panels
	wakepanels = Foil.geom.wakepanels 		# wake panels
	xPoints = Foil.geom.coord[:, 0] 		# x coordinates of the different nodes

	s_totSup = SplineLen(geom_sup, 1) 		# upper geometry total length
	s_totInf = SplineLen(geom_inf, 1) 		# lower geometry total length

	x_sup = xPoints[:int((Foil.N+1)/2)] 	# upper geometry points
	x_inf = xPoints[int((Foil.N+1)/2):]		# lowe geometry points

	s_sup = np.array(abs(s_totInf + SplineLen(geom_sup, x_sup)))
	s_inf = np.array(abs(s_totInf - SplineLen(geom_inf, x_inf[::-1])))
	s_sup = np.flip(s_sup, axis=0)

	Foil.geom.s = np.concatenate((s_inf, s_sup))

	s_stg = s_totSup + SplineLen(geom_inf, Foil.isol.x_stg)
	Foil.isol.xi = np.array(abs(s_stg - Foil.isol.svalue))

	x_sup1 = [pannel.midx for pannel in panneles if pannel.ident < int(Foil.N/2)]
	x_sup1.reverse()
	x_sup1 = np.array(x_sup1)
	x_sup2 = [pannel.midx for pannel in panneles if int(Foil.N/2) <= pannel.ident <= Foil.isol.stgpannel]
	x_sup2.reverse()
	x_sup2 = np.array(x_sup2)
	x_sup = np.concatenate((x_sup1, x_sup2))
	x_inf = np.array(
		[pannel.midx for pannel in panneles if pannel.ident >= Foil.isol.stgpannel])
	x_wake = np.array([pannel.midx for pannel in wakepanels])

	Foil.isol.chorddist = np.array([x_sup, x_inf, x_wake],dtype = object)

def trailing_specs(Foil):
	"""
		Gives the trailing edge specs

		INPUT:
		Foil : solving airfoil

		OUTPUT:
		t : bisector of the trailing edge
		hTE : trailing edge gap
		dtdx : trailing edge slope
		tcp : t x p, gives the trailing edge source intensity
		tdp : t · p, gives the trailing edge vortex intensity

	"""
	panneles = Foil.geom.panneles			# airfoil panels
	t1 = -1*panneles[0].t 				#  first pannel direction vector (lower trailing edge)
	t2 = 1*panneles[-2].t  				# last pannel direction vector (upper trailing edge)
	t = 0.5*(t1+t2)
	t = t/np.linalg.norm(t) 
	s = -1*panneles[-1].t*panneles[-1].len
	hTE = -s[0]*t[1] + s[1]*t[0] 
	dtdx = t1[0]*t2[1] - t2[0]*t1[1] 
	p = s/np.linalg.norm(s)
	tcp = np.abs(t[0]*p[1]-t[1]*p[0])
	tdp = np.dot(t, p)
	return t, hTE, dtdx, tcp, tdp


def panel_division_CVPM(coord, N, foil_name):
	"""
	Divides the airfoil in the especified pannel number for CVPM.

	INPUT:
	coord : foil coordinate array (2 x N)
	N : number of panels
	foil_name: foil name

	OUTPUT:
	panels : array of panels
	coords : array of nodes that define the panels

	"""
	
	x, y = coord[:, 0], coord[:, 1]
	x, y = np.flip(x), np.flip(y)


	if x[0] != 0.000000:
		index = np.nonzero(x == 0.0)[0][0]
	else:
		index = np.nonzero(x == 1.0)[0][0]

	xsup = np.empty(index, float)
	ysup = np.empty(index, float)
	xinf = np.empty(len(x)-index, float)
	yinf = np.empty(len(x)-index, float)
	if x[0] != 0.000000:  		# Counterclockwise
		for i in range(index):
			xsup[i] = x[index-i]
			ysup[i] = y[index-i]
		for i in range(len(x)-index):
			xinf[i] = x[index+i]
			yinf[i] = y[index+i]
	else:  						# Clockwise
		for i in range(index):
			xsup[i] = x[i]
			ysup[i] = y[i]
			xinf[i] = x[i+index+1]
			yinf[i] = y[i+index+1]
	# Spline generation
	if foil_name[0] == '0':
		spline_sup = interp.CubicSpline(xsup, ysup)
		spline_inf = interp.CubicSpline(xinf, yinf)
	else:
		spline_sup = PchipInterpolator(xsup, ysup)
		spline_inf = PchipInterpolator(xinf, yinf)

	# Spline division
	xpa = 0.5 - 0.5*np.cos(np.linspace(np.pi, 0, int(N/2)+1))
	xpb = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, int(N/2)+1))

	xpa = np.delete(xpa, int(N/2))
	yp = []
	ypi = []
	xp = np.concatenate((xpa, xpb))

	for coord in xpa:
		yp.append(spline_sup(coord))
	for coord in xpb:
		ypi.append(spline_inf(coord))
	for item in ypi:
		yp.append(item)

	# Panneling
	panels = np.empty(N, dtype=object)
	coords = np.array(list(zip(xp, yp)))
	for i in range(N):
		panels[i] = Pannel(coords[i], coords[i+1], i+1)


	return panels, coords
