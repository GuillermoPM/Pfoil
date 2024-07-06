"""
	Main foil class definition where the variables will be stored.
	@Author: Guillermo Peña Martínez
	@Date: 04/05/2023
"""
import os
import sys
from scipy import interpolate as interp
import numpy as np
from Geometry import panel_division, panel_division_CVPM, SplineGeom

class Foil():
	class InvSol:
		"""
		Inviscid solution values and parameters
		"""
		def __init__(self):
			self.gamma = 0 			# constant circulation gamma
			self.gamref = np.array([])
			self.panel_intens = [] 	# pannel intensity
			self.x_stg = 0 			# stagnation point
			self.sstag = 0			# spline len up to the stagnation point
			self.xi = type(object)	# indexes for the nodes close to the stagnation point
			self.Istag = [0, 0]
			self.svalue = np.empty(4)
			self.stgpanel = 0
			self.cpi = []       	# inviscid cp distribution
			self.cl = 0          	# inviscid lift coefficient
			self.uei = []			# inviscid tangential velocity distribution
			self.cdpi = 0       	# near-field pressure drag coefficient
			self.cdp = 0         	# pressure drag coefficient
			self.ue = np.empty(3, dtype=object)
			self.uewi = np.array([])
			self.uewiref = np.array([])
			self.sourceMatrix = type(object)  # source matrix
			self.vMatrix = type(object) # vortex matrix
			self.chorddist = np.array([],dtype=object) # distance in chord length
	
	class ConstantVortexSol:
		"CVPM solution values and parameters"
		def __init__(self):
			self.vortexMatrix = type(object)	# vortex mapping matrix
			self.sourceMatrix = type(object)	# source mapping matrix
			self.kuttaCond = type(object)		# Kutta condition mapping matrix
			self.Vt = type(object)				# velocity distribution
			self.KJcl = 0.0						# lift coefficient from the Kutta-Jowjowsky theorem
			self.cl = 0.0						# lif coefficietnt from the cp
			self.Cpi = type(object)				# cp distribution over the airfoil


	class OperCond:
		"Operating conditions"
		def __init__(self):
			self.Vinf = 1			# freestream velocity
			self.alpha = 0			# angle of attack
			self.rho = 1			# freestream density
			self.Re = 100000		# Reynolds number
			self.Ma = 0				# freestream Mach
			
			self.model = 0			# model
			self.initbl = True		# boundary layer initialization
			self.viscous = False	# viscous / inviscid flag

	class Wake:
		"Wake parameters"
		def __init__(self):
			self.N = 0
			self.x = np.array([])
			self.s = np.array([])
			self.t = np.array([])
			self.wpanels = np.array([])

	class ViscSol:
		"Viscous solution parameters and results"
		def __init__(self):
			self.th = []           # theta = momentum thickness [Nsys]
			self.ds = []           # delta star = displacement thickness [Nsys]
			self.Is = [[], [], []]  # 3 cell arrays of surface indices
			self.wgap = []         # wake gap over wake points
			self.ue_m = []         # linearization of ue w.r.t. mass (all nodes)
			self.sigma_m = []      # d(source)/d(mass) matrix
			self.ue_sigma = []     # d(ue)/d(source) matrix
			# flag over nodes indicating if turbulent (1) or lam (0)
			self.turb = np.zeros([])
			# transition location (xi) on current surface under consideration
			self.xt = 0.
			# transition xi/x for lower and upper surfaces
			self.Xt = np.zeros((2, 4))
			self.dMatrix = np.array([])

	class GeomParam:
		"Geometry parameters"
		def __init__(self):
			self.chord = 1
			self.s = np.array([])
			self.wakelen = 1
			self.foil_name = 'noname'
			self.coord = np.array([])  				# foil coordinates x · N, y · N
			self.xref = np.array([0.25, 0] ) 		# centre for momentum calculation
			self.panels = np.array([])  			# foil panels
			self.wakepanels = np.array([])  		# wakepanels
			self.totalpanels = np.concatenate((self.panels, self.wakepanels))
			self.spline_sup = interp._cubic.CubicSpline
			self.spline_inf = interp._cubic.CubicSpline
			self.presc = True
			self.nPoints = 0

	class Results:
		"Results"
		def __init__(self):
			self.cp = []         # cp distribution
			self.cl = 0          # lift coefficient
			self.cl_ue = []      # linearization of cl w.r.t. ue [N, airfoil only]
			self.cl_alpha = 0    # linearization of cl w.r.t. alpha
			self.cm = 0          # moment coefficient
			self.cdpi = 0        # near-field pressure drag coefficient
			self.cd = 0          # total drag coefficient
			self.cdf = 0         # skin friction drag coefficient
			self.cdp = 0         # pressure drag coefficient

			# distributions
			self.th = []         # theta = momentum thickness distribution
			self.ds = []         # delta* = displacement thickness distribution
			self.sa = []         # amplification factor/shear lag coeff distribution
			self.ue = []         # edge velocity (compressible) distribution
			self.cf = []         # skin friction distribution
			self.Ret = []        # Re_theta distribution
			self.Hk = []         # kinematic shape parameter distribution

	class Param:
		"General parameters"
		def __init__(self):
			self.verb = 1
			self.rtol = 1e-10	# Newton tolerance
			self.niglob = 200
			self.doplot = True
			self.axplot = []

			# Viscous case parameters
			self.ncrit = 9.0
			self.Cuq = 1.0
			self.Dlr = 0.9
			self.SlagK = 5.6

			# initial Ct after transition
			self.CtauC = 1.8
			self.CtauE = 3.3

			# G Beta constants
			self.GA = 6.7
			self.GB = 0.75
			self.GC = 18.0

			self.Minf = 0.
			self.Vinf = 0.
			self.muinf = 0.
			self.mu0 = 0.
			self.rho0 = 1.
			self.H0 = 0.
			self.Tsrat = 0.35
			self.gam = 1.4
			self.KTb = 1.
			self.KTl = 0.
			self.cps = 0.

			# flags
			self.simi = False
			self.turb = False
			self.wake = False

	class GlobalCond:
		"Global conditions"
		def __init__(self):
			self.Nsys = 0
			self.U = type(object)
			self.dU = np.array([])
			self.dalpha = 0
			self.conv = True
			self.R = np.array([])
			self.R_U = np.array([])
			self.R_x = np.array([])

	class Data:
		"Stores the geometry directory for data saving"
		def __init__(self):
			self.foil_dir = ''

	def __init__(self, coord, N=199, foil='NACA 0012'):
		# foil variable initialization
		self.geom = self.GeomParam()
		self.param = self.Param()
		self.isol = self.InvSol()
		self.wake = self.Wake()
		self.oper = self.OperCond()
		self.post = self.Results()
		self.vsol = self.ViscSol()
		self.glob = self.GlobalCond()
		self.cvortm = self.ConstantVortexSol()
		self.data = self.Data()

		# input parameters initialization
		self.N = N  # panel number
		self.geom.foil_name = foil
		self.geom.coord = coord

	def FoilInit(self):
		self.FoilSpline()
			
	def FoilSpline(self):	
		self.geom.spline_sup, self.geom.spline_inf = SplineGeom( # type: ignore
			coord=self.geom.coord,
			foil_name=self.geom.foil_name)

	def PanelDiv(self):
		"""
		Choses the panel division system. There are 2 available, one for CVPM and the other for the LVPM and transpiration method.
		"""
		if self.oper.model == 0:
			self.geom.panels = panel_division_CVPM(
				coord=self.geom.coord,
				N=self.N,
				foil_name=self.geom.foil_name)[0]
		else:
			self.geom.panels = panel_division(
				coord=self.geom.coord,
				N=self.N,
				foil_name=self.geom.foil_name)[0]
