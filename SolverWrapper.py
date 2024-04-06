"""
	Viscous and inviscid solvers
	@Author: Guillermo Peña Martínez
	@Date: 05/05/2023
"""

from LinearVortexPanelMethod.LinVortexBuilder import *
from ConstantVortexPanelMethod.ConstVortexSolver import *
from LinearVortexPanelMethod.ParamInit import *
from ViscousFlow.ViscBuilder import *
from Pfoil import *
from FileManager import *

def PfoilBuilder(GUIParameters):
	"""
		Generates the foil coordinates from file or from NACA expression.
	
	"""
	
	if GUIParameters.nPanels%2 == 0 and GUIParameters.model != 0:
		GUIParameters.nPanels += 1		# odd pannels set

	fromfile = GUIParameters.fromfile	# foil from file flag
	file = GUIParameters.foil_file 		# dat file directory

	if not fromfile:  # NACA generated airfoil
		naca_foil = NacaFoil(foil_name=GUIParameters.nacafoil, model = GUIParameters.model)[2]
		if GUIParameters.model == 0:
			paneles, coords = panel_division_CVPM(
                    coord=naca_foil,
                    N=GUIParameters.nPanels,
                    foil_name=GUIParameters.nacafoil)
		else:
			paneles, coords = panel_division(
						coord = naca_foil,
						N=GUIParameters.nPanels,
						foil_name=GUIParameters.nacafoil,
						presc = GUIParameters.presc)
		
		foil = Foil(coord=coords, N=GUIParameters.nPanels,
		            foil=GUIParameters.nacafoil)
		foil.geom.panels = paneles
		
	
	else:
		file_coords = OpenFile(file)
		if all(file_coords[1:20, 1] > 0):
			file_coords = np.flip(file_coords,axis= 0)
		paneles, coords = panel_division(
                    coord=file_coords,
                    N=GUIParameters.nPanels,
                    foil_name=GUIParameters.nacafoil,
               		presc=GUIParameters.presc)
		foil = Foil(coord=coords, N=GUIParameters.nPanels,
		            foil=GUIParameters.nacafoil)
		foil.geom.panels = paneles
	
		
	foil.FoilInit()  # Inicialización del perfil
	foil.geom.nPoints = foil.N+1
	return foil


def PfoilExe(Foil, GUIParameters):
	"""
		Executes the program 
	
		INPUT: 
		- GUIParameters : operating parameters introduced in the GUI

		Three models:
		- Model 0 : Constant Vortex Panel Method
		- Model 1 : Linear Vortex Panel Method
		- Model 2 : LVPM + Transpiration Method (viscous)
	"""

	Foil.oper.model = GUIParameters.model   # model
	Foil.oper.alpha = GUIParameters.alpha   # angle of attack
	Foil.oper.Re = GUIParameters.Re         # Reynolds number
	Foil.oper.Ma = GUIParameters.Mach		# Mach

	Sval(Foil)								# spline length
	build_term(Foil)						# thermodynamics
	if GUIParameters.model == 0:
		CVPM_solver(Foil)
	elif GUIParameters.model == 1:
		LVSolver(Foil)
	else:
		ViscousSolver(Foil)
	
	FolderDir(Foil)			# creates a folder for the foil in case there's not already one
	DataSave(Foil)			# Excel file 
	
	return Foil