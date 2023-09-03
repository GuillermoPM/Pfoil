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

	foil = Foil


	foil.oper.model = GUIParameters.model   # model
	foil.oper.alpha = GUIParameters.alpha   # angle of attack
	foil.oper.Re = GUIParameters.Re         # Reynolds number
	foil.oper.Ma = GUIParameters.Mach		# Mach

	Sval(foil)								# spline length
	build_term(foil)						# thermodynamics
	if GUIParameters.model == 0:
		CVPM_solver(foil)
	elif GUIParameters.model == 1:
		LVSolver(foil)
	else:
		ViscousSolver(foil)
	
	FolderDir(foil)			# creates a folder for the foil in case there's not already one
	DataSave(foil)			# Excel file 
	
	return foil