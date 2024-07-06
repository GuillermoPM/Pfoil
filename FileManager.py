import numpy as np
import pandas as pd
from os import mkdir


# Paths
dir_main = 'C:/Pfoil/'					# Main path
dir_results = 'C:/Pfoil/Results/'		# Results path
dir_airfoils = 'C:/Pfoil/Airfoils/'		# Foil path

def OpenFile(foil_name):
	with open(dir_airfoils + foil_name) as file_name:
		x, y = np.loadtxt(file_name, dtype=float, delimiter='\t', unpack=True)
	file_coords = np.array(list(zip(x, y)))

	return file_coords

def DataSave(Foil):
	"""
		Saves the results in an Excel sheet
	"""

	# Create a dataframe with the results
	if Foil.oper.viscous:
		totalcoord = np.append(Foil.geom.coord[:,0],Foil.wake.x[0,:])
		df = pd.DataFrame({'Coords X': totalcoord,
		                  'Cp': Foil.post.cp, 'Cpi': Foil.post.cpi, 'Mom thickness': Foil.post.th, 'Displ Thickness':Foil.post.ds, 'Cf': Foil.post.cf, 'AmplFactor / ShearLag':Foil.post.sa, 'Ret':Foil.post.Ret,'Hk':Foil.post.Hk})
		writer = pd.ExcelWriter(Foil.data.foil_dir + '/' + 'Coef ' + str(
                    Foil.oper.alpha) + ' deg' + ' '+str(Foil.geom.foil_name) + ' viscous' + '.xlsx', engine='xlsxwriter')
	else:
		if Foil.oper.model != 0:
			df = pd.DataFrame({'Coords X': Foil.geom.coord[:, 0], 'Pressure Coef': Foil.post.cp})
			writer = pd.ExcelWriter(Foil.data.foil_dir + '/' + 'Coef ' + str(
                            Foil.oper.alpha) + ' deg' + ' '+str(Foil.geom.foil_name) + ' LVPM' + '.xlsx', engine='xlsxwriter')
		else:
			df = pd.DataFrame(
				{'Cooords X': np.array([panel.midx for panel in Foil.geom.paneles]), 'Pressure Coef': Foil.cvortm.Cpi})
			writer = pd.ExcelWriter(Foil.data.foil_dir + '/' + 'Coeficientes ' + str(
                            Foil.oper.alpha) + ' deg' + ' '+str(Foil.geom.foil_name) + ' CVPM' + '.xlsx', engine='xlsxwriter')
	
	# Creates the Excel sheet from the dataframe
	df.to_excel(writer, sheet_name='Results', index=False)
	writer.close()

def FolderDir(Foil):
	"""
		Creates the folder to store the results in the directory given for the results in case there's not already one
	"""
	foil_dir = dir_results + str(Foil.geom.foil_name)
	try:
		mkdir(foil_dir)
	except (FileExistsError):
		print("Foil folder already exists")
		pass
	
	Foil.data.foil_dir = foil_dir