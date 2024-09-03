import numpy as np
import pandas as pd
from os import mkdir
import os
import sys


def OpenFile(foil_name):
	"""
	Opens a file containing airfoil coordinates and returns the coordinates as a numpy array.
	Parameters:
	- foil_name (str): The name of the airfoil file to open.
	Returns:
	- file_coords (numpy.ndarray): An array containing the x and y coordinates of the airfoil.
	Raises:
	- FileNotFoundError: If the specified airfoil file does not exist.
	"""
	if getattr(sys, 'frozen', False):
		base_path = sys._MEIPASS
	else:
		base_path = os.path.dirname(os.path.abspath(__file__))

	foil_dir = os.path.join(base_path, "Airfoils", foil_name)
	
	with open(foil_dir) as file_name:
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
				{'Coords X': np.array([panel.midx for panel in Foil.geom.paneles]), 'Pressure Coef': Foil.cvortm.Cpi})
			writer = pd.ExcelWriter(Foil.data.foil_dir + '/' + 'Coefficients ' + str(
							Foil.oper.alpha) + ' deg' + ' '+str(Foil.geom.foil_name) + ' CVPM' + '.xlsx', engine='xlsxwriter')
	
	# Creates the Excel sheet from the dataframe
	df.to_excel(writer, sheet_name='Results', index=False)
	writer.close()

def FolderDir(Foil):
    """
    Creates the folder to store the results in the directory given for the results
    in case there's not already one.
    """
    # Determine base path depending on whether running in a frozen bundle or not
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define the directory path for results
    foil_dir = os.path.join(base_path, "Results", Foil.geom.foil_name)

    # Create directory if it doesn't exist
    if not os.path.exists(foil_dir):
        try:
            os.makedirs(foil_dir)  # Use os.makedirs to create directories recursively
            print(f"Created directory: {foil_dir}")
        except Exception as e:
            print(f"Failed to create directory: {e}")
    else:
        print("Foil folder already exists")

    # Set the directory path for use in the application
    Foil.data.foil_dir = foil_dir