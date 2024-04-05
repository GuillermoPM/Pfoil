"""
	## Graphic User Interface ##

	It allows the user to introduce the parameters for the analysis, choose the panneling model and show
	the different plots and results. It has been made using the Tkinter module.
"""

# Generic Python modules
import tkinter as tk
import tkinter.scrolledtext as tkst
import tkinter.filedialog as tkfd
from SolverWrapper import *
from DataPlotter import *

# Files
import sys
import os
sys.path.append("C:/Pfoil")

class GUI_parameters:
	"""
		Different parameters that can be selected in the GUI and are used in the calculations.
	
	"""
	def __init__(self):
		self.model = 0					# Model: 0 = CVPM, 1 = LVPM, 2 = Viscous
		self.nacafoil = "0012"			# Airfoil name
		self.nPanels = 199				# Number of pannels
		self.alpha = 0					# Angle of attack
		self.Mach = 0.0					# Mach number
		self.Re = 10000					# Reynolds number		
		self.geomname = "NACA"			# Geometry name
		self.fromfile = False			# Flag that shows if the geometry is from a dat file
		self.foil_file = "direc"		# dat file directory


class VirtualTerminal(object):
	"""
		Virtual terminal in the GUI to allow data visualization.
	"""
	def __init__(self,text_widget):
		self.text_widget = text_widget

	def write(self, string):
		self.text_widget.insert('end', string)
		self.text_widget.see('end')

class GUI():
	def __init__(self):
		self.root = tk.Tk() 			# GUI creation
		self.root.geometry("1000x700") 	# Main window geometry
		self.root.title('Pfoil') 		# Main window title

		self.root.iconbitmap("C:/Pfoil Github/Config/icon.ico")
	
		# Creating the frames
		self.frame1 = tk.Frame(self.root, width=350, height=450)
		self.frame2 = tk.Frame(self.root,  width=350, height=450)
		self.frame3 = tk.Frame(self.root,  width=350, height=450)
		self.frame4 = tk.Frame(self.root,  width=350, height=450)

		self.root.grid_columnconfigure(0, minsize=50, weight=1)
		self.root.grid_columnconfigure(1, minsize=50, weight=1)
		self.root.grid_rowconfigure(0, minsize=50, weight=1)
		self.root.grid_rowconfigure(1, minsize=50, weight=1)

		# Setting the frames in the main window
		self.frame1.grid(row=0, column=0,sticky="nsew")
		self.frame2.grid(row=0, column=1,sticky="nsew")
		self.frame3.grid(row=1, column=0,sticky="nsew")
		self.frame4.grid(row=1, column=1,sticky="nsew")

		self.emptyframe = tk.Frame(self.frame2,width=0,height=0)
		self.emptyframe.grid(row=0, column= 1)

		# Generic variables
		self.GUIParam = GUI_parameters()
		self.foil = Foil

		# Widgets creation
		self.opercond_widgets()
		self.model_widgets()
		self.terminal_widgets()
		self.geom_widgets()
		self.plot_widgets()

		self.root.mainloop()

	def exe(self):
		"""
			Executes the program after pressing the button
		"""
		self.GUIParam.Re = self.entryRe.get()
		self.GUIParam.Mach = self.entryMach.get()
		self.GUIParam.alpha = self.entryAlpha.get()
		self.GUIParam.nPanels = self.entryPanels.get()
		self.foil = PfoilExe(self.foil ,GUIParameters= self.GUIParam)

	def PlotCp(self):
		"""
			Shows the pressure distritubion plot using the Cp_plot function from DataPlotter.
		"""
		Cp_plot(self.foil)
		

	def GeomPlot(self):
		"""
			Shows the plot of the geometry
		"""
		FoilPlot(self.foil)

	def BlayerPlot(self):
		"""
			Shows the displacement thickness plot over the airfoil
		"""
		PlotBlayer(self.foil)

	def Data(self):
		"""
			Saves the data
		"""
		DataSave(self.foil)
		
	def opercond_widgets(self):
		"""
			Creates the widgets to set the operating contidions

			- Reynolds
			- Angle of attack
			- Mach

			It also creates the exe button
		"""

		# Reynolds
		self.entryRe = tk.IntVar(value = 1300000)
		self.Re_lbl = tk.Label(self.frame2,text = "Reynolds")
		self.Re_lbl.grid(row = 2, column= 3)
		self.Re_enter = tk.Entry(self.frame2, textvariable=self.entryRe)
		self.Re_enter.grid(row = 2, column = 4)

		# Angle of attack
		self.entryAlpha = tk.IntVar(value=0)
		self.Alpha_lbl = tk.Label(self.frame2, text="Angle of attack")
		self.Alpha_lbl.grid(row=3, column=3)
		self.Alpha_enter = tk.Entry(self.frame2, textvariable=self.entryAlpha)
		self.Alpha_enter.grid(row=3, column=4)
		
		# Mach
		self.entryMach = tk.DoubleVar(value=0)
		self.Mach_lbl = tk.Label(self.frame2, text="Mach")
		self.Mach_lbl.grid(row=4, column=3)
		self.Mach_enter = tk.Entry(self.frame2, textvariable=self.entryMach)
		self.Mach_enter.grid(row=4, column=4)

		# Calculate
		self.calc_btn = tk.Button(self.frame2, text = "Calculate", command= self.exe)
		self.calc_btn.grid(row = 5, column = 7)

	def model_widgets(self):
		"""
			Allows the model selection:

			- Potential: \n
				> Constant Vortex Panel Method\n
				> Linear Vortex Panel Method
			
			- Viscous
		"""
		models = ["CVPM", "LVPM", "Viscous flow"]
		selected_model = tk.StringVar()
		selected_model.set(models[0])
		selector = tk.OptionMenu(self.frame2, selected_model, *models)
		selector.grid(row = 5 , column= 3)
		def imprimir_modelo(*args):
			modelo = selected_model.get()
			self.GUIParam.model = models.index(modelo)
			print("Selected model: ", modelo)
		
		selected_model.trace("w",imprimir_modelo)
	
	def terminal_widgets(self):
		terminal = tkst.ScrolledText(master = self.frame4)
		terminal.grid(row = 6, column = 2)
		sys.stdout = VirtualTerminal(terminal)

	def open_foil(self):
		"""
			Opens the foil file and changes the parameter fromfile to true.
		"""
		foil_file = tkfd.askopenfilename()
		self.GUIParam.foil_file = os.path.basename(foil_file)
		self.GUIParam.fromfile = True
		self.GUIParam.nacafoil = self.GUIParam.foil_file[5:-4]
		self.foil = PfoilBuilder(self.GUIParam)

	def geom_widgets(self):
		"""
			It allows the user to select the geometry:

			- NACA foil: 4 digit NACA foil

			- Foil file: dat coordinate file
		
		"""

		# NACA Foil
		self.entryNACAFoil = tk.StringVar(value="0012")
		self.NACA_lbl = tk.Label(self.frame1, text="NACA foil: ")
		self.NACA_lbl.grid(row=2, column=1)
		self.NACA_enter = tk.Entry(self.frame1, textvariable=self.entryNACAFoil)
		self.NACA_enter.grid(row=2, column=2)

	
		
		def nacaFoil(*args):
			self.GUIParam.nacafoil = self.entryNACAFoil.get()
			self.GUIParam.nPanels = self.entryPanels.get()
			self.foil = PfoilBuilder(self.GUIParam)

		self.NACA_enter.bind('<Return>',nacaFoil)

		# Foil file
		self.file_selector = tk.Button(
			self.frame1, text= "File", command=self.open_foil).grid(row=4, column=1)

		# Pannels
		self.entryPanels = tk.IntVar(value=200)
		self.Panels_lbl = tk.Label(self.frame1, text="Panel Number")
		self.Panels_lbl.grid(row=3, column=1)
		self.Panels_enter = tk.Entry(self.frame1, textvariable=self.entryPanels)
		self.Panels_enter.grid(row=3, column=2)
		def get_panels(*args):
			self.GUIParam.nPanels = self.entryPanels.get()
			self.foil = PfoilBuilder(self.GUIParam)

		self.Panels_enter.bind('<Return>', get_panels)


	def plot_widgets(self):
		"""
			Creates the buttons and labels for plot generation
		"""
		# Cp plot
		self.cpshow_btn = tk.Button(self.frame3, text="Cp distribution", command=self.PlotCp)
		self.cpshow_btn.grid(row=5, column=1)

		# Geomplot
		self.geomplot_show = tk.Button(self.frame1, text="GeomPlot", command=self.GeomPlot)
		self.geomplot_show.grid(row=7, column=1)

		# Blayer Plot
		self.blayer_show = tk.Button(self.frame3, text="Blayer Plot", command=self.BlayerPlot)
		self.blayer_show.grid(row=8, column=1)

		# Blayer variables
		variables = ["Displ Thick", "Mom Thick", "Velocity", "Cf"]
		selected_variable = tk.StringVar()
		selected_variable.set(variables[0])
		varlbl = tk.Label(self.frame3, text= "Boundary layer parameters:")
		varlbl.grid(row=11, column=1)
		var_selector = tk.OptionMenu(self.frame3, selected_variable, *variables)
		var_selector.grid(row=11, column=2)

		def spgeomselect(*args):
			modelo = selected_variable.get()
			PlotBlayerParam(self.foil,variables.index(modelo))

			
		selected_variable.trace("w", spgeomselect)
		

def main():
	user_interface = GUI()
	with open('C:/Pfoil/README.md','r') as file:
		readme_content = file.read()
		print(readme_content)
	return 0

if __name__ == '__main__':
	main()


