import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as tkst
import tkinter.filedialog as tkfd
import sys
import os
from SolverWrapper import *
from DataPlotter import *

class GUIParameters:
    def __init__(self):
        self.model = 0
        self.nacafoil = "0012"
        self.nPanels = 199
        self.alpha = 0
        self.Mach = 0.0
        self.Re = 10000
        self.geomname = "NACA"
        self.fromfile = False
        self.foil_file = "direc"

class VirtualTerminal:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert('end', string)
        self.text_widget.see('end')

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1000x700")
        self.root.title('Pfoil')
        
        # Determine base path depending on whether the script is bundled (using PyInstaller) or not
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        # Reference the icon file
        icon_path = os.path.join(base_path, "Config", "icon.ico")
        try:
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Error loading icon: {e}")

        self.GUIParam = GUIParameters()
        self.foil = None

        # Configure the main frame
        self.main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Widgets Creation
        self.create_widgets()

        self.root.mainloop()

    def create_widgets(self):
        self.opercond_widgets()
        self.model_widgets()
        self.terminal_widgets()
        self.geom_widgets()
        self.plot_widgets()

    def exe(self):
        self.GUIParam.Re = int(self.entryRe.get())
        self.GUIParam.Mach = float(self.entryMach.get())
        self.GUIParam.alpha = int(self.entryAlpha.get())
        self.GUIParam.nPanels = int(self.entryPanels.get())
        self.foil = PfoilExe(self.foil, GUIParameters=self.GUIParam)

    def PlotCp(self):
        Cp_plot(self.foil)

    def GeomPlot(self):
        FoilPlot(self.foil)

    def BlayerPlot(self):
        PlotBlayer(self.foil)

    def Data(self):
        DataSave(self.foil)

    def opercond_widgets(self):
        frame = ttk.Labelframe(self.main_frame, text="Operating Conditions")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        # Reynolds
        ttk.Label(frame, text="Reynolds:").grid(row=0, column=0, sticky=tk.W)
        self.entryRe = ttk.Entry(frame, width=10)
        self.entryRe.insert(0, "1300000")
        self.entryRe.grid(row=0, column=1, sticky=tk.W)

        # Angle of attack
        ttk.Label(frame, text="Angle of Attack:").grid(row=1, column=0, sticky=tk.W)
        self.entryAlpha = ttk.Entry(frame, width=10)
        self.entryAlpha.insert(0, "0")
        self.entryAlpha.grid(row=1, column=1, sticky=tk.W)

        # Mach
        ttk.Label(frame, text="Mach:").grid(row=2, column=0, sticky=tk.W)
        self.entryMach = ttk.Entry(frame, width=10)
        self.entryMach.insert(0, "0.0")
        self.entryMach.grid(row=2, column=1, sticky=tk.W)

        # Calculate Button
        self.calc_btn = ttk.Button(frame, text="Calculate", command=self.exe)
        self.calc_btn.grid(row=3, column=1, pady=10, sticky=tk.W)

    def model_widgets(self):
        frame = ttk.Labelframe(self.main_frame, text="Model Selection")
        frame.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        models = ["CVPM", "LVPM", "Viscous flow"]
        self.selected_model = tk.StringVar()
        self.selected_model.set(models[0])

        model_selector = ttk.Combobox(frame, textvariable=self.selected_model, values=models, state="readonly")
        model_selector.grid(row=0, column=0, pady=5, padx=5)
        model_selector.bind("<<ComboboxSelected>>", self.update_model)

    def update_model(self, event=None):
        model_index = ["CVPM", "LVPM", "Viscous flow"].index(self.selected_model.get())
        self.GUIParam.model = model_index

    def terminal_widgets(self):
        frame = ttk.Labelframe(self.main_frame, text="Terminal Output")
        frame.grid(row=2, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        terminal = tkst.ScrolledText(frame, wrap=tk.WORD, height=15)
        terminal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        sys.stdout = VirtualTerminal(terminal)

    def open_foil(self):
        foil_file = tkfd.askopenfilename()
        self.GUIParam.foil_file = os.path.basename(foil_file)
        self.GUIParam.fromfile = True
        self.GUIParam.nacafoil = self.GUIParam.foil_file[5:-4]
        self.foil = PfoilBuilder(self.GUIParam)

    def geom_widgets(self):
        frame = ttk.Labelframe(self.main_frame, text="Geometry")
        frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky=(tk.W, tk.E))

        # NACA Foil
        ttk.Label(frame, text="NACA foil:").grid(row=0, column=0, sticky=tk.W)
        self.entryNACAFoil = ttk.Entry(frame, width=10)
        self.entryNACAFoil.insert(0, "0012")
        self.entryNACAFoil.grid(row=0, column=1, sticky=tk.W)
        self.entryNACAFoil.bind('<Return>', self.update_foil)

        # Panels
        ttk.Label(frame, text="Panel Number:").grid(row=1, column=0, sticky=tk.W)
        self.entryPanels = ttk.Entry(frame, width=10)
        self.entryPanels.insert(0, "199")
        self.entryPanels.grid(row=1, column=1, sticky=tk.W)
        self.entryPanels.bind('<Return>', self.update_foil)

        # Foil file
        self.file_selector = ttk.Button(frame, text="Select Foil File", command=self.open_foil)
        self.file_selector.grid(row=2, column=1, pady=5, sticky=tk.W)

    def update_foil(self, *args):
        self.GUIParam.nacafoil = self.entryNACAFoil.get()
        self.GUIParam.nPanels = int(self.entryPanels.get())
        self.foil = PfoilBuilder(self.GUIParam)

    def plot_widgets(self):
        frame = ttk.Labelframe(self.main_frame, text="Plotting")
        frame.grid(row=2, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        # Cp plot
        self.cpshow_btn = ttk.Button(frame, text="Cp Distribution", command=self.PlotCp)
        self.cpshow_btn.grid(row=0, column=0, pady=5, sticky=tk.W)

        # Geomplot
        self.geomplot_show = ttk.Button(frame, text="Geometry Plot", command=self.GeomPlot)
        self.geomplot_show.grid(row=1, column=0, pady=5, sticky=tk.W)

        # Blayer Plot
        self.blayer_show = ttk.Button(frame, text="Boundary Layer Plot", command=self.BlayerPlot)
        self.blayer_show.grid(row=2, column=0, pady=5, sticky=tk.W)

        # Blayer variables
        variables = ["Displ Thick", "Mom Thick", "Velocity", "Cf"]
        self.selected_variable = tk.StringVar()
        self.selected_variable.set(variables[0])

        var_selector = ttk.Combobox(frame, textvariable=self.selected_variable, values=variables, state="readonly")
        var_selector.grid(row=3, column=0, pady=5, sticky=tk.W)
        var_selector.bind("<<ComboboxSelected>>", self.update_blayer_plot)

    def update_blayer_plot(self, event=None):
        variable_index = ["Displ Thick", "Mom Thick", "Velocity", "Cf"].index(self.selected_variable.get())
        PlotBlayerParam(self.foil, variable_index)


def main():
    GUI()
    with open('README.md', 'r') as file:
        readme_content = file.read()
        print(readme_content)
    return 0

if __name__ == '__main__':
    main()
