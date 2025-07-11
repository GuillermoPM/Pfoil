import numpy as np
from Pfoil import *

"""
Plots of the results
@Author: Guillermo Peña Martínez
@Date: 06/05/2023
"""
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plots

import matplotlib.pyplot as plt

# Font settings
font = {'family': 'calibri', 'size': 12}
plt.rcParams['font.family'] = font['family']
plt.rcParams['font.size'] = font['size']

def Cp_plot(Foil):
	"""
	Generates the Cp distribution plot.
	"""
	fig, ax = plt.subplots()

	if Foil.oper.model == 2:
		cp = Foil.post.cp[:Foil.N+1+int(Foil.wake.N/2)]
		cpi = Foil.post.cpi[:Foil.N+1+int(Foil.wake.N/2)]
		x = np.append(Foil.geom.coord[:, 0], Foil.wake.x[0, :int(Foil.wake.N/2)])
		ax.plot(x, cpi, '-.', linewidth=3)
		ax.plot(x, cp, linewidth=3)
		results = f"Coefficients:\ncl: {round(Foil.post.cl, 3)}\ncd: {round(Foil.post.cd, 3)}"
		param = f"Parameters\nRe = {Foil.oper.Re:.2e}\nα = {Foil.oper.alpha} deg"
		ax.text(0.9, 0.95, results, ha='left', va='top', transform=plt.gca().transAxes,
				bbox=dict(facecolor='orange', alpha=0.3), fontdict=font)
		ax.text(0.9, 0.8, param, ha='left', va='top', fontdict=font, transform=plt.gca().transAxes,
				bbox=dict(facecolor='orange', alpha=0.3))
		plt.legend(["Inviscid", "Viscous"], loc='lower center', ncol=2)
	else:
		if Foil.oper.model == 0:
			cp = Foil.cvortm.Cpi
		else:
			cp = Foil.post.cp
		if Foil.oper.model == 1:
			x = Foil.geom.coord[:, 0]
		else:
			x = [panel.midx for panel in Foil.geom.panels]

		ax.plot(x, cp, linewidth=3)
		text = f"Coefficients:\ncl: {round(Foil.post.cl, 3)}\ncd: {abs(round(Foil.post.cdpi, 5))}"
		ax.text(0.95, 0.95, text, ha='left', va='top', transform=plt.gca().transAxes,
				bbox=dict(facecolor='yellow', alpha=0.5), fontdict=font)
	
	ax.set_xticks([])
	ax.set_yticks([])
	ax.axhline(0, color='black', lw=0.5)
	ax.axvline(0, color='black', lw=0.5)
	
	# Axis division
	x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	y = np.linspace(-3, 1, 11)
	
	# Default axis off
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	
	# New axis
	ax.axhline(0, color='black')
	ax.axvline(0, color='black')
	
	ax.text(1.04, 0.2, "x", fontdict=font)
	ax.text(-0.05, -3.2, "Cp", fontdict=font)
	
	# x axis division
	for val in x:
		ax.text(val, 0.15, str(val), ha='center', va='top')
		ax.plot([val, val], [0.1, 0.05], color='black', linewidth=0.7)
	
	# y axis division
	for val in y:
		ax.text(-0.01, val, str(round(val, 2)), ha='right', va='center')
		ax.plot([-0.0001, -0.005], [val, val], color='black', linewidth=0.7)
	
	# Grid off
	ax.grid(False)
	ax.invert_yaxis()
	plt.axis('off')
	
	plt.title("Cp distribution", fontdict=font)
	plt.show()

def PlotBlayer(Foil):
	"""
	Displacement thickness plot over the foil
	"""
	ds = Foil.post.ds
	x, y = Foil.geom.coord[:, 0], Foil.geom.coord[:, 1]
	lower, upper, wake = Foil.vsol.Is
	x_lower, y_lower, ds_lower = x[lower], y[lower], ds[lower]
	x_upper, y_upper, ds_upper = x[upper], y[upper], ds[upper]
	y_lower = y_lower - ds_lower
	y_upper = y_upper + ds_upper
	
	x_wake, y_wake, ds_wake = Foil.wake.x[0, :], Foil.wake.x[1, :], ds[wake]
	coef = y_upper[-1] / ds_wake[0]
	
	y_wake1 = y_wake - (1 - coef) * ds_wake
	y_wake2 = y_wake + coef * ds_wake
	
	y_wake1[0] = y_lower[-1]
	y_wake2[0] = y_upper[-1]
	
	plt.plot(x, y, "-k", linewidth=3)
	plt.plot(x_wake, y_wake, "-k", linewidth=3)
	plt.plot(x_upper, y_upper, "-g", linewidth=3)
	plt.plot(x_lower, y_lower, "-b", linewidth=3)
	plt.plot(x_wake, y_wake1, "-r", linewidth=3)
	plt.plot(x_wake, y_wake2, "-r", linewidth=3)
	plt.scatter(Foil.vsol.Xt[1, 1], Foil.geom.spline_sup(Foil.vsol.Xt[1, 1]), color="orange")
	
	if Foil.vsol.Xt[0, 1] != 0:
		plt.scatter(Foil.vsol.Xt[0, 1], Foil.geom.spline_inf(Foil.vsol.Xt[0, 1]), color="orange")
	else:
		plt.scatter(1, Foil.geom.spline_inf(1), color="orange")
	
	plt.ylim(-0.4, 0.4)
	plt.xlim(-0.01, 1.25)
	plt.title(f"Re = {Foil.oper.Re:.2e}        α = {Foil.oper.alpha} deg", fontsize='x-large', loc='center', fontdict=font)
	plt.axis("off")
	plt.show()

def FoilPlot(Foil):
	"""
	Plots the foil with the center of momentum.
	"""
	plt.plot(Foil.geom.coord[:, 0], Foil.geom.coord[:, 1], '-ko', markersize=3)
	plt.plot(Foil.geom.xref[0], Foil.geom.xref[1], 'ro')
	plt.axhline(0, color='black', linewidth=1)
	plt.ylim(-0.4, 0.4)
	plt.xlim(-0.05, 1.1)
	plt.axis('off')
	plt.legend(["Foil geometry", "Center of momentum", "x axis"])
	text = f"Geometry: {Foil.geom.foil_name}"
	plt.text(0.05, 0.95, text, ha='left', va='top', transform=plt.gca().transAxes,
			 bbox=dict(facecolor='yellow', alpha=0.5))
	plt.show()

def PlotBlayerParam(Foil, index):
	"""
	Generates the plots of the boundary layer state vector variables given the index:
	1 -> Displacement thickness
	2 -> Momentum thickness
	3 -> Velocity
	4 -> Skin friction coefficient
	"""
	fig, ax = plt.subplots()
	Is = Foil.vsol.Is
	x_lower = Foil.geom.coord[Is[0], 0]
	x_upper = Foil.geom.coord[Is[1], 0]
	x_wake = Foil.wake.x[0, :]
	
	parameters = [Foil.post.ds, Foil.post.th, Foil.post.ue, Foil.post.cf]
	titles = ["Displacement Thickness", "Momentum Thickness", "Edge velocity (ue)", "Skin Friction Coefficient (cf)"]
	ylabels = ["δ*", "θ", "$u_e$", "$C_f$"]
	
	param = parameters[index]
	param_lower = param[Is[0]]
	param_upper = param[Is[1]]
	param_wake = param[Is[2]]
	
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	
	ax.plot(x_upper, param_upper, linewidth=3)
	ax.plot(x_lower, param_lower, linewidth=3)
	ax.plot(x_wake, param_wake, linewidth=3)
	
	ax.legend(["Upper", "Lower", "Wake"])
	ax.grid(True)
	
	plt.title(titles[index], fontdict=font)
	plt.xlabel("x", fontdict=font)
	plt.ylabel(ylabels[index], fontdict=font)
	plt.show()
