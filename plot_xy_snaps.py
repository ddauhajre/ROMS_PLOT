

########################################
import os
import sys
sys.path.append('/home/dauhajre/ROMS_PLOT/ROMS_PY/')
from netCDF4 import Dataset as netcdf
import numpy as np
import ROMS_depths as RD
import ROMS_tools as RT
import matplotlib.pyplot as plt
import cmocean as cmocean

#RTP is my main plotting library
# it sets a aesthetics of plots and has many plotting functions
import R_tools_plot as RTP
from mpl_toolkits.axes_grid1 import AxesGrid
#####################################


#Set some global plotting parameters (tick sizes , line thicknesses,etc.)
RTP.set_rcs()
axis_font,tick_size,cbar_tick_size = RTP.set_fonts(axis=22,cbar_tick=14)


path_code = '/home/dauhajre/ROMS_PLOT/'

##########################
#Solution and figure name / files to plot
'''
Convention should be that your solution is in a 
directory that is
path_sims + sim_name + '/'

history files need to be in a subdirectory called '/his/'
input files (grd, mag, etc.) need to be in a subdirectory
called '/Input/'

You can change this in xy_snaps.py (which is called from this setup script
'''

path_sims = '/mnt/kamiya/dauhajre/Kelp_Sandbox/'
#Solution name
sim_name = 'IFarm5sink'
grd_name = 'IFarm5_grd.nc'
kelp_name = 'IFarm5_mag.nc'
#Base name to save snapshots
fig_name = sim_name + '_xy_snaps'


#AXES LIMITS FOR PLOTTING (CHOOSE YOURSELF)
#Commented out for now, see associated commented out lines
# in xy_snaps.py 
#[x0_lim,x1_lim]=[0,20]
#[y0_lim,y1_lim]=[-4.8,4.8]


#Output files to plot over
n1 = 8 #first file
n2 = 9 #last file
#######################

#####################################
#Hardcoded locations of kelp (a box)
#####################################
#Center xi axis point
ic_farm=10
#Center eta axis point
jc_farm=16
#Length of farm in xi (x)
ni_farm=10
#Length of farm in eta (y)
nj_farm=6

#################################################
#Variables to plot and associated limits/ticks/etc.
'''
Below is where you choose what variables to plot
and their colorbar limits

Hopefully the comments are self-explanatory

These lists are used in the (hopefully) generalized 
function in xy_snaps.py that grabs variables from roms
files and plots with the limits/colormaps given here
'''
#Variable names, if raw data use same as netcdf name
vars_plot = ['dye1','dye2'] 
#variable in roms or bgc output file
var_file_type = ['roms','roms']
#Vertical coordinate type ('rho', or 'w')
z_type = ['rho','rho']
#Horizontal coordinate type ('rho', 'u', 'v')
h_type = ['rho','rho']
#Sigma level to plot xy (-1=surface)
k_var  = [-20,-20]
#Take from history file (his) or calculate (calc)
var_type  = ['his','his']
#Colorbar limits
xy_lims = [[0,1],[0,1]]
#Amplitude of variable levels (so it will plot var_amps[n] * var)
var_amps = [1,1]
#Colorbar labels (strings)
var_labels = ['Dye1', 'Dye2']
var_cmap =[cmocean.cm.thermal,cmocean.cm.thermal]
#Contour extend for (x,z)
cf_extend = ['both', 'both']
#Nullformat (True) for x or y-axis
nullx_var = [True,False]
nully_var = [False,False]


##################################################
#Automated tick maker
nticks =4. 
nlevs = 100. #contour levels
xy_ticks = [[] for va in range(len(vars_plot))]
xy_levs = [[] for va in range(len(vars_plot))]


for va in range(len(vars_plot)):
    xy_ticks[va] = RTP.make_levs(xy_lims[va][0],xy_lims[va][1],nticks,aslist=True)
    xy_levs[va] = RTP.make_levs(xy_lims[va][0],xy_lims[va][1],nlevs,aslist=False) 
###############################################




################################################


#################################################
#Figure parameters
aspect = 1/2.  #H/L of L3 mag grid with sponge layer cut out 
H = 4 
L = H / aspect
#+2 on L leaves room for colorbar
L_fig = L+2
H_fig = H*2 

#AxesGrid subplot parameters edit to make orientation you want (e.g., (2,2,1) or (4,1,1), etc)
#subp = AxesGrid(fig,(subp[0], subp[1],va+1)
subp = [2,1] 
ax_pad = 0.75 #axes pad
cbar_loc = "right" #colorbar location
cbar_p = 0.04 #colorbar pad
cbar_s = "5%" #colorbar size

#Time text location
x_time = 0.04
y_time = 0.1

#################################################



#############################
# Run plotting
##############################
execfile(path_code + 'xy_snaps.py')

