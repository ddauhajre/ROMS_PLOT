'''
Make snapshots of multiple (x,y) fields
from ROMS solution output (his or avg)
'''
########################################################
#Called with master script, e.g., plot_snaps_4panel.py
####################################################
#Load output
path_grd = path_sims + sim_name + '/Input/'
grd_nc = netcdf(path_grd + grd_name, 'r')

path_his = path_sims + sim_name + '/his/'
path_diag = path_sims + sim_name + '/diag/'

#History
path_files, dirs_out,file_names = os.walk(path_his).next()
file_names.sort()
nfiles = len(file_names)


#Get kelp file
kelp_nc = netcdf(path_grd + kelp_name, 'r')
PlantMask = kelp_nc.variables['mask_kelp'][:,:]
#Get first and last points of farm
#xfarms = np.where(PlantMask[j0,:]==1)[0]
#if0 = xfarms[0]
#if1 = xfarms[-1]


############
#Grid parameters
mask_rho = grd_nc.variables['mask_rho'][:,:]
mask_nan = np.copy(mask_rho)
mask_nan[mask_nan==0.] = np.nan

pm = grd_nc.variables['pm'][:,:]
pn = grd_nc.variables['pn'][:,:]
dx = np.nanmean(1./pm)
dy = np.nanmean(1./pn)


xr_1 = grd_nc.variables['x_rho'][0,:] / 1e3
xr = xr_1 - xr_1[0]
yr = grd_nc.variables['y_rho'][:,0] / 1e3


[ny,nx] = grd_nc.variables['mask_rho'].shape
y0 = yr[0]
y1 = yr[-1]
x0 = xr[0] 
x1 = xr[-1] 


Lx_km = nx * dx / 1e3
Ly_km = ny * dy / 1e3
im_ext_km = [x0,x1,y0,y1]
aspect = np.float(ny) / nx

f0 = grd_nc.variables['f'][:,:]
km_label = r'$[\rm{km}]$'
z_label = r'$z\;[\rm{m}]$'

################################
# Plotting
################################
RT.change_dir_py(fig_name)
#tind=0
counter=0
for n in range(n1,n2):
    print 'Loading file: ' + file_names[n]
    roms_nc = netcdf(path_his + file_names[n], 'r')
    try:
       print 'Loading file: ' + diag_names[n]
       diag_nc = netcdf(path_diag + diag_names[n], 'r')
    except:
       diag_nc = roms_nc
    nt_nc = len(roms_nc.variables['ocean_time'][:])

    #Get number of file for saving
    ind1 = file_names[n].find('.')+1
    str_temp = file_names[n][ind1::]
    ind2 = str_temp.find('.')
    str_t = str_temp[0:ind2]
    tind = int(str_t) 
    for t in range(nt_nc):
        fig = plt.figure(figsize=[L_fig,H_fig])
        ax_vars = [AxesGrid(fig,(subp[0], subp[1],va+1),nrows_ncols=(1,1),axes_pad=ax_pad,share_all=False,label_mode="L",cbar_location=cbar_loc,cbar_mode="single",cbar_pad=cbar_p, cbar_size=cbar_s,aspect=False) for va in range(len(vars_plot))]
        
          
        for va in range(len(vars_plot)):
            ########################
            #Grab/calculate variable
            ########################
            var_nc = roms_nc
            if var_file_type[va]=='diag':
               var_nc = diag_nc
           
            var_xy= RTP.get_var_2D(vars_plot[va],var_type[va],t,var_nc,grd_nc,kv=k_var[va]) * mask_nan
            
            #Get time
            otime_hr = roms_nc.variables['ocean_time'][t] / 3600.
            otime_str = r'$t=$'+str(otime_hr) + ' hr'
           
            ################################
            #Plotting
            ################################
            ax_in = ax_vars[va]
            #x,y plot
            RTP.im_map_ax(0,ax_in,var_xy*var_amps[va], xy_lims[va][0],xy_lims[va][1],var_cmap[va],im_ext_km,var_labels[va],axis_font,cbar_tick_size,add_cbar=True,spec_ticks=True,cticks=xy_ticks[va])          #ax_in = plt.gca()
            RTP.set_tick_params(ax_in[0],tick_size)
            RTP.set_patch(ax_in[0])
            if va ==0:
               RTP.add_text(ax_in[0],otime_str,x_time,y_time,fs=16,a_in=0.5,l_in=1) 
            RTP.set_ax_labels(ax_in[0],km_label,km_label,axis_font,nullx=nullx_var[va],nully=nully_var[va])

            #Constrain axis limits
            #ax_in[0].set_xlim([x0_lim,x1_lim])
            #ax_in[0].set_ylim([y0_lim,y1_lim])




            [ny,nx] = grd_nc.variables['h'].shape
            dx = 1./ grd_nc.variables['pm'][10,10]
            #xr = grd_nc.variables['x_rho'][0,:] / 1e3
            #yr = grd_nc.variables['y_rho'][:,0] / 1e3
            [if0,if1] = [ic_farm-ni_farm/2, ic_farm+ni_farm/2]
            [jf0,jf1] = [jc_farm-nj_farm/2,jc_farm+nj_farm/2]
            ax_in[0].vlines(x=xr[if0],ymin=yr[jf0],ymax=yr[jf1],color='k',linestyle='--')
            ax_in[0].vlines(x=xr[if1],ymin=yr[jf0],ymax=yr[jf1],color='k',linestyle='--')
            ax_in[0].hlines(y=yr[jf0],xmin=xr[if0],xmax=xr[if1],color='k',linestyle='--')
            ax_in[0].hlines(y=yr[jf1],xmin=xr[if0],xmax=xr[if1],color='k',linestyle='--')


        plt.savefig(fig_name + '_'+str(tind).zfill(5),bbox_inches='tight')
        plt.close()
        print 'Saved figure at t = ' + str(tind)
        tind+=1
        counter+=1
    #    break
    #break

