####################################
# Plotting tools for ROMS solutions
#Daniel Dauhajre, 2022
####################################
import os
import sys
from netCDF4 import Dataset as netcdf
import numpy as np
import ROMS_depths as RD
import ROMS_tools as RT
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import NullFormatter
import cmocean as cmocean
import custom_cmap as cm_cust
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1 import AxesGrid
####################################################


def make_levs(vmin,vmax,nlevs,aslist=False):
    dv = (vmax - vmin) / nlevs
    levs =  np.arange(vmin,vmax+dv,dv)
    if aslist:
       return list(levs)
    else:
        return levs


def set_rcs(tick_dir="out"):
    '''
    set basic plotting aesthetics
    '''
    plt.rc('text', usetex=True) 
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    sns.set()
    sns.set_style("ticks", {"axes.linewidth":2.5,"xtick.direction":tick_dir, "ytick.direction":tick_dir,"ytick.major.size":7, "xtick.major.size":7})
    sns.set_style("ticks", {"axes.linewidth":2.5,"xtick.direction":tick_dir, "ytick.direction":tick_dir,"ytick.minor.size":3.5, "xtick.minor.size":3.5})


    plt.rcParams['xtick.major.width']=2.5
    plt.rcParams['ytick.major.width']=2.5
    plt.rcParams['xtick.minor.width']=1.5
    plt.rcParams['ytick.minor.width']=1.5


    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
   #Set monospace family to Courier new
    #plt.rcParams["font.monospace"]="Courier New"
    #Set default font family to monospace
    #plt.rcParams["font.family"] = "monospace"



def set_fonts(axis=24,tick=22,cbar_tick=22):
    '''
    set default axis/tick/cbar font sizes
    '''
    axis_font = axis
    tick_font = tick
    cbar_tick_font = cbar_tick
    return axis_font, tick_font, cbar_tick_font


def get_var_2D(vname,vtype,t,roms_nc,grd_nc,kv=-1):
    '''
    Based on name of variable from raw ROMS output
    variables return a 2-D variable (e.g., relative vorticity normalized by f0)

    #Variable can be calculated or from history file
    
    '''
    ######################
    #Raw history variable
    ####################
    if vtype=='his':
       #Check 2D or 3D
       if len(roms_nc.variables[vname].shape)==4:
           #3-D variable
           var_out = roms_nc.variables[vname][t,kv,:,:]
           if vname=='u':
              var_out = RT.u2rho(roms_nc.variables[vname][t,kv,:,:])
           if vname=='v':
              var_out = RT.v2rho(roms_nc.variables[vname][t,kv,:,:])
           if vname == 'rho':
              #Potential density
              var_out = (var_out + roms_nc.rho0) - 1000.
       else:
           #2-D variable
           var_out = roms_nc.variables[vname][t,:,:] 
           if vname=='ubar':
              var_out = RT.u2rho(roms_nc.variables[vname][t,:,:])
           if vname=='vbar':
              var_out = RT.v2rho(roms_nc.variables[vname][t,:,:])

    ###################################################


    #######################
    #Calculated from Output
    ########################
    if vtype=='calc':
        pm = grd_nc.variables['pm'][:,:]
        pn = grd_nc.variables['pn'][:,:]
        f0 = grd_nc.variables['f'][:,:]
        if vname =='velocity_mag':
           u = RT.u2rho(roms_nc.variables['u'][t,kv,:,:])
           v = RT.v2rho(roms_nc.variables['v'][t,kv,:,:])
           var_out = np.sqrt(u**2 + v**2)
        if vname == 'u_minus_ubar':
           var_out = RT.u2rho(roms_nc.variables['u'][t,kv,:,:] - roms_nc.variables['ubar'][t,:,:])
        #Buoyancy gradient magnitude
        if vname == 'grad_b_mag':
           try:
              b = -9.81 * roms_nc.variables['rho'][t,kv,:,:] / roms_nc.rho0
           except:
              #Hardcoded for now.
              print '   USING TEMPERATURE FOR BUOYANCY!!!  '
              b = 9.81 * 2e-4 * roms_nc.variables['temp'][t,kv,:,:] / roms_nc.rho0
           var_out = RT.grad_mag_2D(b,pm,pn)
        #Variables based on horizontal velocities     
        if vname=='div' or vname=='vort' or vname=='strain':
           u = roms_nc.variables['u'][t,kv,:,:]
           v = roms_nc.variables['v'][t,kv,:,:]
           if vname=='div':
              var_out = RT.div_2D(u,v,pm,pn,f0) 
           if vname=='vort':
              var_out = RT.vort_2D(u,v,pm,pn,f0)  
           if vname=='strain':
              var_out=RT.curvilinear_strain(u,v,pm,pn) / f0
        if vname == 'wbar' or vname=='bvfbar':
           [ny,nx] = roms_nc.variables['temp'][t,-1,:,:].shape
           z_r, z_w = RD.get_zr_zw_tind(roms_nc,grd_nc,t,[0,ny,0,nx])
           if vname == 'wbar':
             w_full = roms_nc.variables['w'][t,:,:,:]
             var_out = np.trapz(w_full,z_r,axis=0) / grd_nc.variables['h'][:,:]
           if vname == 'bvfbar':
              dens = roms_nc.variables['rho'][t,:,:,:]
              b = -9.81 * dens / roms_nc.rho0
              db_dz = (b[1:,:,:] - b[:-1,:,:]) / (z_r[1:,:,:] - z_r[:-1,:,:])
              var_out = np.trapz(db_dz,z_w[1:-1,:,:],axis=0) / grd_nc.variables['h'][:,:] 
    ###############################################

    return var_out
    ##################################################

def get_var_xzyz_ana(vname,vtype,htype,ztype,t,var_nc,roms_nc,grd_nc,xz=True,yz=False,i=0,j=0):
    '''
    Get 2-D cross-section (transect)
    
    for (x,z) idealized analysis (no eta direction)

    returns [nx (or ny), nz] type arrays


    Applicable for multiple roms output netcdf (roms_nc, bgc_nc, plant_nc, etc.)
    '''
    #Caculate depths
    [ny,nx] = grd_nc.variables['pm'].shape
    z_r, z_w = RD.get_zr_zw_tind(roms_nc,grd_nc,t,[0,ny,0,nx])
    nz = z_r.shape[0]
    ######################
    #Raw history variable
    ####################
    var_out = var_nc.variables[vname][t,:,:]

    ###############
    #Generate mesh
    ################
    if xz:
       if ztype=='rho':
          zz = z_r[:,j,:]
       if ztype=='w':
          zz = z_w[:,j,:]
       if 'x_rho' in grd_nc.variables.keys():
           xc = np.asarray(grd_nc.variables['x_rho'][j,:])
       else:
           xc = np.arange(nx) 
       dx = np.nanmean(1./grd_nc.variables['pm'][j,:])

    if yz:
       if ztype=='rho':
          zz = z_r[:,:,i]
       if ztype=='w':
          zz = z_w[:,:,i]
       if 'y_rho' in grd_nc.variables.keys():
           xc = np.asarray(grd_nc.variables['y_rho'][:,i])
       else:
           xc = np.arange(ny) 
       dx = np.nanmean(1./grd_nc.variables['pn'][:,i])

    [xx,z_grd] = np.mgrid[0:len(xc),0:zz.shape[0]]
    xx = (xx * dx/1e3) + xc[0]/1e3

    #print xx.shape
    #print zz.shape
    #print z_grd.shape
    #for k  in range(zz.shape[0]):
    #    xx[:,k] = (xc* dx/1e3)

    return var_out.T,xx,zz.T


def get_var_xzyz_bgc(vname,vtype,htype,ztype,t,var_nc,roms_nc,grd_nc,xz=True,yz=False,i=0,j=0):
    '''
    Get 2-D cross-section (transect)
    variable across xi-axis (xz=True,yz=False)
    or eta-axis (xz=False,yz=True)
    i,j --> xi,eta point to take transect across

    returns [nx (or ny), nz] type arrays


    Applicable for multiple roms output netcdf (roms_nc, bgc_nc, plant_nc, etc.)
    '''
    #Caculate depths
    [ny,nx] = grd_nc.variables['pm'].shape
    z_r, z_w = RD.get_zr_zw_tind(roms_nc,grd_nc,t,[0,ny,0,nx])
    nz = z_r.shape[0]
    ######################
    #Raw history variable
    ####################
    if vtype=='his':      
       if xz:
          if htype=='u':
             var_out = RT.u2rho(var_nc.variables[vname][t,:,:,:])[:,j,:]
          if htype=='v':
             var_out = RT.v2rho(var_nc.variables[vname][t,:,:,:])[:,j,:]
          if htype=='rho':
             var_out = var_nc.variables[vname][t,:,j,:]
       if yz:
          if htype=='u':
              var_out = RT.u2rho(var_nc.variables[vname][t,:,:,:])[:,:,i]
          if htype=='v':
             var_out = RT.v2rho(var_nc.variables[vname][t,:,:,:])[:,:,i]
          if htype=='rho':
             var_out = var_nc.variables[vname][t,:,:,i]
       if vname=='rho':
          #Potential density
          var_out = (var_out + roms_nc.rho0) - 1000.

    if vtype=='calc':
       if vname=='u_minus_ubar':
           u = RT.u2rho(var_nc.variables['u'][t,:,:,:])[:,j,:]
           ubar = RT.u2rho(var_nc.variables['ubar'][t,:,:])[j,:]
           var_out = np.zeros(u.shape)
           for k in range(var_out.shape[0]):
               var_out[k,:] = u[k,:] - ubar
       if vname=='velocity_mag':
          if xz:
             u = RT.u2rho(var_nc.variables['u'][t,:,:,:])[:,j,:]
             v = RT.v2rho(var_nc.variables['v'][t,:,:,:])[:,j,:]
          if yz:
             u = RT.u2rho(var_nc.variables['u'][t,:,:,:])[:,:,i]
             v = RT.v2rho(var_nc.variables['v'][t,:,:,:])[:,:,i]
          var_out = np.sqrt(u**2 + v**2)

       
       #Variables based on horizontal velocities     
       if vname=='div' or vname=='vort' or vname=='strain':
           u = roms_nc.variables['u'][t,:,:,:]
           v = roms_nc.variables['v'][t,:,:,:]
           pm = grd_nc.variables['pm'][:,:]
           pn = grd_nc.variables['pn'][:,:]
           f0 = grd_nc.variables['f'][:,:]
           if vname=='div':
              var_3D = RT.div_3D(u,v,pm,pn,f0) 
           if vname=='vort':
              var_3D = RT.vort_3D(u,v,pm,pn,f0) 
           
           if xz:
              var_out = var_3D[:,j,:]
           if yz:
              var_out = var_3D[:,:,i]
 
       #Buoyancy gradient magnitude
       if vname == 'grad_b_mag':
           pm = grd_nc.variables['pm'][:,:]
           pn = grd_nc.variables['pn'][:,:]
           try:
             b = -9.81 * roms_nc.variables['rho'][t,:,:,:] / roms_nc.rho0
           except:
             #Hardcoded for now.
             print '   USING TEMPERATURE FOR BUOYANCY!!!  '
             b = 9.81 * 2e-4 * roms_nc.variables['temp'][t,:,:,:] / roms_nc.rho0
           
           var_3D = RT.grad_mag_3D(b,pm,pn)
           
           if xz:
              var_out = var_3D[:,j,:]
           if yz:
              var_out = var_3D[:,:,i]
 
        #print 'Sorry, not functional yet'
        #sys.exit()
 
    ###############
    #Generate mesh
    ################
    if xz:
       if ztype=='rho':
          zz = z_r[:,j,:]
       if ztype=='w':
          zz = z_w[:,j,:]
       if 'x_rho' in grd_nc.variables.keys():
           xc = np.asarray(grd_nc.variables['x_rho'][j,:])
       else:
           xc = np.arange(nx) 
       dx = np.nanmean(1./grd_nc.variables['pm'][j,:])

    if yz:
       if ztype=='rho':
          zz = z_r[:,:,i]
       if ztype=='w':
          zz = z_w[:,:,i]
       if 'y_rho' in grd_nc.variables.keys():
           xc = np.asarray(grd_nc.variables['y_rho'][:,i])
       else:
           xc = np.arange(ny) 
       dx = np.nanmean(1./grd_nc.variables['pn'][:,i])

    [xx,z_grd] = np.mgrid[0:len(xc),0:zz.shape[0]]
    xx = (xx * dx/1e3) + xc[0]/1e3

    #print xx.shape
    #print zz.shape
    #print z_grd.shape
    #for k  in range(zz.shape[0]):
    #    xx[:,k] = (xc* dx/1e3)

    return var_out.T,xx,zz.T



def get_var_xzyz(vname,vtype,ztype,t,roms_nc,grd_nc,xz=True,yz=False,i=0,j=0):
    '''
    Get 2-D cross-section (transect)
    variable across xi-axis (xz=True,yz=False)
    or eta-axis (xz=False,yz=True)
    i,j --> xi,eta point to take transect across

    returns [nx (or ny), nz] type arrays
    '''
    #Caculate depths
    [ny,nx] = grd_nc.variables['pm'].shape
    z_r, z_w = RD.get_zr_zw_tind(roms_nc,grd_nc,t,[0,ny,0,nx])
    nz = z_r.shape[0]
    ######################
    #Raw history variable
    ####################
    if vtype=='his':      
       if xz:
          if vname=='u':
             var_out = RT.u2rho(roms_nc.variables[vname][t,:,:,:])[:,j,:]
          elif vname=='v':
             var_out = RT.v2rho(roms_nc.variables[vname][t,:,:,:])[:,j,:]
          else:
             var_out = roms_nc.variables[vname][t,:,j,:]
       if yz:
          if vname=='u':
             var_out = RT.u2rho(roms_nc.variables[vname][t,:,:,:])[:,:,i]
          elif vname=='v':
             var_out = RT.v2rho(roms_nc.variables[vname][t,:,:,:])[:,:,i]
          else:
             var_out = roms_nc.variables[vname][t,:,:,i]
       if vname=='rho':
          #Potential density
          var_out = (var_out + roms_nc.rho0) - 1000.

    if vtype=='calc':
       if vname=='u_minus_ubar':
           u = RT.u2rho(roms_nc.variables['u'][t,:,:,:])[:,j,:]
           ubar = RT.u2rho(roms_nc.variables['ubar'][t,:,:])[j,:]
           var_out = np.zeros(u.shape)
           for k in range(var_out.shape[0]):
               var_out[k,:] = u[k,:] - ubar
        
        
        #Variables based on horizontal velocities     
       if vname=='div' or vname=='vort' or vname=='strain':
           u = roms_nc.variables['u'][t,:,:,:]
           v = roms_nc.variables['v'][t,:,:,:]
           if vname=='div':
              var_3D = RT.div_3D(u,v,pm,pn,f0) 
           if vname=='vort':
              var_3D = RT.vort_3D(u,v,pm,pn,f0) 
           
           if xz:
              var_out = var_3D[:,j,:]
           if yz:
              var_out = var_3D[:,:,i]
        
        #print 'Sorry, not functional yet'
        #sys.exit()
 
    ###############
    #Generate mesh
    ################
    if xz:
       if ztype=='rho':
          zz = z_r[:,j,:]
       if ztype=='w':
          zz = z_w[:,j,:]
       if 'x_rho' in grd_nc.variables.keys():
           xc = np.asarray(grd_nc.variables['x_rho'][j,:])
       else:
           xc = np.arange(nx) 
       dx = np.nanmean(1./grd_nc.variables['pm'][j,:])

    if yz:
       if ztype=='rho':
          zz = z_r[:,:,i]
       if ztype=='w':
          zz = z_w[:,:,i]
       if 'y_rho' in grd_nc.variables.keys():
           xc = np.asarray(grd_nc.variables['y_rho'][:,i])
       else:
           xc = np.arange(ny) 
       dx = np.nanmean(1./grd_nc.variables['pn'][:,i])

    [xx,z_grd] = np.mgrid[0:len(xc),0:zz.shape[0]]
    xx = (xx * dx/1e3) + xc[0]/1e3

    #print xx.shape
    #print zz.shape
    #print z_grd.shape
    #for k  in range(zz.shape[0]):
    #    xx[:,k] = (xc* dx/1e3)

    return var_out.T,xx,zz.T




def plot_latlon_bounds(lon,lat,col,lwidth):
    plt.plot(lon[0,:],lat[0,:],color=col,linewidth=lwidth)
    plt.plot(lon[:,-1],lat[:,-1],color=col,linewidth=lwidth)
    plt.plot(lon[:,0],lat[:,0],color=col,linewidth=lwidth)
    plt.plot(lon[-1,:],lat[-1,:],color=col,linewidth=lwidth)

def get_pst_time(ocean_time,rs):
    '''
    get pacific standard time string based on
    ocean_time in seconds and rs (ROMS_solution object)
    '''
    pst_time = rs.get_PST_date_rounded(ocean_time)
    pst_str = pst_time.strftime("%m/%d/%Y") + '\n' + pst_time.strftime("%H:%M")
    return pst_str, pst_time

def add_baths(axi,h,baths,lw=0.75,cb='slategrey',ext=[],add_label=False,fs=12):
    '''
    add bathymetry
    '''
    if not ext:
       ch = axi.contour(h,baths,colors=cb,linewidths=lw)
    else:
       ch = axi.contour(h,baths,colors=cb,linewidths=lw,extent=ext)
    if add_label:
       ch.clabel(inline=1,fontsize=fs,fmt='%1.f')

def im_map(var,var_min,var_max,cmap_var,im_ext,cbar_label,fs,ts,spec_ticks=False,cticks=[],lognorm=False):
    '''
    imshow of a 2D variable 
    '''
    im = plt.imshow(var,vmin=var_min,vmax=var_max,cmap=cmap_var,origin='lower',aspect='auto',extent=im_ext)
    if lognorm:
       from matplotlib import colors
       im = plt.imshow(var,cmap=cmap_var,origin='lower',aspect='auto',extent=im_ext,norm=colors.LogNorm(vmin=var_min,vmax=var_max))
    if spec_ticks:
       cbar1 = plt.colorbar(im,ticks=cticks)
    else:
       cbar1 = plt.colorbar(im)
    im.figure.axes[1].tick_params(axis='both',which='major',labelsize=ts)
    #cbar1.axi.tick_params(axis='both',which='major',labelsize=ts)
    cbar1.set_label(cbar_label,fontsize=fs)

    #return im



def im_map_ax(i,axi,var,var_min,var_max,cmap_var,im_ext,cbar_label,fs,ts,add_cbar=False,spec_ticks=False,cticks=[],lognorm=False):
    '''
    imshow of a 2D variable
    with ax generally to be used with AxesGrid subplots
    '''
    if lognorm:
       from matplotlib import colors
       im = axi[i].imshow(var,cmap=cmap_var,origin='lower',aspect='auto',extent=im_ext,norm=colors.LogNorm(vmin=var_min,vmax=var_max))
    else:
       im = axi[i].imshow(var,vmin=var_min,vmax=var_max,cmap=cmap_var,origin='lower',aspect='auto',extent=im_ext)
    if add_cbar:
        if spec_ticks:
           cbar1 = axi[0].cax.colorbar(im,ticks=cticks)
        else:
           cbar1 = axi[0].cax.colorbar(im)
        cbar1.set_label_text(cbar_label,fontsize=fs)
        cbar1.ax.tick_params(axis='both',which='major',labelsize=ts)


def contour_xz_ax(i,axi,xx,zz,var,var_levs,cmap_var,cbar_label,fs,ts,add_cbar=False,spec_ticks=False,cticks=[],extend_opt='neither'):
    '''
    Contour transect (horizontal, depth)
    '''
    cf = axi[i].contourf(xx,zz,var,levels=var_levs,cmap=cmap_var,extend=extend_opt)
    if add_cbar:
        if spec_ticks:
           cbar1 = axi[0].cax.colorbar(cf,ticks=cticks)
        else:
           cbar1 = axi[0].cax.colorbar(cf)
        cbar1.set_label_text(cbar_label,fontsize=fs)
        cbar1.ax.tick_params(axis='both',which='major',labelsize=ts)

def contour_xz_ax_lines(i,axi,xx,zz,var,levs_cont,lw=1.5,col='k',label_c=False,fs_clabel=12):
    '''
    Contour lines (e.g., isopycnals)
    '''
    cf = axi[i].contour(xx,zz,var,levels=levs_cont,linewidths=lw,colors=col)
    if label_c:
       cf.clabel(inline=1,fontsize=fs_clabel,fmt='%1.0f')


def plot_BL(i,axi,x,sbl,bbl,cols=['k','darkcyan'], lw=3,lst='--',plot_bbl=True):
    '''
    Plot surface and bottom boundary layers,
    BBL should be height above bottom
    '''
    axi[i].plot(x,-sbl, color=cols[0],linewidth=lw,linestyle=lst)
    if plot_bbl:
       axi[i].plot(x,bbl, color=cols[1], linewidth=lw,linestyle=lst)
     


def plot_xz_box(i,axi,x1,x2,z,col='k',lw=1,lst='--'):
    '''
    Used to plot a kelp farm
    '''
    axi[i].vlines(x=x1,ymin=z,ymax=0,color=col,linestyle=lst,linewidth=lw)
    axi[i].vlines(x=x2,ymin=z,ymax=0,color=col,linestyle=lst,linewidth=lw)
    axi[i].hlines(y=z,xmin=x1,xmax=x2,color=col,linestyle=lst,linewidth=lw)

def im_map_ax_powernorm(i,axi,var,var_min,var_max,cmap_var,im_ext,cbar_label,fs,ts,add_cbar=False,spec_ticks=False,cticks=[],ga=0.5):
    '''
    imshow of a 2D variable
    with ax generally to be used with AxesGrid subplots
    '''
    from matplotlib import colors
    im = axi[i].imshow(var,vmin=var_min,vmax=var_max,cmap=cmap_var,origin='lower',aspect='auto',extent=im_ext,norm=colors.PowerNorm(gamma=ga))
    if add_cbar:
        if spec_ticks:
           cbar1 = axi[0].cax.colorbar(im,ticks=cticks)
        else:
           cbar1 = axi[0].cax.colorbar(im)
        cbar1.set_label_text(cbar_label,fontsize=fs)
        cbar1.ax.tick_params(axis='both',which='major',labelsize=ts)




def set_ax_labels(axi,xlab, ylab,fs,nullx=False,nully=False):
    '''
    Set axis labels

    nullx = nullformat x-axis
    nully = nullformat y-axis
    '''
    if nullx:
       axi.xaxis.set_major_formatter(NullFormatter())
    else:
       axi.set_xlabel(xlab,fontsize=fs)
    if nully:
       axi.yaxis.set_major_formatter(NullFormatter())
    else:
       axi.set_ylabel(ylab,fontsize=fs)

def add_text(axi,text_str,xt,yt,fs=18,a_in=0,l_in=0,col='k'):
    #font = {'family': 'sans-serif',
    #    'color':  'black',
    #    'weight': 'normal',
    #    'size': fs,
    #    }
    props = dict(boxstyle='round', facecolor='white', alpha=a_in,lw=l_in)
    axi.text(xt,yt,  text_str, transform=axi.transAxes, fontsize=fs,verticalalignment='top', bbox=props,color=col)


def add_cbar(axi,cbar_label,fs,ts,spec_ticks=False,cticks=[]):
    if spec_ticks:
       cbar1 = plt.colorbar(ticks=cticks)
    else:
       cbar1 = plt.colorbar()
    cbar1.set_label(cbar_label,fontsize=fs)
    return cbar1
    #cbar1.axi.tick_params(axis='both',which='major',labelsize=ts)

def set_patch(axi,patch_col='grey'):
    axi.patch.set_color('grey')

def set_tick_params(axi,ts):
    axi.tick_params(axis='both',which='major',labelsize=ts)
    axi.tick_params(axis='both',which='minor',labelsize=ts)
    
