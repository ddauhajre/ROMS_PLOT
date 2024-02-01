
#  Adapted from Jon Gula's code
#Daniel Dauhajre, UCLA 2013
#module containing various roms tools(no depths in here, depths.py is module for that)

##########################################################################################3

#load necessary modules
from pylab import *
from scipy.io.netcdf import netcdf_file as netcdf
import numpy as np
import os
from numba import autojit, prange, jit, njit
import sys
sys.path.append('/home/dauhajre/ROMS_PY/balance_tools_esther/R_tools_fort_routines/')
import tools_fort as toolsF

def make_levs(vmin,vmax,nlevs):
    dv = (vmax - vmin) / nlevs
    return np.arange(vmin,vmax+dv,dv)

def Zeros(*args, **kwargs):
  kwargs.update(dtype=float32)
  return np.zeros(*args, **kwargs) 


def change_dir_py(dir_name):
    if not os.path.exists(dir_name):
       print 'Creating directory: ' + dir_name
       os.makedirs(dir_name)
    print 'Moving into directory: ' + dir_name
    os.chdir(dir_name)
    ####


def mask_2D_var(var_in,mask_rho):
    '''
    Multiply 2D variable by mask
    to nan-out masked regions
    '''
    return var_in*mask_rho
    #######################################


def mask_3D_var(var_in,mask_rho):
    '''
    Multiply 3D variable by mask
    to nan-out masked regions
    '''
    var_masked = np.zeros(var_in.shape)
    for k in range(var_in.shape[0]):
        var_masked[k,:,:] = var_in[k,:,:] * mask_rho
    return var_masked
    #######################################


def int_lin_2d(f,x,y):
    # 2D Linear interpolation
    wt=np.empty((2,2))
    wt[0,0] = (1-x)*(1-y)
    wt[1,0] =    x *(1-y)
    wt[0,1] = (1-x)*   y
    wt[1,1] =    x *   y
    return(np.sum(f*wt))


#################################################
################################################
################################
# From tools.py (kaushik, roy, jon)
##################################

#######################################################
#Get distance from lon-lat grid
#######################################################

def lonlat_to_m(lon,lat):
    '''
    assumes [i,j] order
    '''
    lon = lon*2*np.pi/360.
    lat = lat*2*np.pi/360.
    dx = np.arccos(np.sin(lat[1:,:])*np.sin(lat[:-1,:]) + np.cos(lat[1:,:])*np.cos(lat[:-1,:])*np.cos(lon[1:,:]-lon[:-1,:]))*6371000.
    dy = np.arccos(np.sin(lat[:,1:])*np.sin(lat[:,:-1]) + np.cos(lat[:,1:])*np.cos(lat[:,:-1])*np.cos(lon[:,1:]-lon[:,:-1]))*6371000.
    return dx,dy







def vinterp(var, depths, z_r, z_w=None, mask=None,imin=0,jmin=0,kmin=1, floattype=np.float32,interp_sfc=1,interp_bot=0,below=None,**kwargs):


    if mask==None:  mask = np.ones((z_r.shape[0],z_r.shape[1]), order='F', dtype=floattype); mask[z_r[:,:,-1]==0] = 0

    if z_w is None: 
        print('no z_w specified')
        z_w=Zeros((z_r.shape[0],z_r.shape[1],z_r.shape[2]+1), order='F')
        z_w[:,:,1:-1] = 0.5*(z_r[:,:,1:] + z_r[:,:,:-1])
        z_w[:,:,0] = z_r[:,:,0] - (z_r[:,:,1]-z_r[:,:,0])
        z_w[:,:,-1] = z_r[:,:,-1] + (z_r[:,:,-1]-z_r[:,:,-2])
        
    if np.rank(depths)==1: newz = np.asfortranarray(Zeros((z_r.shape[0],z_r.shape[1],len(depths))) + depths, dtype=floattype)
    else: newz = depths

    if interp_bot==1:
        print("data will be interpolated below ground")
        below=1000.
        vnew=toolsF.sigma_to_z_intr_bot(z_r, z_w,mask,var,newz,below,imin,jmin,kmin,9999.)
    elif interp_sfc==1:
        print("no interpolation below ground")
        print(z_r.shape, z_w.shape,mask.shape,var.shape,newz.shape)
        vnew=toolsF.sigma_to_z_intr_sfc(z_r, z_w,mask,var,newz,imin,jmin,kmin,9999.)
    else:
        print("no interpolation below ground")
        vnew=toolsF.sigma_to_z_intr(z_r, z_w,mask,var,newz,imin,jmin,kmin,9999.)    

    
    vnew[np.abs(vnew)==9999.]=np.nan

    return vnew


########################################################
#     ESTHER:   FROM ARJUN (MARCH 2022)
#     An alternative to the methods below, 
#     because these set the uppermost sigma-level
#     horizontal gradients to 0. (and gives problems)
########################################################
#######################################################
#Compute horizontal derivatives on sigma-levels (1st order)
#######################################################
'''
var on rho-rho grid
dvardxi on psi-rho grid
'''

def diffxi(var,pm,z_r,z_w=None,newz=None,mask=None):


    if z_r.shape[2]<=2:
        dvardxi = diffxi_2d(var,pm,z_r,z_w,newz,mask)
    else:
        dvardxi = diffxi_3d(var,pm,z_r,z_w,newz,mask)

    ##############################################

    return dvardxi

#######################################################
#######################################################


def diffxi_3d(var,pm,z_r,z_w=None,newz=None,mask=None):


    if newz==None: newz = 0.5*(z_r[1:,:,:] + z_r[:-1,:,:])
    else: newz = rho2u(newz)

    dvardxi = Zeros((var.shape[0]-1,var.shape[1],var.shape[2]))

    ##############################################

    varzp = vinterp(var[1:,:,:],newz,z_r[1:,:,:],z_w[1:,:,:],interp_bot=1)
    varzm = vinterp(var[:-1,:,:],newz,z_r[:-1,:,:],z_w[:-1,:,:],interp_bot=1)

    dvardxi = ((varzp - varzm ).T*0.5*(pm[1:,:]+pm[:-1,:]).T ).T

    ##############################################

    return dvardxi


#######################################################
#######################################################


def diffxi_2d(var,pm,z_r,z_w=None,newz=None,mask=None):

    dvardxi = Zeros((z_r.shape[0]-1,z_r.shape[1]))

    ##############################################

    if newz==None: newz = 0.5*(z_r[:-1,:,0] + z_r[1:,:,0])
    else: newz = rho2u(newz)

    dz0 = (z_r[1:,:,0]-newz)
    dz1 = (newz-z_r[1:,:,1])
    varzp = (dz1*var[1:,:,0] + dz0*var[1:,:,1])/(z_r[1:,:,0]-z_r[1:,:,1])

    dz0 = (z_r[:-1,:,0]-newz)
    dz1 = (newz-z_r[:-1,:,1])
    varzm = (dz1*var[:-1,:,0] + dz0*var[:-1,:,1])/(z_r[:-1,:,0]-z_r[:-1,:,1])

    dvardxi = (varzp - varzm )*0.5*(pm[1:,:]+pm[:-1,:])
    ##############################################

    return dvardxi

#######################################################
#Compute horizontal derivatives on sigma-levels (1st order)
#######################################################

'''
var on rho-rho grid
dvardxi on psi-rho grid
'''

def diffeta(var,pn,z_r,z_w=None,newz=None,mask=None):


    if z_r.shape[2]<=2:
        dvardeta = diffeta_2d(var,pn,z_r,z_w,newz,mask)
    else:
        dvardeta = diffeta_3d(var,pn,z_r,z_w,newz,mask)

    ##############################################

    return dvardeta


#######################################################
#######################################################


def diffeta_3d(var,pn,z_r,z_w=None,newz=None,mask=None):


    if newz==None: newz = 0.5*(z_r[:,:-1,:] + z_r[:,1:,:])
    else: newz = rho2v(newz)

    dvardeta = Zeros((var.shape[0],var.shape[1]-1,var.shape[2]))

    ##############################################

    varzp = vinterp(var[:,1:,:],newz,z_r[:,1:,:],z_w[:,1:,:],interp_bot=1)
    varzm = vinterp(var[:,:-1,:],newz,z_r[:,:-1,:],z_w[:,:-1,:],interp_bot=1)

    dvardeta = ((varzp - varzm).T*0.5*(pn[:,:-1]+pn[:,1:]).T).T

    ##############################################


    return dvardeta



#######################################################
#Compute horizontal derivatives on sigma-levels (1st order)
#######################################################

def diffeta_2d(var,pn,z_r,z_w=None,newz=None,mask=None):

    dvardeta = Zeros((z_r.shape[0],z_r.shape[1]-1))

    ##############################################

    if newz==None: newz = 0.5*(z_r[:,:-1,0] + z_r[:,1:,0])
    else: newz = rho2v(newz)

    dz0 = (z_r[:,1:,0]-newz)
    dz1 = (newz-z_r[:,1:,1])
    varzp = (dz1*var[:,1:,0] + dz0*var[:,1:,1])/(z_r[:,1:,0]-z_r[:,1:,1])

    dz0 = (z_r[:,:-1,0]-newz)
    dz1 = (newz-z_r[:,:-1,1])
    varzm = (dz1*var[:,:-1,0] + dz0*var[:,:-1,1])/(z_r[:,:-1,0]-z_r[:,:-1,1])

    dvardeta = (varzp - varzm )*0.5*(pn[:,:-1]+pn[:,1:])

    ##############################################


    return dvardeta


########################################################
#     ESTHER:   END OF ARJUN'S SCRIPTS (MARCH 2022)
########################################################
'''
var on rho-rho grid
dvardxi on psi-rho grid
'''

############################################################
#....Compute horizontal derivatives using the chain rule as below.....# 
#......(d/dx)_z = (d/dx)_sigma - [(dz/dx)_sigma]* [(d/dz)]
#......(d/dy)_z = (d/dy)_sigma - [(dz/dy)_sigma]* [(d/dz)]
#....z_r and z_w passed to the func. must have same shape in x-y as var......#
############################################################

def diffxi_orig(var,pm,z_r,z_w=None,newz=None,mask=None):
    dvardxi = Zeros((var.shape[0]-1,var.shape[1],var.shape[2]))
    dz_r = z_r[:,:,1:]-z_r[:,:,:-1]
    dz_w = z_w[:,:,1:]-z_w[:,:,:-1]
    if (var.shape[2]==z_w.shape[2]):
        #.....var on psi-rho points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2u(var)[:,:,1:] + rho2u(var)[:,:,:-1])
        tmp = w2rho_s(rho2u(var),rho2u(z_r),rho2u(z_w))
        #.............(dvar/dx)|z = (dvar/dx)|sigma -(dvar/dz)(dz/dx)|sigma.........................#
        #.........dvar/dx|sigma...............#
        dvardxi[:,:,1:-1] = ((var[1:,:,1:-1] - var[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        #........-(dvar/dz)*(dz/dx)|sigma
        dvardxi[:,:,1:-1] = dvardxi[:,:,1:-1] - (((z_w[1:,:,1:-1] - z_w[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2u(dz_r))
        tmp2 = (rho2u(var)[:,:,1] - rho2u(var[:,:,0]))/rho2u(dz_w[:,:,0])
        dvardxi[:,:,0] = ((var[1:,:,0] - var[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,0] = dvardxi[:,:,0] - (tmp2 + (rho2u(z_w)[:,:,0] - rho2u(z_r)[:,:,0])*(dvardxi[:,:,1] - tmp2)/(rho2u(z_w[:,:,1]) - rho2u(z_r[:,:,0])))*(((z_w[1:,:,0] - z_w[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)
    elif (var.shape[2]==z_r.shape[2]):
        #.....var on w points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2u(var)[:,:,1:] + rho2u(var)[:,:,:-1])
        tmp = rho2w(rho2u(var),rho2u(z_r),rho2u(z_w))[:,:,1:-1]
        #.........................................................................#
        dvardxi[:,:,1:-1] = ((var[1:,:,1:-1] - var[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,1:-1] = dvardxi[:,:,1:-1] - (((z_r[1:,:,1:-1] - z_r[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2u(dz_w[:,:,1:-1]))
        #tmp = rho2w(rho2u(var),rho2u(z_r),rho2u(z_w))[:,:,1::]
        #dvardxi[:,:,1::] = ((var[1:,:,1::] - var[:-1,:,1::]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        #dvardxi[:,:,1::] = dvardxi[:,:,1::] - (((z_r[1:,:,1::] - z_r[:-1,:,1::]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2u(dz_w[:,:,1::]))
        tmp2 = (rho2u(var)[:,:,1] - rho2u(var[:,:,0]))/rho2u(dz_r[:,:,0])
        dvardxi[:,:,0] = ((var[1:,:,0] - var[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,0] = dvardxi[:,:,0] - (tmp2 + (rho2u(z_r)[:,:,0] - rho2u(z_w)[:,:,1])*(dvardxi[:,:,1] - tmp2)/(rho2u(z_r[:,:,1]) - rho2u(z_w[:,:,1])))*(((z_r[1:,:,0] - z_r[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)

        dvardxi[:,:,-1] = diffx(var[:,:,-1],pm)
    return dvardxi

def diffeta_orig(var,pn,z_r,z_w=None,newz=None,mask=None):
    dvardeta = Zeros((var.shape[0],var.shape[1]-1,var.shape[2]))
    dz_r = z_r[:,:,1:]-z_r[:,:,:-1]
    dz_w = z_w[:,:,1:]-z_w[:,:,:-1]
    if (var.shape[2]==z_w.shape[2]):
        #.....var on rho points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2v(var)[:,:,1:] + rho2v(var)[:,:,:-1])
        tmp = w2rho_s(rho2v(var),rho2v(z_r),rho2v(z_w))
        #.........................................................................#
        dvardeta[:,:,1:-1] = ((var[:,1:,1:-1] - var[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,1:-1] = dvardeta[:,:,1:-1] - (((z_w[:,1:,1:-1] - z_w[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2v(dz_r))
        tmp2 = (rho2v(var)[:,:,1] - rho2v(var[:,:,0]))/rho2v(dz_w[:,:,0])
        dvardeta[:,:,0] = ((var[:,1:,0] - var[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,0] = dvardeta[:,:,0] - (tmp2 + (rho2v(z_w)[:,:,0] - rho2v(z_r)[:,:,0])*(dvardeta[:,:,1] - tmp2)/(rho2v(z_w[:,:,1]) - rho2v(z_r[:,:,0])))*(((z_w[:,1:,0] - z_w[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)
    elif (var.shape[2]==z_r.shape[2]):
        #.....var on rho points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2v(var)[:,:,1:] + rho2v(var)[:,:,:-1])
        tmp = rho2w(rho2v(var),rho2v(z_r),rho2v(z_w))[:,:,1:-1]
        #.........................................................................#
        dvardeta[:,:,1:-1] = ((var[:,1:,1:-1] - var[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,1:-1] = dvardeta[:,:,1:-1] - (((z_r[:,1:,1:-1] - z_r[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2v(dz_w[:,:,1:-1]))
        #Daniel changed to 1:: instead of 1:-1
        #tmp = rho2w(rho2v(var),rho2v(z_r),rho2v(z_w))[:,:,1::]
        #.........................................................................#
        #dvardeta[:,:,1::] = ((var[:,1:,1::] - var[:,:-1,1::]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        #dvardeta[:,:,1::] = dvardeta[:,:,1::] - (((z_r[:,1:,1::] - z_r[:,:-1,1::]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2v(dz_w[:,:,1::]))
        tmp2 = (rho2v(var)[:,:,1] - rho2v(var[:,:,0]))/rho2v(dz_r[:,:,0])
        dvardeta[:,:,0] = ((var[:,1:,0] - var[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,0] = dvardeta[:,:,0] - (tmp2 + (rho2v(z_r)[:,:,0] - rho2v(z_w)[:,:,1])*(dvardeta[:,:,1] - tmp2)/(rho2v(z_r[:,:,1]) - rho2v(z_w[:,:,1])))*(((z_r[:,1:,0] - z_r[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)

        #Testing
        dvardeta[:,:,-1] = diffy(var[:,:,-1],pn)
    return dvardeta




#######################################################
#x-derivative from rho-grid to u-grid
#######################################################

def diffx(var,pm,dn=1):

    if np.rank(var)<3:
        dvardx = diffx_2d(var,pm,dn)
    else:
        dvardx = diffx_3d(var,pm,dn)

    return dvardx

###########################

def diffx_3d(var,pm,dn=1):

    [N,M,L]=var.shape

    dvardx = Zeros((N-dn,M,L))

    for iz in range(0, L):    
        dvardx[:,:,iz]=diffx_2d(var[:,:,iz],pm,dn)

    return dvardx

###########################

def diffx_2d(var,pm,dn=1):

    if (np.rank(pm)==2) and (var.shape[0]==pm.shape[0]): 
        dvardx = (var[dn:,:]-var[:-dn,:])*0.5*(pm[dn:,:]+pm[:-dn,:])/dn
    else: 
        dvardx = (var[dn:,:]-var[:-dn,:])*pm/dn

    return dvardx


#######################################################
#y-derivative from rho-grid to v-grid
#######################################################

def diffy(var,pn,dn=1):

    if np.rank(var)<3: dvardy = diffy_2d(var,pn,dn)
    else: dvardy = diffy_3d(var,pn,dn)

    return dvardy

    #######################

def diffy_3d(var,pn,dn=1):

    [N,M,L]=var.shape
    dvardy = Zeros((N,M-dn,L))
    for iz in range(0, L): dvardy[:,:,iz]=diffy_2d(var[:,:,iz],pn,dn)

    return dvardy

    #######################


def diffy_2d(var,pn,dn=1):

    if (np.rank(pn)==2) and (var.shape[1]==pn.shape[1]):
        dvardy = (var[:,dn:]-var[:,:-dn])*0.5*(pn[:,dn:]+pn[:,:-dn])/dn
    else: 
        dvardy = (var[:,dn:]-var[:,:-dn])*pn/dn

    return dvardy


#######################################################
#Compute absolute vorticity of a 3-D field on psi grid
#######################################################

def get_absvrt(u,v,z_r,z_w,f,pm,pn,mask=None):

##########################
#Absolute vorticity,  [f + (dv/dx - du/dy)]

    vrt = get_vrt(u,v,z_r,z_w,pm,pn,mask)
    
    var =  (rho2psi(f).T + vrt.T).T 
    
    return var

#######################################################
#Compute relative vorticity of a 3-D field on psi grid
#######################################################

def get_vrt(u,v,z_r,z_w,pm,pn,mask=None,norm=False,f=None):

    if len(u.shape)==3:
        #dudy and dvdx on psi grid
        dvdx = diffxi_orig(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask)
        dudy = diffeta_orig(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask)       
    else:      
        dvdx = diffx(v,rho2v(pm))
        dudy = diffy(u,rho2u(pn))  
        
    #vrt on psi grid
    vrt = dvdx - dudy    
    
    if norm:
       return norm_rot(vrt,rho2psi(f))
    else:   
       return vrt

def get_div(u,v,z_r,z_w,pm,pn,mask=None,norm=False,f=None):

    if len(u.shape)==3:
        dudx = diffxi_orig(u2rho(u),pm,z_r,z_w,mask)
        dvdy = diffeta_orig(v2rho(v),pn,z_r,z_w,mask)       
    else:      
        dudx = diffx(u2rho(u),pm)
        dvdy = diffy(v2rho(v),pn)  
        
    #div on psi grid
    div = rho2v(dudx) + rho2u(dvdy)
    
    if norm:
       return norm_rot(div,rho2psi(f))
    else:   
       return div



def norm_rot(var,f0):
    if len(var.shape)==3:
       [nx,ny,nz] = var.shape
       var_norm = Zeros((nx,ny,nz))
       for k in range(nz):
           var_norm[:,:,k] = var[:,:,k] /f0[:,:]
    else:
        var_norm = var / f0
    return var_norm
#######################################################
#Transfert a field at psi points to rho points
#######################################################

def psi2rho(var_psi):

    if np.rank(var_psi)<3:
        var_rho = psi2rho_2d(var_psi)
    else:
        var_rho = psi2rho_3d(var_psi)

    return var_rho


##############################

def psi2rho_2d(var_psi):

    [M,L]=var_psi.shape
    Mp=M+1
    Lp=L+1
    Mm=M-1
    Lm=L-1

    var_rho=Zeros((Mp,Lp))
    var_rho[1:M,1:L]=0.25*(var_psi[0:Mm,0:Lm]+var_psi[0:Mm,1:L]+var_psi[1:M,0:Lm]+var_psi[1:M,1:L])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]

    return var_rho

##############################

def psi2rho_2dp1(var_psi):

    [M,L,Nt]=var_psi.shape
    Mp=M+1
    Lp=L+1
    Mm=M-1
    Lm=L-1

    var_rho=Zeros((Mp,Lp,Nt))
    var_rho[1:M,1:L,:]=0.25*(var_psi[0:Mm,0:Lm,:]+var_psi[0:Mm,1:L,:]+var_psi[1:M,0:Lm,:]+var_psi[1:M,1:L,:])
    var_rho[0,:,:]=var_rho[1,:,:]
    var_rho[Mp-1,:,:]=var_rho[M-1,:,:]
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Lp-1,:]=var_rho[:,L-1,:]

    return var_rho

#############################


def psi2rho_3d(var_psi):

    [Mz,Lz,Nz]=var_psi.shape
    var_rho=Zeros((Mz+1,Lz+1,Nz))

    for iz in range(0, Nz, 1):    
        var_rho[:,:,iz]=psi2rho_2d(var_psi[:,:,iz])

    return var_rho



#######################################################
#Transfert a field at rho points to psi points
#######################################################

def rho2psi(var_rho):

    if np.rank(var_rho)<3:
        var_psi = rho2psi_2d(var_rho)
    else:
        var_psi = rho2psi_3d(var_rho)

    return var_psi


##############################

def rho2psi_2d(var_rho):

    var_psi = 0.25*(var_rho[1:,1:]+var_rho[1:,:-1]+var_rho[:-1,:-1]+var_rho[:-1,1:])

    return var_psi

#############################

def rho2psi_3d(var_rho):

    var_psi = 0.25*(var_rho[1:,1:,:]+var_rho[1:,:-1,:]+var_rho[:-1,:-1,:]+var_rho[:-1,1:,:])

    return var_psi





#######################################################
#Transfert a field at rho points to u points
#######################################################

def rho2u(var_rho):

    if np.rank(var_rho)==1:
        var_u = 0.5*(var_rho[1:]+var_rho[:-1])
    elif np.rank(var_rho)==2:       
        var_u = rho2u_2d(var_rho)
    else:
        var_u = rho2u_3d(var_rho)

    return var_u


##############################

def rho2u_2d(var_rho):

    var_u = 0.5*(var_rho[1:,:]+var_rho[:-1,:])

    return var_u

#############################

def rho2u_3d(var_rho):

    var_u = 0.5*(var_rho[1:,:,:]+var_rho[:-1,:,:])

    return var_u



#######################################################
#Transfert a field at rho points to v points
#######################################################

def rho2v(var_rho):

    if np.rank(var_rho)==1:
        var_v = 0.5*(var_rho[1:]+var_rho[:-1])
    elif np.rank(var_rho)==2:
        var_v = rho2v_2d(var_rho)
    else:
        var_v = rho2v_3d(var_rho)

    return var_v


##############################

def rho2v_2d(var_rho):

    var_v = 0.5*(var_rho[:,1:]+var_rho[:,:-1])

    return var_v

#############################

def rho2v_3d(var_rho):

    var_v = 0.5*(var_rho[:,1:,:]+var_rho[:,:-1,:])

    return var_v





#######################################################
#Transfert a field at u points to the rho points
#######################################################

def v2rho(var_v):


    if np.rank(var_v) == 2:
        var_rho = v2rho_2d(var_v)
    elif np.rank(var_v) == 3:
        var_rho = v2rho_3d(var_v)
    else:
        var_rho = v2rho_4d(var_v)

    return var_rho

#######################################################

def v2rho_2d(var_v):

    [Mp,L]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp))
    var_rho[:,1:L]=0.5*(var_v[:,0:Lm]+var_v[:,1:L])
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]
    return var_rho

#######################################################

def v2rho_3d(var_v):

    [Mp,L,N]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp,N))
    var_rho[:,1:L,:]=0.5*(var_v[:,0:Lm,:]+var_v[:,1:L,:])
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Lp-1,:]=var_rho[:,L-1,:]
    return var_rho


#######################################################
#######################################################

def v2rho_4d(var_v):

    [Mp,L,N,Nt]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp,N,Nt))
    var_rho[:,1:L,:,:]=0.5*(var_v[:,0:Lm,:,:]+var_v[:,1:L,:,:])
    var_rho[:,0,:,:]=var_rho[:,1,:,:]
    var_rho[:,Lp-1,:,:]=var_rho[:,L-1,:,:]
    return var_rho


#######################################################
#######################################################

def v2rho_2dp1(var_v):

    [Mp,L,Nt]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp,Nt))
    var_rho[:,1:L,:]=0.5*(var_v[:,0:Lm,:]+var_v[:,1:L,:])
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Lp-1,:]=var_rho[:,L-1,:]
    return var_rho


#######################################################
#Transfert a 2 or 2-D field at u points to the rho points
#######################################################

def u2rho(var_u):

    if np.rank(var_u) == 2:
        var_rho = u2rho_2d(var_u)
    elif np.rank(var_u) == 3:
        var_rho = u2rho_3d(var_u)
    else:
        var_rho = u2rho_4d(var_u)   
    return var_rho

#######################################################

def u2rho_2d(var_u):

    [M,Lp]=var_u.shape
    Mp=M+1
    Mm=M-1
    var_rho=Zeros((Mp,Lp))
    var_rho[1:M,:]=0.5*(var_u[0:Mm,:]+var_u[1:M,:])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]

    return var_rho

#######################################################

def u2rho_3d(var_u):

    [M,Lp,N]=var_u.shape
    Mp=M+1
    Mm=M-1
    var_rho=Zeros((Mp,Lp,N))
    var_rho[1:M,:,:]=0.5*(var_u[0:Mm,:]+var_u[1:M,:,:])
    var_rho[0,:,:]=var_rho[1,:,:]
    var_rho[Mp-1,:,:]=var_rho[M-1,:,:]

    return var_rho

#################################################################################

def u2rho_4d(var_u):

    [M, Lp, N, Nt]=var_u.shape
    Mp = M+1
    Mm = M-1
    var_rho = Zeros((Mp, Lp, N, Nt))
    var_rho[1:M,:,:,:]=0.5*(var_u[0:Mm,:,:,:]+var_u[1:M,:,:,:])
    var_rho[0,:,:,:]=var_rho[1,:,:,:]
    var_rho[Mp-1,:,:,:]=var_rho[M-1,:,:,:]

    return var_rho
#######################################################
#################################################################################

def u2rho_2dp1(var_u):

    [M, Lp, Nt]=var_u.shape
    Mp = M+1
    Mm = M-1
    var_rho = Zeros((Mp, Lp, Nt))
    var_rho[1:M,:,:]=0.5*(var_u[0:Mm,:,:]+var_u[1:M,:,:])
    var_rho[0,:,:]=var_rho[1,:,:]
    var_rho[Mp-1,:,:]=var_rho[M-1,:,:]

    return var_rho
#######################################################


def w2rho_s(var_w, z_r, z_w):
    #print var_w.shape, z_r.shape
    w_r = z_r * 0
    w_r = var_w[:,:,:-1] * (z_w[:,:,1:] - z_r[:,:,:]) + var_w[:,:,1:] * (z_r[:,:,:] - z_w[:,:,:-1])
    w_r /= (z_w[:,:,1:] - z_w[:,:,:-1])
    return w_r


def rho2w(var_r, z_r, z_w):
    #print var_r.shape, z_w.shape
    w_w = z_w * 0
    w_w[:,:,0] = var_r[:,:,0] + (var_r[:,:,1] - var_r[:,:,0])*(z_w[:,:,0] - z_r[:,:,0])/(z_r[:,:,1] - z_r[:,:,0])
    w_w[:,:,1:-1] = var_r[:,:,:-1] * (z_r[:,:,1:] - z_w[:,:,1:-1]) + var_r[:,:,1:] * (z_w[:,:,1:-1] - z_r[:,:,:-1])
    w_w[:,:,1:-1] /= (z_r[:,:,1:] - z_r[:,:,:-1])
    return w_w

#######################################################
#Transfert a 3-D field from verical w points to vertical rho-points
#######################################################

def w2rho(var_w):


    [M,L,N]=var_w.shape
    
    var_rho = Zeros((M,L,N-1))
    
    for iz in range(1,N-2):
        var_rho[:,:,iz]  = 0.5625*(var_w[:,:,iz+1] + var_w[:,:,iz]) -0.0625*(var_w[:,:,iz+2] + var_w[:,:,iz-1])
    
    var_rho[:,:,0]  = -0.125*var_w[:,:,2] + 0.75*var_w[:,:,1] +0.375*var_w[:,:,0] 
    var_rho[:,:,N-2]  = -0.125*var_w[:,:,N-3] + 0.75*var_w[:,:,N-2] +0.375*var_w[:,:,N-1] 
    

    return var_rho







