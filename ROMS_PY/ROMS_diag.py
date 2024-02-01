#################################################
# Library to create diagnostics from ROMS ouptut

# Generally will make a netcdf file like his and fill
# with offline diagnostics
################################################

########################################
import os
import sys
from netCDF4 import Dataset as netcdf
import ROMS_tools as RT
import ROMS_depths as RD
##############################



####################################
#       File creation/definitions
#####################################

def create_diag_nc(roms_his,diag_name,diags_dict):
    '''
    Create diagnostic file given a roms netcdf
    '''
    roms_diag = netcdf(diag_name + '.nc', 'w')
    print ''
    print 'Creating new netcdf: ' + diag_name + '.nc'
    print ''
    #ADD ATTRIBUTES for z-slice capability
    roms_diag.rho0 = roms_his.rho0
    roms_diag.Zob = roms_his.Zob
    roms_diag.Cs_w = roms_his.Cs_w
    roms_diag.Cs_r = roms_his.Cs_r
    try:
       roms_diag.VertCoordType = roms_his.VertCoordType
    except:
        roms_diag.VertCoordType='SM09'
    roms_diag.theta_b = roms_his.theta_b
    roms_diag.theta_s = roms_his.theta_s
    roms_diag.hc = roms_his.hc

    #COPY DIMENSIONS
    for name, dimension in roms_his.dimensions.iteritems():
        if name=='time':
           roms_diag.createDimension('time', None)
        else:
           roms_diag.createDimension(name, (len(dimension)))

    #CREATE VARIABLES (from diags_dict)
    for name, variable in roms_his.variables.items():
        if name=='temp':
           [nt,nz,ny,nx] = roms_his.variables[name].shape
           for i in range(len(diags_dict['3D_vars'])): 
               roms_diag.createVariable(diags_dict['3D_vars'][i], variable.datatype, variable.dimensions,chunksizes=(1,nz,ny,nx,))
        if name=='zeta':
           [nt,ny,nx] = roms_his.variables[name].shape
           for i in range(len(diags_dict['2D_vars'])): 
               roms_diag.createVariable(diags_dict['2D_vars'][i], variable.datatype, variable.dimensions,chunksizes=(1,ny,nx,))
           #Add ocean time
           if name == 'ocean_time':
              roms_diag.createVariable(name, variable.datatype, variable.dimensions)

    return roms_diag

def create_diag_nc_sample(roms_his,diag_name,diags_dict,i0,i1,j0,j1):
    '''
    Create diagnostic file given a roms netcdf

    Create file with fraction of full roms grid according to i0,i1,j0,j1
    '''
    roms_diag = netcdf(diag_name + '.nc', 'w')
    print ''
    print 'Creating new netcdf: ' + diag_name + '.nc'
    print ''
    #ADD ATTRIBUTES for z-slice capability
    roms_diag.rho0 = roms_his.rho0
    roms_diag.Zob = roms_his.Zob
    roms_diag.Cs_w = roms_his.Cs_w
    roms_diag.Cs_r = roms_his.Cs_r
    try:
       roms_diag.VertCoordType = roms_his.VertCoordType
    except:
       roms_diag.VertCoordType='SM09'
    roms_diag.theta_b = roms_his.theta_b
    roms_diag.theta_s = roms_his.theta_s
    roms_diag.hc = roms_his.hc


    roms_diag.i0 = i0
    roms_diag.i1 = i1
    roms_diag.j0 = j0
    roms_diag.j1 = j1
    #COPY DIMENSIONS
    nx_sample = i1 - i0
    ny_sample = j1 - j0
    [nt,nz,ny,nx] = roms_his.variables['temp'].shape

    #Create sampled dimensions
    roms_diag.createDimension('time', None)
    roms_diag.createDimension('xi_rho', nx_sample)
    roms_diag.createDimension('xi_u', nx_sample-1)
    roms_diag.createDimension('eta_rho', ny_sample)
    roms_diag.createDimension('eta_v', ny_sample-1)
    roms_diag.createDimension('s_rho', nz)
    roms_diag.createDimension('s_w', nz+1)
        
    #CREATE VARIABLES (from diags_dict)
    for name, variable in roms_his.variables.items():
        if name=='temp':
           [nt,nz,ny,nx] = roms_his.variables[name].shape
           ny_chunk = j1 - j0
           nx_chunk = i1 - i0
           for i in range(len(diags_dict['3D_vars'])): 
               roms_diag.createVariable(diags_dict['3D_vars'][i], variable.datatype, variable.dimensions,chunksizes=(1,nz,ny_chunk,nx_chunk,))
        if name=='zeta':
           [nt,ny,nx] = roms_his.variables[name].shape
           ny_chunk = j1 - j0
           nx_chunk = i1 - i0
           for i in range(len(diags_dict['2D_vars'])): 
               roms_diag.createVariable(diags_dict['2D_vars'][i], variable.datatype, variable.dimensions,chunksizes=(1,ny_chunk,nx_chunk,))
           #Add ocean time
           if name == 'ocean_time':
              roms_diag.createVariable(name, variable.datatype, variable.dimensions)
    #Add ocean time manually
    otime_name,otime_var = roms_his.variables.items()[0]
    roms_diag.createVariable(otime_name,otime_var.datatype,otime_var.dimensions)
    return roms_diag















