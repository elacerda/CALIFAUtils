#!/usr/bin/python
#
# Lacerda@Saco - 14/Mar/2016
#
##############################################################################
# Example of usage of the object stack_gals():
# This objects was created to stack all data of a given galaxy sample onto the
# same array with N_zones * N_gals size.
# In this example script we will calculate SFR data for all galaxies and make
# a McorSD vs SFRSD plot using various tSF. 
##############################################################################
from CALIFAUtils.objects import stack_gals
from CALIFAUtils.scripts import sort_gals, read_gal_cubes, \
                                calc_SFR, calc_xY
import time
import numpy as np

t_init_prog = time.clock()

pycasso_cube_dir = '/Users/lacerda/CALIFA/gal_fits/v20_q050.d15a'
pycasso_cube_suffix = '_synthesis_eBR_v20_q050.d15a512.ps03.k1.mE.CCM.Bgsd6e.fits'
eml_cube_dir = '/Users/lacerda/CALIFA/rgb-gas/v20_q050.d15a'
eml_cube_suffix = '_synthesis_eBR_v20_q050.d15a512.ps03.k1.mE.CCM.Bgsd6e.EML.MC100.fits' 
gasprop_cube_dir = '/Users/lacerda/CALIFA/rgb-gas/v20_q050.d15a/prop'
gasprop_cube_suffix = '_synthesis_eBR_v20_q050.d15a512.ps03.k1.mE.CCM.Bgsd6e.EML.MC100.GasProp.fits' 

# Parse arguments 
xOkMin = 0.05

# Reading galaxies file,
gals, _ = sort_gals(gals = '/Users/lacerda/CALIFA/listv20_q050.d15a.txt', order = 1)
N_gals = len(gals)
maxGals = N_gals

# SFR-time-scale array (index __T)
tSF__T = np.array([1, 3.2, 10, 100]) * 1e7
N_T = len(tSF__T)

#Using stack gals to stack data in 1 pass.
G = stack_gals()
G.new1d('McorSD__g')
G.new2d('McorSD__Tg', N_T)
G.new2d('SFR__Tg', N_T)
G.new2d('SFRSD__Tg', N_T)
G.new2d('x_Y__Tg', N_T)

#for iGal, K in loop_cubes(gals.tolist(), imax = 4, EL = True, GP = True, v_run = -1, debug = True):        
for iGal, gal in enumerate(gals[0:maxGals]):
    K = read_gal_cubes(gal, debug = True, 
                       pycasso_cube_dir = pycasso_cube_dir, pycasso_cube_suffix = pycasso_cube_suffix,
                       eml_cube_dir = eml_cube_dir, eml_cube_suffix = eml_cube_suffix,
                       gasprop_cube_dir = gasprop_cube_dir, gasprop_cube_suffix = gasprop_cube_suffix,
                       )
    t_init_gal = time.clock()
    califaID = gals[iGal] 
    # Setup elliptical-rings geometry
    pa, ba = K.getEllipseParams()
    K.setGeometry(pa, ba)
    print '>>> Doing %d %s' % (iGal, califaID)

    # append McorSD data    
    G.append1d('McorSD__g', K.Mcor__z / K.zoneArea_pc2)
    
    for iT, tSF in enumerate(tSF__T):
        x_Y__z, _ = calc_xY(K, tSF)
        # append x_Y data
        G.append2d('x_Y__Tg', iT, x_Y__z)

        bad_zones = np.less(x_Y__z, xOkMin)
        
        # append McorSD data
        G.append2d('McorSD__Tg', iT, K.Mcor__z / K.zoneArea_pc2, bad_zones)
        aux = calc_SFR(K, tSF)
        # append SFR data
        G.append2d('SFR__Tg', iT, aux[0], bad_zones)
        G.append2d('SFRSD__Tg', iT, aux[1], bad_zones)
        
# stack all data 
G.stack1d()
G.stack2d()

print 'total time: %.2f' % (time.clock() - t_init_prog)
from matplotlib import pyplot as plt
for iT, tSF in enumerate(tSF__T):
    f = plt.figure()
    x = np.ma.log10(G.McorSD__Tg[iT])
    y = np.ma.log10(G.SFRSD__Tg[iT] * 1e6) # M_sun/yr/kpc
    z = np.ma.log10(G.x_Y__Tg[iT])
    ax = f.gca()
    sc = ax.scatter(x, y, c = z, cmap = 'viridis')
    cb = plt.colorbar(sc)
    cb.set_label(r'$\log\ x_Y$')
    ax.set_xlabel(r'$\log\ \Sigma_\star$ [$M_\odot\ pc^{-2}$]')
    ax.set_ylabel(r'$\log\ \Sigma_{\mathrm{SFR}}^\star(t_\star\ =\ %.0f\ \mathrm{Myr})$ [$M_\odot\ yr^{-1}\ kpc^{-2}$]' % (tSF/1e6))
    f.savefig('MCorSD_SFRSD_xY_%.2fMyr.pdf' % (tSF/1e6))