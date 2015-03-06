#!/usr/bin/python
#
# Lacerda@Granada - 24/Feb/2015
#
califa_work_dir = '/Users/lacerda/CALIFA/'
baseCode = 'Bgsd6e'
version_config = dict(baseCode = 'Bgsd6e',
#                      versionSuffix = 'v20_q036.d13c', 
#                      versionSuffix = 'v20_q046.d15a', 
#                      versionSuffix = 'px1_q043.d14a', 
                      versionSuffix = 'v20_q043.d14a',
#                      othSuffix = '512.ps03.k2.mC.CCM.',     
                      othSuffix = '512.ps03.k1.mE.CCM.',
                      SuperFitsDir = 'gal_fits/')
tmp_suffix = '_synthesis_eBR_' + version_config['versionSuffix'] + version_config['othSuffix'] + version_config['baseCode']
pycasso_suffix = tmp_suffix + '.fits'
emlines_suffix = tmp_suffix + '.EML.MC100.fits'
gasprop_suffix = tmp_suffix + '.EML.MC100.GasProp.fits'
gasprop_cube_dir = califa_work_dir+ 'rgb-gas/' + version_config['versionSuffix'] + '/prop/'
emlines_cube_dir = califa_work_dir + 'rgb-gas/' + version_config['versionSuffix'] + '/'
pycasso_cube_dir = califa_work_dir + version_config['SuperFitsDir'] + version_config['versionSuffix'] + '/'