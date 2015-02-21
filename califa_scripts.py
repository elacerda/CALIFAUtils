#!/usr/bin/python
#
# Lacerda@Granada - 29/Jan/2015
#
import numpy as np
import sys
import h5py
from pycasso import fitsQ3DataCube
import types
import matplotlib.pyplot as plt
import itertools
        
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
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
emlines_cube_dir = califa_work_dir + 'rgb-gas/' + version_config['versionSuffix'] + '/'
pycasso_cube_dir = califa_work_dir + version_config['SuperFitsDir'] + version_config['versionSuffix'] + '/'
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE


class read_kwargs(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs  
        
    def __getattr__(self, attr):
        return self.kwargs.get(attr)


def read_one_cube(gal, **kwargs):
    args = read_kwargs(**kwargs)
    EL = args.EL
    verbose = args.verbose

    pycasso_cube_filename = pycasso_cube_dir + gal + pycasso_suffix
    K = None
    try:
        K = fitsQ3DataCube(pycasso_cube_filename)
        if verbose:
            print >> sys.stderr, 'PyCASSO: Reading file: %s' % pycasso_cube_filename
        if EL:
            emlines_cube_filename = emlines_cube_dir + gal + emlines_suffix
            try:
                K.loadEmLinesDataCube(emlines_cube_filename)
                if verbose:
                    print >> sys.stderr, 'EL: Reading file: %s' % emlines_cube_filename
            except IOError:
                print >> sys.stderr, 'EL: File does not exists: %s' % emlines_cube_filename
    except IOError:
        print >> sys.stderr, 'PyCASSO: File does not exists: %s' % pycasso_cube_filename
    return K


def sort_gal_list_func(gals, func = None, order = 1, **kwargs):
    '''
    Sort galaxies in txt FILENAME by some ATTRibute processed by MODE in ORDER order.
    ORDER = 0 - sort asc, 1 - sort desc
    MODE can be any numpy array method such as sum, max, min, mean, median, etc...

    '''
    args = read_kwargs(**kwargs)
    verbose = args.verbose

    if type(func) != types.FunctionType:
        if verbose:
            print 'func need to be FunctionType'
        return None
    if verbose:
        print gals
    Ng = len(gals)
    data__g = np.ma.empty((Ng))
    for i, gal_id in enumerate(gals):
        K = read_one_cube(gal_id, **kwargs)
        data__g[i] = func(K, **kwargs)
        if verbose:
            print K.califaID, data__g[i]
        K.close()
    sgals = None
    if data__g.mask.sum() < Ng:
        iS = np.argsort(data__g)
        if order != 0:
            iS = iS[::-1]
        sgals = np.asarray(gals)[iS]
        sdata = data__g[iS]
    return sgals, sdata


def sort_gal_func(filename, func = None, order = 1, **kwargs):
    '''
    Sort galaxies in txt FILENAME by some ATTRibute processed by MODE in ORDER order.
    ORDER = 0 - sort asc, 1 - sort desc
    MODE can be any numpy array method such as sum, max, min, mean, median, etc...

    '''
    args = read_kwargs(**kwargs)
    verbose = args.verbose
    if type(func) != types.FunctionType:
        if verbose:
            print 'func need to be FunctionType'
        return None
    f = open(filename, 'r')
    l = f.readlines()
    f.close()
    Ng = len(l)
    gals = np.asarray([ l[i].strip() for i in np.arange(Ng) ])
    if verbose:
        print gals
    data__g = np.ma.empty((Ng))
    for i, gal_id in enumerate(gals):
        K = read_one_cube(gal_id, **kwargs)
        data__g[i] = func(K, **kwargs)
        if verbose:
            print K.califaID, data__g[i]
        K.close()
    sgals = None
    if data__g.mask.sum() < Ng:
        iS = np.argsort(data__g)
        if order != 0:
            iS = iS[::-1]
        sgals = gals[iS]
        sdata = data__g[iS]
    return sgals, sdata


def sort_gal(filename, attr = None, mode = None, order = 1):
    '''
    Sort galaxies in txt FILENAME by some ATTRibute processed by MODE in ORDER order.
    ORDER = 0 - sort asc, 1 - sort desc
    MODE can be any numpy array method such as sum, max, min, mean, median, etc...

    '''
    f = open(filename, 'r')
    l = f.readlines()
    f.close()
    Ng = len(l)
    gals = np.asarray([ l[i].strip() for i in np.arange(Ng) ])
    data__g = np.ma.empty((Ng))
    for i, gal_id in enumerate(gals):
        K = read_one_cube(gal_id)
        try:
            attribute = K.__getattribute__(attr)
            if mode == None:
                data__g[i] = attribute
            else:
                f_mode = np.__dict__[mode]
                data__g[i] = f_mode(attribute)
        except AttributeError:
            print >> sys.stderr, '%s: %s: non-existent attribute' % (gal_id, attr)
            data__g[i] = np.ma.masked
            continue
        K.close()
    sgals = None
    if data__g.mask.sum() < Ng:
        iS = np.argsort(data__g)
        if order != 0:
            iS = iS[::-1]
        sgals = gals[iS]
    return sgals


def create_dx(x):
    dx = np.empty_like(x)
    dx[1:] = (x[1:] - x[:-1]) / 2.   # dl/2 from right neighbor
    dx[:-1] += dx[1:]               # dl/2 from left neighbor
    dx[0] = 2 * dx[0]
    dx[-1] = 2 * dx[-1]
    #dx[-1]      = x[-1]
    return dx

#XXXXXXXXXXXXXXXXXXXXXXXX#
#TODOTODOTODOTODOTODOTODO#
#XXXXXXXXXXXXXXXXXXXXXXXX#
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
# class MultiGal(object):
#     def __init__(self, list, **kwargs):
#         self._init_kwargs(**kwargs)
#         self._list = list
#         if list == types.StringType:
#             self._set_califaIDs_txt()
#         else:
#             self._set_califaIDs()
#             
#         self._init_data()
#         loop_califa_galaxies(self.califaIDs, acquire_data, **kwargs)
#         
#     def _init_kwargs(self, **kwargs):
#         try: 
#             self._EL = kwargs.pop('EL')
#         except:
#             self._EL = False
#     
#     def _set_califaIDs_txt(self):
#         f = open(self._list, 'r')
#         l = f.readlines()
#         f.close()
#         self.N_gals = len(l)
#         tmp = [ l[i].strip() for i in np.arange(self.N_gals) ]
#         mask = np.empty((N_gals), dtype = np.bool)
#         self.califaIDs = np.ma.masked_array(tmp, mask = mask)
# 
#     def _set_califaIDs(self):
#         self.N_gals = len(self._list)
#         mask = np.empty((self.N_gals), dtype = np.bool)
#         self.califaIDs = np.ma.masked_array(self._list, mask = mask)
#         
#     def _init_data(self):
#         self.Mcor__g = np.ma.empty((self.N_gals))
#         self.McorSD__g = np.ma.empty((self.N_gals))
#         self.tau_V__g = np.ma.empty((self.N_gals))
#         self.Mr__g = np.ma.empty((self.N_gals))
#         self.morph__g = np.ma.empty((self.N_gals))
#         self.at_flux__g = np.ma.empty((self.N_gals))
#         self.ba__g = np.ma.empty((self.N_gals))
#         self.u_r__g = np.ma.empty((self.N_gals))
#         self.redshift__g = np.ma.empty((self.N_gals))
#         if self._EL = True:
#             self.tau_V_neb__g = np.ma.empty((self.N_gals))
#             self.EW_Ha__g = np.ma.empty((self.N_gals))
#             self.F_obs_Ha__g = np.ma.empty((self.N_gals))
#             self.F_obs_Hb__g = np.ma.empty((self.N_gals))
#             self.F_obs_N2__g = np.ma.empty((self.N_gals))
#             self.F_obs_O3__g = np.ma.empty((self.N_gals))
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
            
#TODO
def loop_califa_galaxies(gals, func = None, **kwargs):
    '''
    Loop by all CALIFA galaxies in LIST executing f and returning a HDF5.
    F must be a type.FunctionType.
    '''
    args = read_kwargs(**kwargs)
    verbose = args.verbose
    if type(func) != types.FunctionType:
        if verbose:
            print 'func need to be FunctionType'
        return None
    Ng = len(gals)
    data__g = np.ma.empty((Ng))
    for i, gal_id in enumerate(gals):
        K = read_one_cube(gal_id, **kwargs)
        # TODO think about data structure, maybe an object
        data__g[i] = func(K, **kwargs)
        K.close()
    return data__g


class ALLGals(object):
    def __init__(self, N_gals, NRbins, N_T, N_U):
        self.N_gals = N_gals
        self.NRbins = NRbins
        self.N_T = N_T
        self.N_U = N_U
        self._init_arrays()
        self._init_zones_temporary_lists()
        
    def mask_gal(self, iGal):
        for v in self.__dict__.keys():
            if isinstance(self.__dict__[v], np.ma.core.MaskedArray):
                self.__dict__[v][..., iGal] = np.ma.masked
        
    def _init_arrays(self):
        N_gals = self.N_gals
        NRbins = self.NRbins
        N_T = self.N_T
        N_U = self.N_U
        self.califaIDs = np.ma.empty((N_gals), dtype = '|S5')
        self.N_zones__g = np.ma.empty((N_gals), dtype = int)
        self.morfType_GAL__g = np.ma.empty((N_gals))
        self.at_flux_GAL__g = np.ma.empty((N_gals))
        self.Mcor_GAL__g = np.ma.empty((N_gals))
        self.McorSD_GAL__g = np.ma.empty((N_gals))
        self.ba_GAL__g = np.ma.empty((N_gals))
        self.ba_PyCASSO_GAL__g = np.ma.empty((N_gals))
        self.Mr_GAL__g = np.ma.empty((N_gals))
        self.ur_GAL__g = np.ma.empty((N_gals))
        self.integrated_tau_V__g = np.ma.empty((N_gals))
        self.integrated_tau_V_neb__g = np.ma.empty((N_gals))
        self.integrated_logZ_neb_S06__g = np.ma.empty((N_gals))
        self.integrated_L_int_Ha__g = np.ma.empty((N_gals))
        self.integrated_SFR_Ha__g = np.ma.empty((N_gals))
        self.integrated_SFRSD_Ha__g = np.ma.empty((N_gals))
        self.integrated_SFR__Tg = np.ma.empty((N_T, N_gals))
        self.integrated_SFRSD__Tg = np.ma.empty((N_T, N_gals))
        self.alogZ_mass_GAL__Ug = np.ma.empty((N_U, N_gals))
        self.alogZ_flux_GAL__Ug = np.ma.empty((N_U, N_gals))
        self.tau_V_neb__rg = np.ma.empty((NRbins, N_gals))
        self.aSFRSD_Ha__rg = np.ma.empty((NRbins, N_gals))
        self.McorSD__rg = np.ma.empty((NRbins, N_gals))
        self.logZ_neb_S06__rg = np.ma.empty((NRbins, N_gals))
        self.aSFRSD__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.tau_V__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.McorSD__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.f_gas__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.alogZ_mass__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_flux__Urg = np.ma.empty((N_U, NRbins, N_gals))
        
    def _init_zones_temporary_lists(self):
        self._Mcor__g = []
        self._McorSD__g = []
        self._tau_V_neb__g = []
        self._tau_V_neb_err__g = []
        self._tau_V_neb_mask__g = []
        self._logZ_neb_S06__g = []
        self._logZ_neb_S06_err__g = []
        self._logZ_neb_S06_mask__g = []
        self._SFR_Ha__g = []
        self._SFRSD_Ha__g = []
        self._F_obs_Ha__g = []
        self._L_int_Ha__g = []
        self._L_int_Ha_err__g = []
        self._L_int_Ha_mask__g = []
        self._dist_zone__g = []
        self._EW_Ha__g = []
        self._EW_Hb__g = []
        
        self._tau_V__Tg = []
        self._tau_V_mask__Tg = []
        self._SFR__Tg = []
        self._SFR_mask__Tg = []
        self._SFRSD__Tg = []
        self._SFRSD_mask__Tg = []
        self._at_flux__Tg = []
        self._at_flux_mask__Tg = []
        self._x_Y__Tg = []
        self._Mcor__Tg = []
        self._McorSD__Tg = []
        
        for _ in range(self.N_T):
            self._tau_V__Tg.append([])
            self._tau_V_mask__Tg.append([])
            self._SFR__Tg.append([])
            self._SFR_mask__Tg.append([])
            self._SFRSD__Tg.append([])
            self._SFRSD_mask__Tg.append([])
            self._at_flux__Tg.append([])
            self._at_flux_mask__Tg.append([])
            self._x_Y__Tg.append([])
            self._Mcor__Tg.append([])
            self._McorSD__Tg.append([])
        
        self._alogZ_mass__Ug = []
        self._alogZ_mass_mask__Ug = []
        self._alogZ_flux__Ug = []
        self._alogZ_flux_mask__Ug = []
        
        for _ in range(self.N_U):
            self._alogZ_mass__Ug.append([])
            self._alogZ_mass_mask__Ug.append([])
            self._alogZ_flux__Ug.append([])
            self._alogZ_flux_mask__Ug.append([])
        
        #final Tg and Ug zone-by-zone lists
        self.tau_V__Tg = [] 
        self.SFR__Tg = []
        self.SFRSD__Tg = []
        self.x_Y__Tg = []
        self.alogZ_mass__Ug = []
        self.alogZ_flux__Ug = []
        self.Mcor__Tg = []
        self.McorSD__Tg = []
            
    def stack_zones_data(self):
        self.dist_zone__g = np.ma.masked_array(np.hstack(np.asarray(self._dist_zone__g)))
        
        aux = np.hstack(self._tau_V_neb__g)
        auxMask = np.hstack(self._tau_V_neb_mask__g)
        self.tau_V_neb__g = np.ma.masked_array(aux, mask = auxMask)
        self.tau_V_neb_err__g = np.ma.masked_array(np.hstack(self._tau_V_neb_err__g), mask = auxMask)
        
        aux = np.hstack(self._logZ_neb_S06__g)
        auxMask = np.hstack(self._logZ_neb_S06_mask__g)
        self.logZ_neb_S06__g = np.ma.masked_array(aux, mask = auxMask)
        self.logZ_neb_S06_err__g = np.ma.masked_array(np.hstack(self._logZ_neb_S06_err__g), mask = auxMask)

        aux = np.hstack(self._L_int_Ha__g)
        auxMask = np.hstack(self._L_int_Ha_mask__g)
        self.L_int_Ha__g = np.ma.masked_array(aux, mask = auxMask)
        self.F_obs_Ha__g = np.ma.masked_array(np.hstack(self._F_obs_Ha__g), mask = auxMask)
        self.SFR_Ha__g = np.ma.masked_array(np.hstack(self._SFR_Ha__g), mask = auxMask)
        self.SFRSD_Ha__g = np.ma.masked_array(np.hstack(self._SFRSD_Ha__g), mask = auxMask)
        self.Mcor__g = np.ma.masked_array(np.hstack(self._Mcor__g))
        self.McorSD__g = np.ma.masked_array(np.hstack(self._McorSD__g))

        self.EW_Ha__g = np.ma.masked_array(np.hstack(self._EW_Ha__g))
        self.EW_Hb__g = np.ma.masked_array(np.hstack(self._EW_Hb__g))

        for iT in range(self.N_T):
            aux = np.hstack(self._SFR__Tg[iT])
            auxMask = np.hstack(self._SFR_mask__Tg[iT])        
            self.SFR__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._SFRSD__Tg[iT])
            auxMask = np.hstack(self._SFRSD_mask__Tg[iT])
            self.SFRSD__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._x_Y__Tg[iT])
            self.x_Y__Tg.append(np.ma.masked_array(aux))
            aux = np.hstack(self._tau_V__Tg[iT])
            auxMask = np.hstack(self._tau_V_mask__Tg[iT])
            self.tau_V__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._Mcor__Tg[iT])
            self.Mcor__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._McorSD__Tg[iT])
            self.McorSD__Tg.append(np.ma.masked_array(aux, mask = auxMask))
        
        for iU in np.arange(self.N_U):
            aux = np.hstack(self._alogZ_mass__Ug[iU])
            self.alogZ_mass__Ug.append(np.ma.masked_array(aux))
            aux = np.hstack(self._alogZ_flux__Ug[iU])
            self.alogZ_flux__Ug.append(np.ma.masked_array(aux))
            
    def create_dict_h5(self):
        D = {}
        for v in self.__dict__.keys():
            if v[0] != '_':
                suffix = v.split('_')[-1]
                if isinstance(self.__dict__[v], np.ma.core.MaskedArray):
                    tmp_data = {'/masked/data/%s' % v : self.__dict__[v].data}
                    tmp_mask = {'/masked/mask/%s' % v : self.__dict__[v].mask}
                else:
                    if suffix == 'Tg':
                        tmp_data = {'/masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in range(self.N_T)}
                        tmp_mask = {'/masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in range(self.N_T)}
                    elif suffix == 'Ug':
                        tmp_data = {'/masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in range(self.N_U)}
                        tmp_mask = {'/masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in range(self.N_U)}
                    else:
                        tmp_data = {}
                        tmp_mask = {}
                D.update(tmp_data)
                D.update(tmp_mask)
        return D                    


def SFR_parametrize(flux, wl, ages, tSF):
    '''
    Find the k parameter in the equation SFR = k [M_sun yr^-1] L(Halpha) [(10^8 L_sun)^-1]
    
    TODO: blablabla
    
    Nh__Zt is obtained for all t in AGES differently from Nh__Z, which consists in the number
    of H-ionizing photons from MAX_AGE till today (t < MAX_AGE).
    ''' 
    from pystarlight.util.constants import L_sun, h, c, yr_sec
    
    cmInAA = 1e-8             # cm / AA
    lambda_Ha = 6562.8        # Angstrom
    mask_age = ages <= tSF
    
    y = flux * wl * cmInAA * L_sun / (h * c)
    #y = flux * cmInAA * L_sun #BOL
    
    qh__Zt = (y * create_dx(wl)).sum(axis = 2)
    Nh__Z = (qh__Zt[:, mask_age] * create_dx(ages[mask_age])).sum(axis = 1) * yr_sec
    Nh__Zt = np.cumsum(qh__Zt * create_dx(ages), axis = 1) * yr_sec
         
    k_SFR__Z = 2.226 * lambda_Ha * L_sun * yr_sec / (Nh__Z * h * c) # M_sun / yr
    
    return qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z


def SFR_parametrize_trapz(flux, wl, ages, tSF):
    '''
    Find the k parameter in the equation SFR = k [M_sun yr^-1] L(Halpha) [(10^8 L_sun)^-1]
    
    TODO: blablabla
    
    Nh__Zt is obtained for all t in AGES differently from Nh__Z, which consists in the number
    of H-ionizing photons from MAX_AGE till today (t < MAX_AGE).
    ''' 
    from pystarlight.util.constants import L_sun, h, c, yr_sec
    import scipy.integrate as spi
    
    cmInAA = 1e-8          # cm / AA
    lambda_Ha = 6562.8        # Angstrom
    mask_age = ages <= tSF
    
    y = flux * wl * cmInAA * L_sun / (h * c)

    qh__Zt = np.trapz(y = y, x = wl, axis = 2) # 1 / Msol
    Nh__Zt = spi.cumtrapz(y = qh__Zt, x = ages, initial = 0, axis = 1) * yr_sec
    Nh__Z = np.trapz(y = qh__Zt[:, mask_age], x = ages[mask_age], axis = 1) * yr_sec

    k_SFR__Z = 2.226 * lambda_Ha * L_sun * yr_sec / (Nh__Z * h * c) # M_sun / yr
    
    return qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z


def DrawHLRCircleInSDSSImage(ax, HLR_pix, pa, ba):
    from matplotlib.patches import Ellipse
    center = np.array([ 256 , 256])
    a = HLR_pix * 512.0 / 75.0 
    b_a = ba
    theta = pa * 180 / np.pi 
    e1 = Ellipse(center, height = 2 * a * b_a, width = 2 * a, angle = theta, fill = False, color = 'white', lw = 2, ls = 'dotted')
    e2 = Ellipse(center, height = 4 * a * b_a, width = 4 * a, angle = theta, fill = False, color = 'white', lw = 2, ls = 'dotted')
    ax.add_artist(e1)
    ax.add_artist(e2)

    
def DrawHLRCircle(ax, K, color = 'white', lw = 1.5):
    from matplotlib.patches import Ellipse
    center , a , b_a , theta = np.array([ K.x0 , K.y0]) , K.HLR_pix , K.ba , K.pa * 180 / np.pi 
    e1 = Ellipse(center, height = 2 * a * b_a, width = 2 * a, angle = theta, fill = False, color = color, lw = lw, ls = 'dotted')
    e2 = Ellipse(center, height = 4 * a * b_a, width = 4 * a, angle = theta, fill = False, color = color, lw = lw, ls = 'dotted')
    ax.add_artist(e1)
    ax.add_artist(e2)


class H5SFRData:
    def __init__(self, h5file):
        self.h5file = h5file
        try:
            self.h5 = h5py.File(self.h5file, 'r')
        except IOError:
            print "%s: file does not exists" % h5file
            pass

        self.RbinIni = self.get_data_h5('RbinIni')
        self.RbinFin = self.get_data_h5('RbinFin')
        self.RbinStep = self.get_data_h5('RbinStep')
        self.Rbin__r = self.get_data_h5('Rbin__r')
        self.RbinCenter__r = self.get_data_h5('RbinCenter__r')
        self.NRbins = self.get_data_h5('NRbins')
        self.RColor = self.get_data_h5('RColor')
        self.RRange = self.get_data_h5('RRange')
        self.xOkMin = self.get_data_h5('xOkMin')
        self.tauVOkMin = self.get_data_h5('tauVOkMin')
        self.tauVNebOkMin = self.get_data_h5('tauVNebOkMin')
        self.tauVNebErrMax = self.get_data_h5('tauVNebErrMax')
        self.tSF__T = self.get_data_h5('tSF__T')
        self.tZ__U = self.get_data_h5('tZ__U')
        self.califaIDs_all = self.get_data_h5('califaIDs')
        self.califaIDs = self.califaIDs_all.compressed()
        self.N_zones_all__g = self.get_data_h5('N_zones__g')
        self.N_gals_all = len(self.califaIDs_all)
        self.N_gals = len(self.califaIDs)
        self.N_zones__g = self.N_zones_all__g.compressed()
        self.N_T = len(self.tSF__T)
        self.N_U = len(self.tZ__U)

        self._create_attrs()
        
        
    def _create_attrs(self):
        # Ugly way to fill the arrays since ALLGals have all the
        # arrays in the 
        tmp = ALLGals(1,1,1,1)
        for attr in tmp.__dict__.keys():
            if attr[0] != '_' and not hasattr(self, attr):
                x = self.get_data_h5(attr)
                setattr(self, attr, x)
        del tmp

        
    def reply_arr_by_zones(self, p):
        if isinstance(p, str):
            p = self.get_data_h5(p)
        if isinstance(p, np.ma.core.MaskedArray):
            p = p.tolist()
        ltmp = [ itertools.repeat(p[i], self.N_zones__g[i]) for i in range(self.N_gals) ]
        arr = np.asarray(list(itertools.chain.from_iterable(ltmp)))
        return arr


    def reply_arr_by_radius(self, p, N_dim = None):
        if isinstance(p, str):
            p = self.get_data_h5(p)
        if isinstance(p, np.ma.core.MaskedArray):
            p = p.tolist()
        if N_dim:
            Nloop = N_dim * self.NRbins
            output_shape = (N_dim, self.NRbins, self.N_gals)
        else:
            Nloop = self.NRbins
            output_shape = (self.NRbins, self.N_gals)
        l = [ list(v) for v in [ itertools.repeat(p[i], Nloop) for i in range(self.N_gals) ]]
        return np.asarray([list(i) for i in zip(*l)]).reshape(output_shape)


    def __getattr__(self, attr):
        a = attr.split('_')
        x = None
        # somestr.find(str) returns 0 if str is found in somestr.
        if a[0]:
            if not a[0].find('K0'):
                gal = a[0]
                prop = '_'.join(a[1:])
                x = self.get_prop_gal(prop, gal)
            else:
                x = self.get_data_h5(attr)
            setattr(self, attr, x)
            return x


    def get_data_h5(self, prop):
        h5 = self.h5
        try:
            folder_data = 'masked/data'
            folder_mask = 'masked/mask'
            #if prop in h5[folder_mask].keys():
            node = '%s/%s' % (folder_data, prop)
            node_m = '%s/%s' % (folder_mask, prop)
            ds = h5[node]
            if isinstance(ds, h5py.Dataset):
                arr = np.ma.masked_array(ds.value, mask = h5[node_m].value)
            else:
                suffix = prop[-2:]
                if suffix[0] == 'U':
                    arr = [ 
                        np.ma.masked_array(h5['%s/%d' % (node, iU)].value, mask = h5['%s/%d' % (node_m, iU)].value) 
                        for iU in range(self.N_U) 
                    ]
                elif suffix[0] == 'T':
                    arr = [ 
                        np.ma.masked_array(h5['%s/%d' % (node, iT)].value, mask = h5['%s/%d' % (node_m, iT)].value) 
                        for iT in range(self.N_T) 
                    ]
            return arr
        except:
            try:
                node = '/data/' + prop
                ds = h5[node]
                return ds.value
            except:
                print '%s: property not found' % prop
                return None
        
        
    def get_prop_gal(self, data, gal = None):
        if isinstance(data, str):
            data = self.get_data_h5(data)
        arr = None
        if isinstance(data, list):
            califaIDs = self.reply_arr_by_zones(self.califaIDs)
            where_slice = np.where(califaIDs == gal)
            arr = [ data[iU][where_slice] for iU in range(len(data)) ]
        else:
            d_shape = data.shape
            if len(d_shape) == 3:
                califaIDs = self.reply_arr_by_radius(self.califaIDs, d_shape[0])
                where_slice = np.where(califaIDs == gal)
                prop_shape = d_shape[0:2]
                arr = data[where_slice].reshape(prop_shape)
            elif len(d_shape) == 2:
                califaIDs = self.reply_arr_by_radius(self.califaIDs)
                where_slice = np.where(califaIDs == gal)
                prop_shape = self.NRbins
                arr = data[where_slice].reshape(prop_shape)
            else:
                if data.shape == self.califaIDs.shape:
                    # that's not an array...
                    arr = data[self.califaIDs == gal].item()
                else:
                    califaIDs = self.reply_arr_by_zones(self.califaIDs)
                    where_slice = np.where(califaIDs == gal)
                    arr = data[where_slice]
        return arr
    
    
    def sort_gal_by_prop(self, prop, order = 1):
        '''
        ORDER = 0 - sort asc, 1 - sort desc
        '''
        gals = self.califaIDs
        if not isinstance(prop, str) and prop.shape == self.califaIDs.shape:
            data__g = prop
        else:
            data__g = np.asarray([ self.get_prop_gal(prop, gal) for gal in gals ])
        iS = np.argsort(data__g)
        if order != 0:
            iS = iS[::-1]
        sgals = np.asarray(gals)[iS]
        sdata = data__g[iS]
        return sgals, sdata