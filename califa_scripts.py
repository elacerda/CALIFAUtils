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


def read_one_cube(gal, **kwargs):
    try:
        EL = kwargs.pop('EL')
    except:
        EL = False
    try:
        verbose = kwargs.pop('verbose')
    except:
        verbose = False
    pycasso_cube_filename = pycasso_cube_dir + gal + pycasso_suffix
    K = None
    try:
        K = fitsQ3DataCube(pycasso_cube_filename)
        if verbose:
            print >> sys.stderr, 'PyCASSO: Reading file: %s' % pycasso_cube_filename
        if EL == True:
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
    try:
        verbose = kwargs.pop('verbose')
    except:
        verbose = False
        kwargs['verbose'] = False
    if type(func) != types.FunctionType:
        if verbose:
            print 'func need to be FunctionType'
        return None
    if verbose:
        print gals
    Ng = len(gals)
    data__g = np.ma.zeros((Ng))
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
    try:
        verbose = kwargs.pop('verbose')
    except:
        verbose = False
        kwargs['verbose'] = False
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
    data__g = np.ma.zeros((Ng))
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
    data__g = np.ma.zeros((Ng))
    for i, gal_id in enumerate(gals):
        K = read_one_cube(gal_id)
        try:
            attribute = K.__getattribute__(attr)
            if mode == None:
                data__g[i] = attribute
            else:
                f_mode = np.__getattribute__(mode)
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
    dx = np.zeros_like(x)
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
#         mask = np.zeros((N_gals), dtype = np.bool)
#         self.califaIDs = np.ma.masked_array(tmp, mask = mask)
# 
#     def _set_califaIDs(self):
#         self.N_gals = len(self._list)
#         mask = np.zeros((self.N_gals), dtype = np.bool)
#         self.califaIDs = np.ma.masked_array(self._list, mask = mask)
#         
#     def _init_data(self):
#         self.Mcor__g = np.ma.zeros((self.N_gals))
#         self.McorSD__g = np.ma.zeros((self.N_gals))
#         self.tau_V__g = np.ma.zeros((self.N_gals))
#         self.Mr__g = np.ma.zeros((self.N_gals))
#         self.morph__g = np.ma.zeros((self.N_gals))
#         self.at_flux__g = np.ma.zeros((self.N_gals))
#         self.ba__g = np.ma.zeros((self.N_gals))
#         self.u_r__g = np.ma.zeros((self.N_gals))
#         self.redshift__g = np.ma.zeros((self.N_gals))
#         if self._EL = True:
#             self.tau_V_neb__g = np.ma.zeros((self.N_gals))
#             self.EW_Ha__g = np.ma.zeros((self.N_gals))
#             self.F_obs_Ha__g = np.ma.zeros((self.N_gals))
#             self.F_obs_Hb__g = np.ma.zeros((self.N_gals))
#             self.F_obs_N2__g = np.ma.zeros((self.N_gals))
#             self.F_obs_O3__g = np.ma.zeros((self.N_gals))
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
            
#TODO
def loop_califa_galaxies(gals, func = None, **kwargs):
    '''
    Loop by all CALIFA galaxies in LIST executing f and returning a HDF5.
    F must be a type.FunctionType.
    '''
    try:
        verbose = kwargs.pop('verbose')
    except:
        verbose = False
        kwargs['verbose'] = False
    if type(func) != types.FunctionType:
        if verbose:
            print 'func need to be FunctionType'
        return None
    Ng = len(gals)
    data__g = np.ma.zeros((Ng))
    for i, gal_id in enumerate(gals):
        K = read_one_cube(gal_id, **kwargs)
        # TODO think about data structure, maybe an object
        data__g[i] = func(K, **kwargs)
        K.close()
    return data__g
        

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
    
    qh__Zt = (y * create_dx(l)).sum(axis = 2)
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
    center , a , b_a , theta = np.array([ 256 , 256]) , HLR_pix * 512.0 / 75.0 , ba , pa * 180 / np.pi 
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
        self.h5 = h5py.File(self.h5file, 'r')
        try: 
            self.califaIDs__g = self.get_data_h5('califaID_GAL_zones__g')
            self.califaIDs__rg = self.get_data_h5('califaID__rg')
            self.califaIDs__Trg = self.get_data_h5('califaID__Trg')
            self.califaIDs__Urg = self.get_data_h5('califaID__Urg')
            self.califaIDs = np.unique(self.califaIDs__g)
            self.tSF__T = self.get_data_h5('tSF__T')
            self.N_T = len(self.tSF__T)
            self.tZ__U = self.get_data_h5('tZ__U')
            self.N_U = len(self.tZ__U)
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
        except:
            print >> sys.stderr, 'Missing califaID_GAL_zones__g on h5 file!'

            
    def get_data_h5(self, prop):
        h5 = self.h5
        if any([ prop in s for s in h5['masked/mask'].keys() ]):
            node = '/masked/data/' + prop
            ds = h5[node]
            if type(ds) == h5py.Dataset:
                data = h5.get('/masked/data/' + prop).value
                mask = h5.get('/masked/mask/' + prop).value
                arr = np.ma.masked_array(data, mask = mask)
            else:
                suffix = prop[-2:]
                if suffix[0] == 'U':
                    arr = []
                    for iU, tZ in enumerate(self.tZ__U):
                        group = '%s/%d' % (prop, iU)
                        data = h5.get('/masked/data/' + group).value
                        mask = h5.get('/masked/mask/' + group).value
                        arr.append(np.ma.masked_array(data, mask = mask))
                elif suffix[0] == 'T':
                    arr = []
                    for iT, tSF in enumerate(self.tSF__T):
                        group = '%s/%d' % (prop, iT)
                        data = h5.get('/masked/data/' + group).value
                        mask = h5.get('/masked/mask/' + group).value
                        arr.append(np.ma.masked_array(data, mask = mask))
            return arr 
        else:
            return h5.get('/data/' + prop).value

        
    def get_prop_gal(self, prop, gal = None):
        data = self.get_data_h5(prop)
        prop__dim = None
        suffix = prop[-3:]
        if gal:
            if suffix[1:] == 'rg':
                if suffix[0] == '_':
                    where_slice = np.where(self.califaIDs__rg == gal)
                    prop__dim = data[where_slice]
                elif suffix[0] == 'U':
                    where_slice = np.where(self.califaIDs__Urg == gal)
                    prop__dim = (data[where_slice]).reshape(self.N_U, self.NRbins)
                elif suffix[0] == 'T':
                    where_slice = np.where(self.califaIDs__Trg == gal)
                    prop__dim = (data[where_slice]).reshape(self.N_T, self.NRbins)
            else:
                where_slice = np.where(self.califaIDs__g == gal)
                if type(data) is list:
                    #prop__dim here is prop__Tz
                    prop__dim = []
                    if prop[-2] == 'U':
                        for iU, tZ in enumerate(self.tZ__U):
                            prop__dim.append(data[iU][where_slice])
                    elif prop[-2] == 'T':
                        for iT, tSF in enumerate(self.tSF__T):
                            prop__dim.append(data[iT][where_slice])
                else:
                    # by zone
                    prop__dim = data[where_slice]
        return prop__dim

    
    def get_prop_uniq(self, prop):
        tmp = [np.unique(self.get_prop_gal(prop, g))[0] for g in self.califaIDs]
        return np.ma.masked_array(tmp, dtype = data.dtype)

    
    def sort_gal_by_prop(self, prop, unique = False, desc = False):
        list_uniq_gal = self.califaIDs
        if unique == True:
            data__g = self.get_data_h5(prop).compressed()
        else:
            data__g = self.get_prop_uniq(prop)
        iS = np.ma.argsort(data__g)
        if desc == True:
            iS = iS[::-1]
        list_uniq_gal = self.califaIDs[iS]
        return list_uniq_gal
