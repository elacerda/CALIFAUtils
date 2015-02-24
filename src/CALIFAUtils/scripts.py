#!/usr/bin/python
#
# Lacerda@Granada - 29/Jan/2015
#
import numpy as np
import sys
import h5py
from pycasso import fitsQ3DataCube
import types
import itertools
from .globals import *


def get_morfologia(galName, morf_file = '/Users/lacerda/CALIFA/morph_eye_class.csv') : 
    # Morfologia, incluyendo tipo medio y +- error
    # ES.Enrique . DF . 20120808
    # ES.Enrique . Chiclana . 20140417 . Corrected to distinguish E0 and S0.
    Korder = int(galName[1:])
    # lee el numero de la galaxia, tipo y subtipo morfologico
    id, name, morf0, morf1, morf_m0, morf_m1, morf_p0, morf_p1, bar, bar_m, bar_p = \
        np.loadtxt(morf_file, delimiter = ',', unpack = True, 
                   usecols = (0, 2, 5, 6, 7, 8, 9, 10, 12, 13, 14), 
                   skiprows = 23, 
                   dtype = {
                       'names': ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'), 
                       'formats': ('I3', 'S15', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3')
                   })
    morf = [morf0[i].strip() + morf1[i].strip() for i in range(len(morf0))]
    morf_m = [morf_m0[i].strip() + morf_m1[i].strip() for i in range(len(morf0))]
    morf_p = [morf_p0[i].strip() + morf_p1[i].strip() for i in range(len(morf0))]
    # convierte tipo y subtipo morfologico a valor numerico T (-7:E0 -1:E7 0:S0 5:Sm) en array 'tipo'
    # este algoritmo es una verdadera chapuza, pero funciona.
    gtype = [['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'S0', 'S0a', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'Ir'], 
             [   0,    1,    2,    3,    4,    5,    6,    7,    8,   8.5,    9,   9.5,   10,  10.5,   11,  11.5,   12,  12.5,   13,   14]]
    tipos = morf[Korder - 1] # tipo medio ascii
    tipo = gtype[1][gtype[0].index(morf[Korder - 1])] # tipo medio
    tipo_m = gtype[1][gtype[0].index(morf_m[Korder - 1])] # tipo minimo
    tipo_p = gtype[1][gtype[0].index(morf_p[Korder - 1])] # tipo maximo
    
    etipo_m = tipo - tipo_m  # error INFerior en tipo:  tipo-etipo_m
    etipo_p = tipo_p - tipo  # error SUPerior en tipo:  tipo+etipo_p
    
    return tipos, tipo, tipo_m, tipo_p


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


def loop_cubes(gals, **kwargs):
    for g in gals:
        yield gals.index(g), read_one_cube(g, **kwargs)


def sort_gals(gals, func = None, order = 1, **kwargs):
    '''
    Sort galaxies in txt GALS by some ATTRibute processed by MODE in ORDER order.
    If FUNC = None returns a list of galaxies without sort.
    ORDER = 0 - sort asc, 1 - sort desc
    MODE can be any numpy array method such as sum, max, min, mean, median, etc...

    '''
    args = read_kwargs(**kwargs)
    verbose = args.verbose
    if isinstance(gals, str):
        fname = gals
        f = open(fname, 'r')
        gals = np.asarray([ l.strip() for l in f.readlines() ])
    Ng = len(gals)
    if isinstance(func, types.FunctionType):
        if verbose:
            print gals
        data__g = np.ma.empty((Ng))
        for i, K in loop_cubes(gals.tolist(), **kwargs):
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
    else:
        sgals = gals
        sdata = None
    return sgals, sdata


def create_dx(x):
    dx = np.empty_like(x)
    dx[1:] = (x[1:] - x[:-1]) / 2.   # dl/2 from right neighbor
    dx[:-1] += dx[1:]               # dl/2 from left neighbor
    dx[0] = 2 * dx[0]
    dx[-1] = 2 * dx[-1]
    #dx[-1]      = x[-1]
    return dx

            
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
        self.alogZ_mass__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_flux__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_mass_wei__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_flux_wei__Urg = np.ma.empty((N_U, NRbins, N_gals))
        
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
                    tmp_data = {'masked/data/%s' % v : self.__dict__[v].data}
                    tmp_mask = {'masked/mask/%s' % v : self.__dict__[v].mask}
                else:
                    if suffix == 'Tg':
                        tmp_data = {'masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in range(self.N_T)}
                        tmp_mask = {'masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in range(self.N_T)}
                    elif suffix == 'Ug':
                        tmp_data = {'masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in range(self.N_U)}
                        tmp_mask = {'masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in range(self.N_U)}
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
    
    
def radialProfileWeighted(v__yx, w__yx, **kwargs): 
    args = read_kwargs(**kwargs)
    r_func = args.r_func
    rad_scale = args.rad_scale
    bin_r = args.bin_r
    
    v__r = None

    if r_func:
        w__r = r_func(w__yx, bin_r = bin_r, mode = 'sum', rad_scale = rad_scale)
        v_w__r = r_func(v__yx * w__yx, bin_r = bin_r, mode = 'sum', rad_scale = rad_scale)
        v__r = v_w__r / w__r

    return v__r


def calc_xY(K, tY):
    _, aLow__t, aUpp__t, indY = calc_agebins(K.ageBase, tY)

    # Compute xY__z
    x__tZz = K.popx / K.popx.sum(axis = 1).sum(axis = 0)
    aux1__z = x__tZz[:indY, :, :].sum(axis = 1).sum(axis = 0)
    aux2__z = x__tZz[indY, :, :].sum(axis = 0) * (tY - aLow__t[indY]) / (aUpp__t[indY] - aLow__t[indY])
    return (aux1__z + aux2__z)
    

def calc_SFR(K, tSF):
    '''
    Add up (in Mini and x) populations younger than tSF to compute SFR's and xY.
    First for zones (__z), and then images (__yx).

    tSF is a arbitrary/continuous number; it'll cover full age-bins plus a last one which will be
    only partially covered. The mass (light) within this last age-bin is scaled by the ratio
    (tSF - bin-lower-age) / bin-size. (P ex, if tSF is such that only 34% of the last-bin width is
    covered, then only 34% of its mass is added to the total.)

    Since our bases span so many ages, this "exact" calculation is just a refinement over the simpler
    method of just adding upp all age-bins satisfying K.agebase < tSF.

    OBS: Note that we are NOT dezonifying SFR__z. (Among other reasons, it'll be compared to the un-dezonifiable tauV!)

    Cid@IAA - 27/Jan/2015
    '''
    _, aLow__t, aUpp__t, indSF = calc_agebins(K.ageBase, tSF)
    
    # Compute SFR__z
    aux1__z = K.Mini__tZz[:indSF, :, :].sum(axis = 1).sum(axis = 0)
    aux2__z = K.Mini__tZz[indSF, :, :].sum(axis = 0) * (tSF - aLow__t[indSF]) / (aUpp__t[indSF] - aLow__t[indSF])
    SFR__z = (aux1__z + aux2__z) / tSF
    SFRSD__z = SFR__z / K.zoneArea_pc2

    return SFR__z, SFRSD__z


def calc_alogZ_Stuff(K, tZ, xOkMin):
 #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
 # flag__t, Rbin__r, weiRadProf = False, xOkMin = 0.10):
 #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    '''
    Compute average logZ (alogZ_*) both for each zone (*__z) and the galaxy-wide
    average (*_GAL, computed a la GD14).

    Only st-pops satisfying the input flag__t (ageBase-related) mask are considered!
    This allows us to compute alogZ_* for, say, < 2 Gyr or 1--7 Gyr populations,
    as well as the whole-age range (using flag__t = True for all base ages)
    with a same function and saving the trouble of keeping separate variables for the same thing:-)

    Radial profiles of alogZ_mass and alogZ_flux are also computed. They are/are-not weighted
    (by Mcor__yx & Lobn__yx, respectively) according to weiRadProf.

    ==> return alogZ_mass_GAL, alogZ_flux_GAL, isOkFrac_GAL , alogZ_mass__r, alogZ_flux__r

    Cid@Lagoa - 05/Jun/2014


    !!HELP!! ATT: My way of computing alogZ_*__z produces nan', which are ugly but harmless.
    I tried to fix it using masked arrays:

    alogZ_mass__z  = np.ma.masked_array( numerator__z/(denominator__z+0e-30) , mask = (denominator__z == 0))

    but this did not work!

    Cid@Lagoa - 20/Jun/2014
    
    Correct nan problems using:
    alogZ_mass__z[np.isnan(alogZ_mass__z)] = np.ma.masked
    Lacerda@Granada - 19/Feb/2015
    
    removed radial profiles inside this func.
    Lacerda@Granada - 23/Feb/2015
    '''
    #--------------------------------------------------------------------------
    # Initialization
    Zsun = 0.019
    # Define log of base metallicities **in solar units** for convenience
    logZBase__Z = np.log10(K.metBase / Zsun)
    #--------------------------------------------------------------------------
    flag__t = K.ageBase <= tZ

    #--------------------------------------------------------------------------
    # Define alogZ_****__z: flux & mass weighted average logZ for each zone
    # ==> alogZ_mass__z - ATT: There may be nan's here depending on flag__t!
    numerator__z = np.tensordot(K.Mcor__tZz[flag__t, :, :] , logZBase__Z , (1, 0)).sum(axis = 0)
    denominator__z = K.Mcor__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0)
    alogZ_mass__z = np.ma.masked_array(numerator__z / denominator__z)
    alogZ_mass__z[np.isnan(alogZ_mass__z)] = np.ma.masked

    # ==> alogZ_flux__z - ATT: There may be nan's here depending on flag__t!
    numerator__z = np.tensordot(K.Lobn__tZz[flag__t, :, :] , logZBase__Z , (1, 0)).sum(axis = 0)
    denominator__z = K.Lobn__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0)
    alogZ_flux__z = np.ma.masked_array(numerator__z / denominator__z)
    alogZ_flux__z[np.isnan(alogZ_mass__z)] = np.ma.masked
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Def galaxy-wide averages of alogZ in light & mass, but **discards** zones having
    # too little light fractions in the ages given by flag__t
    isOk__z = np.ones_like(K.Mcor__z, dtype = np.bool)
    
    # Define Ok flag: Zones with light fraction x < xOkMin are not reliable for alogZ (& etc) estimation!
    if xOkMin >= 0.:
        x__tZz = K.popx / K.popx.sum(axis = 1).sum(axis = 0)
        xOk__z = x__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0)
        isOk__z = xOk__z > xOkMin
        
    # Fraction of all zones which are Ok in the isOk__z sense. Useful to censor galaxies whose
    # galaxy-wide averages are based on too few zones (hence not representative)
    # OBS: isOkFrac_GAL is not used in this function, but it's returned to be used by the caller
    isOkFrac_GAL = (1.0 * isOk__z.sum()) / (1.0 * K.N_zone)

    # Galaxy wide averages of logZ - ATT: Only isOk__z zones are considered in this averaging!
    numerator__z = K.Mcor__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0) * alogZ_mass__z
    denominator__z = K.Mcor__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0)
    alogZ_mass_GAL = numerator__z[isOk__z].sum() / denominator__z[isOk__z].sum()

    numerator__z = K.Lobn__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0) * alogZ_flux__z
    denominator__z = K.Lobn__tZz[flag__t, :, :].sum(axis = 1).sum(axis = 0)
    alogZ_flux_GAL = numerator__z[isOk__z].sum() / denominator__z[isOk__z].sum()

    return alogZ_mass__z, alogZ_flux__z, alogZ_mass_GAL, alogZ_flux_GAL, isOkFrac_GAL


def calc_agebins(ages, age):
    # Define ranges for age-bins
    # ToDo: This age-bin-edges thing could be made more elegant & general.
    aCen__t = ages
    aLow__t = np.empty_like(ages)
    aUpp__t = np.empty_like(ages)
    aLow__t[0] = 0.
    aLow__t[1:] = (aCen__t[1:] + aCen__t[:-1]) / 2
    aUpp__t[:-1] = aLow__t[1:]
    aUpp__t[-1] = aCen__t[-1]
    # Find index of age-bin corresponding to the last bin fully within < tSF
    age_index = np.where(aLow__t < age)[0][-1]
    return aCen__t, aLow__t, aUpp__t, age_index 


class H5SFRData(object):
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
        self.N_T = len(self.tSF__T)
        self.N_U = len(self.tZ__U)
        
        self.califaIDs_all = self.get_data_h5('califaIDs')
        self.califaIDs = self.califaIDs_all.compressed()
        self.N_gals_all = len(self.califaIDs_all)
        self.N_gals = len(self.califaIDs)
        self.N_zones_all__g = self.get_data_h5('N_zones__g')
        self.N_zones__g = self.N_zones_all__g.compressed()

        self._create_attrs()
        
        
    def _create_attrs(self):
        # Ugly way to fill the arrays since ALLGals have all the
        # arrays in the 
        tmp = ALLGals(1, 1, 1, 1)
        for attr in tmp.__dict__.keys():
            if attr[0] != '_' and attr not in self.__dict__.keys():
                x = self.get_data_h5(attr)
                setattr(self, attr, x)
        del tmp

        
    def reply_arr_by_zones(self, p):
        if isinstance(p, str):
            p = self.get_data_h5(p)
        if isinstance(p, np.ma.core.MaskedArray):
            p = p.compressed()
        if isinstance(p, np.ndarray):
            p = p.tolist()
        laux1 = [ itertools.repeat(a[0], times = a[1]) for a in zip(p, self.N_zones__g) ]
        return np.asarray(list(itertools.chain.from_iterable(laux1)))


    def reply_arr_by_radius(self, p, N_dim = None):
        if isinstance(p, str):
            p = self.get_data_h5(p)
        if isinstance(p, np.ndarray):
            p = p.tolist()
        if N_dim:
            Nloop = N_dim * self.NRbins
            output_shape = (N_dim, self.NRbins, self.N_gals_all)
        else:
            Nloop = self.NRbins
            output_shape = (self.NRbins, self.N_gals_all)
        l = [ list(v) for v in [ itertools.repeat(prop, Nloop) for prop in p ]]
        return np.asarray([list(i) for i in zip(*l)]).reshape(output_shape)


    def __getattr__(self, attr):
        a = attr.split('_')
        x = None
        if a[0]:
            # somestr.find(str) returns 0 if str is found in somestr.
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
        folder_data = 'masked/data'
        folder_mask = 'masked/mask'
        folder_nomask = 'data'
        if prop in h5[folder_mask].keys():
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
                        for iU in xrange(self.N_U) 
                    ]
                elif suffix[0] == 'T':
                    arr = [ 
                        np.ma.masked_array(h5['%s/%d' % (node, iT)].value, mask = h5['%s/%d' % (node_m, iT)].value) 
                        for iT in xrange(self.N_T) 
                    ]
            return arr
        elif prop in h5[folder_nomask].keys():
            node = '%s/%s' % (folder_nomask, prop)
            ds = h5[node]
            return ds.value
        
        
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
                califaIDs = self.reply_arr_by_radius(self.califaIDs_all, d_shape[0])
                where_slice = np.where(califaIDs == gal)
                prop_shape = d_shape[0:2]
                arr = data[where_slice].reshape(prop_shape)
            elif len(d_shape) == 2:
                califaIDs = self.reply_arr_by_radius(self.califaIDs_all)
                where_slice = np.where(califaIDs == gal)
                prop_shape = self.NRbins
                arr = data[where_slice].reshape(prop_shape)
            else:
                if data.shape == self.califaIDs_all.shape:
                    # that's not an array...
                    arr = data[self.califaIDs_all == gal].item()
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
