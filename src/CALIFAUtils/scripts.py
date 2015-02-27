#!/usr/bin/python
#
# Lacerda@Granada - 29/Jan/2015
#
import sys
import types
import numpy as np
from pycasso import fitsQ3DataCube
from .globals import pycasso_cube_dir
from .globals import pycasso_suffix
from .globals import emlines_cube_dir
from .globals import emlines_suffix
from .objects import read_kwargs

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
    morf = [morf0[i].strip() + morf1[i].strip() for i in xrange(len(morf0))]
    morf_m = [morf_m0[i].strip() + morf_m1[i].strip() for i in xrange(len(morf0))]
    morf_p = [morf_p0[i].strip() + morf_p1[i].strip() for i in xrange(len(morf0))]
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
    imax = kwargs.get('imax', None)
    for g in gals[:imax]:
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


def calc_alogZ_Stuff(K, tZ, xOkMin, Rbin__r):
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
    
    #radial profiles
    alogZ_mass__yx = K.zoneToYX(alogZ_mass__z, extensive = False, surface_density = False)
    alogZ_flux__yx = K.zoneToYX(alogZ_flux__z, extensive = False, surface_density = False)
    alogZ_mass__r = K.radialProfile(alogZ_mass__yx, Rbin__r, rad_scale = K.HLR_pix)
    alogZ_flux__r = K.radialProfile(alogZ_flux__yx, Rbin__r, rad_scale = K.HLR_pix)
    
    Mcor__z = np.ma.masked_array(K.Mcor__z, mask = ~isOk__z)
    Lobn__z = np.ma.masked_array(K.Lobn__z, mask = ~isOk__z)
    Mcor__yx = K.zoneToYX(Mcor__z, extensive = True)
    Lobn__yx = K.zoneToYX(Lobn__z, extensive = True)
    alogZ_mass_wei__r = radialProfileWeighted(alogZ_mass__yx, Mcor__yx, r_func = K.radialProfile, bin_r = Rbin__r, rad_scale = K.HLR_pix)
    alogZ_flux_wei__r = radialProfileWeighted(alogZ_flux__yx, Lobn__yx, r_func = K.radialProfile, bin_r = Rbin__r, rad_scale = K.HLR_pix)

    return alogZ_mass__z, alogZ_flux__z, alogZ_mass_GAL, alogZ_flux_GAL, \
           alogZ_mass__r, alogZ_flux__r, alogZ_mass_wei__r, alogZ_flux_wei__r, \
           isOkFrac_GAL


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