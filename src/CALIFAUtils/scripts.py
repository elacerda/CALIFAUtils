#!/usr/bin/python
#
# Lacerda@Granada - 29/Jan/2015
#
import sys
import types
import numpy as np
import CALIFAUtils as C
from pycasso import fitsQ3DataCube

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
             [   0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14]]
    tipos = morf[Korder - 1] # tipo medio ascii
    tipo = gtype[1][gtype[0].index(morf[Korder - 1])] # tipo medio
    tipo_m = gtype[1][gtype[0].index(morf_m[Korder - 1])] # tipo minimo
    tipo_p = gtype[1][gtype[0].index(morf_p[Korder - 1])] # tipo maximo
    
    etipo_m = tipo - tipo_m  # error INFerior en tipo:  tipo-etipo_m
    etipo_p = tipo_p - tipo  # error SUPerior en tipo:  tipo+etipo_p
    
    return tipos, tipo, tipo_m, tipo_p

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def ma_mask_xyz(x, y, z = None, mask = None):
    m = np.zeros_like(x, dtype = np.bool)
    if isinstance(x, np.ma.core.MaskedArray):
        m |= np.bitwise_or(x.mask, np.isnan(x))
    if isinstance(y, np.ma.core.MaskedArray):
        m |= np.bitwise_or(y.mask, np.isnan(y))
    if mask is not None:
         m |= mask
    if z is not None:
        if isinstance(z, np.ma.core.MaskedArray):
            m |= np.copy(z.mask) | np.isnan(z)
        return np.ma.masked_array(x, mask = m, dtype = np.float_), np.ma.masked_array(y, mask = m, dtype = np.float_), np.ma.masked_array(z, mask = m, dtype = np.float_)
    return np.ma.masked_array(x, mask = m, dtype = np.float_), np.ma.masked_array(y, mask = m, dtype = np.float_)

def calc_running_stats(x, y, **kwargs):
    '''
    Statistics of x & y in equal size x-bins (dx-box).
    Note the mery small default xbinStep, so we have overlapping boxes.. so running stats..

    Cid@Lagoa -
    '''
    # XXX Lacerda@IAA - masked array mess with the method
    debug = kwargs.get('debug', False)
    C.debug_var(debug, kwargs = kwargs)
    overlap = kwargs.get('overlap', 0.2)
    nBox_tmp = kwargs.get('nBox', np.floor(len(x) * 0.1)) 
    xbinIni = kwargs.get('xbinIni', x.min())
    xbinFin = kwargs.get('xbinFin', x.max())
    xbinStep = kwargs.get('xbinStep', (x.max() - x.min()) / (nBox_tmp - 1.))
    dxBox = kwargs.get('dxBox', xbinStep * (1 + overlap))
    if isinstance(x, np.ma.core.MaskedArray):
        x = x[~(x.mask)]
    if isinstance(y, np.ma.core.MaskedArray):
        y = y[~(y.mask)]
    # Def x-bins
    xbin = kwargs.get('xbin', np.arange(xbinIni, xbinFin + xbinStep, xbinStep))
    #xbinCenter = (xbin[:-1] + xbin[1:]) / 2.0
    xbinCenter = np.diff(xbin) / 2.0 + xbin[:-1]
    C.debug_var(debug, pref = 'OOO>',
                dxBox = dxBox,
                xbinIni = xbinIni,
                xbinFin = xbinFin,
                xbinStep = xbinStep,
                xbin = xbin, 
                xbinCenter = xbinCenter,
    ) 
    Nbins = len(xbinCenter)
    # Reset in-bin stats arrays
    xMedian = np.zeros(Nbins)
    xMean = np.zeros(Nbins)
    xStd = np.zeros(Nbins)
    yMedian = np.zeros(Nbins)
    yMean = np.zeros(Nbins)
    yStd = np.zeros(Nbins)
    xPrc5 = np.zeros(Nbins)
    xPrc16 = np.zeros(Nbins)
    xPrc84 = np.zeros(Nbins)
    xPrc95 = np.zeros(Nbins)
    yPrc5 = np.zeros(Nbins)
    yPrc16 = np.zeros(Nbins)
    yPrc84 = np.zeros(Nbins)
    yPrc95 = np.zeros(Nbins)
    nInBin = np.zeros(Nbins)
    # fill up in x & y stats for each x-bin
    bin_radius = dxBox / 2.
    for ixBin in xrange(Nbins):
        #fix the borders
        isInBin = (np.abs(x - xbinCenter[ixBin]) <= bin_radius)
        xx , yy = x[isInBin] , y[isInBin]
        Np = isInBin.sum()
        if (Np >= 2):
            xMedian[ixBin] = np.median(xx)
            xMean[ixBin] = xx.mean()
            xStd[ixBin] = xx.std()
            yMedian[ixBin] = np.median(yy)
            yMean[ixBin] = yy.mean()
            yStd[ixBin] = yy.std()
            xPrc5[ixBin], xPrc16[ixBin], xPrc84[ixBin], xPrc95[ixBin] = np.percentile(xx, [5, 16, 84, 95])
            yPrc5[ixBin], yPrc16[ixBin], yPrc84[ixBin], yPrc95[ixBin] = np.percentile(yy, [5, 16, 84, 95])
        else:
            if ixBin > 0:
                xMedian[ixBin] = xMedian[ixBin - 1]
                xMean[ixBin] = xMean[ixBin - 1]
                xStd[ixBin] = xStd[ixBin - 1]
                yMedian[ixBin] = yMedian[ixBin - 1]
                yMean[ixBin] = yMean[ixBin - 1]
                yStd[ixBin] = yStd[ixBin - 1]
                xPrc5[ixBin] = xPrc5[ixBin - 1]
                xPrc16[ixBin] = xPrc16[ixBin - 1]
                xPrc84[ixBin] = xPrc84[ixBin - 1]
                xPrc95[ixBin] = xPrc95[ixBin - 1]
                yPrc5[ixBin] = yPrc5[ixBin - 1]
                yPrc16[ixBin] = yPrc16[ixBin - 1]
                yPrc84[ixBin] = yPrc84[ixBin - 1]
                yPrc95[ixBin] = yPrc95[ixBin - 1]
            else:
                if Np == 1:
                    xMedian[ixBin] = xx
                    xMean[ixBin] = xx
                    xStd[ixBin] = xx
                    yMedian[ixBin] = yy
                    yMean[ixBin] = yy
                    yStd[ixBin] = yy
                    xPrc5[ixBin] = xx
                    xPrc16[ixBin] = xx
                    xPrc84[ixBin] = xx
                    xPrc95[ixBin] = xx
                    yPrc5[ixBin] = yy
                    yPrc16[ixBin] = yy
                    yPrc84[ixBin] = yy
                    yPrc95[ixBin] = yy
        nInBin[ixBin] = Np
    return xbinCenter, xMedian, xMean, xStd, yMedian, yMean, yStd, nInBin, [xPrc5, xPrc16, xPrc84, xPrc95], [yPrc5, yPrc16, yPrc84, yPrc95]

def gaussSmooth_YofX(x, y, FWHM):
    '''
    Sloppy function to return the gaussian-smoothed version of an y(x) relation.
    Cid@Lagoa - 07/June/2014
    '''

    sig = FWHM / np.sqrt(8. * np.log(2))
    xS , yS = np.zeros_like(x), np.zeros_like(x)
    w__ij = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        # for j in np.arange(len(x)):
        #     w__ij[i,j] = np.exp( -0.5 * ((x[j] - x[i]) / sig)**2  )

        w__ij[i, :] = np.exp(-0.5 * ((x - x[i]) / sig) ** 2)
        w__ij[i, :] = w__ij[i, :] / w__ij[i, :].sum()

        xS[i] = (w__ij[i, :] * x).sum()
        yS[i] = (w__ij[i, :] * y).sum()

    return xS , yS

def calcYofXStats_EqNumberBins(x, y, nPerBin = 25):
    '''
    This gives the statistics of y(x) for x-bins of variable width, but all containing
    the same number of points.
    We 1st sort x, and the y accordingly. Then we compute the median, mean and std
    of x & y in contiguous x-bins in x defined to have nPerBin points each

    Cid@Lagoa - 05/June/2014
    '''

    ind_sx = np.argsort(x)
    xS , yS = x[ind_sx] , y[ind_sx]

    Nbins = len(x) - nPerBin + 1
    xMedian , xMean , xStd = np.zeros(Nbins) , np.zeros(Nbins) , np.zeros(Nbins)
    yMedian , yMean , yStd = np.zeros(Nbins) , np.zeros(Nbins) , np.zeros(Nbins)
    nInBin = np.zeros(Nbins)

    for ixBin in np.arange(0, Nbins):
        xx , yy = xS[ixBin:ixBin + nPerBin] , yS[ixBin:ixBin + nPerBin]
        xMedian[ixBin] , xMean[ixBin] , xStd[ixBin] = np.median(xx) , xx.mean() , xx.std()
        yMedian[ixBin] , yMean[ixBin] , yStd[ixBin] = np.median(yy) , yy.mean() , yy.std()
        nInBin[ixBin] = len(xx)
    return xMedian, xMean, xStd, yMedian, yMean , yStd, nInBin

def data_uniq(list_gal, data):
    list_uniq_gal = np.unique(list_gal)
    NGal = len(list_uniq_gal)
    data__g = np.ones((NGal))
    
    for i, g in enumerate(list_uniq_gal):
        data__g[i] = np.unique(data[np.where(list_gal == g)])
        
    return NGal, list_uniq_gal, data__g        
        
def OLS_bisector(x, y):
    xdev = x - x.mean()
    ydev = y - y.mean()
    Sxx = (xdev ** 2.0).sum()
    Syy = (ydev ** 2.0).sum()
    Sxy = (xdev * ydev).sum()
    b1 = Sxy / Sxx
    b2 = Syy / Sxy
    var1 = 1. / Sxx ** 2.
    var1 *= (xdev ** 2.0 * (ydev - b1 * xdev) ** 2.0).sum()
    var2 = 1. / Sxy ** 2.
    var2 *= (ydev ** 2.0 * (ydev - b2 * xdev) ** 2.0).sum()
    
    cov12 = 1. / (b1 * Sxx ** 2.0)
    cov12 *= (xdev * ydev * (ydev - b1 * ydev) * (ydev - b2 * ydev)).sum() 
    
    bb1 = (1 + b1 ** 2.)
    bb2 = (1 + b2 ** 2.)

    b3 = 1. / (b1 + b2) * (b1 * b2 - 1 + (bb1 * bb2) ** .5)
    
    var = b3 ** 2.0 / ((b1 + b2) ** 2.0 * bb1 * bb2) 
    var *= (bb2 ** 2.0 * var1 + 2. * bb1 * bb2 * cov12 + bb1 ** 2. * var2)
        
    slope = b3
    intercept = y.mean() - slope * x.mean()
    var_slope = var
    
    try: 
        n = (~x.mask).sum()
    except AttributeError:
        n = len(x)
    
    gamma1 = b3 / ((b1 + b2) * (bb1 * bb2) ** 0.5)
    gamma13 = gamma1 * bb2
    gamma23 = gamma1 * bb1
    var_intercept = 1. / n ** 2.0
    var_intercept *= ((ydev - b3 * xdev - n * x.mean() * (gamma13 / Sxx * xdev * (ydev - b1 * xdev) + gamma23 / Sxy * ydev * (ydev - b2 * xdev))) ** 2.0).sum() 
    
    sigma_slope = var_slope ** 0.5
    sigma_intercept = var_intercept ** 0.5
    
    return slope, intercept, sigma_slope, sigma_intercept

def read_one_cube(gal, **kwargs):
    EL = kwargs.get('EL', None)
    GP = kwargs.get('GP', None)
    v_run = kwargs.get('v_run', -1)
    verbose = kwargs.get('verbose', None)
    paths = C.CALIFAPaths()
    paths.set_v_run(v_run)
    pycasso_cube_filename = paths.get_pycasso_file(gal)
    K = None
    try:
        K = fitsQ3DataCube(pycasso_cube_filename)
        if verbose is not None:
            print >> sys.stderr, 'PyCASSO: Reading file: %s' % pycasso_cube_filename
        if EL is True:
            emlines_cube_filename = paths.get_emlines_file(gal)
            try:
                K.loadEmLinesDataCube(emlines_cube_filename)
                if verbose is not None:
                    print >> sys.stderr, 'EL: Reading file: %s' % emlines_cube_filename
            except IOError:
                print >> sys.stderr, 'EL: File does not exists: %s' % emlines_cube_filename
        if GP is True:
            gasprop_cube_filename = paths.get_gasprop_file(gal)
            try:
                K.GP = C.GasProp(gasprop_cube_filename)
                if verbose is not None:
                    print >> sys.stderr, 'GP: Reading file: %s' % gasprop_cube_filename
            except IOError:
                print >> sys.stderr, 'GP: File does not exists: %s' % gasprop_cube_filename
    except IOError:
        print >> sys.stderr, 'PyCASSO: File does not exists: %s' % pycasso_cube_filename
    del paths
    return K

def loop_cubes(gals, **kwargs):
    imax = kwargs.get('imax', None)
    if isinstance(gals, str):
        gals = sort_gals(gals)[0].tolist()
    elif isinstance(gals, np.ndarray):
        gals = gals.tolist()
    for g in gals[:imax]:
        yield gals.index(g), read_one_cube(g, **kwargs)

def debug_var(turn_on = False, **kwargs):
    pref = kwargs.pop('pref', '>>>')
    if turn_on == True:
        for kw, vw in kwargs.iteritems():
            if isinstance(vw, dict):
                print '%s' % pref, kw
                for k, v in vw.iteritems():
                    print '\t%s' % pref, k, ':\t', v
            else:
                print '%s' % pref, '%s:\t' % kw, vw

def sort_gals(gals, func = None, order = 1, **kwargs):
    '''
    Sort galaxies in txt GALS by some ATTRibute processed by MODE in ORDER order.
    If FUNC = None returns a list of galaxies without sort.
    ORDER = 0 - sort asc, 1 - sort desc, < 0 - no sort
    MODE can be any numpy array method such as sum, max, min, mean, median, etc...

    '''
    verbose = kwargs.get('verbose', None)
    if isinstance(gals, str):
        fname = gals
        f = open(fname, 'r')
        g = []
        for line in f.xreadlines():
            l = line.strip()
            if l[0] == '#':
                continue
            g.append(l)
        f.close()
        gals = np.unique(np.asarray(g))
    elif isinstance(gals, list):
        gals = np.unique(np.asarray(gals))
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
        if order >= 0:
            sgals = None
            if data__g.mask.sum() < Ng:
                iS = np.argsort(data__g)
                if order != 0:
                    iS = iS[::-1]
                sgals = gals[iS]
                sdata = data__g[iS]
        else:
            sgals = gals
            sdata = data__g
    else:
        sgals = gals
        sdata = None
    if kwargs.get('return_data_sort', True) == True:
        return sgals, sdata
    else:
        return sgals

def create_dx(x):
    dx = np.empty_like(x)
    dx[1:] = (x[1:] - x[:-1]) / 2.   # dl/2 from right neighbor
    dx[:-1] += dx[1:]               # dl/2 from left neighbor
    dx[0] = 2 * dx[0]
    dx[-1] = 2 * dx[-1]
    #dx[-1]      = x[-1]
    return dx

def SFR_parametrize(flux, wl, ages, tSF, wl_lum = 6562.8):
    '''
    Find the k parameter in the equation SFR = k [M_sun yr^-1] L(Halpha) [(10^8 L_sun)^-1]
    
    TODO: blablabla
    
    Nh__Zt is obtained for all t in AGES differently from Nh__Z, which consists in the number
    of H-ionizing photons from MAX_AGE till today (t < MAX_AGE).
    ''' 
    from pystarlight.util.constants import L_sun, h, c, yr_sec
    
    cmInAA = 1e-8             # cm / AA
    mask_age = ages <= tSF
    
    y = flux * wl * cmInAA * L_sun / (h * c)
    #y = flux * cmInAA * L_sun #BOL
    
    qh__Zt = (y * create_dx(wl)).sum(axis = 2)
    Nh__Z = (qh__Zt[:, mask_age] * create_dx(ages[mask_age])).sum(axis = 1) * yr_sec
    Nh__Zt = np.cumsum(qh__Zt * create_dx(ages), axis = 1) * yr_sec
         
    k_SFR__Z = 2.226 * wl_lum * L_sun * yr_sec / (Nh__Z * h * c) # M_sun / yr
    
    return qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z

def linearInterpol(x1, x2, y1, y2, x):
    '''
    Let S be the matrix:
    
        S = |x x1 x2|
            |y y1 y2|
    
    Now we do:
    
        DET(S) = 0,
    
    to find the linear equation between the points (x1, y1) and (x2, y2). 
    Hence we find the general equation Ax + By + C = 0 where:
    
        A = (y1 - y2)
        B = (x2 - x1)
        C = x1y2 - x2y1
    '''
    return (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)

def SFR_parametrize_trapz(flux, wl, ages, tSF, wl_lum = 6562.8):
    '''
    Find the k parameter in the equation SFR = k [M_sun yr^-1] L(Halpha) [(10^8 L_sun)^-1]
    
    TODO: blablabla
    
    Nh__Zt is obtained for all t in AGES differently from Nh__Z, which consists in the number
    of H-ionizing photons from MAX_AGE till today (t < MAX_AGE).
    ''' 
    from pystarlight.util.constants import L_sun, h, c, yr_sec
    import scipy.integrate as spi
    
    cmInAA = 1e-8          # cm / AA
    mask_age = ages <= tSF
    
    y = flux * wl * cmInAA * L_sun / (h * c)

    qh__Zt = np.trapz(y = y, x = wl, axis = 2) # 1 / Msol
    Nh__Zt = spi.cumtrapz(y = qh__Zt, x = ages, initial = 0, axis = 1) * yr_sec
    Nh__Z = np.trapz(y = qh__Zt[:, mask_age], x = ages[mask_age], axis = 1) * yr_sec

    k_SFR__Z = 2.226 * wl_lum * L_sun * yr_sec / (Nh__Z * h * c) # M_sun / yr
    
    return qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z

def radialProfileWeighted(v__yx, w__yx, **kwargs): 
    r_func = kwargs.get('r_func', None)
    rad_scale = kwargs.get('rad_scale', None)
    bin_r = kwargs.get('bin_r', None)
    
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
    integrated_x__tZ = K.integrated_popx / K.integrated_popx.sum()
    aux1__z = x__tZz[:indY, :, :].sum(axis = 1).sum(axis = 0)
    aux2__z = x__tZz[indY, :, :].sum(axis = 0) * (tY - aLow__t[indY]) / (aUpp__t[indY] - aLow__t[indY])
    integrated_aux1 = integrated_x__tZ[:indY, :].sum()
    integrated_aux2 = integrated_x__tZ[indY, :].sum(axis = 0) * (tY - aLow__t[indY]) / (aUpp__t[indY] - aLow__t[indY])
    return (aux1__z + aux2__z), (integrated_aux1 + integrated_aux2)

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

def redshift_dist(z, H0):
    from pystarlight.util.constants import c
    return  z * c / H0
