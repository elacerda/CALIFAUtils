#!/usr/bin/python
#
# Lacerda@Granada - 29/Jan/2015
#
import sys
import types
import numpy as np
import CALIFAUtils as C
from scipy.linalg import eigh
from CALIFAUtils.lines import Lines


def create_masks_gal(K, tSF__T, args = None, **kwargs):
    from CALIFAUtils.objects import tupperware_none
    from CALIFAUtils.lines import Lines
    debug = kwargs.get('debug', False)
    if args is None:
        args = tupperware_none()
        args.mintauv = kwargs.get('mintauv', np.finfo(np.float_).min)
        args.mintauvneb = kwargs.get('mintauvneb', np.finfo(np.float_).min)
        args.maxtauvneberr = kwargs.get('maxtauvneberr', np.finfo(np.float_).max)
        args.minpopx = kwargs.get('minpopx', np.finfo(np.float_).min)
        args.minEWHb = kwargs.get('minEWHb', np.finfo(np.float_).min)
        args.minSNR = kwargs.get('minSNR', 3)
        args.minSNRHb = kwargs.get('minSNRHb', 3)
        args.nolinecuts = kwargs.get('nolinecuts', args.nolinecuts)
        args.rgbcuts = kwargs.get('rgbcuts', args.rgbcuts)
        args.underS06 = kwargs.get('underS06', args.underS06)
        args.whanSF = kwargs.get('whanSF', args.whanSF)
        args.filter_residual = kwargs.get('filter_residual', args.filter_residual)
        args.gasprop = kwargs.get('gasprop', args.gasprop)
        C.debug_var(debug, pref = 'create_masks_gal() >>>>', args = args)
    args.gasprop = K.__dict__.has_key('GP') and args.gasprop
    #######################
    ### RESID.EML MASKS ###
    #######################
    Hb_central_wl = '4861'
    O3_central_wl = '5007'
    Ha_central_wl = '6563'
    N2_central_wl = '6583'
    lines_central_wl = [Hb_central_wl, O3_central_wl, Ha_central_wl, N2_central_wl]
    i_Hb = K.EL.lines.index(Hb_central_wl)
    i_Ha = K.EL.lines.index(Ha_central_wl)
    mask_lines_dict__Lz = {}
    for l in lines_central_wl:
        if args.nolinecuts is True:
            mask_lines_dict__Lz[l] = np.zeros((K.N_zone), dtype = np.bool_)
        else:
            if (args.rgbcuts and args.gasprop) is True: 
                pos = K.GP._dlcons[l]['pos']
                sigma = K.GP._dlcons[l]['sigma']
                snr = K.GP._dlcons[l]['SN']
            else:
                pos, sigma, snr = 3.0, 3.0, args.minSNR
                C.debug_var(debug, l = l)
            if snr < args.minSNR: snr = args.minSNR
            if l == '4861': snr = args.minSNRHb 
            mask_lines_dict__Lz[l] = K.EL._setMaskLineFluxNeg(l)
            mask_lines_dict__Lz[l] |= K.EL._setMaskLineDisplacement(l, pos)
            mask_lines_dict__Lz[l] |= K.EL._setMaskLineSigma(l, sigma)
            mask_lines_dict__Lz[l] |= K.EL._setMaskLineSNR(l, snr)
    mask_tau_V_neb__z = np.less(K.EL.tau_V_neb__z, args.mintauvneb)
    mask_tau_V_neb__z = np.ma.masked_array(mask_tau_V_neb__z, dtype = np.bool_, fill_value = True)
    mask_tau_V_neb__z = mask_tau_V_neb__z.data
    mask_tau_V_neb_err__z = np.greater(K.EL.tau_V_neb_err__z, args.maxtauvneberr)
    mask_tau_V_neb_err__z = np.ma.masked_array(mask_tau_V_neb_err__z, dtype = np.bool_, fill_value = True)
    mask_tau_V_neb_err__z = mask_tau_V_neb_err__z.data
    mask_EW_Hb__z = np.less(K.EL.EW[i_Hb], args.minEWHb)
    mask_EW_Hb__z = np.ma.masked_array(mask_EW_Hb__z, dtype = np.bool_, fill_value = True)
    mask_EW_Hb__z = mask_EW_Hb__z.data
    mask_bpt__z = np.zeros((K.N_zone), dtype = np.bool_)
    if args.underS06:
        L = Lines()
        N2Ha = np.ma.log10(K.EL.N2_obs__z / K.EL.Ha_obs__z)
        O3Hb = np.ma.log10(K.EL.O3_obs__z / K.EL.Hb_obs__z)
        mask_bpt__z = ~(L.belowlinebpt('S06', N2Ha, O3Hb))
    mask_whan__z = np.zeros((K.N_zone), dtype = np.bool_)
    if args.whanSF is not None:
        N2Ha = np.ma.log10(K.EL.N2_obs__z / K.EL.Ha_obs__z)
        WHa = K.EL.EW[i_Ha, :]
        mask_whan__z = np.bitwise_or(np.less(WHa, 3.), np.greater(N2Ha, -0.4))
    mask_eml__z = np.zeros(K.N_zone, dtype = np.bool_)
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lz[Hb_central_wl])
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lz[O3_central_wl])
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lz[Ha_central_wl])
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lz[N2_central_wl])
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_EW_Hb__z)
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_bpt__z)
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_whan__z)
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_tau_V_neb__z)
    mask_eml__z = np.bitwise_or(mask_eml__z, mask_tau_V_neb_err__z)
    #######################
    ### STARLIGHT MASKS ###
    #######################
    N_T = len(tSF__T)
    mask__Tz = np.zeros((N_T, K.N_zone), dtype = np.bool_)
    mask_syn__Tz = np.zeros((N_T, K.N_zone), dtype = np.bool_)
    mask_popx__Tz = np.zeros((N_T, K.N_zone), dtype = np.bool_)
    mask_tau_V__z = np.less(K.tau_V__z, args.mintauv) 
    mask_residual__z = np.zeros(K.N_zone, dtype = np.bool_)
    if args.filter_residual is True:
        mask_residual__z = ~(K.filterResidual(w2 = 4600))
    for iT, tSF in enumerate(tSF__T):
        mask_popx__Tz[iT] = np.less(calc_xY(K, tSF)[0], args.minpopx)
        mask_syn__Tz[iT] = np.bitwise_or(np.bitwise_or(mask_tau_V__z, mask_popx__Tz[iT]), mask_residual__z)
        #######################
        ### mixing up masks ###
        #######################
        mask__Tz[iT] = np.bitwise_or(mask_syn__Tz[iT], mask_eml__z)
    #######################
    #######################
    #######################
    print '# Mask Summary '              
    print '#'                                        
    print '# Gal: ', K.califaID
    print '# N_Zone: ', K.N_zone
    print '# N_x: ', K.N_x
    print '# N_y: ', K.N_y
    print '# N_pix: ', K.qMask.astype(int).sum()
    print '# Number of not contributing zones: '
    for l in lines_central_wl:
        print '# N_mask_line_', l, ' : ', mask_lines_dict__Lz[l].astype(int).sum()  
    print '# N_mask_tauVNeb: ', mask_tau_V_neb__z.astype(int).sum()
    print '# N_mask_bpt: ',mask_bpt__z.astype(int).sum()
    print '# N_mask_whan: ',mask_whan__z.astype(int).sum()
    print '# N_mask_eml: ', mask_eml__z.astype(int).sum()
    print '# N_mask_residual: ', mask_residual__z.astype(int).sum()
    print '# N_mask_tauV: ', mask_tau_V__z.astype(int).sum()
    for iT, tSF in enumerate(tSF__T):
        print '# N_mask_popx (%.3f Myrs): %d' % ((tSF / 1e6), mask_popx__Tz[iT].astype(int).sum())
        print '# N_mask_syn (%.3f Myrs): %d' % ((tSF / 1e6), mask_syn__Tz[iT].astype(int).sum())
        print '# N_mask_total (%.3f Myrs): %d' % ((tSF / 1e6), mask__Tz[iT].astype(int).sum())        
    return mask__Tz, mask_syn__Tz, mask_eml__z, \
        mask_popx__Tz, mask_tau_V__z, mask_residual__z, \
        mask_tau_V_neb__z, mask_tau_V_neb_err__z, mask_EW_Hb__z, mask_whan__z, mask_bpt__z, mask_lines_dict__Lz

def PCA(arr, reduced = False, arrMean = False, arrStd = False, sort = True):
    '''
    ARR array must have shape (measurements, variables)
    reduced = True:
        each var = (var - var.mean()) / var.std()
    '''
    arr__mv = arr
    nMeasurements, nVars = arr__mv.shape    
    if not arrMean or not arrMean.any():
        arrMean__v = arr.mean(axis = 0)
    else:
        arrMean__v = arrMean
    if not reduced:
        diff__mv = arr__mv - arrMean__v
    else:
        if not arrStd or not arrStd.any():
            arrStd__v = arr.std(axis = 0)
        else:
            arrStd__v = arrStd
        diff__mv = np.asarray([ diff / arrStd__v for diff in (arr__mv - arrMean__v) ])
    covMat__vv = (diff__mv.T).dot(diff__mv) / (nVars - 1)
    eigVal__e, eigVec__ve = eigh(covMat__vv)
    eigValS__e = eigVal__e
    eigVecS__ve = eigVec__ve
    if sort:
        S = np.argsort(eigVal__e)[::-1]
        eigValS__e = eigVal__e[S]
        eigVecS__ve = eigVec__ve[:, S]
    return diff__mv, arrMean__v, arrStd__v, covMat__vv, eigValS__e, eigVecS__ve

def my_morf(morf_in = None):
    mtype = {
        'Sa' : 0,
        'Sab' : 1,
        'Sb' : 2,
        'Sbc' : 3,
        'Sc' : 4,
        'Scd' : 5,
        'Sd' : 6,
        'Sdm' : 7,
        'Sm' : 7,
        'Ir' : 7,
        'E0' : -2,
        'E1' : -2,
        'E2' : -2,
        'E3' : -2,
        'E4' : -2,
        'E5' : -2,
        'E6' : -2,
        'E7' : -2,
        'S0' : -1,
        'S0a' : -1,
    }
    morf_out = mtype[morf_in]
    return morf_out  

def get_h5_data_masked(h5, prop_str, h5_root = '', add_mask = None, **ma_kwargs):
    prop = h5['%sdata/%s' % (h5_root, prop_str)].value
    mask = h5['%smask/%s' % (h5_root, prop_str)].value
    if add_mask is not None:
        mask = np.bitwise_or(mask, add_mask)
    return np.ma.masked_array(prop, mask, **ma_kwargs) 

def get_CALIFAID_by_NEDName(nedname):
    import atpy
    from califa import masterlist
    t = atpy.Table(C.CALIFAPaths().get_masterlist_file(), type = 'califa_masterlist')
    if isinstance(nedname, str):
        nedlist = [ nedname ]
    else:
        nedlist = nedname
    i_ned = []
    for nn in nedlist:
        try:
            i_ned.append(t['ned_name'].tolist().index(nn))
        except:
            print nedname, ': not found'
            #t.close()
            return False
    
    rval = np.copy(t['CALIFA_ID'][i_ned])
    #t.close()
    return rval

def get_NEDName_by_CALIFAID(califaID):
    import atpy
    from califa import masterlist
    t = atpy.Table(C.CALIFAPaths().get_masterlist_file(), type = 'califa_masterlist')
    if isinstance(califaID, str):
        califalist = [ califaID ]
    else:
        califalist = califaID
    i_cal = []
    for ci in califalist:
        try:
            i_cal.append(t['CALIFA_ID'].tolist().index(ci))
        except:
            print califaID, ': not found'
            #t.close()
            return False
    rval = np.copy(t['ned_name'][i_cal])    
    #t.close()
    return rval

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
                       'formats': ('S3', 'S15', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3', 'S3')
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
    Statistics of x & y with a minimal (floor limit) number of points in x
    Note: we have overlapping boxes.. so running stats..

    XXX Lacerda@IAA - masked array mess with the method
    '''
    
    debug = kwargs.get('debug', False)
    if isinstance(x, np.ma.core.MaskedArray) or isinstance(y, np.ma.core.MaskedArray): 
        xm, ym = ma_mask_xyz(x = x, y = y)
        x = xm.compressed()
        y = ym.compressed()
    ind_xs = np.argsort(x)
    xS = x[ind_xs]
    nx = len(x)
    frac = kwargs.get('frac', 0.1)
    minimal_bin_points = kwargs.get('min_np', nx * frac)
    i = 0
    xbin = []
    xbin.append(xS[0])
    while i < nx:
        to_i = i + minimal_bin_points
        delta = (nx - to_i)
        miss_frac = 1. * delta / nx
        if to_i < nx and miss_frac >= frac:
            xbin.append(xS[to_i])
        else:
            to_i = nx
            xbin.append(xS[-1])
        #print i, to_i, delta, miss_frac, xbin
        i = to_i
    # Def x-bins
    xbin = np.asarray(xbin)
    nxbin = len(xbin)
    debug_var(debug,
              minimal_bin_points = minimal_bin_points,
              xbin = xbin,
              n_xbin = nxbin)
    # Reset in-bin stats arrays
    xbinCenter_out = []
    xbin_out = []
    xMedian_out = []
    xMean_out = []
    xStd_out = []
    yMedian_out = []
    yMean_out = []
    yStd_out = []
    xPrc_out = []
    yPrc_out = []
    nInBin_out = []
    ixBin = 0
    while ixBin < (nxbin - 1):
        left = xbin[ixBin]
        xbin_out.append(left)
        right = xbin[ixBin + 1]
        isInBin = np.bitwise_and(np.greater_equal(x, left), np.less(x, right))
        xx , yy = x[isInBin] , y[isInBin]
        center = (right + left) / 2.
        xbin_out.append(right)
        xbinCenter_out.append(center)
        Np = isInBin.astype(np.int).sum()
        nInBin_out.append(Np)
        if Np >= 2:
            xMedian_out.append(np.median(xx))
            xMean_out.append(xx.mean())
            xStd_out.append(xx.std())
            yMedian_out.append(np.median(yy))
            yMean_out.append(yy.mean())
            yStd_out.append(yy.std())
            xPrc_out.append(np.percentile(xx, [5, 16, 84, 95]))
            yPrc_out.append(np.percentile(yy, [5, 16, 84, 95]))
        else:
            if len(xMedian_out) > 0:
                xMedian_out.append(xMedian_out[-1])
                xMean_out.append(xMean_out[-1])
                xStd_out.append(xStd_out[-1])
                yMedian_out.append(yMedian_out[-1])
                yMean_out.append(yMean_out[-1])
                yStd_out.append(yStd_out[-1])
            else:
                xMedian_out.append(0.)
                xMean_out.append(0.)
                xStd_out.append(0.)
                yMedian_out.append(0.)
                yMean_out.append(0.)
                yStd_out.append(0.)
            if len(xPrc_out) > 0:
                xPrc_out.append(xPrc_out[-1])
                yPrc_out.append(xPrc_out[-1])
            else:
                xPrc_out.append(np.asarray([0., 0., 0., 0.]))
                yPrc_out.append(np.asarray([0., 0., 0., 0.]))
        ixBin += 1
    C.debug_var(debug, xbinCenter_out = np.array(xbinCenter_out))
    return xbin, \
        np.array(xbinCenter_out), np.array(xMedian_out), np.array(xMean_out), np.array(xStd_out), \
        np.array(yMedian_out), np.array(yMean_out), np.array(yStd_out), np.array(nInBin_out), \
        np.array(xPrc_out).T, np.array(yPrc_out).T

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
        
def OLS_bisector(x, y, **kwargs):
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
        n = x.count()
    except AttributeError:
        n = len(x)
    gamma1 = b3 / ((b1 + b2) * (bb1 * bb2) ** 0.5)
    gamma13 = gamma1 * bb2
    gamma23 = gamma1 * bb1
    var_intercept = 1. / n ** 2.0
    var_intercept *= ((ydev - b3 * xdev - n * x.mean() * (gamma13 / Sxx * xdev * (ydev - b1 * xdev) + gamma23 / Sxy * ydev * (ydev - b2 * xdev))) ** 2.0).sum() 
    sigma_slope = var_slope ** 0.5
    sigma_intercept = var_intercept ** 0.5
    C.debug_var(kwargs.get('debug', False), 
                slope = slope, intercept = intercept, 
                sigma_slope = sigma_slope, sigma_intercept = sigma_intercept)
    return slope, intercept, sigma_slope, sigma_intercept

def get_HLR_pc(K, **kwargs):
    return K.HLR_pc

def get_HMR_pc(K, **kwargs):
    HMR_pix = K.getHalfRadius(K.McorSD__yx)
    return HMR_pix * K.parsecPerPixel

def get_McorSD_GAL(K, **kwargs):
    return K.McorSD__yx.mean()

def read_one_cube(gal, **kwargs):
    from pycasso import fitsQ3DataCube
    EL = kwargs.get('EL', None)
    GP = kwargs.get('GP', None)
    v_run = kwargs.get('v_run', -1)
    verbose = kwargs.get('verbose', None)
    debug = kwargs.get('debug', None)
    paths = C.CALIFAPaths(v_run = v_run)
    pycasso_cube_filename = paths.get_pycasso_file(gal)
    C.debug_var(debug, pycasso = pycasso_cube_filename)
    K = None
    try:
        K = fitsQ3DataCube(pycasso_cube_filename)
        if verbose is not None:
            print >> sys.stderr, 'PyCASSO: Reading file: %s' % pycasso_cube_filename
        if EL is True:
            emlines_cube_filename = paths.get_emlines_file(gal)
            C.debug_var(debug, emlines = emlines_cube_filename)
            try:
                K.loadEmLinesDataCube(emlines_cube_filename)
                if verbose is not None:
                    print >> sys.stderr, 'EL: Reading file: %s' % emlines_cube_filename
            except IOError:
                print >> sys.stderr, 'EL: File does not exists: %s' % emlines_cube_filename
        if GP is True:
            gasprop_cube_filename = paths.get_gasprop_file(gal)
            C.debug_var(debug, gasprop = gasprop_cube_filename)
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

def sort_gals(gals, func = None, order = 0, **kwargs):
    '''
    Sort galaxies inside GALS in ORDER order.
    
    GALS can be a list or a string with a text file direction. 
    
    If FUNC = None the function should return a list of galaxies sorted by 
    name following ORDER order. Otherwise pass by each CALIFA fits file 
    executing function FUNC that should return a number (or text) that will
    be used to the sort process. 
    After pass by each datacube (fits file) the list of galaxies names is 
    sorted by this data retrieved by FUNC. 
    
    ORDER
    > 0 - sort desc
      0 - no sort
    < 0 - sort asc
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
        if order != 0:
            sgals = None
            if data__g.mask.sum() < Ng:
                iS = np.argsort(data__g)
                if order < 0:
                    iS = iS[::-1]
                sgals = gals[iS]
                sdata = data__g[iS]
        else:
            sgals = gals
            sdata = data__g
    else:
        reverse = False
        if order != 0:
            if order < 0:
                reverse = True
            sgals = np.asarray(sorted(gals.tolist(), reverse = reverse))
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

def calc_xO(K, tO):
    _, aLow__t, aUpp__t, indO = calc_agebins(K.ageBase, tO)
    x__tZz = K.popx / K.popx.sum(axis = 1).sum(axis = 0)
    integrated_x__tZ = K.integrated_popx / K.integrated_popx.sum()
    aux1__z = x__tZz[(indO + 1):, :, :].sum(axis = 1).sum(axis = 0)
    aux2__z = x__tZz[indO, :, :].sum(axis = 0) * (aUpp__t[indO] - tO) / (aUpp__t[indO] - aLow__t[indO]) 
    integrated_aux1 = integrated_x__tZ[(indO + 1):, :].sum()
    integrated_aux2 = integrated_x__tZ[indO, :].sum(axis = 0) * (aUpp__t[indO] - tO) / (aUpp__t[indO] - aLow__t[indO])
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
    
    #aux1__z = K.MiniSD__tZz[:indSF, :, :].sum(axis = 1).sum(axis = 0)
    #aux2__z = K.MiniSD__tZz[indSF, :, :].sum(axis = 0) * (tSF - aLow__t[indSF]) / (aUpp__t[indSF] - aLow__t[indSF])

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

    alogZ_mass__r = None
    alogZ_flux__r = None
    alogZ_mass_oneHLR = None
    alogZ_flux_oneHLR = None
    alogZ_mass_wei__r = None
    alogZ_flux_wei__r = None
    
    # this bin_step only works if bins equally spaced 
    if Rbin__r is not None:
        binstep = Rbin__r[1] - Rbin__r[0]
        Rbin_oneHLR = [1. - binstep, 1. + binstep]
        #radial profiles
        alogZ_mass__yx = K.zoneToYX(alogZ_mass__z, extensive = False)
        alogZ_flux__yx = K.zoneToYX(alogZ_flux__z, extensive = False)
        alogZ_mass__r = K.radialProfile(alogZ_mass__yx, Rbin__r, rad_scale = K.HLR_pix)
        alogZ_flux__r = K.radialProfile(alogZ_flux__yx, Rbin__r, rad_scale = K.HLR_pix)
        alogZ_mass_oneHLR = K.radialProfile(alogZ_mass__yx, Rbin_oneHLR, rad_scale = K.HLR_pix)
        alogZ_flux_oneHLR = K.radialProfile(alogZ_flux__yx, Rbin_oneHLR, rad_scale = K.HLR_pix)
        Mcor__z = np.ma.masked_array(K.Mcor__z, mask = ~isOk__z)
        Lobn__z = np.ma.masked_array(K.Lobn__z, mask = ~isOk__z)
        Mcor__yx = K.zoneToYX(Mcor__z, extensive = True)
        Lobn__yx = K.zoneToYX(Lobn__z, extensive = True)
        alogZ_mass_wei__r = radialProfileWeighted(alogZ_mass__yx, Mcor__yx, r_func = K.radialProfile, bin_r = Rbin__r, rad_scale = K.HLR_pix)
        alogZ_flux_wei__r = radialProfileWeighted(alogZ_flux__yx, Lobn__yx, r_func = K.radialProfile, bin_r = Rbin__r, rad_scale = K.HLR_pix)

    return alogZ_mass__z, alogZ_flux__z, alogZ_mass_GAL, alogZ_flux_GAL, \
           alogZ_mass__r, alogZ_flux__r, alogZ_mass_wei__r, alogZ_flux_wei__r, \
           isOkFrac_GAL, alogZ_mass_oneHLR, alogZ_flux_oneHLR

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

def redshift_dist_Mpc(z, H0):
    from pystarlight.util.constants import c # m/s
    c /= 1e5
    return  z * c / H0

def spaxel_size_pc(z, H0):
    arc2rad = 0.0000048481368111
    return arc2rad * redshift_dist_Mpc(z, H0) * 1e6 
