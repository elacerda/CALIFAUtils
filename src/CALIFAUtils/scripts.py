#!/usr/bin/python
#
# Lacerda@Granada - 29/Jan/2015
#
import os
import sys
import types
import numpy as np
from .objects import GasProp
from scipy.linalg import eigh
from .objects import CALIFAPaths, tupperware_none
from CALIFAUtils import __path__


def stack_spectra(K, sel, v_0=None, segmap__yx=None, noflag=False):
    '''
        XXX TODO:
        This function receive a pycasso fitsQ3DataCube (K).
    '''
    if segmap__yx is not None:
        '''
        Good to remember that if segmap is not None sel_zones is not but an array
        with the zones index. Otherwise sel_zones can be a boolean array with
        K.N_zone length marking True for each zone inside stack.
        '''
        zones = K.qZones[segmap__yx]
        sel = np.zeros(K.N_zone, dtype='bool')
        for z in zones:
            sel[z] = True
    wl_of = K.l_obs
    N = sel.astype('int').sum()
    O_of__lz = K.f_obs[:, sel]
    M_of__lz = K.f_syn[:, sel]
    err_of__lz = K.f_err[:, sel]
    b_of__lz = K.f_flag[:, sel]
    v_0__z = v_0
    if v_0 is None:
        v_0__z = K.v_0[sel]
    bindata = tupperware_none()
    bindata.O_rf__lz = np.zeros((K.Nl_obs, N), dtype='float')
    bindata.M_rf__lz = np.zeros((K.Nl_obs, N), dtype='float')
    bindata.err_rf__lz = np.zeros((K.Nl_obs, N), dtype='float')
    bindata.b_rf__lz = np.zeros((K.Nl_obs, N), dtype='float')
    for iz in range(N):
        #  bring all spectra local rest-frame
        R, bindata.O_rf__lz[:, iz] = doppler_resample_spec(wl_of, v_0__z[iz], O_of__lz[:, iz])
        _, bindata.M_rf__lz[:, iz] = doppler_resample_spec(wl_of, v_0__z[iz], M_of__lz[:, iz], R)
        _, bindata.err_rf__lz[:, iz] = doppler_resample_spec(wl_of, v_0__z[iz], err_of__lz[:, iz], R)
        _, bindata.b_rf__lz[:, iz] = doppler_resample_spec(wl_of, v_0__z[iz], b_of__lz[:, iz], R)
    # set the data to store
    # creating badpixels flag
    if noflag:
        b_tmp = np.zeros_like(bindata.b_rf__lz)
    else:
        b_tmp = np.where(bindata.b_rf__lz == 0., 0., 1.)
    bad_ratio = b_tmp.sum(axis=1)/(1.*N)
    flag_factor = np.where(bad_ratio == 1., 0., 1./(1.-bad_ratio))
    b_rf__l = bindata.b_rf__lz.sum(axis=1)
    bad_ratio__l = bad_ratio
    # improved sum of values for each lambda in this bin
    fmasktmp__l = np.ma.masked_array(bindata.O_rf__lz, mask=b_tmp.astype('bool')).sum(axis=1)
    fsumok__l = np.where(np.ma.is_mask(fmasktmp__l), 0., fmasktmp__l * flag_factor)
    O_rf__l = fsumok__l
    M_rf__l = bindata.M_rf__lz.sum(axis=1)
    # squareroot of the sum of squares
    ferrmasktmp__l = np.square(np.ma.masked_array(bindata.err_rf__lz, mask=b_tmp.astype('bool'))).sum(axis=1)
    ferrsumok__l = np.where(np.ma.is_mask(ferrmasktmp__l), 0., ferrmasktmp__l * flag_factor)
    err_rf__l = ferrsumok__l ** 0.5
    return O_rf__l, M_rf__l, err_rf__l, b_rf__l, bad_ratio__l, bindata


def doppler_resample_spec(lorig, v_0, Fobs__l, R=None):
    from astropy import constants as const
    from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
    # doppler factor to correct wavelength
    dopp_fact = (1.0 + v_0 / const.c.to('km/s').value)
    # resample matrix
    if R is None:
        R = ReSamplingMatrixNonUniform(lorig=lorig / dopp_fact, lresam=lorig)
    return R, np.tensordot(R, Fobs__l * dopp_fact, (1, 0))


# Trying to correctly load q055 cubes inside q054 directory
def try_q055_instead_q054(califaID, **kwargs):
    from pycasso import fitsQ3DataCube
    config = kwargs.get('config', -1)
    EL = kwargs.get('EL', False)
    GP = kwargs.get('GP', False)
    elliptical = kwargs.get('elliptical', False)
    K = None
    P = CALIFAPaths()
    P.set_config(config)
    pycasso_file = P.get_pycasso_file(califaID)
    if not os.path.isfile(pycasso_file):
        P.pycasso_suffix = P.pycasso_suffix.replace('q054', 'q055')
        pycasso_file = P.get_pycasso_file(califaID)
        print(pycasso_file)
        if os.path.isfile(pycasso_file):
            K = fitsQ3DataCube(P.get_pycasso_file(califaID))
            if elliptical:
                K.setGeometry(*K.getEllipseParams())
            if EL:
                emlines_file = P.get_emlines_file(califaID)
                if not os.path.isfile(emlines_file):
                    P.emlines_suffix = P.emlines_suffix.replace('q054', 'q055')
                    emlines_file = P.get_emlines_file(califaID)
                    print(emlines_file)
                    if os.path.isfile(emlines_file):
                        K.loadEmLinesDataCube(emlines_file)
            if GP:
                gasprop_file = P.get_gasprop_file(califaID)
                if not os.path.isfile(gasprop_file):
                    P.gasprop_suffix = P.gasprop_suffix.replace('q054', 'q055')
                    gasprop_file = P.get_gasprop_file(califaID)
                    print(gasprop_file)
                    if os.path.isfile(gasprop_file):
                        K.GP = GasProp(gasprop_file)
    return K


def mask_zones_iT(iT, H, args, maskRadiusOk, gals_slice):
    mask__g = np.zeros_like(np.ma.log10(H.SFRSD_Ha__g * 1e6).mask, dtype=np.bool_)
    mask__g[np.ma.log10(H.SFRSD__Tg[iT] * 1e6).mask] = True
    mask__g[np.ma.log10(H.tau_V__Tg[iT]).mask] = True
    mask__g[np.ma.log10(H.SFRSD_Ha__g * 1e6).mask] = True
    mask__g[np.ma.log10(H.tau_V_neb__g).mask] = True
    mask__g[H.logO3N2_M13__g.mask] = True
    mask__g[np.less(H.reply_arr_by_zones(H.ba_GAL__g), args.bamin)] = True
    mask__g[~maskRadiusOk] = True
    mask__g[~gals_slice] = True
    #mask__g = np.bitwise_or(mask__g, np.less(H.EW_Ha__g, 3.))
    return mask__g


def mask_radius_iT(iT, H, args, maskRadiusOk, gals_slice):
    mask__rg = np.zeros_like(maskRadiusOk, dtype=np.bool_)
    mask__rg[np.less(H.reply_arr_by_radius(H.ba_GAL__g), args.bamin)] = True
    mask__rg[~maskRadiusOk] = True
    mask__rg[~gals_slice] = True
    return mask__rg


def create_zones_masks_gal(K, tSF__T, args=None, **kwargs):
    from CALIFAUtils.objects import tupperware_none
    from CALIFAUtils.lines import Lines
    debug = kwargs.get('debug', False)
    summary = kwargs.get('summary', False)
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
        debug_var(debug, pref='create_masks_gal() >>>>', args=args)
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
    mask_lines_dict__Lmz = {}
    mask_lines_dict__Lz = {}
    for l in lines_central_wl:
        mask_lines_dict__Lmz[l] = {}
        if args.nolinecuts is True:
            mask_lines_dict__Lz[l] = np.zeros((K.N_zone), dtype=np.bool_)
            mask_lines_dict__Lmz[l]['fluxneg'] = np.zeros((K.N_zone), dtype=np.bool_)
            mask_lines_dict__Lmz[l]['pos'] = np.zeros((K.N_zone), dtype=np.bool_)
            mask_lines_dict__Lmz[l]['sigma'] = np.zeros((K.N_zone), dtype=np.bool_)
            mask_lines_dict__Lmz[l]['SNR'] = np.zeros((K.N_zone), dtype=np.bool_)
        else:
            if args.rgbcuts is True:
                pos = K.GP._dlcons[l]['pos']
                sigma = K.GP._dlcons[l]['sigma']
                snr = K.GP._dlcons[l]['SN']
            else:
                pos, sigma, snr = 3.0, 3.0, args.minSNR
                debug_var(debug, l=l)
            if snr < args.minSNR:
                snr = args.minSNR
            if l == '4861':
                snr = args.minSNRHb
            mask_lines_dict__Lmz[l]['fluxneg'] = K.EL._setMaskLineFluxNeg(l)
            mask_lines_dict__Lmz[l]['pos'] = K.EL._setMaskLineDisplacement(l, pos)
            mask_lines_dict__Lmz[l]['sigma'] = K.EL._setMaskLineSigma(l, sigma)
            mask_lines_dict__Lmz[l]['SNR'] = K.EL._setMaskLineSNR(l, snr)
            mask_lines_dict__Lz[l] = K.EL._setMaskLineFluxNeg(l)
            mask_lines_dict__Lz[l] |= K.EL._setMaskLineDisplacement(l, pos)
            mask_lines_dict__Lz[l] |= K.EL._setMaskLineSigma(l, sigma)
            mask_lines_dict__Lz[l] |= K.EL._setMaskLineSNR(l, snr)
    mask_tau_V_neb__z = np.less(K.EL.tau_V_neb__z, args.mintauvneb)
    mask_tau_V_neb__z = np.ma.masked_array(mask_tau_V_neb__z, dtype=np.bool_, fill_value=True)
    mask_tau_V_neb__z = mask_tau_V_neb__z.data
    mask_tau_V_neb_err__z = np.greater(K.EL.tau_V_neb_err__z, args.maxtauvneberr)
    mask_tau_V_neb_err__z = np.ma.masked_array(mask_tau_V_neb_err__z, dtype=np.bool_, fill_value=True)
    mask_tau_V_neb_err__z = mask_tau_V_neb_err__z.data
    mask_EW_Hb__z = np.less(K.EL.EW[i_Hb], args.minEWHb)
    mask_EW_Hb__z = np.ma.masked_array(mask_EW_Hb__z, dtype=np.bool_, fill_value=True)
    mask_EW_Hb__z = mask_EW_Hb__z.data
    mask_bpt__z = np.zeros((K.N_zone), dtype=np.bool_)
    if args.underS06:
        L = Lines()
        N2Ha = np.ma.log10(K.EL.N2_obs__z / K.EL.Ha_obs__z)
        O3Hb = np.ma.log10(K.EL.O3_obs__z / K.EL.Hb_obs__z)
        mask_bpt__z = ~(L.belowlinebpt('S06', N2Ha, O3Hb))
    mask_whan__z = np.zeros((K.N_zone), dtype=np.bool_)
    if args.whanSF is not None:
        N2Ha = np.ma.log10(K.EL.N2_obs__z / K.EL.Ha_obs__z)
        WHa = K.EL.EW[i_Ha, :]
        mask_whan__z = np.bitwise_or(np.less(WHa, 3.), np.greater(N2Ha, -0.4))
    mask_eml__z = np.zeros(K.N_zone, dtype=np.bool_)
    if kwargs.get('mask_lines_snr_only', False):
        mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lmz[Hb_central_wl]['SNR'])
        mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lmz[O3_central_wl]['SNR'])
        mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lmz[Ha_central_wl]['SNR'])
        mask_eml__z = np.bitwise_or(mask_eml__z, mask_lines_dict__Lmz[N2_central_wl]['SNR'])
    else:
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
    mask__Tz = np.zeros((N_T, K.N_zone), dtype=np.bool_)
    mask_syn__Tz = np.zeros((N_T, K.N_zone), dtype=np.bool_)
    mask_popx__Tz = np.zeros((N_T, K.N_zone), dtype=np.bool_)
    mask_tau_V__z = np.less(K.tau_V__z, args.mintauv)
    mask_residual__z = np.zeros(K.N_zone, dtype=np.bool_)
    if args.filter_residual is True:
        mask_residual__z = ~(K.filterResidual(w2=4600))
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
    if summary or debug:
        print('# Mask Summary ')
        print('#')
        print('# Gal: ', K.califaID)
        print('# N_Zone: ', K.N_zone)
        print('# N_x: ', K.N_x)
        print('# N_y: ', K.N_y)
        print('# N_pix: ', K.qMask.astype(int).sum())
        print('# Number of not contributing zones: ')
        for l in lines_central_wl:
            print('# N_mask_line_', l, ' fluxneg: ', mask_lines_dict__Lmz[l]['fluxneg'].astype(int).sum())
            print('# N_mask_line_', l, ' pos: ', mask_lines_dict__Lmz[l]['pos'].astype(int).sum())
            print('# N_mask_line_', l, ' sigma: ', mask_lines_dict__Lmz[l]['sigma'].astype(int).sum())
            print('# N_mask_line_', l, ' SNR: ', mask_lines_dict__Lmz[l]['SNR'].astype(int).sum())
            print('# N_mask_line_', l, ' : ', mask_lines_dict__Lz[l].astype(int).sum())
        print('# N_mask_tauVNeb: ', mask_tau_V_neb__z.astype(int).sum())
        print('# N_mask_bpt: ',mask_bpt__z.astype(int).sum())
        print('# N_mask_whan: ',mask_whan__z.astype(int).sum())
        print('# N_mask_eml: ', mask_eml__z.astype(int).sum())
        print('# N_mask_residual: ', mask_residual__z.astype(int).sum())
        print('# N_mask_tauV: ', mask_tau_V__z.astype(int).sum())
        for iT, tSF in enumerate(tSF__T):
            print('# N_mask_popx (%.3f Myrs): %d' % ((tSF / 1e6), mask_popx__Tz[iT].astype(int).sum()))
            print('# N_mask_syn (%.3f Myrs): %d' % ((tSF / 1e6), mask_syn__Tz[iT].astype(int).sum()))
            print('# N_mask_total (%.3f Myrs): %d' % ((tSF / 1e6), mask__Tz[iT].astype(int).sum()))
    retmask_lines = mask_lines_dict__Lz
    k = 'mask_lines_dict__Lz'
    k_oth = 'mask_lines_dict__Lmz'
    if kwargs.get('return_mask_lines_separated', False):
        retmask_lines = mask_lines_dict__Lmz
        k = 'mask_lines_dict__Lmz'
        k_oth = 'mask_lines_dict__Lz'
    if kwargs.get('return_dict', False):
        D = {}
        D['mask__Tz'] = mask__Tz
        D['mask_syn__Tz'] = mask_syn__Tz
        D['mask_eml__z'] = mask_eml__z
        D['mask_popx__Tz'] = mask_popx__Tz
        D['mask_tau_V__z'] = mask_tau_V__z
        D['mask_residual__z'] = mask_residual__z
        D['mask_tau_V_neb__z'] = mask_tau_V_neb__z
        D['mask_tau_V_neb_err__z'] = mask_tau_V_neb_err__z
        D['mask_EW_Hb__z'] = mask_EW_Hb__z
        D['mask_whan__z'] = mask_whan__z
        D['mask_bpt__z'] = mask_bpt__z
        D[k] = retmask_lines
        D[k_oth] = None
    else:
        return mask__Tz, mask_syn__Tz, mask_eml__z, \
            mask_popx__Tz, mask_tau_V__z, mask_residual__z, \
            mask_tau_V_neb__z, mask_tau_V_neb_err__z, mask_EW_Hb__z, mask_whan__z, mask_bpt__z, retmask_lines


def PCA(arr, reduced=False, arrMean=False, arrStd=False, sort=True):
    '''
    ARR array must have shape (measurements, variables)
    reduced = True:
        each var = (var - var.mean()) / var.std()
    '''
    arr__mv = arr
    nMeasurements, nVars = arr__mv.shape
    if not arrMean or not arrMean.any():
        arrMean__v = arr.mean(axis=0)
    else:
        arrMean__v = arrMean
    if not reduced:
        diff__mv = arr__mv - arrMean__v
    else:
        if not arrStd or not arrStd.any():
            arrStd__v = arr.std(axis=0)
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


def my_morf(morf_in=None, get_dict=False):
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
    if get_dict is True:
        if morf_in is None:
            return mtype
        else:
            return mtype[morf_in], mtype
    else:
        return mtype[morf_in]


def get_h5_data_masked(h5, prop_str, h5_root='', add_mask=None, **ma_kwargs):
    prop = h5['%sdata/%s' % (h5_root, prop_str)].value
    mask = h5['%smask/%s' % (h5_root, prop_str)].value
    if add_mask is not None:
        mask = np.bitwise_or(mask, add_mask)
    return np.ma.masked_array(prop, mask, **ma_kwargs)


def get_CALIFAID_by_NEDName(nedname):
    import atpy
    from califa import masterlist
    t = atpy.Table(CALIFAPaths().get_masterlist_file(), type='califa_masterlist')
    if isinstance(nedname, str):
        nedlist = [ nedname ]
    else:
        nedlist = nedname
    i_ned = []
    for nn in nedlist:
        try:
            i_ned.append(t['ned_name'].tolist().index(nn))
        except:
            print(nedname, ': not found')
            #t.close()
            return False

    rval = np.copy(t['CALIFA_ID'][i_ned])
    #t.close()
    return rval


def get_NEDName_by_CALIFAID(califaID, work_dir=None):
    import atpy
    from califa import masterlist
    t = atpy.Table(CALIFAPaths(work_dir=work_dir).get_masterlist_file(), type='califa_masterlist')
    if isinstance(califaID, str):
        califalist = [ califaID ]
    else:
        califalist = califaID
    i_cal = []
    for ci in califalist:
        try:
            i_cal.append(t['CALIFA_ID'].tolist().index(ci))
        except:
            print(califaID, ': not found')
            #t.close()
            return False
    rval = np.copy(t['ned_name'][i_cal])
    #t.close()
    return rval


def get_morfologia(galName, morph_file=None) :
    if morph_file == None:
        morph_file = '%s/morph_eye_class.csv' % __path__[0]
    # Morfologia, incluyendo tipo medio y +- error
    # ES.Enrique . DF . 20120808
    # ES.Enrique . Chiclana . 20140417 . Corrected to distinguish E0 and S0.
    Korder = int(galName[1:])
    # lee el numero de la galaxia, tipo y subtipo morfologico
    id, name, morf0, morf1, morf_m0, morf_m1, morf_p0, morf_p1, bar, bar_m, bar_p = \
        np.loadtxt(morph_file, delimiter=',', unpack=True,
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


def ma_mask_xyz(x, y, z=None, mask=None, copy=True):
    m = np.zeros_like(x, dtype=np.bool)
    m |= np.isnan(x)
    m |= np.isnan(y)
    m |= np.isinf(x)
    m |= np.isinf(y)
    if isinstance(x, np.ma.core.MaskedArray):
        m |= np.bitwise_or(x.mask, np.isnan(x))
    if isinstance(y, np.ma.core.MaskedArray):
        m |= np.bitwise_or(y.mask, np.isnan(y))
    if mask is not None:
        m |= mask
    if z is not None:
        m |= np.isnan(z)
        m |= np.isinf(z)
        if isinstance(z, np.ma.core.MaskedArray):
            m |= np.copy(z.mask) | np.isnan(z)
        return np.ma.masked_array(x, mask=m, dtype=np.float_, copy=copy), np.ma.masked_array(y, mask=m, dtype=np.float_, copy=copy), np.ma.masked_array(z, mask=m, dtype=np.float_, copy=copy)
    return np.ma.masked_array(x, mask=m, dtype=np.float_, copy=copy), np.ma.masked_array(y, mask=m, dtype=np.float_, copy=copy)


def calc_running_stats(x, y, **kwargs):
    '''
    Statistics of x & y with a minimal (floor limit) number of points in x
    Note: we have overlapping boxes.. so running stats..

    XXX Lacerda@IAA - masked array mess with the method
    '''

    debug = kwargs.get('debug', False)
    if isinstance(x, np.ma.core.MaskedArray) or isinstance(y, np.ma.core.MaskedArray):
        xm, ym = ma_mask_xyz(x=x, y=y)
        x = xm.compressed()
        y = ym.compressed()
    ind_xs = np.argsort(x)
    xS = x[ind_xs]
    nx = len(x)
    frac = kwargs.get('frac', 0.1)
    minimal_bin_points = kwargs.get('min_np', nx * frac)
    i = 0
    xbin = kwargs.get('xbin', [])
    debug_var(debug, xbin=xbin)
    if xbin == []:
        xbin.append(xS[0])
        min_next_i = int(np.ceil(minimal_bin_points))
        next_i = min_next_i
        while i < nx:
            to_i = i + next_i
            delta = (nx - to_i)
            miss_frac = 1. * delta / nx
            #print(to_i, int(to_i), xS[to_i], xS[int(to_i)])
            if to_i < nx:
                if (xS[to_i] != xbin[-1]) and (miss_frac >= frac):
                    xbin.append(xS[to_i])
                    next_i = min_next_i
                else:
                    next_i += 1
            else:
                #### last bin will be the xS.max()
                to_i = nx
                xbin.append(xS[-1])
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
        Np = isInBin.astype(np.int).sum()
        nInBin_out.append(Np)
        xx , yy = x[isInBin] , y[isInBin]
        # print(ixBin, Np, xx, yy)
        center = (right + left) / 2.
        xbin_out.append(right)
        xbinCenter_out.append(center)
        if Np == 1:
            xMedian_out.append(np.median(xx))
            xMean_out.append(xx.mean())
            xStd_out.append(xx.std())
            yMedian_out.append(np.median(yy))
            yMean_out.append(yy.mean())
            yStd_out.append(yy.std())
            if len(xPrc_out) > 0:
                xPrc_out.append(xPrc_out[-1])
                yPrc_out.append(xPrc_out[-1])
            else:
                xPrc = np.median(xx)
                yPrc = np.median(yy)
                xPrc_out.append(np.asarray([xPrc, xPrc, xPrc, xPrc]))
                yPrc_out.append(np.asarray([yPrc, yPrc, yPrc, yPrc]))
        elif Np >= 2:
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
    debug_var(
        debug,
        xbinCenter_out = np.array(xbinCenter_out),
        xMedian_out = np.array(xMedian_out),
        yMedian_out = np.array(yMedian_out),
        nInBin_out = nInBin_out,
    )
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


def calcYofXStats_EqNumberBins(x, y, nPerBin=25):
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
    debug_var(kwargs.get('debug', False),
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
    print(kwargs)
    from pycasso import fitsQ3DataCube
    EL = kwargs.get('EL', None)
    GP = kwargs.get('GP', None)
    config = kwargs.get('config', kwargs.get('v_run', -1))
    verbose = kwargs.get('verbose', None)
    debug = kwargs.get('debug', None)
    work_dir = kwargs.get('work_dir', None)
    paths = CALIFAPaths(work_dir=work_dir, config=config)
    pycasso_cube_filename = paths.get_pycasso_file(gal)
    debug_var(debug, pycasso=pycasso_cube_filename)
    elliptical = kwargs.get('elliptical', False)
    K = None
    try:
        K = fitsQ3DataCube(pycasso_cube_filename)
        K._fits_filename = pycasso_cube_filename
        if elliptical:
            pa, ba = K.getEllipseParams()
            print(K.pa, K.ba, pa, ba)
            K.setGeometry(*K.getEllipseParams())
            print(K.pa, K.ba)
        if verbose is not None:
            print('PyCASSO: Reading file: %s' % pycasso_cube_filename, file=sys.stderr)
        if not isinstance(EL, type(None)):
            if EL is True:
                emlines_cube_filename = paths.get_emlines_file(gal)
            else:
                emlines_cube_filename = EL
            debug_var(debug, emlines=emlines_cube_filename)
            try:
                K.loadEmLinesDataCube(emlines_cube_filename)
                K.EL._fits_filename = emlines_cube_filename
                if verbose is not None:
                    print('EL: Reading file: %s' % emlines_cube_filename, file=sys.stderr)
            except IOError:
                print('EL: File does not exists: %s' % emlines_cube_filename, file=sys.stderr)
        if GP is True:
            gasprop_cube_filename = paths.get_gasprop_file(gal)
            debug_var(debug, gasprop=gasprop_cube_filename)
            try:
                K.GP = GasProp(gasprop_cube_filename)
                K.GP._fits_filename = gasprop_cube_filename
                if verbose is not None:
                    print('GP: Reading file: %s' % gasprop_cube_filename, file=sys.stderr)
            except IOError:
                print('GP: File does not exists: %s' % gasprop_cube_filename, file=sys.stderr)
    except IOError:
        print('PyCASSO: File does not exists: %s' % pycasso_cube_filename, file=sys.stderr)
    del paths
    return K


def F_to_L(flux, distance_Mpc):
    Mpc_in_cm = 3.08567758e24  # cm
    solidAngle = 4. * np.pi * (distance_Mpc * Mpc_in_cm) ** 2.0
    return solidAngle * flux


def loop_cubes(gals, **kwargs):
    imax = kwargs.get('imax', None)
    if isinstance(gals, str):
        gals = sort_gals(gals)[0].tolist()
    elif isinstance(gals, np.ndarray):
        gals = gals.tolist()
    for g in gals[:imax]:
        yield gals.index(g), read_one_cube(g, **kwargs)


def debug_var(turn_on=False, **kwargs):
    pref = kwargs.pop('pref', '>>>')
    if turn_on:
        for kw, vw in kwargs.iteritems():
            if isinstance(vw, dict):
                print('%s' % pref, kw)
                for k, v in vw.iteritems():
                    print('\t%s' % pref, k, ':\t', v)
            else:
                print('%s' % pref, '%s:\t' % kw, vw)


def get_data_gals(gals, func=None, **kwargs):
    '''
    Retreive data from galaxies inside GALS in using func.

    GALS can be a list or a string with a text file direction.

    This method will pass by each CALIFA fits file executing function FUNC that
    should return the data requested.
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
            print(gals)
        data__g = []
        for i, K in loop_cubes(gals.tolist(), **kwargs):
            data__g.append(func(K, **kwargs))
            if verbose:
                print(K.califaID, data__g[i])
            K.close()
        sgals = gals
        sdata = data__g
    else:
        sgals = gals
        sdata = None
    return sgals, sdata


def sort_gals(gals, func=None, order=0, **kwargs):
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
            print(gals)
        data__g = np.ma.empty((Ng))
        for i, K in loop_cubes(gals.tolist(), **kwargs):
            data__g[i] = func(K, **kwargs)
            if verbose:
                print(K.califaID, data__g[i])
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
            sgals = np.asarray(sorted(gals.tolist(), reverse=reverse))
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


def SFR_parametrize(flux, wl, ages, tSF, wl_lum=6562.8, qh__Zt=None):
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

    if qh__Zt is None:
        qh__Zt = (y * create_dx(wl)).sum(axis=2)
    Nh__Z = (qh__Zt[:, mask_age] * create_dx(ages[mask_age])).sum(axis=1) * yr_sec
    Nh__Zt = np.cumsum(qh__Zt * create_dx(ages), axis=1) * yr_sec

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


def SFR_parametrize_trapz(flux, wl, ages, tSF, wl_lum=6562.8):
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

    qh__Zt = np.trapz(y=y, x=wl, axis=2) # 1 / Msol
    Nh__Zt = spi.cumtrapz(y=qh__Zt, x=ages, initial=0, axis=1) * yr_sec
    Nh__Z = np.trapz(y=qh__Zt[:, mask_age], x=ages[mask_age], axis=1) * yr_sec

    k_SFR__Z = (1./0.453) * wl_lum * L_sun * yr_sec / (Nh__Z * h * c) # M_sun / yr

    return qh__Zt, Nh__Zt, Nh__Z, k_SFR__Z


def radialProfileWeighted(v__yx, w__yx, **kwargs):
    r_func = kwargs.get('r_func', None)
    rad_scale = kwargs.get('rad_scale', None)
    bin_r = kwargs.get('bin_r', None)

    v__r = None

    if r_func:
        w__r = r_func(w__yx, bin_r=bin_r, mode='sum', rad_scale=rad_scale)
        v_w__r = r_func(v__yx * w__yx, bin_r=bin_r, mode='sum', rad_scale=rad_scale)
        v__r = v_w__r / w__r

    return v__r


def calc_agebins(ages, age=None):
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
    age_index = -1
    if age is not None:
        age_index = np.where(aLow__t < age)[0][-1]
    return aCen__t, aLow__t, aUpp__t, age_index


def prop_Y(prop, tY, age_base):
    # prop must have dimension __tZz or __tZyx
    _, aLow__t, aUpp__t, indY = calc_agebins(age_base, tY)
    aux1 = prop[:indY, ...].sum(axis=1).sum(axis=0)
    aux2 = prop[indY, ...].sum(axis=0) * (tY - aLow__t[indY]) / (aUpp__t[indY] - aLow__t[indY])
    return (aux1 + aux2)


def integrated_prop_Y(prop, tY, age_base):
    # prop must have dimension __tZ
    _, aLow__t, aUpp__t, indY = calc_agebins(age_base, tY)
    aux1__z = prop[:indY, :].sum()
    aux2__z = prop[indY, :].sum(axis=0) * (tY - aLow__t[indY]) / (aUpp__t[indY] - aLow__t[indY])
    return (aux1__z + aux2__z)


def calc_xY(K=None, tY=32e6, popx=None, ageBase=None):
    integrated_x_Y = None
    if K is not None:
        popx = K.popx
        ageBase = K.ageBase
        integrated_x__tZ = K.integrated_popx / K.integrated_popx.sum()
        integrated_x_Y = integrated_prop_Y(integrated_x__tZ, tY, ageBase)
    # Compute xY
    if popx is not None:
        x__tZdim = popx / popx.sum(axis=1).sum(axis=0)
    return prop_Y(x__tZdim, tY, ageBase), integrated_x_Y


def calc_xO(K, tO):
    _, aLow__t, aUpp__t, indO = calc_agebins(K.ageBase, tO)
    x__tZz = K.popx / K.popx.sum(axis=1).sum(axis=0)
    integrated_x__tZ = K.integrated_popx / K.integrated_popx.sum()
    aux1__z = x__tZz[(indO + 1):, :, :].sum(axis=1).sum(axis=0)
    aux2__z = x__tZz[indO, :, :].sum(axis=0) * (aUpp__t[indO] - tO) / (aUpp__t[indO] - aLow__t[indO])
    integrated_aux1 = integrated_x__tZ[(indO + 1):, :].sum()
    integrated_aux2 = integrated_x__tZ[indO, :].sum(axis=0) * (aUpp__t[indO] - tO) / (aUpp__t[indO] - aLow__t[indO])
    return (aux1__z + aux2__z), (integrated_aux1 + integrated_aux2)


def calc_SFR(K, tSF):
    """
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
    """
    _, aLow__t, aUpp__t, indSF = calc_agebins(K.ageBase, tSF)

    # Compute SFR__z
    aux1__z = K.Mini__tZz[:indSF, :, :].sum(axis=1).sum(axis=0)
    aux2__z = K.Mini__tZz[indSF, :, :].sum(axis=0) * (tSF - aLow__t[indSF]) / (aUpp__t[indSF] - aLow__t[indSF])
    SFR__z = (aux1__z + aux2__z) / tSF
    SFRSD__z = SFR__z / K.zoneArea_pc2

    #aux1__z = K.MiniSD__tZz[:indSF, :, :].sum(axis=1).sum(axis=0)
    #aux2__z = K.MiniSD__tZz[indSF, :, :].sum(axis=0) * (tSF - aLow__t[indSF]) / (aUpp__t[indSF] - aLow__t[indSF])

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

    radial profiles back to this func.
    Lacerda@Corrego
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
    numerator__z = np.tensordot(K.Mcor__tZz[flag__t, :, :] , logZBase__Z , (1, 0)).sum(axis=0)
    denominator__z = K.Mcor__tZz[flag__t, :, :].sum(axis=1).sum(axis=0)
    alogZ_mass__z = np.ma.masked_array(numerator__z / denominator__z)
    alogZ_mass__z[np.isnan(alogZ_mass__z)] = np.ma.masked

    # ==> alogZ_flux__z - ATT: There may be nan's here depending on flag__t!
    numerator__z = np.tensordot(K.Lobn__tZz[flag__t, :, :] , logZBase__Z , (1, 0)).sum(axis=0)
    denominator__z = K.Lobn__tZz[flag__t, :, :].sum(axis=1).sum(axis=0)
    alogZ_flux__z = np.ma.masked_array(numerator__z / denominator__z)
    alogZ_flux__z[np.isnan(alogZ_mass__z)] = np.ma.masked
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Def galaxy-wide averages of alogZ in light & mass, but **discards** zones having
    # too little light fractions in the ages given by flag__t
    isOk__z = np.ones_like(K.Mcor__z, dtype=np.bool)

    # Define Ok flag: Zones with light fraction x < xOkMin are not reliable for alogZ (& etc) estimation!
    if xOkMin >= 0.:
        x__tZz = K.popx / K.popx.sum(axis=1).sum(axis=0)
        xOk__z = x__tZz[flag__t, :, :].sum(axis=1).sum(axis=0)
        isOk__z = xOk__z > xOkMin

    # Fraction of all zones which are Ok in the isOk__z sense. Useful to censor galaxies whose
    # galaxy-wide averages are based on too few zones (hence not representative)
    # OBS: isOkFrac_GAL is not used in this function, but it's returned to be used by the caller
    isOkFrac_GAL = (1.0 * isOk__z.sum()) / (1.0 * K.N_zone)

    # Galaxy wide averages of logZ - ATT: Only isOk__z zones are considered in this averaging!
    numerator__z = K.Mcor__tZz[flag__t, :, :].sum(axis=1).sum(axis=0) * alogZ_mass__z
    denominator__z = K.Mcor__tZz[flag__t, :, :].sum(axis=1).sum(axis=0)
    alogZ_mass_GAL = numerator__z[isOk__z].sum() / denominator__z[isOk__z].sum()

    numerator__z = K.Lobn__tZz[flag__t, :, :].sum(axis=1).sum(axis=0) * alogZ_flux__z
    denominator__z = K.Lobn__tZz[flag__t, :, :].sum(axis=1).sum(axis=0)
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
        alogZ_mass__yx = K.zoneToYX(alogZ_mass__z, extensive=False)
        alogZ_flux__yx = K.zoneToYX(alogZ_flux__z, extensive=False)
        alogZ_mass__r = K.radialProfile(alogZ_mass__yx, Rbin__r, rad_scale=K.HLR_pix)
        alogZ_flux__r = K.radialProfile(alogZ_flux__yx, Rbin__r, rad_scale=K.HLR_pix)
        alogZ_mass_oneHLR = K.radialProfile(alogZ_mass__yx, Rbin_oneHLR, rad_scale=K.HLR_pix)
        alogZ_flux_oneHLR = K.radialProfile(alogZ_flux__yx, Rbin_oneHLR, rad_scale=K.HLR_pix)
        Mcor__z = np.ma.masked_array(K.Mcor__z, mask=~isOk__z)
        Lobn__z = np.ma.masked_array(K.Lobn__z, mask=~isOk__z)
        Mcor__yx = K.zoneToYX(Mcor__z, extensive=True)
        Lobn__yx = K.zoneToYX(Lobn__z, extensive=True)
        alogZ_mass_wei__r = radialProfileWeighted(alogZ_mass__yx, Mcor__yx, r_func=K.radialProfile, bin_r=Rbin__r, rad_scale=K.HLR_pix)
        alogZ_flux_wei__r = radialProfileWeighted(alogZ_flux__yx, Lobn__yx, r_func=K.radialProfile, bin_r=Rbin__r, rad_scale=K.HLR_pix)

    return alogZ_mass__z, alogZ_flux__z, alogZ_mass_GAL, alogZ_flux_GAL, \
           alogZ_mass__r, alogZ_flux__r, alogZ_mass_wei__r, alogZ_flux_wei__r, \
           isOkFrac_GAL, alogZ_mass_oneHLR, alogZ_flux_oneHLR


def redshift_dist_Mpc(z, H0):
    from pystarlight.util.constants import c  # m/s
    c /= 1e5
    return z * c / H0


def spaxel_size_pc(dist_Mpc, arcsec=1):
    arc2rad = 4.84814e-6
    return arcsec * arc2rad * dist_Mpc * 1e6


def spaxel_size_pc_hubblelaw(z, H0, arcsec=1):
    return spaxel_size_pc(redshift_dist_Mpc(z, H0), arcsec)
