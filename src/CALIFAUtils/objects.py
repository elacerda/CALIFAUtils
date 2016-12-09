import numpy as np
import itertools
import pyfits
import h5py
import os


def get_h5_data_masked(h5, prop_str, h5_root='', add_mask=None, **ma_kwargs):
    prop = h5['%sdata/%s' % (h5_root, prop_str)].value
    mask = h5['%smask/%s' % (h5_root, prop_str)].value
    if add_mask is not None:
        mask = np.bitwise_or(mask, add_mask)
    return np.ma.masked_array(prop, mask, **ma_kwargs)


class CALIFAPaths(object):
    _superfits_dir = [
        'synth',
        'superfits'
    ]
    _spacialSampling = [
        'v20',
        'pix'
    ]
    _qVersion = [
        'q036',
        'q043',
        'q045',
        'q046',
        'q050',
        'q051',
        'q053',
        'q054'
    ]
    _dVersion = [
        'd13c',
        'd14a',
        'd15a',
        'd22a'
    ]
    _othSuffix = [
        '512.ps03.k1.mE.CCM.',
    ]
    _bases = [
        'Bgsd6e',
        'Bzca6e',
        'Bgstf6e'
    ]
    _config = [
        [
            _superfits_dir.index('superfits'),
            _spacialSampling.index('v20'),
            _qVersion.index('q050'),
            _dVersion.index('d15a'),
            0,
            _bases.index('Bgsd6e')
        ],
        [
            _superfits_dir.index('superfits'),
            _spacialSampling.index('v20'),
            _qVersion.index('q054'),
            _dVersion.index('d22a'),
            0,
            _bases.index('Bgstf6e')
        ],
    ]
    _masterlist_file = 'califa_master_list_rgb.txt'

    def __init__(self,
                 qVersion='q050',
                 dVersion='d15a',
                 base='Bgsd6e',
                 spacialSampling='v20',
                 othSuffix='512.ps03.k1.mE.CCM.',
                 work_dir=None,
                 config=None):
        if work_dir is None:
            work_dir = '%s/califa' % os.getenv('HOME')
        self.califa_work_dir = work_dir
        self.set_config(config)

    def _config_run(self):
        config = self.get_config()
        tmp_suffix = '_synthesis_eBR_'
        tmp_suffix += config['spacialSampling'] + '_' + config['qVersion'] + '.' + config['dVersion']
        tmp_suffix += config['othSuffix'] + config['base']
        self.pycasso_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + '/' + config['superfits_dir'] + '/'
        self.emlines_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + '/EML/'
        self.gasprop_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + '/EML/prop/'
        self.pycasso_suffix = tmp_suffix + '.fits'
        self.emlines_suffix = tmp_suffix + '.EML.MC100.fits'
        self.gasprop_suffix = tmp_suffix + '.EML.MC100.GasProp.fits'

    def set_config(self, config):
        if config is not None:
            if config == 'last':
                self.config = -1
            else:
                self.config = config
        else:
            self.config = -1
        self._config_run()

    def get_masterlist_file(self):
        return self.califa_work_dir + self._masterlist_file

    def get_config(self):
        config = self._config[self.config]
        return dict(superfits_dir=self._superfits_dir[config[0]],
                    spacialSampling=self._spacialSampling[config[1]],
                    qVersion=self._qVersion[config[2]],
                    dVersion=self._dVersion[config[3]],
                    othSuffix=self._othSuffix[config[4]],
                    base=self._bases[config[5]])

    def get_image_file(self, gal):
        return self.califa_work_dir + '/images/' + gal + '.jpg'

    def get_emlines_file(self, gal):
        return '%s%s%s' % (self.emlines_cube_dir, gal, self.emlines_suffix)

    def get_gasprop_file(self, gal):
        return '%s%s%s' % (self.gasprop_cube_dir, gal, self.gasprop_suffix)

    def get_pycasso_file(self, gal):
        return '%s%s%s' % (self.pycasso_cube_dir, gal, self.pycasso_suffix)


class tupperware_none(object):
    def __init__(self):
        pass

    def __getattr__(self, attr):
        r = self.__dict__.get(attr, None)
        return r


class tupperware(object):
    pass


class stack_gals(object):
    def __init__(self):
        self.keys1d = []
        self.keys1d_masked = []
        self.keys2d = []
        self.keys2d_masked = []

    def new1d(self, k):
        self.keys1d.append(k)
        setattr(self, '_%s' % k, [])

    def new1d_masked(self, k):
        self.keys1d_masked.append(k)
        setattr(self, '_%s' % k, [])
        setattr(self, '_mask_%s' % k, [])

    def new2d(self, k, N):
        self.keys2d.append(k)
        setattr(self, '_N_%s' % k, N)
        setattr(self, '_%s' % k, [[] for _ in xrange(N)])

    def new2d_masked(self, k, N):
        self.keys2d_masked.append(k)
        setattr(self, '_N_%s' % k, N)
        setattr(self, '_%s' % k, [[] for _ in xrange(N)])
        setattr(self, '_mask_%s' % k, [[] for _ in xrange(N)])

    def append1d(self, k, val):
        attr = getattr(self, '_%s' % k)
        attr.append(val)

    def append1d_masked(self, k, val, mask_val=None):
        attr = getattr(self, '_%s' % k)
        attr.append(val)
        m = getattr(self, '_mask_%s' % k)
        if mask_val is None:
            mask_val = np.zeros_like(val, dtype=np.bool_)
        m.append(mask_val)

    def append2d(self, k, i, val):
        key = '_N_%s' % k
        if key in self.__dict__:
            attr = getattr(self, '_%s' % k)
            attr[i].append(val)

    def append2d_masked(self, k, i, val, mask_val=None):
        key = '_N_%s' % k
        if key in self.__dict__:
            attr = getattr(self, '_%s' % k)
            attr[i].append(val)
            m = getattr(self, '_mask_%s' % k)
            if mask_val is None:
                mask_val = np.zeros_like(val, dtype=np.bool_)
            m[i].append(mask_val)

    def stack(self):
        if len(self.keys1d) > 0:
            self._stack1d()
        if len(self.keys1d_masked) > 0:
            self._stack1d_masked()
        if len(self.keys2d) > 0:
            self._stack2d()
        if len(self.keys2d_masked) > 0:
            self._stack2d_masked()

    def _stack1d(self):
        for k in self.keys1d:
            print k
            attr = np.hstack(getattr(self, '_%s' % k))
            setattr(self, k, np.array(attr, dtype=np.float_))

    def _stack1d_masked(self):
        for k in self.keys1d_masked:
            print k
            attr = np.hstack(getattr(self, '_%s' % k))
            mask = np.hstack(getattr(self, '_mask_%s' % k))
            setattr(self, k, np.ma.masked_array(attr, mask=mask, dtype=np.float_))

    def _stack2d(self):
        for k in self.keys2d:
            print k
            N = getattr(self, '_N_%s' % k)
            attr = getattr(self, '_%s' % k)
            setattr(self, k, np.asarray([np.array(np.hstack(attr[i]), dtype=np.float_) for i in xrange(N)]))

    def _stack2d_masked(self):
        for k in self.keys2d_masked:
            print k
            N = getattr(self, '_N_%s' % k)
            attr = getattr(self, '_%s' % k)
            mask = getattr(self, '_mask_%s' % k)
            setattr(self, k, np.ma.asarray([np.ma.masked_array(np.hstack(attr[i]), mask=np.hstack(mask[i]), dtype=np.float_) for i in xrange(N)]))


class ALLGals(object):
    def __init__(self, N_gals, NRbins, N_T, N_U):
        self.N_gals = N_gals
        self.NRbins = NRbins
        self.N_T = N_T
        self.N_U = N_U
        self.range_T = xrange(self.N_T)
        self.range_U = xrange(self.N_U)
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
        self.califaIDs = np.ma.masked_all((N_gals), dtype = '|S5')

        self.N_zones__g = np.ma.masked_all((N_gals), dtype = int)
        self.morfType_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.at_flux_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.Mcor_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.McorSD_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.McorSD_oneHLR__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.McorSD__rg = np.ma.masked_all((NRbins, N_gals), dtype = np.float_)
        self.ba_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.ba_PyCASSO_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.Mr_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.ur_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.HLR_pix_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.HMR_pix_GAL__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.parsecPerPixel__g = np.ma.masked_all((N_gals), dtype = np.float_)

        self.McorSD__rg = np.ma.masked_all((NRbins, N_gals), dtype = np.float_)
        self.at_flux__rg = np.ma.masked_all((NRbins, N_gals), dtype = np.float_)
        self.at_mass__rg = np.ma.masked_all((NRbins, N_gals), dtype = np.float_)
        self.alogZ_mass_GAL__Ug = np.ma.masked_all((N_U, N_gals), dtype = np.float_)
        self.alogZ_flux_GAL__Ug = np.ma.masked_all((N_U, N_gals), dtype = np.float_)
        self.at_flux_oneHLR__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.at_mass_oneHLR__g = np.ma.masked_all((N_gals), dtype = np.float_)

        self.tau_V_neb__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.aSFRSD_Ha__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.EW_Ha__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.EW_Hb__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.EW_Ha_wei__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.EW_Hb_wei__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_obs_Ha__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_obs_Hb__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_obs_O3__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_obs_N2__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_int_Ha__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_int_Hb__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_int_O3__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.F_int_N2__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.logO3N2_M13__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.x_Y__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.aSFRSD__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.tau_V__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.McorSD__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.at_flux__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.at_mass__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.at_flux_dezon__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.at_mass_dezon__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.at_flux_wei__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.at_mass_wei__Trg = np.ma.masked_all((N_T, NRbins, N_gals), dtype = np.float_)
        self.alogZ_mass__Urg = np.ma.masked_all((N_U, NRbins, N_gals), dtype = np.float_)
        self.alogZ_flux__Urg = np.ma.masked_all((N_U, NRbins, N_gals), dtype = np.float_)
        self.alogZ_mass_wei__Urg = np.ma.masked_all((N_U, NRbins, N_gals), dtype = np.float_)
        self.alogZ_flux_wei__Urg = np.ma.masked_all((N_U, NRbins, N_gals), dtype = np.float_)

        self.x_Y_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.aSFRSD_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.tau_V_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.McorSD_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.at_flux_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.at_mass_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.at_flux_dezon_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.at_mass_dezon_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.tau_V_neb_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.aSFRSD_Ha_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.aSFRSD_Ha_masked_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.EW_Ha_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.EW_Hb_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_obs_Ha_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_obs_Hb_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_obs_O3_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_obs_N2_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_int_Ha_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_int_Hb_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_int_O3_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.F_int_N2_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.logO3N2_M13_oneHLR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.alogZ_mass_oneHLR__Ug = np.ma.masked_all((N_U, N_gals), dtype = np.float_)
        self.alogZ_flux_oneHLR__Ug = np.ma.masked_all((N_U, N_gals), dtype = np.float_)

        self.integrated_x_Y__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.integrated_SFR__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.integrated_SFRSD__Tg = np.ma.masked_all((N_T, N_gals), dtype = np.float_)
        self.integrated_EW_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_EW_Hb__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_tau_V__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_tau_V_neb__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_tau_V_neb_err__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_obs_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_obs_Hb__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_obs_O3__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_obs_N2__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_eF_obs_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_eF_obs_Hb__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_eF_obs_O3__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_eF_obs_N2__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_int_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_int_Hb__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_int_O3__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_F_int_N2__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_baseline_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_baseline_Hb__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_baseline_O3__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_baseline_N2__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_L_int_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_L_obs_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_SFR_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_SFRSD_Ha__g = np.ma.masked_all((N_gals), dtype = np.float_)
        self.integrated_logO3N2_M13__g = np.ma.masked_all((N_gals), dtype = np.float_)

    def _init_zones_temporary_lists(self):
        self._Mcor__g = []
        self._McorSD__g = []
        self._tau_V_neb__g = []
        self._tau_V_neb_err__g = []
        self._tau_V_neb_mask__g = []
        self._SFR_Ha__g = []
        self._SFR_Ha_mask__g = []
        self._SFRSD_Ha__g = []
        self._SFRSD_Ha_mask__g = []

        self._F_obs_Hb__g = []
        self._F_obs_O3__g = []
        self._F_obs_Ha__g = []
        self._F_obs_N2__g = []
        self._eF_obs_Hb__g = []
        self._eF_obs_O3__g = []
        self._eF_obs_Ha__g = []
        self._eF_obs_N2__g = []
        self._F_obs_Hb_mask__g = []
        self._F_obs_O3_mask__g = []
        self._F_obs_Ha_mask__g = []
        self._F_obs_N2_mask__g = []
        self._F_int_Ha__g = []
        self._F_int_Hb__g = []
        self._F_int_N2__g = []
        self._F_int_O3__g = []

        self._baseline_Hb__g = []
        self._baseline_O3__g = []
        self._baseline_Ha__g = []
        self._baseline_N2__g = []

        self._L_int_Ha__g = []
        self._L_int_Ha_err__g = []
        self._L_int_Ha_mask__g = []
        self._L_obs_Ha__g = []
        self._L_obs_Ha_err__g = []
        self._L_obs_Ha_mask__g = []
        self._zone_area_pc2__g = []
        self._zone_dist_HLR__g = []
        self._EW_Ha__g = []
        self._EW_Hb__g = []
        self._EW_Ha_mask__g = []
        self._EW_Hb_mask__g = []
        self._at_flux__g = []
        self._at_mass__g = []
        self._maskOkRadius__g = []
        self._tau_V__Tg = [[] for _ in self.range_T]
        self._tau_V_mask__Tg = [[] for _ in self.range_T]
        self._SFR__Tg = [[] for _ in self.range_T]
        self._SFR_mask__Tg = [[] for _ in self.range_T]
        self._SFRSD__Tg = [[] for _ in self.range_T]
        self._SFRSD_mask__Tg = [[] for _ in self.range_T]
        self._x_Y__Tg = [[] for _ in self.range_T]
        self._Mcor__Tg = [[] for _ in self.range_T]
        self._McorSD__Tg = [[] for _ in self.range_T]
        self._at_flux__Tg = [[] for _ in self.range_T]
        self._at_mass__Tg = [[] for _ in self.range_T]
        self._alogZ_mass__Ug = [[] for _ in self.range_U]
        self._alogZ_mass_mask__Ug = [[] for _ in self.range_U]
        self._alogZ_flux__Ug = [[] for _ in self.range_U]
        self._alogZ_flux_mask__Ug = [[] for _ in self.range_U]
        #final Tg and Ug zone-by-zone lists
        self.tau_V__Tg = []
        self.SFR__Tg = []
        self.SFRSD__Tg = []
        self.x_Y__Tg = []
        self.alogZ_mass__Ug = []
        self.alogZ_flux__Ug = []
        self.Mcor__Tg = []
        self.McorSD__Tg = []
        self.at_flux__Tg = []
        self.at_mass__Tg = []
        self._logO3N2_M13__g = []
        self._logO3N2_M13_mask__g = []

    def stack_zones_data(self):
        self.zone_dist_HLR__g = np.ma.masked_array(np.hstack(np.asarray(self._zone_dist_HLR__g)), dtype = np.float_)
        self.zone_area_pc2__g = np.ma.masked_array(np.hstack(np.asarray(self._zone_area_pc2__g)), dtype = np.float_)

        aux = np.hstack(self._tau_V_neb__g)
        auxMask = np.hstack(self._tau_V_neb_mask__g)
        self.tau_V_neb__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        self.tau_V_neb_err__g = np.ma.masked_array(np.hstack(self._tau_V_neb_err__g), mask = auxMask, dtype = np.float_)

        aux = np.hstack(self._L_int_Ha__g)
        auxMask = np.hstack(self._L_int_Ha_mask__g)
        self.L_int_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        self.L_int_Ha_err__g = np.ma.masked_array(np.hstack(self._L_int_Ha_err__g), mask = auxMask, dtype = np.float_)

        aux = np.hstack(self._L_obs_Ha__g)
        auxMask = np.hstack(self._L_obs_Ha_mask__g)
        self.L_obs_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        self.L_obs_Ha_err__g = np.ma.masked_array(np.hstack(self._L_obs_Ha_err__g), mask = auxMask, dtype = np.float_)

        aux = np.hstack(self._F_obs_Ha__g)
        auxMask = np.hstack(self._F_obs_Ha_mask__g)
        self.F_obs_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._baseline_Ha__g)
        self.baseline_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._eF_obs_Ha__g)
        self.eF_obs_Ha__g = np.ma.masked_array(aux, mask = np.zeros_like(aux, dtype = np.bool), dtype = np.float_)

        aux = np.hstack(self._F_obs_Hb__g)
        auxMask = np.hstack(self._F_obs_Hb_mask__g)
        self.F_obs_Hb__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._baseline_Hb__g)
        self.baseline_Hb__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._eF_obs_Hb__g)
        self.eF_obs_Hb__g = np.ma.masked_array(aux, mask = np.zeros_like(aux, dtype = np.bool), dtype = np.float_)

        aux = np.hstack(self._F_obs_O3__g)
        auxMask = np.hstack(self._F_obs_O3_mask__g)
        self.F_obs_O3__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._baseline_O3__g)
        self.baseline_O3__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._eF_obs_O3__g)
        self.eF_obs_O3__g = np.ma.masked_array(aux, mask = np.zeros_like(aux, dtype = np.bool), dtype = np.float_)

        aux = np.hstack(self._F_obs_N2__g)
        auxMask = np.hstack(self._F_obs_N2_mask__g)
        self.F_obs_N2__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._baseline_N2__g)
        self.baseline_N2__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._eF_obs_N2__g)
        self.eF_obs_N2__g = np.ma.masked_array(aux, mask = np.zeros_like(aux, dtype = np.bool), dtype = np.float_)

        aux = np.hstack(self._F_int_Ha__g)
        self.F_int_Ha__g = np.ma.masked_array(aux, dtype = np.float_)
        aux = np.hstack(self._F_int_Hb__g)
        self.F_int_Hb__g = np.ma.masked_array(aux, dtype = np.float_)
        aux = np.hstack(self._F_int_O3__g)
        self.F_int_O3__g = np.ma.masked_array(aux, dtype = np.float_)
        aux = np.hstack(self._F_int_N2__g)
        self.F_int_N2__g = np.ma.masked_array(aux, dtype = np.float_)

        aux = np.hstack(self._SFR_Ha__g)
        auxMask = np.hstack(self._SFR_Ha_mask__g)
        self.SFR_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)

        aux = np.hstack(self._SFRSD_Ha__g)
        auxMask = np.hstack(self._SFRSD_Ha_mask__g)
        self.SFRSD_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)

        self.Mcor__g = np.ma.masked_array(np.hstack(self._Mcor__g), dtype = np.float_)
        self.McorSD__g = np.ma.masked_array(np.hstack(self._McorSD__g), dtype = np.float_)

        aux = np.hstack(self._EW_Ha__g)
        auxMask = np.hstack(self._EW_Ha_mask__g)
        self.EW_Ha__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)

        aux = np.hstack(self._EW_Hb__g)
        auxMask = np.hstack(self._EW_Hb_mask__g)
        self.EW_Hb__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)

        aux = np.hstack(self._logO3N2_M13__g)
        auxmask = np.hstack(self._logO3N2_M13_mask__g)
        self.logO3N2_M13__g = np.ma.masked_array(aux, mask = auxmask, dtype = np.float_)

        aux = np.hstack(self._at_flux__g)
        auxMask = np.zeros_like(aux, dtype = np.bool_)
        self.at_flux__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        aux = np.hstack(self._at_mass__g)
        self.at_mass__g = np.ma.masked_array(aux, mask = auxMask, dtype = np.float_)
        for iT in self.range_T:
            aux = np.hstack(self._SFR__Tg[iT])
            auxMask = np.hstack(self._SFR_mask__Tg[iT])
            self.SFR__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
            aux = np.hstack(self._SFRSD__Tg[iT])
            auxMask = np.hstack(self._SFRSD_mask__Tg[iT])
            self.SFRSD__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
            aux = np.hstack(self._x_Y__Tg[iT])
            self.x_Y__Tg.append(np.ma.masked_array(aux, dtype = np.float_))
            # all arrays below are using the same tau_V_mask
            aux = np.hstack(self._tau_V__Tg[iT])
            auxMask = np.hstack(self._tau_V_mask__Tg[iT])
            self.tau_V__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
            aux = np.hstack(self._Mcor__Tg[iT])
            self.Mcor__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
            aux = np.hstack(self._McorSD__Tg[iT])
            self.McorSD__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
            aux = np.hstack(self._at_flux__Tg[iT])
            self.at_flux__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
            aux = np.hstack(self._at_mass__Tg[iT])
            self.at_mass__Tg.append(np.ma.masked_array(aux, mask = auxMask, dtype = np.float_))
        for iU in self.range_U:
            aux = np.hstack(self._alogZ_mass__Ug[iU])
            self.alogZ_mass__Ug.append(np.ma.masked_array(aux, mask = np.isnan(aux), dtype = np.float_))
            aux = np.hstack(self._alogZ_flux__Ug[iU])
            self.alogZ_flux__Ug.append(np.ma.masked_array(aux, mask = np.isnan(aux), dtype = np.float_))

    def integrated_mask(self):
        aux = np.less(self.integrated_tau_V_neb__g, 0)
        self.integrated_tau_V_neb__g[aux] = np.ma.masked

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
                        tmp_data = {'masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in self.range_T}
                        tmp_mask = {'masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in self.range_T}
                    elif suffix == 'Ug':
                        tmp_data = {'masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in self.range_U}
                        tmp_mask = {'masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in self.range_U}
                    else:
                        tmp_data = {}
                        tmp_mask = {}
                D.update(tmp_data)
                D.update(tmp_mask)
        return D


class H5SFRData(object):
    def __init__(self, h5file, create_attrs = False):
        self.h5file = h5file

        try:
            self.h5 = h5py.File(self.h5file, 'r')
        except IOError:
            print "%s: file does not exists" % h5file
            return None

        self.RbinIni = self.get_data_h5('RbinIni')
        self.RbinFin = self.get_data_h5('RbinFin')
        self.RbinStep = self.get_data_h5('RbinStep')
        self.Rbin__r = self.get_data_h5('Rbin__r')
        self.RbinCenter__r = self.get_data_h5('RbinCenter__r')
        self.NRbins = self.get_data_h5('NRbins', dtype = np.int)
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

        self.N_zones_all__g = self.get_data_h5('N_zones__g', dtype = np.int)
        self.N_zones__g = self.N_zones_all__g.compressed()
        self.califaIDs_all = self.get_data_h5('califaIDs', dtype = '|S5')
        #self.califaIDs = np.ma.masked_array(self.califaIDs_all, mask = self.N_zones_all__g.mask, dtype = self.califaIDs_all.dtype).compressed()
        self.califaIDs = self.califaIDs_all.compress(~(self.N_zones_all__g.mask))
        self.N_gals_all = len(self.califaIDs_all)
        self.N_gals = len(self.califaIDs)
        self.zones_map = np.asarray([ i for j in xrange(self.N_gals) for i in xrange(self.N_zones__g[j]) ])

        if create_attrs is not False:
            self._create_attrs()

    def Rtoplot(self, shape = None):
        if shape == None:
            shape = (self.NRbins, self.N_gals_all)
        return self.RbinCenter__r[..., np.newaxis] * np.ones(shape)

    def _create_attrs(self):
        ds = self.h5['masked/data']
        for k in ds.iterkeys():
            if k not in self.__dict__.keys():
                v = self.get_data_h5(k)
                setattr(self, k, v)

    def reply_arr_by_zones(self, p, add_mask = None):
        if isinstance(p, str):
            p = self.get_data_h5(p, add_mask = add_mask)
        if isinstance(p, np.ma.core.MaskedArray):
            mask = p.mask
            #if add_mask is not None: mask = np.bitwise_or(mask, add_mask)
            p = np.ma.masked_array(p, mask = mask).compressed()
        if isinstance(p, np.ndarray):
            p = p.tolist()
        laux1 = [ itertools.repeat(a[0], times = a[1]) for a in zip(p, self.N_zones__g) ]
        return np.asarray(list(itertools.chain.from_iterable(laux1)))

    def reply_arr_by_radius(self, p, N_dim = None, add_mask = None):
        if isinstance(p, np.ma.core.MaskedArray):
            if N_dim:
                Nloop = N_dim * self.NRbins
                output_shape = (N_dim, self.NRbins, self.N_gals_all)
            else:
                Nloop = self.NRbins
                output_shape = (self.NRbins, self.N_gals_all)
            ld = [ list(v) for v in [ itertools.repeat(prop, Nloop) for prop in p.data ]]
            if not isinstance(p.mask, np.ndarray):
                mask = np.zeros_like(p.data, dtype = np.bool_)
            else:
                mask = p.mask
            #if add_mask is not None: mask = np.bitwise_or(mask, add_mask)
            lm = [ list(v) for v in [ itertools.repeat(prop, Nloop) for prop in mask ]]
            od = np.asarray([list(i) for i in zip(*ld)]).reshape(output_shape)
            om = np.asarray([list(i) for i in zip(*lm)]).reshape(output_shape)
            o = np.ma.masked_array(od, mask = om)
        else:
            if isinstance(p, str):
                p = self.get_data_h5(p, add_mask = add_mask)
            elif isinstance(p, np.ndarray):
                p = p.tolist()
            if N_dim:
                Nloop = N_dim * self.NRbins
                output_shape = (N_dim, self.NRbins, self.N_gals_all)
            else:
                Nloop = self.NRbins
                output_shape = (self.NRbins, self.N_gals_all)
            l = [ list(v) for v in [ itertools.repeat(prop, Nloop) for prop in p ]]
            o = np.asarray([list(i) for i in zip(*l)]).reshape(output_shape)
        return o

    def __getattr__(self, attr):
        a = attr.split('_')
        x = None
        if a[0]:
            # somestr.find(str) returns 0 if str is found in somestr.
            if a[0].find('K0') == 0:
                gal = a[0]
                prop = '_'.join(a[1:])
                x = self.get_prop_gal(prop, gal)
            else:
                x = self.get_data_h5(attr)
                if x is not None: setattr(self, attr, x)
            return x

    def get_data_h5(self, prop, dtype = np.float_, add_mask = None):
        h5 = self.h5
        h5_root = 'masked/'
        folder_data = '%sdata' % h5_root
        folder_mask = '%smask' % h5_root
        folder_nomask = 'data'
        if prop in h5[folder_mask].keys():
            node = '%s/%s' % (folder_data, prop)
            ds = h5[node]
            if isinstance(h5[node], h5py.Dataset):
                arr = get_h5_data_masked(h5, prop, h5_root, add_mask, **dict(dtype = dtype))
            else:
                suffix = prop[-2:]
                if suffix[0] == 'U':
                    arr = [
                        get_h5_data_masked(h5, '%s/%d' % (prop, iU), h5_root, add_mask, **dict(dtype = dtype))
                        for iU in xrange(self.N_U)
                    ]
                elif suffix[0] == 'T':
                    arr = [
                        get_h5_data_masked(h5, '%s/%d' % (prop, iT), h5_root, add_mask, **dict(dtype = dtype))
                        #np.ma.masked_array(h5['%s/%d' % (node, iT)].value, mask = h5['%s/%d' % (node_m, iT)].value, dtype = dtype)
                        for iT in xrange(self.N_T)
                    ]
            return arr
        elif prop in h5[folder_nomask].keys():
            node = '%s/%s' % (folder_nomask, prop)
            ds = h5[node]
            return ds.value
        else:
            return None

    def _get_valid_gals(self, l_gals):
        if isinstance(l_gals, str):
            fname = l_gals
            f = open(fname, 'r')
            g = []
            for line in f.xreadlines():
                lin = line.strip()
                if lin[0] == '#':
                    continue
                g.append(lin)
            f.close()
            l_gals = np.unique(np.asarray(g))
        return [g for g in l_gals if g in self.califaIDs]

    # this method only works if self.califaIDs is sorted also
    def get_mask_zones_list(self, l_gals, return_ngals = False):
        l_valid_gals = self._get_valid_gals(l_gals)
        maskOKGals = np.zeros(self.N_zones__g.sum(), dtype = np.bool_)
        for g in l_valid_gals:
            i = self.califaIDs.tolist().index(g)
            N_zones = self.N_zones__g[i]
            N_zones_before = self.N_zones__g[:i].sum()
            N_zones_step = N_zones_before + N_zones
            maskOKGals[N_zones_before:N_zones_step] = True
        if return_ngals is False:
            return maskOKGals
        else:
            return maskOKGals, len(l_valid_gals)

    # this method only works if self.califaIDs is sorted also
    def get_mask_radius_list(self, l_gals, return_ngals = False):
        l_valid_gals = self._get_valid_gals(l_gals)
        shape = self.NRbins, self.N_gals_all
        maskOKGals = np.zeros((shape), dtype = np.bool_)
        for g in l_valid_gals:
            i = self.califaIDs_all.data.tolist().index(g)
            maskOKGals[:, i] = True
        if return_ngals is False:
            return maskOKGals
        else:
            return maskOKGals, len(l_valid_gals)

    def get_mask_integrated_list(self, l_gals, return_ngals = False):
        l_valid_gals = self._get_valid_gals(l_gals)
        aux = [ True if g in l_valid_gals else False for g in self.califaIDs_all ]
        maskOKGals = np.asarray(aux, dtype = np.bool_)
        if return_ngals is False:
            return maskOKGals
        else:
            return maskOKGals, len(l_gals)

    def sum_prop_gal(self, data, mask_zones = None):
        if isinstance(data, str):
            data = self.get_data_h5(data)
        if mask_zones is None:
            mask_zones = np.zeros(data.shape, dtype = np.bool_)
        dm = np.ma.masked_array(data, mask = (data.mask | mask_zones))
        prop_sum__g = np.ma.masked_all(self.califaIDs_all.shape)
        for iGal, gal in enumerate(self.califaIDs_all):
            if gal is not np.ma.masked:
                prop_sum__g[iGal] = self.get_prop_gal(dm, gal).sum()
        return prop_sum__g

    # XXX: TODO: ADD_MASK
    def get_prop_gal(self, data, gal = None, return_slice = False):
        if isinstance(data, str):
            data = self.get_data_h5(data)
        arr = None
        if isinstance(data, list):
            califaIDs = self.reply_arr_by_zones(self.califaIDs)
            where_slice = np.where(califaIDs == gal)
            range_data = xrange(len(data))
            if isinstance(data[0], np.ma.core.MaskedArray):
                arr_data = [ data[i][where_slice].data for i in range_data ]
                arr_mask = [
                    data[i][where_slice].mask if isinstance(data[i][where_slice].mask, np.ndarray)
                    else np.zeros(data[i][where_slice].shape, dtype = np.bool_)
                    for i in range_data
                ]
                arr = np.ma.masked_array(arr_data, mask = arr_mask)
            else:
                arr_data = [ data[i][where_slice] for i in range_data ]
                arr = np.asarray(arr_data)
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
        if return_slice:
            return arr, where_slice
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

    def get_plot_dict(self, iT = 11, iU = -1, key = None, mask = None):
        pd = {
            #########################
            ### GAL or Integrated ###
            #########################
            'logintMcor' : dict(v = np.ma.log10(self.Mcor_GAL__g), legendname = r'$\log\ M_\star$', label = r'$\log\ M_\star$ [$M_\odot$]', lim = [8.5, 12.], majloc = 1., minloc = 0.2),
            'logintSFRSD' : dict(v = np.ma.log10(self.integrated_SFRSD__Tg[iT]), legendname = r'$\log\ \Sigma_{SFR}^\star$', label = r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} pc^{-2}]$', lim = [-12, -7.5], majloc = 0.5, minloc = 0.1),
            'logintSFRSDHa' : dict(v = np.ma.log10(self.integrated_SFRSD_Ha__g), legendname = r'$\log\ \Sigma_{SFR}^{neb}', label = r'$\log\ \Sigma_{SFR}^{neb}\ [M_\odot yr^{-1} pc^{-2}]$', lim = [-12, -7.5], majloc = 0.5, minloc = 0.1),
            #########################
            ### zones ###############
            #########################
            'atflux' : dict(v = self.at_flux__g, legendname = r'$\langle \log\ t_\star \rangle_L$', label = r'$\langle \log\ t \rangle_L$ [yr]', lim = [7, 10], majloc = 0.6, minloc = 0.12,),
            'alogZmass' : dict(v = self.alogZ_mass__Ug[iU], legendname = r'$\langle \log\ Z_\star \rangle_M$', label = r'$\langle \log\ Z_\star \rangle_M$ [$Z_\odot$]', lim = [ -0.75, 0.25], majloc = 0.25, minloc = 0.05),
            'OHIICHIM' : dict(v = self.O_HIICHIM__g, legendname = r'12 + $\log\ O/H$', label = r'12 + $\log\ O/H$ (HII-CHI-mistry, EPM, 2014)', lim = [7., 9.5], majloc = 0.5, minloc = 0.1),
            'logO3N2S06' : dict(v = self.logZ_neb_S06__g, legendname = r'$\log\ Z_{neb}$', label = r'$\log\ Z_{neb}$ [$Z_\odot$] (Stasinska, 2006)', lim = [-0.5, 0.1], majloc = 0.5, minloc = 0.1),
            'logO3N2M13' : dict(v = self.O_O3N2_M13__g, legendname = r'12 + $\log\ O/H$', label = r'12 + $\log\ O/H$ (logO3N2, Marino, 2013)', lim = [8.2, 8.7], majloc = 0.25, minloc = 0.05),
            'logMcorSD' : dict(v = np.ma.log10(self.McorSD__g), legendname = r'$\log\ \mu_\star$', label = r'$\log\ \mu_\star$ [$M_\odot \ pc^{-2}$]', lim = [1, 4.6], majloc = 1., minloc = 0.2),
            'logMcor' : dict(v = np.ma.log10(self.Mcor__g), legendname = r'$\log\ M_\star$', label = r'$\log\ M_\star$ [$M_\odot$]', lim = None, majloc = 1., minloc = 0.2),
            'xY' : dict(v = self.x_Y__Tg[iT], legendname = r'$x_Y$', label = r'$x_Y$', lim = [0, .50], majloc = .10, minloc = .02),
            'tauVdiff' : dict(v = self.tau_V_neb__g - self.tau_V__Tg[iT], legendname = '$\tau_V^{neb}\ -\ \tau_V^\star$', label = r'$\tau_V^{neb}\ -\ \tau_V^\star$', lim = [-1.2, 2.6], majloc = 0.75, minloc = 0.15),
            'tauVRatio' : dict(v = self.tau_V_neb__g / self.tau_V__Tg[iT], legendname = r'$\frac{\tau_V^{neb}}{\tau_V^\star}$', label = r'$\frac{\tau_V^{neb}}{\tau_V^\star}$', lim = [0, 6], majloc = 1., minloc = 0.2),
            'logWHaWHb' : dict(v = np.ma.log10(self.EW_Ha__g / self.EW_Hb__g), legendname = r'$\log\ \frac{W_{H\alpha}}{W_{H\beta}}$', label = r'$\log\ \frac{W_{H\alpha}}{W_{H\beta}}$', lim = [0.2, 0.8], majloc = 0.12, minloc = 0.024,),
            'logO3N2PP04' : dict(v = self.O_O3N2_PP04__g, legendname = r'12 + $\log\ O/H$', label = r'12 + $\log\ O/H$ (PP, 2004)', lim = [8, 8.6]),
            'logtauV' : dict(v = np.ma.log10(self.tau_V__Tg[iT]), legendname = r'$\log\ \tau_V^\star$', label = r'$\log\ \tau_V^\star$', lim = [ -1.5, 0.5 ], majloc = 0.5, minloc = 0.1),
            'logtauVNeb' : dict(v = np.ma.log10(self.tau_V_neb__g), legendname = r'$\log\ \tau_V^{neb}$', label = r'$\log\ \tau_V^{neb}$', lim = [ -1.5, 0.5 ], majloc = 0.5, minloc = 0.1),
            'tauV' : dict(v = self.tau_V__Tg[iT], legendname = r'$\tau_V^\star$', label = r'$\tau_V^\star$', lim = [ 0., 1.5 ], majloc = 0.5, minloc = 0.1),
            'tauVNeb' : dict(v = self.tau_V_neb__g, legendname = r'$\tau_V^{neb}$', label = r'$\tau_V^{neb}$', lim = [ 0., 3. ], majloc = 1., minloc = 0.2),
            'logZoneArea' : dict(v = np.ma.log10(self.zone_area_pc2__g), legendname = r'$\log\ A_{zone}$', label = r'$\log\ A_{zone}$ [$pc^2$]', lim = [ 3.5, 8.5 ], majloc = 1.0, minloc = 0.2),
            'zoneDistHLR' : dict(v = self.zone_dist_HLR__g, legendname = r'$R_{z}$', label = r'$R_{z}$ [HLR]', lim = [ 0, 2 ], majloc = 0.5, minloc = 0.1),
            'logSFRSD' : dict(v = np.ma.log10(self.SFRSD__Tg[iT]), legendname = r'$\log\ \Sigma_{SFR}^\star$', label = r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} pc^{-2}]$', lim = [-8.5, -6.0], majloc = 0.5, minloc = 0.1),
            'logSFRSDHa' : dict(v = np.ma.log10(self.SFRSD_Ha__g), legendname = r'$\log\ \Sigma_{SFR}^{neb}$', label = r'$\log\ \Sigma_{SFR}^{neb}\ [M_\odot yr^{-1} pc^{-2}]$', lim = [-8.5, -6.0], majloc = 0.5, minloc = 0.1),
            'morfType' : dict(v = self.reply_arr_by_zones(self.morfType_GAL__g), legendname = 'mt', label = 'morph. type', mask = False, lim = [9, 11.5]),
            'ba' : dict(v = self.reply_arr_by_zones(self.ba_GAL__g), legendname = r'$b/a$', label = r'$\frac{b}{a}$', mask = False, lim = [0, 1.]),
            #########################
            ### Radius ##############
            #########################
            'atfluxR' : dict(v = self.at_flux__rg, label = r'$\langle \log\ t_\star \rangle_L (R)$ [yr]', lim = [7, 10], majloc = 0.6, minloc = 0.12,),
            'alogZmassR' : dict(v = self.alogZ_mass__Urg[-1], label = r'$\langle \log\ Z_\star \rangle_M (R)$ [$Z_\odot$]', lim = [-1.5, 0.5], majloc = 0.5, minloc = 0.1),
            'OHIICHIMR' : dict(v = self.O_HIICHIM__rg, label = r'12 + $\log\ O/H(R)$ (HII-CHI-mistry, EPM, 2014)', lim = [7., 9.5], majloc = 0.5, minloc = 0.1),
            'logO3N2S06R' : dict(v = self.logZ_neb_S06__rg, label = r'$\log\ Z_{neb} (R)$ [$Z_\odot$] (Stasinska, 2006)', lim = [-2., 0.5], majloc = 0.5, minloc = 0.1),
            'logO3N2M13R' : dict(v = np.ma.masked_array(self.logO3N2_M13__Trg[iT] - 8.69, mask = np.isnan(self.logO3N2_M13__Trg[iT])), label = r'$\log\ \left(\frac{(O/H)}{(O/H)_\odot}\right)(R)$', lim = [-0.6, 0.], majloc = 0.1, minloc = 0.025),
            'logMcorSDR' : dict(v = np.ma.log10(self.McorSD__rg), label = r'$\log\ \mu_\star (R)$ [$M_\odot \ pc^{-2}$]', lim = [0, 4], majloc = 1., minloc = 0.2),
            'xYR' : dict(v = self.x_Y__Trg[iT], label = '$x_Y (R)$', lim = [0, 1], majloc = 0.1, minloc = 0.02),
            'tauVdiffR' : dict(v = self.tau_V_neb__Trg[iT] - self.tau_V__Trg[iT], label = r'$\tau_V^{neb} (R)\ -\ \tau_V^\star (R)$', lim = [-1.2, 2.6], majloc = 0.75, minloc = 0.15),
            'tauVRatioR' : dict(v = self.tau_V_neb__Trg[iT] / self.tau_V__Trg[iT], label = r'$\frac{\tau_V^{neb}(R)}{\tau_V^\star(R)}$', lim = [0, 6], majloc = 1., minloc = 0.2),
            'logWHaWHbR' : dict(v = np.ma.log10(self.EW_Ha__Trg[iT] / self.EW_Hb__Trg[iT]), label = r'$\log\ \frac{W_{H\alpha} (R)}{W_{H\beta} (R)}$', lim = [0.2, 0.8], majloc = 0.12, minloc = 0.024,),
            'logO3N2PP04R' : dict(v = self.O_O3N2_PP04__rg, label = r'12 + $\log\ O/H (R)$ (PP, 2004)', lim = [8, 8.6]),
            'logtauVR' : dict(v = np.ma.log10(self.tau_V__Trg[iT]), label = r'$\log\ \tau_V^\star (R)$', lim = [ -1.5, 0.5 ], majloc = 0.5, minloc = 0.1),
            'logtauVNebR' : dict(v = np.ma.log10(self.tau_V_neb__Trg[iT]), label = r'$\log\ \tau_V^{neb} (R)$', lim = [ -1.5, 0.5 ], majloc = 0.5, minloc = 0.1),
            'tauVR' : dict(v = self.tau_V__Trg[iT], label = r'$\tau_V^\star (R)$', lim = [ 0., 1.5 ], majloc = 0.5, minloc = 0.1),
            'tauVNebR' : dict(v = self.tau_V_neb__Trg[iT], label = r'$\tau_V^{neb} (R)$', lim = [ 0., 3. ], majloc = 1., minloc = 0.2),
            'alogSFRSDR' : dict(v = np.ma.log10(self.aSFRSD__Trg[iT]), label = r'$\log\ \Sigma_{SFR}^\star (t_\star, R)\ [M_\odot yr^{-1} pc^{-2}]$', lim = [-9.5, -5], majloc = 0.5, minloc = 0.1),
            'alogSFRSDHaR' : dict(v = np.ma.log10(self.aSFRSD_Ha__Trg[iT]), label = r'$\log\ \Sigma_{SFR}^{neb} (R)\ [M_\odot yr^{-1} pc^{-2}]$', lim = [-9.5, -5], majloc = 0.5, minloc = 0.1),
            'alogSFRSDkpcR' : dict(v = np.ma.log10(self.aSFRSD__Trg[iT] * 1e6), label = r'$\log\ \Sigma_{SFR}^\star (t_\star, R)\ [M_\odot yr^{-1} kpc^{-2}]$', lim = [-3.5, 0], majloc = 0.5, minloc = 0.1),
            'alogSFRSDHakpcR' : dict(v = np.ma.log10(self.aSFRSD_Ha__Trg[iT] * 1e6), label = r'$\log\ \Sigma_{SFR}^{neb} (R)\ [M_\odot yr^{-1} kpc^{-2}]$', lim = [-3.5, 0], majloc = 0.5, minloc = 0.1),
            'morfTypeR' : dict(v = self.reply_arr_by_radius(self.morfType_GAL__g), label = 'morph. type', mask = False, lim = [9, 12]),
            'baR' : dict(v = self.reply_arr_by_radius(self.ba_GAL__g), label = r'$b/a$', mask = False, lim = [0, 1.]),
        }
        if key is not None:
            try:
                return key, pd.get(key)
            except:
                print '%s: key not found' % key
                return None
        return pd

    def plot_xyz_iter(self, xd, yd = None, zd = None):
        for xk, xv in xd.iteritems():
            for yk, yv in yd.iteritems():
                if zd is not None:
                    for zk, zv in zd.iteritems():
                        yield '%s_%s_%s' % (xk, yk, zk), xv, yv, zv
                else:
                    yield '%s_%s' % (xk, yk), xv, yv

    def plot_xyz_keys_iter(self, xkeys, ykeys = None, zkeys = None, iT = 11, iU = -1):
        xd = { '%s' % k : self.get_plot_dict(iT = iT, iU = iU, key = k)[1] for k in xkeys }
        yd = { '%s' % k : self.get_plot_dict(iT = iT, iU = iU, key = k)[1] for k in ykeys }
        if zkeys is not None:
            zd = { '%s' % k : self.get_plot_dict(iT = iT, iU = iU, key = k)[1] for k in zkeys }
        for xk, xv in xd.iteritems():
            for yk, yv in yd.iteritems():
                if zkeys is not None:
                    for zk, zv in zd.iteritems():
                        yield '%s_%s_%s' % (xk, yk, zk), xv, yv, zv
                else:
                    yield '%s_%s' % (xk, yk), xv, yv


class GasProp(object):
    def __init__(self, filename=None):
        try:
            self._hdulist = pyfits.open(filename)
        except:
            print 'pyfits: %s: file error' % filename
            self._hdulist = None

        if self._hdulist is not None:
            self.header = self._hdulist[0].header
            self._excluded_hdus = ['FLINES', 'NAMEFILES', 'ICF']
            self._nobs = self.header['NOBS']
            self._create_attrs()
            self._dlcons = eval(self._hdulist[-1].header['DLCONS'])

        self.cte_av_tau = 1. / (2.5 * np.log10(np.exp(1.)))

    def close(self):
        self._hdulist.close()
        self._hdulist = None

    def _iter_hdus(self):
        for i in xrange(1, len(self._hdulist)):
            n = self._hdulist[i].name
            if n in self._excluded_hdus:
                continue
            h = self._hdulist[i].data
            yield n, h

    def _create_attrs(self):
        for hname, h in self._iter_hdus():
            setattr(self, hname, tupperware())
            tmp = getattr(self, hname)
            names = h.names
            attrs = [name.replace('[', '_').replace(']', '').replace('.', '_') for name in names]
            for attr, k in zip(attrs, names):
                if len(h[k]) == self._nobs:
                    data = np.copy(h[k][1:])
                    setattr(tmp, attr, data)
                    int_attr = 'integrated_%s' % attr
                    int_data = np.copy(h[k][0])
                    setattr(tmp, int_attr, int_data)

    def AVtoTau(self, AV):
        return AV * self.cte_av_tau

    def TautoAV(self, tau):
        return tau * 1. / self.cte_av_tau

    def CtoAV(self, c, Rv=3.1, extlaw=1.443):
        return c * (Rv / extlaw)

    def CtoTau(self, c, Rv=3.1, extlaw=1.443):
        return self.AVtoTau(self.CtoAV(c, Rv, extlaw))
