import numpy as np
import itertools
import cPickle
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
        'pix',
        'px1'
    ]
    _qVersion = [
        'q036',
        'q043',
        'q045',
        'q046',
        'q050',
        'q051',
        'q053',
        'q054',
        'q055',
        'q057'
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
        [
            _superfits_dir.index('superfits'),
            _spacialSampling.index('px1'),
            _qVersion.index('q057'),
            _dVersion.index('d22a'),
            0,
            _bases.index('Bgstf6e')
        ],
    ]
    _masterlist_file = 'califa_masterlist.txt'

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

        superfits_dir = '/' + config['superfits_dir'] + '/'
        eml_dir = '/EML/'
        if config['spacialSampling'] == 'px1':
            superfits_dir += config['spacialSampling']
            eml_dir += config['spacialSampling']
        superfits_dir += config['base']
        eml_dir += config['base']

        self.pycasso_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + superfits_dir + '/'
        self.emlines_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + eml_dir + '/'
        self.gasprop_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + eml_dir + '/prop/'
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
        return '%s/%s' % (self.califa_work_dir, self._masterlist_file)

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
    def __init__(self, keys1d=None, keys1d_masked=None, keys2d=None, keys2d_masked=None):
        self.keys1d = []
        self.keys1d_masked = []
        self.keys2d = []
        self.keys2d_masked = []
        if keys1d is not None:
            self.addkeys1d(keys1d)
        if keys1d_masked is not None:
            self.addkeys1d_masked(keys1d_masked)
        if keys2d is not None:
            self.addkeys2d(keys2d)
        if keys2d_masked is not None:
            self.addkeys2d_masked(keys2d_masked)
        pass

    def load(self, filename=None):
        with open(filename, 'r') as f:
            return cPickle.load(f)

    def addkeys1d(self, keys):
        for k in keys:
            self.new1d(k)
            # self.keys1d.append(k)

    def addkeys1d_masked(self, keys):
        for k in keys:
            self.new1d_masked(k)
            # self.keys1d_masked.append(k)

    def addkeys2d(self, keys):
        print keys
        for kN in keys:
            k, N = kN
            self.new2d(k, N)
            # self.keys2d.append(k)

    def addkeys2d_masked(self, keys):
        for kN in keys:
            k, N = kN
            self.new2d_masked(k, N)
            # self.keys2d_masked.append(k)

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
            print 'keys1d'
            self._stack1d()
        if len(self.keys1d_masked) > 0:
            print 'keys1d_masked'
            self._stack1d_masked()
        if len(self.keys2d) > 0:
            print 'keys2d'
            self._stack2d()
        if len(self.keys2d_masked) > 0:
            print 'keys2d_masked'
            self._stack2d_masked()

    def _stack1d(self):
        for k in self.keys1d:
            attr = np.hstack(getattr(self, '_%s' % k))
            print k, attr, attr.dtype
            setattr(self, k, np.array(attr, dtype=attr.dtype))

    def _stack1d_masked(self):
        for k in self.keys1d_masked:
            attr = np.hstack(getattr(self, '_%s' % k))
            mask = np.hstack(getattr(self, '_mask_%s' % k))
            new_attr = np.ma.masked_array(attr, mask=mask, dtype=attr.dtype, copy=True)
            print k, len(attr), mask.sum(), new_attr.count(), new_attr
            setattr(self, k, new_attr)

    def _stack2d(self):
        for k in self.keys2d:
            N = getattr(self, '_N_%s' % k)
            attr = getattr(self, '_%s' % k)
            setattr(self, k, np.asarray([np.array(np.hstack(attr[i]), dtype=np.hstack(attr[i]).dtype) for i in xrange(N)]))

    def _stack2d_masked(self):
        for k in self.keys2d_masked:
            N = getattr(self, '_N_%s' % k)
            attr = getattr(self, '_%s' % k)
            mask = getattr(self, '_mask_%s' % k)
            list__N = []
            for i in xrange(N):
                attr_stacked = np.hstack(attr[i])
                mask_stacked = np.hstack(mask[i])
                list__N.append(np.ma.masked_array(attr_stacked, mask=mask_stacked, dtype=np.hstack(attr[i]).dtype, copy=True))
            setattr(self, k, np.ma.asarray(list__N))

    def dump(self, filename):
        with open(filename, 'w') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)

    def get_gal_prop(self, gal='K0001', prop_in=None):
        if isinstance(prop_in, np.ndarray):
            prop = prop_in
            prop_N = prop.shape[-1]
            N_pixel = self.califaID__yx.shape[0]
            if prop_N == N_pixel:
                gals = self.califaID__yx
            else:
                gals = self.califaID__z
        else:
            prop = getattr(self, prop_in)
            if prop_in.endswith('yx'):
                gals = self.califaID__yx
            else:
                gals = self.califaID__z
        return prop[np.where(gals == gal)]

    def get_gal_prop_unique(self, gal='K0001', prop_in=None):
        if isinstance(prop_in, np.ndarray):
            prop = prop_in
        else:
            prop = getattr(self, prop_in)
        gals = getattr(self, 'califaID__z')
        _, ind = np.unique(gals, return_index=True)
        gals_inorder = gals[sorted(ind)]
        return prop[np.where(gals_inorder == gal)][0]


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
