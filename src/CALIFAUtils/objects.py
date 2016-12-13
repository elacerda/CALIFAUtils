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
        'q054',
        'q055'
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
        self.pycasso_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + '/' + config['superfits_dir'] + '/' + config['base'] + '/'
        self.emlines_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + '/EML/' + config['base'] + '/'
        self.gasprop_cube_dir = self.califa_work_dir + '/legacy/' + config['qVersion'] + '/EML/' + config['base'] + '/prop/'
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
            attr = np.hstack(getattr(self, '_%s' % k))
            setattr(self, k, np.array(attr, dtype=attr.dtype))

    def _stack1d_masked(self):
        for k in self.keys1d_masked:
            attr = np.hstack(getattr(self, '_%s' % k))
            mask = np.hstack(getattr(self, '_mask_%s' % k))
            setattr(self, k, np.ma.masked_array(attr, mask=mask, dtype=attr.dtype))

    def _stack2d(self):
        for k in self.keys2d:
            N = getattr(self, '_N_%s' % k)
            attr = getattr(self, '_%s' % k)
            setattr(self, k, np.asarray([np.array(np.hstack(attr[i]), dtype=attr.dtype) for i in xrange(N)]))

    def _stack2d_masked(self):
        for k in self.keys2d_masked:
            N = getattr(self, '_N_%s' % k)
            attr = getattr(self, '_%s' % k)
            mask = getattr(self, '_mask_%s' % k)
            setattr(self, k, np.ma.asarray([np.ma.masked_array(np.hstack(attr[i]), mask=np.hstack(mask[i]), dtype=attr.dtype) for i in xrange(N)]))


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
