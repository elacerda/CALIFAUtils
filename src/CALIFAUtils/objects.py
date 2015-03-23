import numpy as np
import itertools
import pyfits
import h5py

class empty: pass

class GasProp(object):
    def __init__(self, filename = None):
        try:
            self._hdulist = pyfits.open(filename)
        except:
            print 'pyfits: %s: file error' % filename
            self._hdulist = None
            
        if self._hdulist is not None:
            self.header = self._hdulist[0].header
            self._excluded_hdus = [ 'FLINES', 'NAMEFILES', 'ICF' ]
            self._nobs = self.header['NOBS']
            self._create_attrs()
            self._dlcons = eval(self._hdulist[-1].header['DLCONS'])
            
        self.cte_av_tau = 1. / (2.5 * np.log10(np.exp(1.)))
        
    def _iter_hdus(self):
        for i in xrange(1, len(self._hdulist)):
            n = self._hdulist[i].name
            if n in self._excluded_hdus:
                continue
            h = self._hdulist[i].data
            yield n, h 
            
    def _create_attrs(self):
        for hname, h in self._iter_hdus():
            setattr(self, hname, empty())
            tmp = getattr(self, hname)
            names = h.names
            attrs = [ name.replace('[', '_').replace(']', '').replace('.', '_') for name in names ]
            for attr, k in zip(attrs, names):
                if len(h[k]) == self._nobs:
                    data = h[k][1:]
                    setattr(tmp, attr, data)
                    int_attr = 'integrated_%s' % attr
                    int_data = h[k][0]
                    setattr(tmp, int_attr, int_data)
                    
    def AVtoTau(self, AV):
        return AV * self.cte_av_tau 
    
    def CtoAV(self, c, Rv = 3.1, extlaw = 1.443):
        return c * (Rv / extlaw)
                        
    def CtoTau(self, c, Rv = 3.1, extlaw = 1.443):
        return self.AVtoTau(self.CtoAV(c, Rv, extlaw))
    
class read_kwargs(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs  
        
    def __getattr__(self, attr):
        return self.kwargs.get(attr)
            
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
        self.aSFRSD_Ha_masked__rg = np.ma.empty((NRbins, N_gals))
        self.McorSD__rg = np.ma.empty((NRbins, N_gals))
        self.logZ_neb_S06__rg = np.ma.empty((NRbins, N_gals))
        self.at_flux__rg = np.ma.empty((NRbins, N_gals))
        self.at_mass__rg = np.ma.empty((NRbins, N_gals))
        self.aSFRSD__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.tau_V__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.McorSD__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.at_flux__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.at_mass__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.at_flux_dezon__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.at_mass_dezon__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.at_flux_wei__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.at_mass_wei__Trg = np.ma.empty((N_T, NRbins, N_gals))
        self.alogZ_mass__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_flux__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_mass_wei__Urg = np.ma.empty((N_U, NRbins, N_gals))
        self.alogZ_flux_wei__Urg = np.ma.empty((N_U, NRbins, N_gals))
        #GasProp
        self.integrated_chb_in__g = np.ma.empty((N_gals))
        self.integrated_c_Ha_Hb__g = np.ma.empty((N_gals))
        self.integrated_O_HIICHIM__g = np.ma.empty((N_gals))
        self.integrated_O_O3N2_M13__g = np.ma.empty((N_gals))
        self.integrated_O_O3N2_PP04__g = np.ma.empty((N_gals))
        self.integrated_O_direct_O_23__g = np.ma.empty((N_gals))
        self.O_HIICHIM__rg = np.ma.empty((NRbins, N_gals))
        self.O_O3N2_M13__rg = np.ma.empty((NRbins, N_gals))
        self.O_O3N2_PP04__rg = np.ma.empty((NRbins, N_gals))
        self.O_direct_O_23__rg = np.ma.empty((NRbins, N_gals))
        
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
        self._SFR_Ha_mask__g = []
        self._SFRSD_Ha__g = []
        self._SFRSD_Ha_mask__g = []
        self._F_obs_Ha__g = []
        self._F_obs_Ha_mask__g = []
        self._L_int_Ha__g = []
        self._L_int_Ha_err__g = []
        self._L_int_Ha_mask__g = []
        self._dist_zone__g = []
        self._EW_Ha__g = []
        self._EW_Hb__g = []
        self._EW_Ha_mask__g = []
        self._EW_Hb_mask__g = []
        self._at_flux__g = []
        self._at_mass__g = []
        self._tau_V__Tg = [[] for _ in xrange(self.N_T)]
        self._tau_V_mask__Tg = [[] for _ in xrange(self.N_T)]
        self._SFR__Tg = [[] for _ in xrange(self.N_T)]
        self._SFR_mask__Tg = [[] for _ in xrange(self.N_T)]
        self._SFRSD__Tg = [[] for _ in xrange(self.N_T)]
        self._SFRSD_mask__Tg = [[] for _ in xrange(self.N_T)]
        self._x_Y__Tg = [[] for _ in xrange(self.N_T)]
        self._Mcor__Tg = [[] for _ in xrange(self.N_T)]
        self._McorSD__Tg = [[] for _ in xrange(self.N_T)]
        self._at_flux__Tg = [[] for _ in xrange(self.N_T)]
        self._at_mass__Tg = [[] for _ in xrange(self.N_T)]
        self._alogZ_mass__Ug = [[] for _ in xrange(self.N_U)]
        self._alogZ_mass_mask__Ug = [[] for _ in xrange(self.N_U)]
        self._alogZ_flux__Ug = [[] for _ in xrange(self.N_U)]
        self._alogZ_flux_mask__Ug = [[] for _ in xrange(self.N_U)]
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
        #GasProp
        self._chb_in__g = []
        self._c_Ha_Hb__g = []
        self._O_HIICHIM__g = []
        self._O_O3N2_M13__g = []
        self._O_O3N2_PP04__g = []
        self._O_direct_O_23__g = []
            
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

        aux = np.hstack(self._F_obs_Ha__g)
        auxMask = np.hstack(self._F_obs_Ha_mask__g)
        self.F_obs_Ha__g = np.ma.masked_array(aux, mask = auxMask)

        aux = np.hstack(self._SFR_Ha__g)
        auxMask = np.hstack(self._SFR_Ha_mask__g)
        self.SFR_Ha__g = np.ma.masked_array(aux, mask = auxMask)

        aux = np.hstack(self._SFRSD_Ha__g)
        auxMask = np.hstack(self._SFRSD_Ha_mask__g)
        self.SFRSD_Ha__g = np.ma.masked_array(aux, mask = auxMask)

        self.Mcor__g = np.ma.masked_array(np.hstack(self._Mcor__g))
        self.McorSD__g = np.ma.masked_array(np.hstack(self._McorSD__g))

        aux = np.hstack(self._EW_Ha__g)
        auxMask = np.hstack(self._EW_Ha_mask__g)
        self.EW_Ha__g = np.ma.masked_array(aux, mask = auxMask)

        aux = np.hstack(self._EW_Hb__g)
        auxMask = np.hstack(self._EW_Hb_mask__g)
        self.EW_Hb__g = np.ma.masked_array(aux, mask = auxMask)
        
        aux = np.hstack(self._at_flux__g)
        auxMask = np.zeros_like(aux, dtype = np.bool)
        self.at_flux__g = np.ma.masked_array(aux, mask = auxMask)
        aux = np.hstack(self._at_mass__g)
        self.at_mass__g = np.ma.masked_array(aux, mask = auxMask)
        for iT in xrange(self.N_T):
            aux = np.hstack(self._SFR__Tg[iT])
            auxMask = np.hstack(self._SFR_mask__Tg[iT])        
            self.SFR__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._SFRSD__Tg[iT])
            auxMask = np.hstack(self._SFRSD_mask__Tg[iT])
            self.SFRSD__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._x_Y__Tg[iT])
            self.x_Y__Tg.append(np.ma.masked_array(aux))
            # all arrays below are using the same tau_V_mask
            aux = np.hstack(self._tau_V__Tg[iT])
            auxMask = np.hstack(self._tau_V_mask__Tg[iT])
            self.tau_V__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._Mcor__Tg[iT])
            self.Mcor__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._McorSD__Tg[iT])
            self.McorSD__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._at_flux__Tg[iT])
            self.at_flux__Tg.append(np.ma.masked_array(aux, mask = auxMask))
            aux = np.hstack(self._at_mass__Tg[iT])
            self.at_mass__Tg.append(np.ma.masked_array(aux, mask = auxMask))
        for iU in np.arange(self.N_U):
            aux = np.hstack(self._alogZ_mass__Ug[iU])
            self.alogZ_mass__Ug.append(np.ma.masked_array(aux))
            aux = np.hstack(self._alogZ_flux__Ug[iU])
            self.alogZ_flux__Ug.append(np.ma.masked_array(aux))
        #GasProp
        aux = np.hstack(self._chb_in__g)
        self.chb_in__g = np.ma.masked_array(aux, mask = np.isnan(aux))
        aux = np.hstack(self._c_Ha_Hb__g)
        self.c_Ha_Hb__g = np.ma.masked_array(aux, mask = np.isnan(aux))
        aux = np.hstack(self._O_HIICHIM__g)
        self.O_HIICHIM__g = np.ma.masked_array(aux, mask = np.isnan(aux))
        aux = np.hstack(self._O_O3N2_M13__g)
        self.O_O3N2_M13__g = np.ma.masked_array(aux, mask = np.isnan(aux))
        aux = np.hstack(self._O_O3N2_PP04__g)
        self.O_O3N2_PP04__g = np.ma.masked_array(aux, mask = np.isnan(aux))
        aux = np.hstack(self._O_direct_O_23__g)
        self.O_direct_O_23__g = np.ma.masked_array(aux, mask = np.isnan(aux))
            
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
                        tmp_data = {'masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in xrange(self.N_T)}
                        tmp_mask = {'masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in xrange(self.N_T)}
                    elif suffix == 'Ug':
                        tmp_data = {'masked/data/%s/%d' % (v, i) : self.__dict__[v][i].data for i in xrange(self.N_U)}
                        tmp_mask = {'masked/mask/%s/%d' % (v, i) : self.__dict__[v][i].mask for i in xrange(self.N_U)}
                    else:
                        tmp_data = {}
                        tmp_mask = {}
                D.update(tmp_data)
                D.update(tmp_mask)
        return D      
                  
class H5SFRData(object):
    def __init__(self, h5file):
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
        self.zones_map = np.asarray([ i for j in xrange(self.N_gals) for i in xrange(self.N_zones__g[j]) ])

        self._create_attrs()
        
    def _create_attrs(self):
        ds = self.h5['masked/data']
        for k in ds.iterkeys():
            if k not in self.__dict__.keys():
                v = self.get_data_h5(k)
                setattr(self, k, v)
        
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
        
    def get_prop_gal(self, data, gal = None, return_slice = False):
        if isinstance(data, str):
            data = self.get_data_h5(data)
        arr = None
        if isinstance(data, list):
            califaIDs = self.reply_arr_by_zones(self.califaIDs)
            where_slice = np.where(califaIDs == gal)
            arr = [ data[iU][where_slice] for iU in xrange(len(data)) ]
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
            return where_slice, arr
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
    