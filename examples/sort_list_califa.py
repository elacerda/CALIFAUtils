#!/usr/bin/python
import sys
import numpy as np
from califa_scripts import sort_gal, sort_gal_func


def sort_by_Mcor(K, **kwargs):
    return K.Mcor_tot.sum()


def sort_by_McorSD(K, **kwargs):
    return K.Mcor_tot.sum() / K.zoneArea_pc2.sum()


def sort_by_Mr(K, **kwargs):
    return np.float(K.masterListData['Mr'])


def sort_by_morph(K, **kwargs):
    from get_morfologia import get_morfologia
    return get_morfologia(K.califaID)[1]


def sort_by_ellipse_ba(K, **kwargs):
    return K.masterListData['ba']
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # pa, ba = K.getEllipseParams() 
    # return ba
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE


def sort_by_dhubax(K, **kwargs):
    ##some_var = kwargs.pop('some_var')
    pa, ba = K.getEllipseParams() 
    return ba

def ret_u_r(K, **kwargs):
    return np.float(K.masterListData['u-r'])

def ret_integrated_at_flux(K, **kwargs):
    return K.integrated_at_flux

def sort_by_integrated_Fobs_Ha(K, **kwargs):
    return K.EL.integrated_fluxmedian[K.EL.lines.index('6563')]


def example(listfile, sortby, order, **kwargs):
    #gals = sort_gal(listfile, mode = sort_funcs[sortby], order = 1)
    gals = sort_gal_func(listfile, sort_funcs[sortby], order = order) #, **kwargs)

    for i, g in enumerate(gals):
        print g
        
def sort_by_WHa(K, **kwargs):
    try:
        nuc_R = kwargs.pop('nuc_R')
    except:
        nuc_R = 0.2
    try:
        minSNR = kwargs.pop('minSNR')
    except:
        minSNR = 3
        
    i_Hb = K.EL.lines.index('4861')
    i_O3 = K.EL.lines.index('5007')
    i_Ha = K.EL.lines.index('6563')
    i_N2 = K.EL.lines.index('6583')
    
    Ha = K.EL.flux[i_Ha, :]
    eHa = K.EL.eflux[i_Ha, :]
    Hb = K.EL.flux[i_Hb, :]
    eHb = K.EL.eflux[i_Hb, :]
    O3 = K.EL.flux[i_O3, :]
    eO3 = K.EL.eflux[i_O3, :]
    N2 = K.EL.flux[i_N2, :]
    eN2 = K.EL.eflux[i_N2, :]
    
    HbOk = np.array((Hb / eHb) >= minSNR, dtype = np.bool)
    O3Ok = np.array((O3 / eO3) >= minSNR, dtype = np.bool)
    HaOk = np.array((Ha / eHa) >= minSNR, dtype = np.bool)
    N2Ok = np.array((N2 / eN2) >= minSNR, dtype = np.bool)
    
    maskLinesSNR__z = HbOk & O3Ok & HaOk & N2Ok
    maskFluxOk__z = (Ha >= 0) & (Hb >= 0) & (O3 >= 0) & (N2 >= 0)

    dist_HLR__z = np.sqrt((K.zonePos['x'] - K.x0) ** 2. + (K.zonePos['y'] - K.y0) ** 2.) / K.HLR_pix
    WHa_masked = np.ma.masked_array(K.EL.EW[i_Ha], mask = ~(maskLinesSNR__z & maskFluxOk__z))
    
    return WHa_masked[dist_HLR__z <= nuc_R].sum() / WHa_masked.sum()    


def sort_by_fluxHaRatioNuc(K, **kwargs):
    try:
        nuc_R = kwargs.pop('nuc_R')
    except:
        nuc_R = 0.2
    try:
        minSNR = kwargs.pop('minSNR')
    except:
        minSNR = 3
        
    i_Hb = K.EL.lines.index('4861')
    i_O3 = K.EL.lines.index('5007')
    i_Ha = K.EL.lines.index('6563')
    i_N2 = K.EL.lines.index('6583')
    
    Ha = K.EL.flux[i_Ha, :]
    eHa = K.EL.eflux[i_Ha, :]
    Hb = K.EL.flux[i_Hb, :]
    eHb = K.EL.eflux[i_Hb, :]
    O3 = K.EL.flux[i_O3, :]
    eO3 = K.EL.eflux[i_O3, :]
    N2 = K.EL.flux[i_N2, :]
    eN2 = K.EL.eflux[i_N2, :]
    
    HbOk = np.array((Hb / eHb) >= minSNR, dtype = np.bool)
    O3Ok = np.array((O3 / eO3) >= minSNR, dtype = np.bool)
    HaOk = np.array((Ha / eHa) >= minSNR, dtype = np.bool)
    N2Ok = np.array((N2 / eN2) >= minSNR, dtype = np.bool)
    
    maskLinesSNR__z = HbOk & O3Ok & HaOk & N2Ok
    maskFluxOk__z = (Ha >= 0) & (Hb >= 0) & (O3 >= 0) & (N2 >= 0)

    dist_HLR__z = np.sqrt((K.zonePos['x'] - K.x0) ** 2. + (K.zonePos['y'] - K.y0) ** 2.) / K.HLR_pix
    Ha_masked = np.ma.masked_array(Ha, mask = ~(maskLinesSNR__z & maskFluxOk__z))
    
    return Ha_masked[dist_HLR__z <= nuc_R].sum() / Ha_masked.sum()


def sort_by_integrated_tau_V_neb(K, **kwargs):
    i_Ha = K.EL.lines.index('6563')
    i_Hb = K.EL.lines.index('4861')
    HaHb = K.EL.integrated_fluxmedian[i_Ha]/K.EL.integrated_fluxmedian[i_Hb]
    q = K.EL._qCCM['4861'] - K.EL._qCCM['6563'] 
    return np.ma.log(HaHb / 2.86) / q


def sort_by_mask_tau_V_neb(K, **kwargs):
    try:
        minSNR = kwargs.pop('minSNR')
    except:
        minSNR = 3

    tauVNebOkMin = 0.05
        
    i_Hb = K.EL.lines.index('4861')
    i_O3 = K.EL.lines.index('5007')
    i_Ha = K.EL.lines.index('6563')
    i_N2 = K.EL.lines.index('6583')
    
    Ha = K.EL.flux[i_Ha, :]
    eHa = K.EL.eflux[i_Ha, :]
    Hb = K.EL.flux[i_Hb, :]
    eHb = K.EL.eflux[i_Hb, :]
    O3 = K.EL.flux[i_O3, :]
    eO3 = K.EL.eflux[i_O3, :]
    N2 = K.EL.flux[i_N2, :]
    eN2 = K.EL.eflux[i_N2, :]
    
    HbOk = np.array((Hb / eHb) >= minSNR, dtype = np.bool)
    O3Ok = np.array((O3 / eO3) >= minSNR, dtype = np.bool)
    HaOk = np.array((Ha / eHa) >= minSNR, dtype = np.bool)
    N2Ok = np.array((N2 / eN2) >= minSNR, dtype = np.bool)
    
    maskLinesSNR__z = HbOk & O3Ok & HaOk & N2Ok
    maskFluxOk__z = (Ha >= 0) & (Hb >= 0) & (O3 >= 0) & (N2 >= 0)

    maskOkTauVNeb__z = np.ones((K.N_zone), dtype = np.bool)
    
    if tauVNebOkMin >= 0:
        maskOkTauVNeb__z = (K.EL.tau_V_neb__z >= tauVNebOkMin)
    
    maskOkNeb__z = (maskOkTauVNeb__z & maskLinesSNR__z & maskFluxOk__z)
    N_zones_tau_V = len(K.EL.tau_V_neb__z[maskOkNeb__z])

    return N_zones_tau_V / (1. * K.N_zone) 


sort_funcs = {
    'Mcor' : sort_by_Mcor,
    'McorSD' : sort_by_McorSD,
    'morph' : sort_by_morph,
    'Mr' : sort_by_Mr,
    'ba' : sort_by_ellipse_ba,
    'dhubax' : sort_by_dhubax,
    'FobsHa' : sort_by_integrated_Fobs_Ha,
    'WHa_ratio': sort_by_WHa,
    'FHa_ratio' : sort_by_fluxHaRatioNuc,
    'tauVneb_zoneratio' : sort_by_mask_tau_V_neb, 
    'integrated_tauVneb' : sort_by_integrated_tau_V_neb, 
}


def histogram_califa_remove(var__G, var__g, var_gal__r, xlabel, bins, fname):
    H = plt.hist(var__G, bins = bins, color = 'b')
    plt.hist(var__g, bins = H[1], color = 'g')
    plt.hist(var_gal__r, bins = H[1], color = 'r')
    plt.xlabel(xlabel)
    plt.ylabel('number of galaxies')
    plt.savefig(fname)
    plt.clf()
    
def histogram_califa(var__G, var__g, xlabel, bins, fname):
    H = plt.hist(var__G, bins = bins, color = 'b')
    plt.hist(var__g, bins = H[1], color = 'g')
    plt.xlabel(xlabel)
    plt.ylabel('number of galaxies')
    plt.savefig(fname)
    plt.clf()



        
if __name__ == '__main__':
    try:
        listfile = sys.argv[1]
        sortby = sys.argv[2]
    except IndexError:
        print 'usage: %s FILE %s' % (sys.argv[0], sort_funcs.keys())
        exit(1)
        
    #example(listfile, sortby, order = 1)
    f = open('/Users/lacerda/CALIFA/list_SFR/rem_ba_morph/list_1234.txt', 'r')
    l = f.readlines()
    f.close()
    listgals_toberemoved = [ l[i].strip() for i in np.arange(len(l)) ]

    f = open('/Users/lacerda/CALIFA/list_SFR/rem_ba_morph/list_0_si_si.txt', 'r')
    l = f.readlines()
    f.close()
    listgals_zero = [ l[i].strip() for i in np.arange(len(l)) ]

    gals, data__g = sort_gal_func(listfile, sort_funcs[sortby], order = 1, EL = True, nuc_R = 0.15)
    
    for i, g in enumerate(gals):
        j = -1

        try:
            j = listgals_zero.index(g)
            mark = '0'
        except:
            mark = None
                
        #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # if mark == None:
        #     try:
        #         j = listgals_toberemoved.index(g)
        #         mark = '*'
        #     except:
        #         mark = None
        #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        
        if mark == None:
            mark = ''
                
        print g, data__g[i], mark
