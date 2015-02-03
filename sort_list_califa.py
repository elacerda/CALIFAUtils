#!/usr/bin/python
import sys
import numpy as np
from califa_scripts import sort_gal, sort_gal_func

def sort_by_Mcor(K):
    return K.Mcor_tot.sum()


def sort_by_McorSD(K):
    return K.Mcor_tot.sum() / K.zoneArea_pc2.sum()


def sort_by_Mr(K):
    return np.float(K.masterListData['Mr'])


def sort_by_morph(K):
    from get_morfologia import get_morfologia
    return get_morfologia(K.califaID)[1]


def sort_by_ellipse_ba(K):
    pa, ba = K.getEllipseParams() 
    return ba

def sort_by_dhubax(K, **kwargs):
    ##some_var = kwargs.pop('some_var')
    pa, ba = K.getEllipseParams() 
    return ba

sort_funcs = {
    'Mcor' : sort_by_Mcor,
    'McorSD' : sort_by_McorSD,
    'morph' : sort_by_morph,
    'Mr' : sort_by_Mr,
    'ba' : sort_by_ellipse_ba,
    'dhubax' : sort_by_dhubax,
}

if __name__ == '__main__':
    try:
        listfile = sys.argv[1]
        sortby = sys.argv[2]
    except IndexError:
        print 'usage: %s FILE %s' % (sys.argv[0], sort_funcs.keys())
        exit(1)

    #gals = sort_gal(listfile, mode = sort_funcs[sortby], order = 1)
    gals = sort_gal_func(listfile, sort_funcs[sortby], order = 1) #, some_var = some_val) 

    for i, g in enumerate(gals):
        print g
