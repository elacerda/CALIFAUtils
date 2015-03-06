#!/usr/bin/python
#
# Lacerda@Granada - 13/Oct/2014
#
import h5py
import numpy as np
import matplotlib as mpl
import scipy.optimize as so
from scipy import stats as st
from matplotlib.pyplot import cm
#from sklearn import linear_model
from matplotlib import pyplot as plt
from CALIFAUtils.scripts import debug_var
from CALIFAUtils.scripts import OLS_bisector
from CALIFAUtils.scripts import read_one_cube
from matplotlib.pyplot import MultipleLocator
from CALIFAUtils.scripts import gaussSmooth_YofX
from CALIFAUtils.scripts import calc_running_stats
from CALIFAUtils.scripts import find_confidence_interval
from CALIFAUtils.scripts import DrawHLRCircleInSDSSImage

def plot_linreg_params(param, x, xlabel, ylabel, fname, best_param = None, fontsize = 12):
    y = param
    xm = x
    ym = y
    f = plt.figure()
    ax = f.gca()
    ax.scatter(xm, ym, c = 'k', marker = 'o', s = 10., edgecolor = 'none', alpha = 0.6)
    if best_param != None:
        ax.axhline(y = best_param, c = 'k', ls = '--')
        delta = np.abs(ym - best_param)
        where = np.where(delta == delta.min())
        txt = r'$x_{best}:$ %.2f ($\Delta$:%.2f)' % (x[where], delta[where])
        textbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.)
        x_pos = xm[where]
        y_pos = ym[where]
        xlim_inf, xlim_sup = ax.get_xlim()
        ylim_inf, ylim_sup = ax.get_ylim() 
        arrow_size_x = (xlim_sup - x_pos) / 6
        arrow_size_y = (ylim_sup - y_pos) / 3 
        ax.annotate(txt,
            xy = (x_pos, y_pos), xycoords = 'data',
            xytext = (x_pos + arrow_size_x, y_pos + arrow_size_y),
            textcoords = 'data',
            verticalalignment = 'top', horizontalalignment = 'left',
            bbox = textbox,
            arrowprops = dict(arrowstyle = "->",
                            #linestyle="dashed",
                            color = "0.5",
                            connectionstyle = "angle3,angleA=90,angleB=0",
                            ),
            )
#        ax.text(xm[where], ym[where], txt, fontsize = fontsize,
#                verticalalignment = 'top', horizontalalignment = 'left',
#                bbox = textbox)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    f.savefig(fname)


def plot_text_ax(ax, txt, xpos, ypos, fontsize, va, ha, color = 'k'):
    textbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.)
    ax.text(xpos, ypos, txt, fontsize = fontsize, color = color,
            transform = ax.transAxes,
            verticalalignment = va, horizontalalignment = ha,
            bbox = textbox)


def plotRunningStatsAxis(ax, x, y, ylegend, plot_stats = 'mean', color = 'black', errorbar = True, nBox = 25):
    dxBox = (x.max() - x.min()) / (nBox - 1.)
    aux = calc_running_stats(x, y, dxBox = dxBox, xbinIni = x.min(), xbinFin = x.max(), xbinStep = dxBox)
    xbinCenter = aux[0]
    xMedian = aux[1]
    xMean = aux[2]
    xStd = aux[3]
    yMedian = aux[4]
    yMean = aux[5]
    yStd = aux[6]
    nInBin = aux[7]
    
    if plot_stats == 'median':
        xx = xMedian
        yy = yMedian
    else:
        xx = xMean
        yy = yMean

    ax.plot(xx, yy, color, lw = 2, label = ylegend)
    
    if errorbar:
        ax.errorbar(xx, yy, yerr = yStd, xerr = xStd, c = color)

 
def density_contour(xdata, ydata, binsx, binsy, ax = None, **contour_kwargs):
    """ Create a density contour plot.
 
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """    
    #nbins_x = len(binsx) - 1
    #nbins_y = len(binsy) - 1

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins = [binsx, binsy], normed = True)
    x_bin_sizes = (xedges[1:] - xedges[:-1])
    y_bin_sizes = (yedges[1:] - yedges[:-1])
 
    pdf = (H * (x_bin_sizes * y_bin_sizes))
 
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.99))
    levels = [one_sigma, two_sigma, three_sigma]
 
    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T
 
    if ax == None:
        contour = plt.contour(X, Y, Z, levels = levels, origin = "lower", **contour_kwargs)
    else:
        contour = ax.contour(X, Y, Z, levels = levels, origin = "lower", **contour_kwargs)
 
    return contour


def plotStatCorreAxis(ax, x, y, pos_x, pos_y, fontsize):
    rhoSpearman, pvalSpearman = st.spearmanr(x, y)
    txt = '<y/x>:%.3f - (y/x) median:%.3f - $\sigma(y/x)$:%.3f - Rs: %.2f' % (np.mean(y / x), np.ma.median((y / x)), np.ma.std(y / x), rhoSpearman)
    plot_text_ax(ax, txt, pos_x, pos_y, fontsize, 'top', 'left')


def plotOLSbisectorAxis(ax, x, y, **kwargs):
    pos_x = kwargs.get('pos_x', 0.99)
    pos_y = kwargs.get('pos_y', 0.00)
    fontsize = kwargs.get('fontsize', kwargs.get('fs', 10))
    color = kwargs.get('color', kwargs.get('c', 'r'))
    rms = kwargs.get('rms', True)
    label = kwargs.get('label', None)
    txt = kwargs.get('text', True)
    kwargs_plot = dict(c = color, ls = '-', lw = 1.5, label = label)
    kwargs_plot.update(kwargs.get('kwargs_plot', {}))
    a, b, sigma_a, sigma_b = OLS_bisector(x, y)
    Yrms_str = ''
    if rms == True:
        Yrms = (y - (a * x + b)).std()
        Yrms_str = r' : $y_{rms}$:%.2f' % Yrms
    ax.plot(ax.get_xlim(), a * np.asarray(ax.get_xlim()) + b, **kwargs_plot)
    if b > 0:
        txt_y = r'$y_{OLS}$ = %.2f$x$ + %.2f%s' % (a, b, Yrms_str)
    else:
        txt_y = r'$y_{OLS}$ = %.2f$x$ - %.2f%s' % (a, b * -1., Yrms_str)
    if txt == True:
        plot_text_ax(ax, txt_y, pos_x, pos_y, fontsize, 'bottom', 'right', color = color)
    else:
        print txt_y
    return a, b, sigma_a, sigma_b


def plot_gal_img_ax(ax, imgfile, gal, pos_x, pos_y, fontsize):
    galimg = plt.imread(imgfile)[::-1, :, :]
    plt.setp(ax.get_xticklabels(), visible = False)
    plt.setp(ax.get_yticklabels(), visible = False)
    ax.imshow(galimg, origin = 'lower')
    K = read_one_cube(gal)
    pa, ba = K.getEllipseParams()
    DrawHLRCircleInSDSSImage(ax, K.HLR_pix, pa, ba)
    K.close()
    txt = '%s' % gal
    plot_text_ax(ax, txt, pos_x, pos_y, fontsize, 'top', 'left', color = 'w')
    return ax


def plotSFR(x, y, xlabel, ylabel, xlim, ylim, age, fname):
    f = plt.figure()
    f.set_size_inches(10, 8)
    ax = f.gca()
    scat = ax.scatter(x, y, c = 'black', edgecolor = 'none', alpha = 0.5)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls = "--", c = ".3")
    rhoSpearman, pvalSpearman = st.spearmanr(x, y)
    yxlabel = r'$%s /\ %s $ ' % (ylabel.split('[')[0].strip('$ '), xlabel.split('[')[0].strip('$ '))
    txt = '%s mean:%.3f  median:%.3f  $\sigma(y/x)$:%.3f  Rs: %.2f' % (yxlabel, (y / x).mean(), np.ma.median((y / x)), np.ma.std(y / x), rhoSpearman)
    plot_text_ax(ax, txt, 0.03, 0.97, 16, 'top', 'left')
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    ax.set_title(r'$%s$ Myr' % str(age / 1.e6))
    if fname:
        f.savefig(fname)
    else:
        f.show()
    plt.close(f)


def plotTau(x, y, xlabel, ylabel, xlim, ylim, age, fname):
    f = plt.figure()
    f.set_size_inches(10, 8)
    ax = f.gca()
    scat = ax.scatter(x, y, c = 'black', edgecolor = 'none', alpha = 0.5)
    #ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
    rhoSpearman, pvalSpearman = st.spearmanr(x, y)
    yxlabel = r'$%s /\ %s $ ' % (ylabel.split('[')[0].strip('$ '), xlabel.split('[')[0].strip('$ '))
    txt = '%s mean:%.3f  median:%.3f  $\sigma(y/x)$:%.3f  Rs: %.2f' % (yxlabel, (y / x).mean(), np.ma.median((y / x)), np.ma.std(y / x), rhoSpearman)
    plot_text_ax(ax, txt, 0.03, 0.97, 15, 'top', 'left')
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    ax.set_title(r'$%s$ Myr' % str(age / 1.e6))
    if fname:
        f.savefig(fname)
    else:
        f.show()
    plt.close(f)


def plotScatterColorAxis(f, x, y, z, xlabel, ylabel, zlabel, xlim, ylim,
                         zlim = None, age = None,
                         contour = True, run_stats = True, OLS = False):
    
    ax = f.gca()
    
    if zlim != None:
        sc = ax.scatter(x, y, c = z, cmap = 'spectral_r', vmin = zlim[0], vmax = zlim[1], marker = 'o', s = 5., edgecolor = 'none')
    else:
        sc = ax.scatter(x, y, c = z, cmap = 'spectral_r', marker = 'o', s = 5., edgecolor = 'none')
        
    cb = f.colorbar(sc)
    cb.set_label(zlabel)
    
    if contour == True:
        binsx = np.linspace(min(x), max(x), 21)
        binsy = np.linspace(min(y), max(y), 21)
        density_contour(x, y, binsx, binsy, ax = ax, color = 'k')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    if OLS == True:
        A, B, sigma_A, sigma_B = plotOLSbisectorAxis(ax, x, y, 0.92, 0.05, 16)
        
    if run_stats == True:
        nBox = 20
        dxBox = (x.max() - x.min()) / (nBox - 1.)
        aux = calc_running_stats(x, y, dxBox = dxBox, xbinIni = x.min(), xbinFin = x.max(), xbinStep = dxBox)
        xbinCenter = aux[0]
        xMedian = aux[1]
        xMean = aux[2]
        xStd = aux[3]
        yMedian = aux[4]
        yMean = aux[5]
        yStd = aux[6]
        nInBin = aux[7]
        xPrc = aux[8]
        yPrc = aux[9]
        ax.plot(xMedian, yMedian, 'k', lw = 2)
        ax.plot(xPrc[0], yPrc[0], 'k--', lw = 2)
        ax.plot(xPrc[1], yPrc[1], 'k--', lw = 2)
        
    if age != None:
        txt = r'$%s$ Myr' % str(age / 1.e6)
        plot_text_ax(ax, txt, 0.02, 0.98, 14, 'top', 'left')
    
    
def plotScatterColor(x, y, z, xlabel, ylabel, zlabel, xlim, ylim,
                     fname = 'PlotScatter.png',
                     zlim = None, age = None,
                     contour = True, run_stats = True, OLS = False):
    
    f = plt.figure()
    f.set_size_inches(10, 8)
    ax = f.gca()
    
    if zlim != None:
        sc = ax.scatter(x, y, c = z, cmap = 'spectral_r', vmin = zlim[0], vmax = zlim[1], marker = 'o', s = 5., edgecolor = 'none')
    else:
        sc = ax.scatter(x, y, c = z, cmap = 'spectral_r', marker = 'o', s = 5., edgecolor = 'none')
        
    cb = f.colorbar(sc)
    cb.set_label(zlabel)
    
    if contour == True:
        binsx = np.linspace(min(x), max(x), 21)
        binsy = np.linspace(min(y), max(y), 21)
        density_contour(x, y, binsx, binsy, ax = ax, color = 'k')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    if OLS == True:
        A, B, sigma_A, sigma_B = plotOLSbisectorAxis(ax, x, y, 0.98, 0.02, 16)
        
    if run_stats == True:
        nBox = 40
        dxBox = (x.max() - x.min()) / (nBox - 1.)
        aux = calc_running_stats(x, y, dxBox = dxBox, xbinIni = x.min(), xbinFin = x.max(), xbinStep = dxBox)
        xbinCenter = aux[0]
        xMedian = aux[1]
        xMean = aux[2]
        xStd = aux[3]
        yMedian = aux[4]
        yMean = aux[5]
        yStd = aux[6]
        nInBin = aux[7]
        xPrc = aux[8]
        yPrc = aux[9]
        ax.plot(xMedian, yMedian, 'k', lw = 2)
        ax.plot(xPrc[0], yPrc[0], 'k--', lw = 2)
        ax.plot(xPrc[1], yPrc[1], 'k--', lw = 2)
        if OLS == True:
            xMm = np.ma.masked_array(xMedian, mask = np.isnan(xMedian))
            yMm = np.ma.masked_array(yMedian, mask = np.isnan(yMedian))
            A, B, sigma_A, sigma_B = plotOLSbisectorAxis(ax, xMm, yMm, 0.98, 0.08, 16, color = 'b')
        
    if age != None:
        txt = r'$%s$ Myr' % str(age / 1.e6)
        plot_text_ax(ax, txt, 0.02, 0.98, 14, 'top', 'left')
    
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # ax.xaxis.set_major_locator(MultipleLocator(0.2))
    # ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    # ax.grid(which = 'both')
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        
    f.savefig(fname)
    plt.close(f)


def plot_contour_axis(ax, x, y, n = 21, color = 'k'):
    binsx = np.linspace(min(x), max(x), n)
    binsy = np.linspace(min(y), max(y), n)
    density_contour(x, y, binsx, binsy, ax = ax)


def plotScatter(x, y, xlabel, ylabel, xlim, ylim, fname = 'PlotScatter.png',
                age = None, color = 'grey', contour = True, run_stats = True,
                OLS = False):
    f = plt.figure()
    f.set_size_inches(10, 8)
    ax = f.gca()
    
    sc = ax.scatter(x, y, c = color, marker = 'o', s = 5., edgecolor = 'none')

    if contour == True:
        binsx = np.linspace(min(x), max(x), 21)
        binsy = np.linspace(min(y), max(y), 21)
        density_contour(x, y, binsx, binsy, ax = ax, color = 'k')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
        
    if OLS == True:
        a, b, sigma_a, sigma_b = plotOLSbisectorAxis(ax, x, y, 0.92, 0.05, 16, 'b', rms = True)
        
    if run_stats == True:
        nBox = 20
        dxBox = (x.max() - x.min()) / (nBox - 1.)
        aux = calc_running_stats(x, y, dxBox = dxBox, xbinIni = x.min(), xbinFin = x.max(), xbinStep = dxBox)
        xbinCenter = aux[0]
        xMedian = aux[1]
        xMean = aux[2]
        xStd = aux[3]
        yMedian = aux[4]
        yMean = aux[5]
        yStd = aux[6]
        nInBin = aux[7]
        ax.plot(xMedian, yMedian, 'k', lw = 2)
        
    if age != None:
        txt = r'$%s$ Myr' % str(age / 1.e6)
        plot_text_ax(ax, txt, 0.02, 0.98, 14, 'top', 'left')
        
    f.savefig(fname)
    plt.close(f)


def plotLinRegAge(x, y, xlabel, ylabel, xlim, ylim, age, fname):
    f = plt.figure()
    f.set_dpi(100)
    f.set_size_inches(11.69, 8.27) 
    plot_suptitle = '%.2f Myr' % (age / 1e6)
    f.suptitle(plot_suptitle)
    ax = f.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlim != None:
        ax.set_xlim(xlim)
        
    if ylim != None:
        ax.set_ylim(ylim)

    ax.scatter(x, y, c = 'black', marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = '')
    
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # c = 'b'
    # step = (x.max() - x.min()) / len(x)
    # A1, B1, Rp, pval, std_err = st.linregress(x, y)
    # X = np.linspace(x.min(), x.max() + step, len(x))
    # Y = A1 * X + B1
    # Yrms = (Y - y).std()
    # ax.plot(X, Y, c = c, ls = '--', lw = 2, label = 'least squares')
    # txt = '%.2f Myr' % (age / 1e6)
    # plot_text_ax(ax, txt, 0.05, 0.92, 14, 'top', 'left')
    # txt = r'$y = %.2f\ x\ +\ (%.2f)\ (y_{rms}:%.2f)$' %  (A1, B1, Yrms)
    # plot_text_ax(ax, txt, 0.98, 0.21, 14, 'bottom', 'right', c)
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
     
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # c = 'g'
    # model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    # model_ransac.fit(np.vstack(x),np.vstack(y))
    # Y = model_ransac.predict(X[:, np.newaxis])
    # Yrms = (Y - y).std()
    # ax.plot(X, Y, c = c, ls = '--', lw = 2, label = 'RANSAC')
    # A = model_ransac.estimator_.coef_
    # B = model_ransac.estimator_.intercept_
    # inlier_mask = model_ransac.inlier_mask_
    # #outlier_mask = np.logical_not(inlier_mask)
    # txt = r'$y = %.2f\ x\ +\ (%.2f)\ (y_{rms}:%.2f)$' %  (A, B, Yrms)
    # plot_text_ax(ax, txt, 0.98, 0.14, 14, 'bottom', 'right', c)
    # ax.scatter(x[inlier_mask], y[inlier_mask], c = c, marker = 'x', s = 20, facecolor = 'k', edgecolor = c, alpha = 0.3, label='')
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    
    c = 'r'
    plotOLSbisectorAxis(ax, x, y, 0.98, 0.07, 14, c, True)
    
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls = '--', label = '', c = 'k')
    ax.legend()
    f.savefig(fname)
    plt.close(f)

def plotLinRegAxis(ax, x, y, xlabel, ylabel, xlim, ylim):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.scatter(x, y, c = 'black', marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = '')    
    plotOLSbisectorAxis(ax, x, y, 0.98, 0.02, 14, 'b', True)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls = '--', label = '', c = 'k')
    ax.legend()


def plot_zbins(**kwargs):
    debug = kwargs.get('debug', False)
    debug_var(debug, kwargs = kwargs)
    ###### vars begin ######
    x = kwargs.get('x', None)
    xlabel = kwargs.get('xlabel', None)
    xlim = kwargs.get('xlim', None)
    y = kwargs.get('y', None)
    ylabel = kwargs.get('ylabel', None)
    ylim = kwargs.get('ylim', None)
    z = kwargs.get('z', None)
    zmask = kwargs.get('zmask', False)
    zbins = kwargs.get('zbins', False)
    zlim = kwargs.get('zlim', None)
    zlimprc = kwargs.get('zlimprc', None)
    zname = kwargs.get('zname', 'z')
    z2 = kwargs.get('z2', None)
    z2mask = kwargs.get('z2mask', False)
    zcolor = kwargs.get('zcolor', 'k')
    zbins_mask = kwargs.get('zbins_mask', None) 
    zbins_labels = kwargs.get('zbins_labels', None)
    zbins_colors = kwargs.get('zbins_colors', None)
    zticklabels = kwargs.get('zticklabels', None)
    zticks = kwargs.get('zticks', None)
    ###### vars end ######
    if z is not None: 
        if zmask == True:
            mask = x.mask | y.mask | z.mask
            if z2mask == True:
                mask = mask | z2.mask
    else:
        mask = x.mask | y.mask
    xm = np.ma.masked_array(x, mask = mask)
    ym = np.ma.masked_array(y, mask = mask)
    kwargs_scatter = {}
    if z is not None:
        zm = np.ma.masked_array(z, mask = mask)
        zcolor = zm
        if z2 is not None:
            z2m = np.ma.masked_array(z2, mask = mask)
        kwargs_scatter = dict(c = zcolor)
        zcmap = cm.ScalarMappable()
        zcmap.set_cmap(cm.spectral_r)
        kwargs_scatter.update(dict(cmap = zcmap.cmap))
        if zticks is not None:
            zlim = [ zticks[0], zticks[-1] ]
        else:
            if zlim is None:
                zlim = [ zm.compressed().min(), zm.compressed().max() ]
        if zlimprc is not None:
            zlim = np.percentile(zm.compressed(), zlimprc)
        norm = mpl.colors.Normalize(vmin = zlim[0], vmax = zlim[1])
        zcmap.set_norm(norm)
        kwargs_scatter.update(dict(norm = norm))
    ###### fig begin ######
    f = plt.figure()
    ax = f.gca()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    kwargs_scatter.update(kwargs.get('kwargs_scatter', {}))
    debug_var(debug, kwargs_scatter = kwargs_scatter)
    sc = ax.scatter(xm, ym, **kwargs_scatter)
    zlabel = kwargs.get('zlabel', None)
    if z is not None and zlabel is not None:
        kwargs_tmp = {}
        if zticks is not None:
            kwargs_tmp.update(dict(ticks = zticks))
        cb = f.colorbar(sc, **kwargs_tmp)
        cb.set_label(zlabel)
        if zticklabels is not None:
            cb.ax.set_yticklabels(zticklabels)
    if kwargs.get('ols', False) is not False:
        kwargs_ols = dict(pos_x = 0.98, pos_y = 0.00, fs = 10, c = 'k', rms = True, text = True)
        kwargs_ols.update(kwargs.get('kwargs_ols', {}))
        kwargs_ols_plot = dict(ls = '--', lw = 1.5, label = 'OLS', c = 'k')
        kwargs_ols_plot.update(kwargs.get('kwargs_ols_plot', {})) 
        kwargs_ols.update(dict(kwargs_plot = kwargs_ols_plot))
        debug_var(debug, kwargs_ols = kwargs_ols)
        a, b, sa, sb = plotOLSbisectorAxis(ax, xm, ym, **kwargs_ols)
    if kwargs.get('running_stats', False) is not False:
        nBox = 20
        dxBox = (xm.max() - xm.min()) / (nBox - 1.)
        kwargs_rs = dict(dxBox = dxBox, xbinIni = xm.min(), xbinFin = xm.max(), xbinStep = dxBox)
        kwargs_rs.update(kwargs.get('kwargs_rs', {})) 
        debug_var(debug, kwargs_rs = kwargs_rs)
        xbinCenter, xMedian, xMean, xStd, yMedian, yMean, yStd, nInBin, xPrc, yPrc = calc_running_stats(xm, ym, **kwargs_rs)
        kwargs_plot_rs = kwargs.get('kwargs_plot_rs', dict(c = 'k', lw = 2))
        if kwargs.get('rs_errorbar', False) is not False:
            kwargs_plot_rs.update(dict(xerr = xStd, yerr = yStd))
        debug_var(debug, kwargs_plot_rs = kwargs_plot_rs)
        if kwargs.get('rs_gaussian_smooth', None) is None:
            plt.errorbar(xMedian, yMedian, **kwargs_plot_rs)
        else:
            FWHM = kwargs.get('rs_gs_fwhm', 0.4)
            xM = np.ma.masked_array(xMedian)
            yM = np.ma.masked_array(yMedian)
            m_gs = np.isnan(xM) | np.isnan(yM) 
            Xs, Ys = gaussSmooth_YofX(xM[~m_gs], yM[~m_gs], FWHM)
            ax.plot(Xs, Ys, **kwargs_plot_rs)
        ax.plot(xPrc[0], yPrc[0], 'k--')
        ax.plot(xPrc[1], yPrc[1], 'k--')
        if kwargs.get('ols_rs', False) is not False:
            kwargs_ols_rs = dict(pos_x = 0.98, pos_y = 0.03, fs = 10, c = 'k', rms = True, label = 'OLS(tend)', text = True)
            kwargs_ols_rs.update(kwargs.get('kwargs_ols_rs', {}))
            debug_var(debug, kwargs_ols_rs = kwargs_ols_rs)
            a, b, sa, sb = plotOLSbisectorAxis(ax, xMedian, yMedian, **kwargs_ols_rs)
    if zbins != False:
        zbinsrange = xrange(zbins)
        if zbins_mask is None:
            nprc = kwargs.get('zbinsprc', [ (i + 1) * (100 / zbins) for i in zbinsrange ])
            center_nprc = [ (nprc[i] + nprc[i - 1]) / 2. for i in zbinsrange ]
            center_nprc[0] = nprc[0] / 2.
            zprc = np.percentile(zm.compressed(), nprc)
            center_prc = [ np.percentile(zm.compressed(), center_nprc[i]) for i in zbinsrange ]
            debug_var(debug, nprc = nprc)
            debug_var(debug, zprc = zprc)
            debug_var(debug, center_prc = center_prc)
            zbins_mask = [ (zm > zprc[i - 1]) & (zm <= zprc[i]) for i in zbinsrange ]
            zbins_mask[0] = (zm <= zprc[0])
            if zbins_labels is None:
                zbins_labels = [ '%.2f < %s <= %.2f' % (zprc[i - 1], zname, zprc[i]) for i in zbinsrange ]
                zbins_labels[0] = '%s <= %.2f' % (zname, zprc[0])
            if zbins_colors is None:
                zbins_colors = [ zcmap.to_rgba(center_prc[i]) for i in zbinsrange ]
        if zbins_labels is None or zbins_colors is None:
            listrange = xrange(len(zbins_mask))
            zmsk = [ zm[np.where(np.asarray(np.asarray(zbins_mask[i])) == True)] for i in listrange ]
            if zbins_labels is None:
                zbins_labels = []
                for i in listrange:
                    zmskmax = zmsk[i].max()
                    zmskmin = zmsk[i].min()
                    debug(debug, zmskmax = zmskmax)
                    debug(debug, zmskmin = zmskmin)
                    if zmskmin == zmskmax:
                        zbins_labels.append('%s = %.2f' % (zname, zmskmin))
                    else:
                        zbins_labels.append('%.2f <= %s <= %.2f' % (zmskmin, zname, zmskmax))         
            if zbins_colors is None:
                zbins_colors = []
                for i in listrange:
                    zmskmax = zmsk[i].max()
                    zmskmin = zmsk[i].min()
                    if zmskmin == zmskmax:
                        zbins_colors.append(zcmap.to_rgba(zmskmax))
                    else:
                        zbins_colors.append(zcmap.to_rgba(0.5 * (zmskmax + zmskmin)))
        debug_var(debug, zbins_labels = zbins_labels)
        debug_var(debug, zbins_colors = zbins_colors)
        for i, msk in enumerate(zbins_mask):
            y_pos = (0.03 * i) + 0.03
            X = xm[msk]
            Y = ym[msk]
            NX = len(X)
            NY = len(Y)
            debug_var(debug, bin = i)
            debug_var(debug, N = (~(X.mask | Y.mask)).sum())
            if NX > 3 and NY > 3:
                nBox = 50
                dxBox = (X.max() - X.min()) / (nBox - 1.)
                kwargs_zbins_rs = dict(dxBox = dxBox, xbinIni = X.min(), xbinFin = X.max(), xbinStep = dxBox)
                kwargs_zbins_rs.update(kwargs.get('kwargs_zbins_rs', {})) 
                debug_var(debug, kwargs_zbins_rs = kwargs_zbins_rs)
                xbinCenter, xMedian, xMean, xStd, yMedian, yMean, yStd, nInBin, xPrc, yPrc = calc_running_stats(X, Y, **kwargs_zbins_rs)
                if kwargs.get('zbins_rs_gaussian_smooth', None) is None:
                    plt.errorbar(xMedian, yMedian, c = zbins_colors[i], lw = 2, label = '%s' % zbins_labels[i])
                else:
                    FWHM = kwargs.get('zbins_rs_gs_fwhm', 0.4)
                    xM = np.ma.masked_array(xMedian)
                    yM = np.ma.masked_array(yMedian)
                    m_gs = np.isnan(xM) | np.isnan(yM) 
                    Xs, Ys = gaussSmooth_YofX(xM[~m_gs], yM[~m_gs], FWHM)
                    ax.plot(Xs, Ys, c = zbins_colors[i], lw = 2, label = '%s' % zbins_labels[i])
    x_major_locator = kwargs.get('x_major_locator', (xm.max() - xm.min()) / 6.)
    x_minor_locator = kwargs.get('x_minor_locator', (xm.max() - xm.min()) / 30.)
    y_major_locator = kwargs.get('y_major_locator', (ym.max() - ym.min()) / 6.)
    y_minor_locator = kwargs.get('y_minor_locator', (ym.max() - ym.min()) / 30.)
    debug_var(debug, x_major_locator = x_major_locator)
    debug_var(debug, x_minor_locator = x_minor_locator)
    debug_var(debug, y_major_locator = y_major_locator)
    debug_var(debug, y_minor_locator = y_minor_locator)
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_locator))
    ax.grid(which = 'major')
    if xlim is not None:
        xlim = ax.set_xlim(xlim)
    if ylim is not None:
        xlim = ax.set_ylim(ylim)
    kwargs_legend = dict(loc = 'upper left', fontsize = 14)
    kwargs_legend.update(kwargs.get('kwargs_legend', {}))
    debug_var(debug, kwargs_legend = kwargs_legend)
    ax.legend(**kwargs_legend)
    kwargs_suptitle = dict(fontsize = 10)
    kwargs_suptitle.update(kwargs.get('kwargs_suptitle', {}))
    debug_var(debug, kwargs_suptitle = kwargs_suptitle)
    f.suptitle(kwargs.get('suptitle'), **kwargs_suptitle)
    filename = kwargs.get('filename')
    debug_var(debug, filename = filename)
    if filename is not None:
        f.savefig(filename)
    else:
        f.show()
    plt.close(f)  


if __name__ == '__main__':
    import sys
    from CALIFAUtils.plots import plot_zbins
    from CALIFAUtils.scripts import read_one_cube, sort_gals
    from CALIFAUtils.objects import GasProp, H5SFRData
    filename = '/Users/lacerda/dev/astro/EmissionLines/SFR_0.05_0.05_0.05_999.0_listOf300GalPrefixes.h5'
    H = H5SFRData(filename)
    iT = 11
    iU = -1
    tSF = H.tSF__T[iT]
    tZ = H.tZ__U[iU]

    DGR = 10. ** (-2.21)
    k = 0.2 / DGR
    SigmaGas__g = k * H.tau_V__Tg[iT]
    f_gas__g = 1. / (1. + (H.McorSD__Tg[iT] / SigmaGas__g))
    
    P = GasProp()
    tau_V_GP__g = np.ma.masked_where(P.CtoTau(H.chb_in__g) < H.xOkMin, P.CtoTau(H.chb_in__g), copy = True)
    
    Z = { 
        'OHIICHIM' : dict(v = H.O_HIICHIM__g, label = r'12 + $\log\ O/H$ (HII-CHI-mistry, EPM, 2014)'),
        'logO3N2S06' : dict(v = H.logZ_neb_S06__g + np.log10(4.9e-4) + 12, label = r'12 + $\log\ O/H$ (logO3N2, Stasinska, 2006)'),
        'logO3N2M13' : dict(v = H.O_O3N2_M13__g, label = r'12 + $\log\ O/H$ (logO3N2, Marino, 2013)'),
    }
    tau = {
        'tauV' : dict(v = H.tau_V__Tg[iT], label = r'$\tau_V^\star$'),
        'tauVneb' : dict(v = H.tau_V_neb__g, label = r'$\tau_V^{neb}$'),
        'tauVGP' : dict(v = tau_V_GP__g, label = r'$\tau_V^{GP}$'),
    }
    logtau = {
        'logtauV' : dict(v = np.ma.log10(H.tau_V__Tg[iT]), label = r'$\log\ \tau_V^\star$'),
        'logtauVneb' : dict(v = np.ma.log10(H.tau_V_neb__g), label = r'$\log\ \tau_V^{neb}$'),
        'logtauVGP' : dict(v = np.ma.log10(tau_V_GP__g), label = r'$\log\ \tau_V^{GP}$'),
    }
    SFRSD = {
        'logSFRSD' : dict(v = np.ma.log10(H.SFRSD__Tg[iT] * 1e6), label = r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} kpc^{-2}]$'),
        'logSFRSDHa' : dict(v = np.ma.log10(H.SFRSD_Ha__g * 1e6), label = r'$\log\ \Sigma_{SFR}^{neb}\ [M_\odot yr^{-1} kpc^{-2}]$'),
    }
    
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # for Zname, Zval in Z.iteritems():
    #     for tauname, tauval in tau.iteritems():
    #         plot_zbins(
    #                    debug = True,
    #                    x = tauval['v'],
    #                    xlabel = tauval['label'],
    #                    y = Zval['v'],
    #                    ylabel = Zval['label'],
    #                    z = H.dist_zone__g,
    #                    zlabel = r'zone distance [HLR]',
    #                    kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
    #                    running_stats = True,
    #                    kwargs_plot_rs = dict(c = 'k', lw = 2),
    #                    rs_errorbar = False,
    #                    suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
    #                    filename = '%s_%s_zoneDistance_%.2fMyr.png' % (tauname, Zname, tSF / 1e6),
    #                    kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
    #                    x_major_locator = 1.,
    #                    x_minor_locator = 0.25,
    #                    y_major_locator = 0.25,
    #                    y_minor_locator = 0.05,
    #                    kwargs_legend = dict(fontsize = 8),               
    #         )
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    for sfrsdname, sfrsdval in SFRSD.iteritems():
        for tauname, tauval in logtau.iteritems():
            for Zname, Zval in Z.iteritems():
                plot_zbins(
                           debug = True,
                           x = tauval['v'],
                           xlabel = tauval['label'],
                           y = sfrsdval['v'],
                           ylabel = sfrsdval['label'],
                           z = Zval['v'],
                           zlabel = Zval['label'],
                           zmask = True,
                           zlimprc = [ 2, 98 ],
                           zbins = 4,
                           zbins_rs_gaussian_smooth = True,
                           zbins_rs_gs_fwhm = 0.4,
                           kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
                           running_stats = True,
                           rs_gaussian_smooth = True,
                           rs_gs_fwhm = 0.4,
                           kwargs_plot_rs = dict(c = 'k', lw = 2, label = 'Median (run. stats)'),
                           rs_errorbar = False,
                           suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
                           filename = '%s_%s_%s_%.2fMyr.png' % (tauname, sfrsdname, Zname, tSF / 1e6),
                           x_major_locator = 0.5,
                           x_minor_locator = 0.125,
                           y_major_locator = 0.5,
                           y_minor_locator = 0.125,
                           kwargs_legend = dict(fontsize = 8),
                )

    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # plot_zbins(
    #            debug = True,
    #            x = np.ma.log10(H.tau_V__Tg[iT]),
    #            xlim = [-1.5, 0.75],
    #            xlabel = r'$\log\ \tau_V^\star$',
    #            y = np.ma.log10(H.SFRSD__Tg[iT] * 1e6),
    #            ylim = [-3.5, 1],
    #            ylabel = r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} kpc^{-2}]$',
    #            z = H.alogZ_mass__Ug[iU],
    #            zlabel = r'$\langle \log\ Z_\star \rangle_M (R)\ (t\ <\ %.2f\ Gyr)\ [Z_\odot]$' % (tZ / 1e9),
    #            zmask = True,
    #            zlimprc = [ 2, 98 ],
    #            zbins = 5,
    #            kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
    #            ols_rs = False,
    #            ols = True,
    #            running_stats = True,
    #            #kwargs_rs = dict(), 
    #            kwargs_plot_rs = dict(c = 'k', lw = 2),
    #            rs_errorbar = False,
    #            suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
    #            filename = 'logTauV_logSFRSD_alogZmass_%.2fMyr.png' % (tSF / 1e6),
    #            kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
    #            x_major_locator = 0.25,
    #            x_minor_locator = 0.05,
    #            y_major_locator = 0.25,
    #            y_minor_locator = 0.05,
    #            kwargs_legend = dict(fontsize = 8),               
    # )
    # plot_zbins(
    #            debug = True,
    #            x = H.tau_V__Tg[iT],
    #            xlim = [0, 1.5],
    #            xlabel = r'$\tau_V^\star$',
    #            y = H.tau_V_neb__g,
    #            ylim = [0, 2.5],
    #            ylabel = r'$\tau_V^{neb}$',
    #            z = H.alogZ_mass__Ug[iU],
    #            zlabel = r'$\langle \log\ Z_\star \rangle_M (R)\ (t\ <\ %.2f\ Gyr)\ [Z_\odot]$' % (tZ / 1e9),
    #            zlimprc = [5,95],
    #            zmask = False,
    #            kwargs_scatter = dict(marker = 'o', s = 6, edgecolor = 'none', alpha = 0.4, label = ''),
    #            suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
    #            filename = 'tauV_tauVneb_alogZmass_%.2fMyr.png' % (tSF / 1e6),
    #            ols_rs = False,
    #            ols = True,
    #            running_stats = True,
    #            kwargs_plot_rs = dict(c = 'k', lw = 2),
    #            rs_errorbar = False,
    #            zbins = 6,
    #            kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
    #            x_major_locator = 0.25,
    #            x_minor_locator = 0.05,
    #            y_major_locator = 0.25,
    #            y_minor_locator = 0.05,
    #            kwargs_legend = dict(fontsize = 8),
    # )
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
 #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
 #    x = H.tau_V__Tg[iT]
 #    y = H.tau_V_neb__g
 #    z =  H.reply_arr_by_zones(H.morfType_GAL__g)
 #    mask = x.mask | y.mask
 #    zm = np.ma.masked_array(z, mask = mask)
 #    zticks_mask = [(zm > 8.9) & (zm <= 9.5), (zm == 10), (zm == 10.5), (zm >= 11.) & (zm > 9) & (zm <= 11.5)]
 #    zticks = [9., 9.5, 10, 10.5, 11., 11.5]
 #    zticklabels = ['Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd']
 #    #zbinsticklabels = ['Sa + Sab', 'Sb', 'Sbc', 'Sc + Scd']
 # 
 #    plot_zbins(
 #               zname ='morph',
 #               debug = True, 
 #               x = H.tau_V__Tg[iT],
 #               xlim = [0, 1.5],
 #               xlabel = r'$\tau_V^\star$',
 #               y = H.tau_V_neb__g,
 #               ylim = [0, 2.5],
 #               ylabel = r'$\tau_V^{neb}$',
 #               z =  H.reply_arr_by_zones(H.morfType_GAL__g),
 #               zlabel = 'morph. type',
 #               zbins = len(zticks_mask),
 #               zbins_mask = zticks_mask,
 #               zticks = zticks,  
 #               zticklabels = zticklabels,
 #               #zbinsticklabels = zbinsticklabels, 
 #               zlim = [9, 11.5],
 #               kwargs_scatter = dict(marker = 'o', s = 6, edgecolor = 'none', alpha = 0.4, label = ''),
 #               suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
 #               filename = 'tauV_tauVneb_morphType_%.2fMyr.png' % (tSF / 1e6),
 #               ols = True,
 #               running_stats = True,
 #               kwargs_plot_rs = dict(c = 'k', lw = 2),
 #               rs_errorbar = False,
 #               kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
 #               x_major_locator = 0.25,
 #               x_minor_locator = 0.05,
 #               y_major_locator = 0.25,
 #               y_minor_locator = 0.05,
 #               kwargs_legend = dict(fontsize = 8),
 #    )
 #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

