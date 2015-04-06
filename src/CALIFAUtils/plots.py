#!/usr/bin/python
#
# Lacerda@Granada - 13/Oct/2014
#
import h5py
import numpy as np
import CALIFAUtils as C
import matplotlib as mpl
import scipy.optimize as so
from scipy import stats as st
from matplotlib.pyplot import cm
#from sklearn import linear_model
from matplotlib import pyplot as plt
from CALIFAUtils.scripts import OLS_bisector
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
    K = C.read_one_cube(gal)
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

def add_subplot_axes(ax, rect, axisbg = 'w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg = axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize = x_labelsize)
    subax.yaxis.set_tick_params(labelsize = y_labelsize)

    return subax

def plot_zbins(**kwargs):
    debug = kwargs.get('debug', False)
    C.debug_var(debug, kwargs = kwargs)
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
    xlimprc = kwargs.get('xlimprc', None)
    ylimprc = kwargs.get('ylimprc', None)
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
    mask = x.mask | y.mask
    C.debug_var(debug, mask_xy = mask, mask_xy_len = len(mask), mask_xy_sum = mask.sum())
    if z is not None and zmask == True:
        mask |= z.mask
        C.debug_var(debug, mask_xyz = mask, mask_xyz_len = len(mask), mask_xyz_sum = mask.sum())
    if z2 is not None and z2mask == True:
        mask |= z2.mask
        C.debug_var(debug, mask_xyz2 = mask, mask_xyz2_sum = mask.sum())
    xm = np.ma.masked_array(x, mask = mask, dtype = x.dtype)
    ym = np.ma.masked_array(y, mask = mask, dtype = y.dtype)
    kwargs_scatter = {}
    if z is not None:
        zm = np.ma.masked_array(z, mask = mask, dtype = np.float64)
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
    f_kw = kwargs.get('figure', kwargs.get('f', None))
    if f_kw is not None:
        f = f_kw
    else:
        kwargs_figure = kwargs.get('kwargs_figure', {})
        f = plt.figure(**kwargs_figure)
    ax_kw = kwargs.get('axis', kwargs.get('ax', None)) 
    if ax_kw is not None:
        ax = ax_kw
    else:
        ax = f.gca()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    kwargs_scatter.update(kwargs.get('kwargs_scatter', {}))    
    C.debug_var(debug, kwargs_scatter = kwargs_scatter)
    sc = ax.scatter(xm, ym, **kwargs_scatter)
    zlabel = kwargs.get('zlabel', None)
    if z is not None:
        kwargs_tmp = {}
        if zticks is not None:
            kwargs_tmp.update(dict(ticks = zticks))
        cb = f.colorbar(sc, **kwargs_tmp)
        if zlabel is not None:
            cb.set_label(zlabel)
        if zticklabels is not None:
            cb.ax.set_yticklabels(zticklabels)
    if kwargs.get('ols', False) is not False:
        kwargs_ols = dict(pos_x = 0.98, pos_y = 0.00, fs = 10, c = 'k', rms = True, text = True)
        kwargs_ols.update(kwargs.get('kwargs_ols', {}))
        kwargs_ols_plot = dict(ls = '--', lw = 1.5, label = 'OLS', c = 'k')
        kwargs_ols_plot.update(kwargs.get('kwargs_ols_plot', {})) 
        kwargs_ols.update(dict(kwargs_plot = kwargs_ols_plot))
        C.debug_var(debug, kwargs_ols = kwargs_ols)
        a, b, sa, sb = plotOLSbisectorAxis(ax, xm, ym, **kwargs_ols)
    if kwargs.get('running_stats', False) is not False:
        nBox = 20
        dxBox = (xm.max() - xm.min()) / (nBox - 1.)
        kwargs_rs = dict(dxBox = dxBox, xbinIni = xm.min(), xbinFin = xm.max(), xbinStep = dxBox)
        kwargs_rs.update(kwargs.get('kwargs_rs', {})) 
        C.debug_var(debug, kwargs_rs = kwargs_rs)
        xbinCenter, xMedian, xMean, xStd, yMedian, yMean, yStd, nInBin, xPrc, yPrc = calc_running_stats(xm, ym, **kwargs_rs)
        kwargs_plot_rs = kwargs.get('kwargs_plot_rs', dict(c = 'k', lw = 2))
        if kwargs.get('rs_errorbar', False) is not False:
            kwargs_plot_rs.update(dict(xerr = xStd, yerr = yStd))
        C.debug_var(debug, kwargs_plot_rs = kwargs_plot_rs)
        if kwargs.get('rs_gaussian_smooth', None) is None:
            plt.errorbar(xMedian, yMedian, **kwargs_plot_rs)
        else:
            FWHM = kwargs.get('rs_gs_fwhm', 0.4)
            xM = np.ma.masked_array(xMedian)
            yM = np.ma.masked_array(yMedian)
            m_gs = np.isnan(xM) | np.isnan(yM) 
            Xs, Ys = gaussSmooth_YofX(xM[~m_gs], yM[~m_gs], FWHM)
            ax.plot(Xs, Ys, **kwargs_plot_rs)
        if kwargs.get('rs_percentiles', None) is not None:
            c = kwargs_plot_rs.get('c', 'k')
            ax.plot(xPrc[0], yPrc[0], c = c, ls = '--', label = '16 perc. (run. stats)')
            ax.plot(xPrc[1], yPrc[1], c = c, ls = '--', label = '84 perc. (run. stats)')
        if kwargs.get('rs_ols', False) is not False:
            pos_x = kwargs_ols.get('pos_x', 0.99)
            pos_y = kwargs_ols.get('pos_y', 0.) + 0.03
            kwargs_rs_ols = dict(pos_x = pos_x, pos_y = pos_y, fs = 10, c = 'k', rms = True, label = 'OLS(tend)', text = True)
            kwargs_rs_ols.update(kwargs.get('kwargs_rs_ols', {}))
            C.debug_var(debug, kwargs_rs_ols = kwargs_rs_ols)
            a, b, sa, sb = plotOLSbisectorAxis(ax, xMedian, yMedian, **kwargs_rs_ols)
        if kwargs.get('rs_yx', False) is not False:
            nBox = 20
            dxBox = (ym.max() - ym.min()) / (nBox - 1.)
            kwargs_rs_yx = dict(dxBox = dxBox, xbinIni = ym.min(), xbinFin = ym.max(), xbinStep = dxBox)
            kwargs_rs_yx.update(kwargs.get('kwargs_rs_yx', {})) 
            C.debug_var(debug, kwargs_rs_yx = kwargs_rs_yx)
            xbinCenter, xMedian, xMean, xStd, yMedian, yMean, yStd, nInBin, xPrc, yPrc = calc_running_stats(ym, xm, **kwargs_rs)
            kwargs_plot_rs_yx = kwargs.get('kwargs_plot_rs_yx', dict(c = 'k', lw = 2))
            if kwargs.get('rs_yx_errorbar', False) is not False:
                kwargs_plot_rs_yx.update(dict(xerr = xStd, yerr = yStd))
            C.debug_var(debug, kwargs_plot_rs_yx = kwargs_plot_rs_yx)
            if kwargs.get('rs_yx_gaussian_smooth', None) is None:
                plt.errorbar(xMedian, yMedian, **kwargs_plot_rs_yx)
            else:
                FWHM = kwargs.get('rs_yx_gs_fwhm', 0.4)
                xM = np.ma.masked_array(xMedian)
                yM = np.ma.masked_array(yMedian)
                m_gs = np.isnan(xM) | np.isnan(yM) 
                Xs, Ys = gaussSmooth_YofX(xM[~m_gs], yM[~m_gs], FWHM)
                ax.plot(Xs, Ys, **kwargs_plot_rs_yx)
            if kwargs.get('rs_yx_percentiles', None) is not None:
                c = kwargs_plot_rs_yx.get('c', 'k')
                ax.plot(xPrc[0], yPrc[0], c = c, ls = '--', label = '16 perc. (run. stats)')
                ax.plot(xPrc[1], yPrc[1], c = c, ls = '--', label = '84 perc. (run. stats)')
    if zbins != False:
        if zbins_mask is None:
            if isinstance(zbins, list):
                ### XXX: TODO
                zbinsrange = xrange(len(zbins))
                zprc = np.asarray(zbins, dtype = np.double)
                center_prc = (zprc[1:] + zprc[0:-1]) / 2
                C.debug_var(debug, zprc = zprc)
                C.debug_var(debug, center_prc = center_prc)
            else:
                zbinsrange = xrange(zbins)
                nprc = kwargs.get('zbinsprc', [ (i + 1) * (100 / zbins) for i in zbinsrange ])
                nprc[-1] = 100
                center_nprc = [ (nprc[i] + nprc[i - 1]) / 2. for i in zbinsrange ]
                center_nprc[0] = nprc[0] / 2.
                zprc = np.percentile(zm.compressed(), nprc)
                center_prc = [ np.percentile(zm.compressed(), center_nprc[i]) for i in zbinsrange ]
                C.debug_var(debug, nprc = nprc)
                C.debug_var(debug, zprc = zprc)
                C.debug_var(debug, center_prc = center_prc)
            zbins_mask = [ (zm > zprc[i - 1]) & (zm <= zprc[i]) for i in zbinsrange ]
            zbins_mask[0] = (zm <= zprc[0])
            if zbins_labels is None:
                zbins_labels = [ '%.2f < %s <= %.2f' % (zprc[i - 1], zname, zprc[i]) for i in zbinsrange ]
                zbins_labels[0] = '%s <= %.2f' % (zname, zprc[0])
            if zbins_colors is None:
                zbins_colors = [ zcmap.to_rgba(center_prc[i]) for i in zbinsrange ]
        if zbins_labels is None or zbins_colors is None:
            listrange = xrange(len(zbins_mask))
            zmsk = [ zm[np.where(np.asarray(zbins_mask[i]) == True)] for i in listrange ]
            if zbins_labels is None:
                zbins_labels = []
                for i in listrange:
                    zmskmax = zmsk[i].max()
                    zmskmin = zmsk[i].min()
                    C.debug_var(debug, zmskmax = zmskmax)
                    C.debug_var(debug, zmskmin = zmskmin)
                    if zmskmin == zmskmax:
                        zbins_labels.append('%s = %.2f' % (zname, zmskmin))
                    else:
                        zbins_labels.append('%.2f <= %s <= %.2f' % (zmskmin, zname, zmskmax))         
            if zbins_colors is None:
                zbins_colors = []
                for i in listrange:
                    zmskmax = zmsk[i].max()
                    zmskmin = zmsk[i].min()
                    C.debug_var(debug, zmskmax = zmskmax)
                    C.debug_var(debug, zmskmin = zmskmin)
                    if zmskmin == zmskmax:
                        zbins_colors.append(zcmap.to_rgba(zmskmax))
                    else:
                        zbins_colors.append(zcmap.to_rgba(0.5 * (zmskmax + zmskmin)))
        C.debug_var(debug, zbins_labels = zbins_labels)
        C.debug_var(debug, zbins_colors = zbins_colors)
        for i, msk in enumerate(zbins_mask):
            y_pos = (0.03 * i) + 0.03
            X = xm[msk]
            Y = ym[msk]
            NX = len(X)
            NY = len(Y)
            C.debug_var(debug, bin = i)
            C.debug_var(debug, N = (~(X.mask | Y.mask)).sum())
            if NX > 3 and NY > 3:
                nBox = 50
                dxBox = (X.max() - X.min()) / (nBox - 1.)
                kwargs_zbins_rs = dict(dxBox = dxBox, xbinIni = X.min(), xbinFin = X.max(), xbinStep = dxBox)
                kwargs_zbins_rs.update(kwargs.get('kwargs_zbins_rs', {})) 
                C.debug_var(debug, kwargs_zbins_rs = kwargs_zbins_rs)
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
    if xlim is not None:
        ax.set_xlim(xlim)
    elif xlimprc is not None:
        xlim = np.percentile(xm.compressed(), xlimprc)
        ax.set_xlim(xlim)
    else:
        xlim = [ xm.min(), xm.max() ]
    if ylim is not None:
        ax.set_ylim(ylim)
    elif ylimprc is not None:
        ylim = np.percentile(ym.compressed(), ylimprc)
        ax.set_ylim(ylim)
    else:
        ylim = [ ym.min(), ym.max() ]
    C.debug_var(debug, xlim = xlim)
    C.debug_var(debug, ylim = ylim)
    x_major_locator = kwargs.get('x_major_locator', (xlim[1] - xlim[0]) / 5.)
    x_minor_locator = kwargs.get('x_minor_locator', (xlim[1] - xlim[0]) / 25.)
    y_major_locator = kwargs.get('y_major_locator', (ylim[1] - ylim[0]) / 5.)
    y_minor_locator = kwargs.get('y_minor_locator', (ylim[1] - ylim[0]) / 25.)
    C.debug_var(debug, x_major_locator = x_major_locator)
    C.debug_var(debug, x_minor_locator = x_minor_locator)
    C.debug_var(debug, y_major_locator = y_major_locator)
    C.debug_var(debug, y_minor_locator = y_minor_locator)
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_locator))
    ax.grid(which = 'major')
    kwargs_legend = dict(loc = 'upper left', fontsize = 14)
    kwargs_legend.update(kwargs.get('kwargs_legend', {}))
    C.debug_var(debug, kwargs_legend = kwargs_legend)
    ax.legend(**kwargs_legend)
    kwargs_suptitle = dict(fontsize = 10)
    kwargs_suptitle.update(kwargs.get('kwargs_suptitle', {}))
    C.debug_var(debug, kwargs_suptitle = kwargs_suptitle)
    if ax_kw is None:
        f.suptitle(kwargs.get('suptitle'), **kwargs_suptitle)
        filename = kwargs.get('filename')
        C.debug_var(debug, filename = filename)
        if filename is not None:
            f.savefig(filename)
        else:
            f.show()
        plt.close(f)  
    if kwargs.get('return_kwargs', None) is not None:
        return kwargs

if __name__ == '__main__':
    import sys
    from CALIFAUtils.plots import plot_zbins
    filename = '/Users/lacerda/dev/astro/EmissionLines/SFR_0.05_0.05_0.05_999.0_listOf300GalPrefixes.h5'
    H = C.H5SFRData(filename)
    iT = 11
    iU = -1
    tSF = H.tSF__T[iT]
    tZ = H.tZ__U[iU]

    DGR = 10. ** (-2.21)
    k = 0.2 / DGR
    SigmaGas__g = k * H.tau_V__Tg[iT]
    f_gas__g = 1. / (1. + (H.McorSD__Tg[iT] / SigmaGas__g))
    
    P = C.GasProp()
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
    
    for Zname, Zval in Z.iteritems():
        for tauname, tauval in tau.iteritems():
            plot_zbins(
                       debug = True,
                       x = tauval['v'],
                       xlabel = tauval['label'],
                       y = Zval['v'],
                       ylabel = Zval['label'],
                       z = H.dist_zone__g,
                       zlabel = r'zone distance [HLR]',
                       kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
                       running_stats = True,
                       kwargs_plot_rs = dict(c = 'k', lw = 2),
                       rs_errorbar = False,
                       suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
                       filename = '%s_%s_zoneDistance_%.2fMyr.png' % (tauname, Zname, tSF / 1e6),
                       kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
                       x_major_locator = 1.,
                       x_minor_locator = 0.25,
                       y_major_locator = 0.25,
                       y_minor_locator = 0.05,
                       kwargs_legend = dict(fontsize = 8),               
            )
            
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

    ##############################################################################
    ##############################################################################
    ##############################################################################

    plot_zbins(
               debug = True,
               x = np.ma.log10(H.tau_V__Tg[iT]),
               xlim = [-1.5, 0.75],
               xlabel = r'$\log\ \tau_V^\star$',
               y = np.ma.log10(H.SFRSD__Tg[iT] * 1e6),
               ylim = [-3.5, 1],
               ylabel = r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} kpc^{-2}]$',
               z = H.alogZ_mass__Ug[iU],
               zlabel = r'$\langle \log\ Z_\star \rangle_M (R)\ (t\ <\ %.2f\ Gyr)\ [Z_\odot]$' % (tZ / 1e9),
               zmask = True,
               zlimprc = [ 2, 98 ],
               zbins = 5,
               kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
               ols_rs = False,
               ols = True,
               running_stats = True,
               #kwargs_rs = dict(), 
               kwargs_plot_rs = dict(c = 'k', lw = 2),
               rs_errorbar = False,
               suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
               filename = 'logTauV_logSFRSD_alogZmass_%.2fMyr.png' % (tSF / 1e6),
               kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
               x_major_locator = 0.25,
               x_minor_locator = 0.05,
               y_major_locator = 0.25,
               y_minor_locator = 0.05,
               kwargs_legend = dict(fontsize = 8),               
    )
    
    ##############################################################################
    ##############################################################################
    ##############################################################################

    
    plot_zbins(
               debug = True,
               x = H.tau_V__Tg[iT],
               xlim = [0, 1.5],
               xlabel = r'$\tau_V^\star$',
               y = H.tau_V_neb__g,
               ylim = [0, 2.5],
               ylabel = r'$\tau_V^{neb}$',
               z = H.alogZ_mass__Ug[iU],
               zlabel = r'$\langle \log\ Z_\star \rangle_M (R)\ (t\ <\ %.2f\ Gyr)\ [Z_\odot]$' % (tZ / 1e9),
               zlimprc = [5,95],
               zmask = False,
               kwargs_scatter = dict(marker = 'o', s = 6, edgecolor = 'none', alpha = 0.4, label = ''),
               suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
               filename = 'tauV_tauVneb_alogZmass_%.2fMyr.png' % (tSF / 1e6),
               ols_rs = False,
               ols = True,
               running_stats = True,
               kwargs_plot_rs = dict(c = 'k', lw = 2),
               rs_errorbar = False,
               zbins = 6,
               kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
               x_major_locator = 0.25,
               x_minor_locator = 0.05,
               y_major_locator = 0.25,
               y_minor_locator = 0.05,
               kwargs_legend = dict(fontsize = 8),
    )

    ##############################################################################
    ##############################################################################
    ##############################################################################

    x = H.tau_V__Tg[iT]
    y = H.tau_V_neb__g
    z =  H.reply_arr_by_zones(H.morfType_GAL__g)
    mask = x.mask | y.mask
    zm = np.ma.masked_array(z, mask = mask)
    zticks_mask = [(zm > 8.9) & (zm <= 9.5), (zm == 10), (zm == 10.5), (zm >= 11.) & (zm > 9) & (zm <= 11.5)]
    zticks = [9., 9.5, 10, 10.5, 11., 11.5]
    zticklabels = ['Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd']
    #zbinsticklabels = ['Sa + Sab', 'Sb', 'Sbc', 'Sc + Scd']
  
    plot_zbins(
               zname ='morph',
               debug = True, 
               x = H.tau_V__Tg[iT],
               xlim = [0, 1.5],
               xlabel = r'$\tau_V^\star$',
               y = H.tau_V_neb__g,
               ylim = [0, 2.5],
               ylabel = r'$\tau_V^{neb}$',
               z =  H.reply_arr_by_zones(H.morfType_GAL__g),
               zlabel = 'morph. type',
               zbins = len(zticks_mask),
               zbins_mask = zticks_mask,
               zticks = zticks,  
               zticklabels = zticklabels,
               #zbinsticklabels = zbinsticklabels, 
               zlim = [9, 11.5],
               kwargs_scatter = dict(marker = 'o', s = 6, edgecolor = 'none', alpha = 0.4, label = ''),
               suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1.e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax),
               filename = 'tauV_tauVneb_morphType_%.2fMyr.png' % (tSF / 1e6),
               ols = True,
               kwargs_ols_plot = dict(ls = '--', lw = 0.7, label = ''),
               running_stats = True,
               kwargs_plot_rs = dict(c = 'k', lw = 2),
               rs_errorbar = False,
               x_major_locator = 0.25,
               x_minor_locator = 0.05,
               y_major_locator = 0.25,
               y_minor_locator = 0.05,
               kwargs_legend = dict(fontsize = 8),
    )
    
    ##############################################################################
    ##############################################################################
    ##############################################################################
    
    suptitle = r'NGals:%d  tSF:%.2f Myr  $x_Y$(min):%.0f%%  $\tau_V^\star$(min):%.2f  $\tau_V^{neb}$(min):%.2f  $\epsilon\tau_V^{neb}$(max):%.2f' % (H.N_gals, (tSF / 1e6), H.xOkMin * 100., H.tauVOkMin, H.tauVNebOkMin, H.tauVNebErrMax)

    xaxis = {
        'logtauV' : dict(
                        v = np.ma.log10(H.tau_V__Tg[iT]),
                        label = r'$\log\ \tau_V^\star$', 
                        lim = [ -1.5, 0.5 ],  
                        majloc = 0.5, 
                        minloc = 0.1, 
                    ), 
        'logtauVNeb' : dict(
                        v = np.ma.log10(H.tau_V_neb__g),
                        label = r'$\log\ \tau_V^{neb}$', 
                        lim = [ -1.5, 0.5 ],  
                        majloc = 0.5, 
                        minloc = 0.1, 
                    ), 
        'logZoneArea' : dict(
                            v = np.ma.log10(H.zone_area_pc2__g),
                            label = r'$\log\ A_{zone}$ [$pc^2$]', 
                            lim = [ 3.5, 8.5 ],
                            majloc = 1.0, 
                            minloc = 0.2, 
                        ),
    }
    yaxis = {
        'logSFRSD' : dict(
                        v = np.ma.log10(H.SFRSD__Tg[iT]),
                        label = r'$\log\ \Sigma_{SFR}^\star(t_\star)\ [M_\odot yr^{-1} pc^{-2}]$', 
                        lim = [-9.5, -5],  
                        majloc = 0.5, 
                        minloc = 0.1, 
                    ), 
        'logSFRSDHa' : dict(
                        v = np.ma.log10(H.SFRSD_Ha__g),
                        label = r'$\log\ \Sigma_{SFR}^{neb}\ [M_\odot yr^{-1} pc^{-2}]$', 
                        lim = [-9.5, -5],  
                        majloc = 0.5, 
                        minloc = 0.1, 
                    ), 
    }

    x = H.tau_V__Tg[iT]
    y = H.tau_V_neb__g
    morfType =  H.reply_arr_by_zones(H.morfType_GAL__g)
    mask = x.mask | y.mask
    zm = np.ma.masked_array(morfType, mask = mask)
    zticks_mask = [(zm > 8.9) & (zm <= 9.5), (zm == 10), (zm == 10.5), (zm >= 11.) & (zm > 9) & (zm <= 11.5)],
    zticks = [9., 9.5, 10, 10.5, 11., 11.5]
    zticklabels = ['Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd']

    zaxis = {
        'alogZmass' : dict(
                        v = H.alogZ_mass__Ug[-1], 
                        label = r'$\langle \log\ Z_\star \rangle_M$ (t < %.2f Gyr) [$Z_\odot$]' % (H.tZ__U[-1] / 1e9),
                        lim = [-1.2, 0.22],
                        majloc = 0.28,
                        minloc = 0.056,
                      ),
        'logO3N2M13' : dict(
                        v = H.O_O3N2_M13__g, 
                        label = r'12 + $\log\ O/H$ (logO3N2, Marino, 2013)',
                        lim = [8.25, 8.6],
                        majloc = 0.07,
                        minloc = 0.014,
                       ),
        'xY' : dict(
                v = H.x_Y__Tg[iT] * 100.,
                label = r'$x_Y$ [%]',
                lim = [0., 30],
                majloc = 6,
                minloc = 1.2,
               ),
        'morfType' : dict(
                        v = H.reply_arr_by_zones(H.morfType_GAL__g),
                        label = 'morph. type',
                        mask = False,
                        ticks_mask = [(zm > 8.9) & (zm <= 9.5), (zm == 10), (zm == 10.5), (zm >= 11.) & (zm > 9) & (zm <= 11.5)],
                        ticks = zticks,
                        ticklabels = zticklabels,
                        bins = len(zticks_mask),
                        bins_mask = zticks_mask,
                        lim = [9, 11.5],
                     ),
    }

    #Rmin = 0.1
    #mask = (H.zone_dist_HLR__g > Rmin)

    for xk, xv in xaxis.iteritems():
        for yk, yv in yaxis.iteritems():
            for zk, zv in zaxis.iteritems():
                plot_zbins(
                    debug = True,
                    x = xv['v'],
                    y = yv['v'],
                    z = zv['v'], 
                    zbins = zv.get('bins', 4),
                    zbins_mask = zv.get('ticks_mask', None),
                    zticklabels = zv.get('ticklabels', None),
                    zticks = zv.get('ticks', None),
                    zmask = zv.get('mask', None), 
                    running_stats = True,
                    rs_gaussian_smooth = True,
                    rs_percentiles = True,
                    rs_gs_fwhm = 0.4,
                    zbins_rs_gaussian_smooth = True,
                    zbins_rs_gs_fwhm = 0.4,
                    kwargs_figure = dict(figsize=(10,8), dpi = 100),
                    kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
                    kwargs_plot_rs = dict(c = 'k', lw = 2, label = 'Median (xy) (run. stats)'),
                    kwargs_legend = dict(loc = 'best'),
                    kwargs_suptitle = dict(fontsize = 14),
                    xlabel = xv['label'],
                    ylabel = yv['label'],
                    zlabel = zv['label'], 
                    xlim = xv['lim'], 
                    ylim = yv['lim'],
                    zlim = zv['lim'], 
                    x_major_locator = xv['majloc'], 
                    x_minor_locator = xv['minloc'], 
                    y_major_locator = yv['majloc'], 
                    y_minor_locator = yv['minloc'], 
                    suptitle = suptitle,
                    filename = '%s_%s_%s_%.2fMyrs.png' % (xk, yk, zk, tSF / 1e6),
                )

    ##############################################################################
    ##############################################################################
    ##############################################################################
                
    xaxis = {
        'OHIICHIM' : dict(
                        v = H.O_HIICHIM__g,
                        label = r'12 + $\log\ O/H$ (HII-CHI-mistry, EPM, 2014)',
                        lim = [7.5, 8.6],
                     ),
        'logO3N2S06' : dict(
                        v = 12 + H.logZ_neb_S06__g + np.log10(4.9e-4),
                        label = r'12 + $\log\ O/H$ (Stasinska, 2006)',
                        lim = [8, 8.7],
                       ),
        'logO3N2PP04' : dict(
                        v = H.O_O3N2_PP04__g,
                        label = r'12 + $\log\ O/H$ (PP, 2004)',
                        lim = [8, 8.6],
                       ),
        'logO3N2M13' : dict(
                        v = H.O_O3N2_M13__g,
                        label = r'12 + $\log\ O/H$ (logO3N2, Marino, 2013)',
                        lim = [8, 8.6],
                       ),
    }
    y = 12. + np.ma.log10(H.O_direct_O_23__g)
    ylim = [7.0, 8.6]
    for xk, xv in xaxis.iteritems():
        x = xv['v']
        xlim = xv['lim']
        mask = y.mask | x.mask
        not_masked = len(y) - mask.sum() 
        kw = plot_zbins(
            return_kwargs = True,
            x = x,
            y = y,
            xlabel = xv['label'],
            xlim = xlim,
            ylabel = r'12 + $\log\ O/H$ (O23Direct)',
            ylim = ylim,
            kwargs_figure = dict(figsize = (10, 8), dpi = 100),
            kwargs_scatter = dict(marker = 'o', s = 10, edgecolor = 'none', alpha = 0.5, label = ''),
            running_stats = True,
            rs_percentiles = True,
            #rs_errorbar = True,
            #rs_gaussian_smooth = True,
            #rs_gs_fwhm = 0.4,
            kwargs_plot_rs = dict(c = 'k', lw = 2, label = 'Median (run. stats)'),
            x_major_locator = (xlim[1] - xlim[0]) / 5,
            x_minor_locator = (xlim[1] - xlim[0]) / 25,
            y_major_locator = (ylim[1] - ylim[0]) / 5,
            y_minor_locator = (ylim[1] - ylim[0]) / 25,
            suptitle = 'zones: %d' % not_masked,
            kwargs_suptitle = dict(fontsize = 12),
            filename = '%s_%s.png' % (xk, 'O23direct'),
            kwargs_legend = dict(fontsize = 12),
        )
        filetxt = kw['filename'].replace('.png', '.txt')
        f = open(filetxt, 'w') 
        f.write('CALIFAID\tZONE\tO/H\tOH_direct\n')
        for g, z, OH, OHdir in zip(H.reply_arr_by_zones(H.califaIDs)[~mask], H.zones_map[~mask], xv['v'][~mask], y[~mask]):
            f.write('%s\t%d\t%.4f\t%.4f' % (g, z, OH, OHdir))
            if OH >= 8.1 and OHdir <= 7.9:
                f.write('\t*\n')
            else:
                f.write('\n')
        f.close()
    
    ##############################################################################
    ##############################################################################
    ##############################################################################
    
     
        
        