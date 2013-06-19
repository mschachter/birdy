import copy
import numpy as np
import matplotlib.pyplot as plt
import operator

def compute_R2(x, y):
    return(((x - x.mean()) * (y - y.mean())).mean() / (x.std()*y.std()))**2

def multi_plot(data_list, plot_func, title=None, nrows=4, ncols=5):

    nsp = 0
    fig = None
    plots_per_page = nrows*ncols
    for pdata in data_list:
        if nsp % plots_per_page == 0:
            fig = plt.figure()
            fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20)
            if title is not None:
                plt.suptitle(title)

        nsp += 1
        sp = nsp % plots_per_page
        ax = fig.add_subplot(nrows, ncols, sp)
        plot_func(pdata, ax)


def plot_pairwise_analysis(data_mat, feature_columns, dependent_column, column_names):
    """
        Does a basic pairwise correlation analysis between features and a dependent variable,
        meaning it plots a scatter plot with a linear curve fit through it, with the R^2.
        Then it plots a correlation matrix for all features and the dependent variable.

        data_mat: an NxM matrix, where there are N samples, M-1 features, and 1 dependent variable.

        feature_columns: the column indices of the features in data_mat that are being examined

        dependent_column: the column index of the dependent variable in data_mat

        column_names: a list of len(feature_columns)+1 feature/variable names. The last element is
                      the name of the dependent variable.
    """

    plot_data = list()
    for k,fname in enumerate(column_names[:-1]):
        fi = feature_columns[k]

        pdata = dict()
        pdata['x'] = data_mat[:, fi]
        pdata['y'] = data_mat[:, dependent_column]
        pdata['xlabel'] = column_names[fi]
        pdata['ylabel'] = column_names[-1]
        pdata['R2'] = compute_R2(pdata['x'], pdata['y'])
        plot_data.append(pdata)

    #sort by R^2
    plot_data.sort(key=operator.itemgetter('R2'), reverse=True)
    multi_plot(plot_data, plot_pairwise_scatter, title=None, nrows=3, ncols=3)

    all_columns = copy.copy(feature_columns)
    all_columns.append(dependent_column)
    
    C = np.corrcoef(data_mat[:, all_columns].transpose())

    Cy = C[:, -1]
    corr_list = [(column_names[k], np.abs(Cy[k]), Cy[k]) for k in range(len(column_names)-1)]
    corr_list.sort(key=operator.itemgetter(1), reverse=True)

    print 'Correlations  with %s' % column_names[-1]
    for cname,abscorr,corr in corr_list:
        print '\t%s: %0.6f' % (cname, corr)

    fig = plt.figure()
    plt.subplots_adjust(top=0.99, bottom=0.15, left=0.15)
    ax = fig.add_subplot(1, 1, 1)
    fig.autofmt_xdate(rotation=45)
    im = ax.imshow(C, interpolation='nearest', aspect='auto', vmin=-1.0, vmax=1.0, origin='lower')
    plt.colorbar(im)
    ax.set_yticks(range(len(column_names)))
    ax.set_yticklabels(column_names)
    ax.set_xticks(range(len(column_names)))
    ax.set_xticklabels(column_names)


def plot_pairwise_scatter(plot_data, ax):

    x = plot_data['x']
    y = plot_data['y']
    if 'R2' not in plot_data:
        R2 = compute_R2(x, y)
    else:
        R2 = plot_data['R2']
    slope,bias = np.polyfit(x, y, 1)
    sp = (x.max() - x.min()) / 25.0
    xrng = np.arange(x.min(), x.max(), sp)

    clr = '#aaaaaa'
    if 'color' in plot_data:
        clr = plot_data['color']
    ax.plot(x, y, 'o', mfc=clr)
    ax.plot(xrng, slope*xrng + bias, 'k-')
    ax.set_title('%s: R2=%0.2f' % (plot_data['xlabel'], R2))
    if 'ylabel' in plot_data:
        ax.set_ylabel(plot_data['ylabel'])
    ax.set_ylim(y.min(), y.max())


def plot_histogram_categorical(x, xname='x', y=None, yname='y', color='g'):
    """
        Makes a histogram of the variable x, which is an array of categorical variables in their native representation
        (string or intger) . If a dependent continuous variable y is specified, it will make another plot which
        is a bar graph showing the mean and standard deviation of the continuous variable y.
    """

    ux = np.unique(x)
    xfracs = np.array([(x == xval).sum() for xval in ux]) / float(len(x))

    nsp = 1 + (y is not None)
    ind = range(len(ux))

    fig = plt.figure()
    ax = fig.add_subplot(nsp, 1, 1)
    ax.bar(ind, xfracs, facecolor=color, align='center', ecolor='black')
    ax.set_xticks(ind)
    ax.set_xticklabels(ux)
    ax.set_xlabel(xname)
    ax.set_ylabel('Fraction of Samples')

    if y is not None:
        y_per_x = dict()
        for xval in ux:
            indx = x == xval
            y_per_x[xval] = y[indx]

        ystats = [ (xval, y_per_x[xval].mean(), y_per_x[xval].std()) for xval in ux]
        ystats.sort(key=operator.itemgetter(0), reverse=True)

        xvals = [x[0] for x in ystats]
        ymeans = np.array([x[1] for x in ystats])
        ystds = np.array([x[2] for x in ystats])

        ax = fig.add_subplot(nsp, 1, 2)
        ax.bar(ind, ymeans, yerr=ystds, facecolor=color, align='center', ecolor='black')
        ax.set_xticks(ind)
        ax.set_xticklabels(xvals)
        #fig.autofmt_xdate()
        ax.set_ylabel('Mean %s' % yname)
        ax.set_xlabel(xname)
        ax.set_ylim(0, (ymeans+ystds).max())


def whist(x, **kwds):
    return plt.hist(x, weights=np.ones([len(x)]) / float(len(x)), **kwds)


def plot_confusion_matrix_single(pdata, ax):
    plt.imshow(pdata['cmat'], interpolation='nearest', aspect='auto', origin='upper', vmin=0, vmax=1)
    plt.title('p=%0.3f' % pdata['p'])