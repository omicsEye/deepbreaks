import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from scipy import stats
from deepBreaks.models import importance_from_pipe
from deepBreaks.utils import stacked_barplot, box_plot


def plot_scatter(summary_result, report_dir):
    """
    Plots a scatter plot with -log of (p-value) column as the x-axis and the values of the other columns (start at 3) as
    the points color by each column name.

    :param report_dir: str, path to save the figure
    :param summary_result: pandas.DataFrame object that contains feature, p_value, score, and dynamic columns.
    :return: matplotlib.pyplot object that displays the scatter plot.
    """
    # Define the columns to plot
    cols_to_plot = list(summary_result.columns)[3:]

    if len(cols_to_plot) <= 7:
        color_list = ['#E69F00', '#56B4E9', '#cc79a7', '#009E73', '#0072b2', '#F0E449', '#d55e00']
    else:
        n = len(cols_to_plot)
        cmap = plt.get_cmap('viridis')
        cmap_max = cmap.N
        color_list = [cmap(int(k * cmap_max / (n - 1))) for k in range(n)]

    if summary_result['p_value'].min() == 0:
        min_value = min(summary_result.loc[summary_result['p_value'] > 0, 'p_value'].min()/10, 1e-300)
        summary_result.loc[summary_result['p_value'] == 0, 'p_value'] = min_value

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=300)
    for i, col in enumerate(cols_to_plot):
        # Plot each column's values
        ax.scatter(-1 * np.log10(summary_result['p_value']), summary_result[col],
                   c=color_list[i], label=col, alpha=0.5, edgecolor='black', linewidth=0.1)

    # Add legend and labels
    ax.legend(fontsize='xx-small', loc='upper left', bbox_to_anchor=(0, 1.02), ncol=2)
    ax.set_xlabel('-log(p_value)', fontsize=8)
    ax.set_ylabel('Relative Importance', fontsize=8)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(report_dir + '/pvalue_importance_scatter.pdf', bbox_inches='tight')
    # plt.show()
    return fig, ax


# function to fix the position of annotations
def best_position(x, y, xlist, ylist):
    xlist = np.array(xlist)
    ylist = np.array(ylist)

    if any((xlist > x * .9) & (xlist < x * 1.1)):
        sub_ylist = ylist[(xlist > x * .9) & (xlist < x * 1.1)]
        while any((sub_ylist > y - .1) & (sub_ylist < y + .1)):
            y = y * 1.15
    if y > 1.05:
        y = 1.05

    return x, y


# lollipop plot of positions and their relative importance
def dp_plot(importance, imp_col, model_name,
            figsize=(7.2, 3), dpi=350,
            ylab='Relative Importance', xlab='Positions',
            title_fontsize=10, xlab_fontsize=8, ylab_fontsize=8,
            xtick_fontsize=6, ytick_fontsize=6,
            annotate=1,
            report_dir='.'):
    """
    Plots the importance bar plot. x-axis is the positions and y-axis is the importance values.
    Parameters
    ----------
    importance : dict or pandas.DataFrame
        a dictionary or a dataframe containing information of `feature`, `importance` given in the `imp_col`
    imp_col : str
        name of the key or column that should be considered as the importance value
    model_name : str
        string to be added to the plot title
    figsize : tuple, default = (7.2, 3)
        a tuple for the size of the plot
    dpi : int, default = 350
    ylab : str, default = 'Relative Importance'
    xlab : str, default = 'Positions'
    title_fontsize : int, default = 10
    xlab_fontsize : int, default = 8
    ylab_fontsize : int, default = 8
    xtick_fontsize : int, default = 6
    ytick_fontsize : int, default = 6
    annotate : int, default = 1
        number of top positions to annotate
    report_dir : str
        path to directory to save the plots
    Returns
    -------
    str
        It saves the plot as both `pdf` and `pickle` object in `report_dir`
    """
    pl_title = "Important Positions - " + model_name

    if type(importance) is dict:
        importance = pd.DataFrame.from_dict(importance)

    assert isinstance(importance, pd.DataFrame), 'Please provide a dictionary or a dataframe for importance values'

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.vlines(x=importance['feature'], ymin=0,
               ymax=importance[imp_col], color='black',
               linewidth=.7, alpha=0.8)

    plt.title(pl_title, loc='center', fontsize=title_fontsize)
    plt.xlabel(xlab, fontsize=xlab_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.ylabel(ylab, fontsize=ylab_fontsize)
    plt.ylim(0, 1.1)
    plt.xlim(1, importance['feature'].max() + 10)
    plt.yticks(fontsize=ytick_fontsize)
    plt.grid(True, linewidth=.3)
    plt.grid(visible=True, which='minor', axis='x', color='r', linestyle='-', linewidth=2)

    # annotating top positions
    if annotate > 0:
        features = importance.sort_values(by=imp_col, ascending=False).head(annotate)['feature'].tolist()
        xtext_list = []
        ytext_list = []
        for n, ft in enumerate(features):
            x = int(ft)
            y = importance.loc[importance['feature'] == ft, imp_col].tolist()[0]
            if n > 0:
                xtext, ytext = best_position(x, y + .1, xlist=xtext_list, ylist=ytext_list)
            else:
                xtext = x
                ytext = y + .1

            if y == 1:
                ytext = y + .05
                xtext = x * 1.1

            xtext_list.append(xtext)
            ytext_list.append(ytext)

            plt.annotate(text=x,
                         xy=(x, y),
                         xytext=(xtext, ytext), fontsize=5,
                         arrowprops=dict(arrowstyle='->',
                                         color='green',
                                         lw=1,
                                         ls='-'))

    plt.savefig(str(report_dir + '/' + str(model_name) + '_' + str(dpi) + '.pdf'), bbox_inches='tight')

    with open(str(report_dir + '/' + model_name + '_' + str(dpi) + '.pickle'), 'wb') as handle:
        pickle.dump(fig, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return print(str(model_name) + ' Done')


# plot top 4 positions in a model
def plot_imp_model(importance, X_train, y_train, model_name,
                   meta_var, model_type, report_dir):
    """
    Plots the top 4 positions of a model. For regression (`model_type='reg'`), it plots box-plot and for classification
    (`model_type='cl'`) it plots stacked bar plot.
    Parameters
    ----------
    importance : dict or pandas.DataFrame
        a dictionary or a dataframe containing information of `feature`, `importance` given in the `imp_col`
    X_train : pandas.DataFrame
        a numeric dataframe
    y_train : 1D array
        an 1D array of values of the response variable (phenotype)
    model_name : str
        string to be added to the plot title
    meta_var : str
        the name of the feature under study (phenotype)
    model_type :  str
        'reg' for regression and 'cl' for classification
    report_dir : str
        path to directory to save the plots
    Returns
    -------
    str
        It saves the plot as `pdf` in `report_dir`
    """
    dat = X_train.copy()
    dat.loc[:, meta_var] = y_train

    if type(importance) is dict:
        temp = pd.DataFrame.from_dict(importance)
    else:
        temp = importance

    temp = temp[['feature', 'standard_value']]
    features = temp.sort_values(by='standard_value', ascending=False).head(4)['feature'].tolist()
    features = ['p' + str(f) for f in features]

    if model_type == 'reg':

        fig, axes = plt.subplots(figsize=(7.5, 7.5), dpi=350, constrained_layout=True, nrows=2, ncols=2)
        axes = axes.ravel()
        fig.suptitle(meta_var + ' VS important positions', fontsize=10)
        for nm, cl in enumerate(features):
            ax = axes[nm]
            box_plot(data=dat, group_col=cl, response_var=meta_var, ax=ax)

        plt.savefig(str(report_dir + '/' + model_name + '_positions_box_' + str(350) + '.pdf'), bbox_inches='tight')

    else:

        fig, axes = plt.subplots(figsize=(7.5, 7.5), dpi=350, constrained_layout=True, nrows=2, ncols=2)
        axes = axes.ravel()
        plt.suptitle(meta_var + ' VS important positions', fontsize=10)
        #         plt.subplots_adjust(wspace=0.3)
        for nm, cl in enumerate(features):
            ax = axes[nm]
            stacked_barplot(data=dat, group_col=cl, response_var=meta_var, ax=ax)
        plt.savefig(str(report_dir + '/' + model_name + '_top_positions' + str(350) + '.pdf'), bbox_inches='tight')
    return print(model_name, ' Done')


def plot_imp_all(final_models, X_train, y_train,
                 model_type, report_dir,
                 meta_var = 'meta_var', n_positions=None, grouped_features=None,
                 max_plots=100,
                 figsize=(3, 3)):
    """
    plots all the important position across all the top selected models (up to `max_plots`).
    Parameters
    ----------
    final_models : list
        a list of sklearn model objects. Can also be a pipeline that the last layer is a model.
    X_train : pandas.DataFrame
        a numeric dataframe
    y_train : 1D array
        an 1D array of values of the response variable (phenotype)
    model_type :  str
        'reg' for regression and 'cl' for classification
    report_dir : str
        path to directory to save the plots
    n_positions : int
        Number of positions in the initial sequence file. Only needed when the model object
         does not have a preprocessing step.
    grouped_features : dict
        a dictionary that has information of the clusters of the positions. Only needed when the model object
         does not have a preprocessing step.
    max_plots : int
        maximum number of plots to create
    figsize : tuple
        a tuple for the size of the plot
    Returns
    -------
    dict
        a dictionary of all plots. Keys are position names and values are the plots.
    """
    plot_dir = str(report_dir + '/significant_positions_plots')

    if os.path.exists(plot_dir):
        cn = 1
        plot_dir_temp = plot_dir
        while os.path.exists(plot_dir_temp):
            plot_dir_temp = plot_dir + '_' + str(cn)
            cn += 1
        plot_dir = plot_dir_temp

    os.makedirs(plot_dir)

    dat = X_train.copy()
    dat.loc[:, meta_var] = y_train

    feature_list = []
    plots = {}
    cn_p = 0  # plot counter

    for model in final_models:

        tmp2 = pd.DataFrame.from_dict(importance_from_pipe(model, n_positions=n_positions,
                                                           grouped_features=grouped_features))
        temp = tmp2.loc[:, ['feature', 'standard_value']]
        features = temp.sort_values(by='standard_value', ascending=False)['feature'].tolist()
        features = ['p' + str(f) for f in features]

        p = 0
        cn_f = 0  # feature counter
        check = 0

        while (p < 0.05 or check < 10) and (cn_f < len(features)) and (cn_p < max_plots):

            cl = features[cn_f]
            cn_f += 1
            cn_p += 1
            if cl not in feature_list:
                if model_type == 'reg':
                    try:
                        k, p = stats.kruskal(*[group[meta_var].values for name, group in dat.groupby(cl)])
                        if p < 0.05:
                            feature_list.append(cl)
                            fig, ax = plt.subplots(figsize=figsize, dpi=350)
                            box_plot(data=dat, group_col=cl, response_var=meta_var, ax=ax, p=p)
                            plt.savefig(str(plot_dir + '/' + cl + '_boxplot_' + str(350) + '.pdf'),
                                        bbox_inches='tight')
                            plots[cl] = fig, ax
                    except:
                        pass

                else:
                    try:

                        cross_tb = pd.crosstab(dat[meta_var], dat[cl])

                        # chi-square test
                        chi2, p, dof, expected = stats.chi2_contingency(cross_tb)

                        if p < 0.05:
                            feature_list.append(cl)
                            fig, ax = plt.subplots(figsize=figsize, dpi=350)
                            stacked_barplot(cross_table=cross_tb, group_col=cl, ax=ax)
                            plt.savefig(str(plot_dir + '/' + cl + '_stacked_barplot_' + str(350) + '.pdf'),
                                        bbox_inches='tight')
                            plots[cl] = fig, ax
                    except:
                        pass
            if p >= 0.05:
                check += 1
    with open(str(plot_dir + '/plots.pickle'), 'wb') as handle:
        pickle.dump(plots, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return plots
