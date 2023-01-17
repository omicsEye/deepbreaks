import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _importance_to_df(imp_dic):
    return pd.DataFrame.from_dict(imp_dic)


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
        importance = _importance_to_df(importance)

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
        temp = _importance_to_df(importance)
    else:
        temp = importance

    temp = temp[['feature', 'standard_value']]
    features = temp.sort_values(by='standard_value', ascending=False).head(4)['feature'].tolist()
    features = ['p' + str(f) for f in features]

    if model_type == 'reg':

        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=350, constrained_layout=True)
        fig.suptitle(meta_var + ' VS important positions', fontsize=10)
        for nm, cl in enumerate(features):
            k, p = stats.kruskal(*[group[meta_var].values for name, group in dat.groupby(cl)])

            color_dic = {}
            key_list = ['A', 'C', 'G', 'R',
                        'T', 'N', 'D', 'E',
                        'Q', 'H', 'I', 'L',
                        'K', 'M', 'F', 'P',
                        'S', 'W', 'Y', 'V', 'GAP']
            color_list = ['#0273b3', '#de8f07', '#029e73', '#d55e00',
                          '#cc78bc', '#ca9161', '#fbafe4', '#ece133',
                          '#56b4e9', '#bcbd21', '#aec7e8', '#ff7f0e',
                          '#ffbb78', '#98df8a', '#d62728', '#ff9896',
                          '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                          '#dbdb8d']  # sns.color_palette('husl', 21)
            for n, key in enumerate(key_list):
                color_dic[key] = color_list[n]
            color_dic['U'] = color_dic['T']
            # add gray to pallet for mixed combinations
            for let in set(dat.loc[:, cl]):
                if let not in color_dic:
                    color_dic[let] = '#808080'  # hex code for gray color

            ax = plt.subplot(2, 2, nm + 1)
            ax.grid(color='gray', linestyle='-', linewidth=0.2, axis='y')
            ax.set_axisbelow(True)

            ax = sns.boxplot(x=cl, y=meta_var, data=dat, showfliers=False,
                             width=.6, linewidth=.6, palette=color_dic)
            sns.despine(ax=ax)
            ax = sns.stripplot(x=cl, y=meta_var, data=dat, size=5,
                               alpha=0.3, linewidth=.5, palette=color_dic)

            ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize=8)
            ax.set_xlabel('')

        plt.savefig(str(report_dir + '/' + model_name + '_positions_box_' + str(350) + '.pdf'), bbox_inches='tight')

    else:
        np.random.seed(123)
        color_dic = {}
        key_list = ['A', 'C', 'G', 'R',
                    'T', 'N', 'D', 'E',
                    'Q', 'H', 'I', 'L',
                    'K', 'M', 'F', 'P',
                    'S', 'W', 'Y', 'V', 'GAP']
        color_list = ['#0273b3', '#de8f07', '#029e73', '#d55e00',
                      '#cc78bc', '#ca9161', '#fbafe4', '#ece133',
                      '#56b4e9', '#bcbd21', '#aec7e8', '#ff7f0e',
                      '#ffbb78', '#98df8a', '#d62728', '#ff9896',
                      '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                      '#dbdb8d']  # sns.color_palette('husl', 21)
        for n, key in enumerate(key_list):
            color_dic[key] = color_list[n]
        color_dic['U'] = color_dic['T']

        plt.figure(figsize=(7.5, 7.5), dpi=350)
        plt.suptitle(meta_var + ' VS important positions', fontsize=10)
        plt.subplots_adjust(wspace=0.3)
        for nm, cl in enumerate(features):
            ax = plt.subplot(2, 2, nm + 1)
            # Creating crosstab
            crosstb = pd.crosstab(dat[meta_var], dat[cl])

            # chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(crosstb)
            colors = []
            for let in crosstb.columns.tolist():
                if let in color_dic.keys():
                    colors.append(color_dic[let.upper()])
                else:
                    color_dic[let.upper()] = '#808080'
                    colors.append(color_dic[let.upper()])

            crosstb.plot(kind="bar", stacked=True, rot=0, ax=ax, color=colors, width=.3)
            # ax.title.set_text(cl + ', P-value of Chi-square test: ' + str(round(p, 3)))
            ax.set_title(cl + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize=8)
            plt.xlabel('')
            plt.xticks(fontsize=8, rotation=90)
            plt.ylabel('Counts', fontsize=8)
            plt.yticks(fontsize=6)
            plt.legend(title=None, fontsize=6)
            plt.savefig(str(report_dir + '/' + model_name + 'top_positions' + str(350) + '.pdf'), bbox_inches='tight')
    return print(model_name, ' Done')


# All significant positions - individual box/bar plots
def plot_imp_all(trained_models, X_train, y_train,
                 meta_var, model_type, report_dir, max_plots=100,
                 figsize=(3, 3)):
    """
    plots all the important position across all the top selected models (up to `max_plots`).
    Parameters
    ----------
    trained_models : dict
        a nested dictionary. The first layer of the keys are the name of the models and each of them
        is a dictionary with keys ['metrics', 'importance', 'model']. For example, evaluation values can be accessed
        as: `dict['model']['metrics']`
    X_train : pandas.DataFrame
        a numeric dataframe
    y_train : 1D array
        an 1D array of values of the response variable (phenotype)
    meta_var : str
        the name of the feature under study (phenotype)
    model_type :  str
        'reg' for regression and 'cl' for classification
    report_dir : str
        path to directory to save the plots
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

    for key in trained_models.keys():
        if key != 'mean':
            tmp2 = pd.DataFrame.from_dict(trained_models[key]['importance'])
            temp = tmp2.loc[:, ['feature', 'standard_value']]
            features = temp.sort_values(by='standard_value', ascending=False)['feature'].tolist()
            features = ['p' + str(f) for f in features]

            p = 0
            cn_f = 0  # feature counter
            check = 0

            while (p < 0.05 or check < 10) and (cn_f <= len(features)) and (cn_p < max_plots):
                cl = features[cn_f]
                cn_f += 1
                cn_p += 1
                if cl not in feature_list:
                    if model_type == 'reg':
                        try:
                            k, p = stats.kruskal(*[group[meta_var].values for name, group in dat.groupby(cl)])
                            if p < 0.05:
                                feature_list.append(cl)
                                color_dic = {}
                                key_list = ['A', 'C', 'G', 'R',
                                            'T', 'N', 'D', 'E',
                                            'Q', 'H', 'I', 'L',
                                            'K', 'M', 'F', 'P',
                                            'S', 'W', 'Y', 'V', 'GAP']
                                color_list = ['#0273b3', '#de8f07', '#029e73', '#d55e00',
                                              '#cc78bc', '#ca9161', '#fbafe4', '#ece133',
                                              '#56b4e9', '#bcbd21', '#aec7e8', '#ff7f0e',
                                              '#ffbb78', '#98df8a', '#d62728', '#ff9896',
                                              '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                                              '#dbdb8d']  # sns.color_palette('husl', 21)
                                for n, let in enumerate(key_list):
                                    color_dic[let] = color_list[n]
                                color_dic['U'] = color_dic['T']

                                # add gray to pallet for mixed combinations
                                for let in set(dat.loc[:, cl]):
                                    if let not in color_dic:
                                        color_dic[let] = '#808080'  # hex code for gray color

                                fig, ax = plt.subplots(figsize=figsize, dpi=350)
                                ax.grid(color='gray', linestyle='-', linewidth=0.2, axis='y')
                                ax.set_axisbelow(True)
                                ax = sns.boxplot(x=cl, y=meta_var, data=dat, showfliers=False,
                                                 width=.6, linewidth=.6, palette=color_dic)
                                sns.despine(ax=ax)
                                ax = sns.stripplot(x=cl, y=meta_var, data=dat, size=3,
                                                   alpha=0.3, linewidth=0.3, palette=color_dic)

                                ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize=8)
                                ax.set_xlabel('')

                                plt.savefig(str(plot_dir + '/' + cl + '_boxplot_' + str(350) + '.pdf'),
                                            bbox_inches='tight')
                                plots[cl] = fig, ax
                        except:
                            pass

                    else:
                        try:
                            np.random.seed(123)
                            color_dic = {}
                            key_list = ['A', 'C', 'G', 'R',
                                        'T', 'N', 'D', 'E',
                                        'Q', 'H', 'I', 'L',
                                        'K', 'M', 'F', 'P',
                                        'S', 'W', 'Y', 'V', 'GAP']
                            color_list = ['#0273b3', '#de8f07', '#029e73', '#d55e00',
                                          '#cc78bc', '#ca9161', '#fbafe4', '#ece133',
                                          '#56b4e9', '#bcbd21', '#aec7e8', '#ff7f0e',
                                          '#ffbb78', '#98df8a', '#d62728', '#ff9896',
                                          '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                                          '#dbdb8d']  # sns.color_palette('husl', 21)
                            for n, let in enumerate(key_list):
                                color_dic[let] = color_list[n]
                            color_dic['U'] = color_dic['T']
                            # Creating crosstab
                            crosstb = pd.crosstab(dat[meta_var], dat[cl])

                            # chi-square test
                            chi2, p, dof, expected = stats.chi2_contingency(crosstb)

                            if p < 0.05:
                                feature_list.append(cl)
                                fig, ax = plt.subplots(figsize=figsize, dpi=350)
                                colors = []
                                for let in crosstb.columns.tolist():
                                    if let in color_dic.keys():
                                        colors.append(color_dic[let.upper()])
                                    else:
                                        color_dic[let.upper()] = '#808080'  # hex code for gray color
                                        colors.append(color_dic[let.upper()])
                                crosstb.plot(kind="bar", stacked=True, rot=0, ax=ax, color=colors, width=.3)
                                ax.set_title(cl + ', P-value: ' + str(round(p, 3)), fontsize=8)
                                ax.set_xlabel('')
                                plt.xticks(fontsize=8, rotation=90)
                                plt.ylabel('Counts', fontsize=8)
                                plt.yticks(fontsize=6)
                                plt.legend(title=None, fontsize=6)
                                plt.tight_layout()

                                plt.savefig(str(plot_dir + '/' + cl + '_stackedbarplot_' + str(350) + '.pdf'),
                                            bbox_inches='tight')
                                plots[cl] = fig, ax
                        except:
                            pass
                if p >= 0.05:
                    check += 1
    with open(str(plot_dir + '/plots.pickle'), 'wb') as handle:
        pickle.dump(plots, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return plots
