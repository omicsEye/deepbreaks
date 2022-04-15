import numpy as np
import pandas as pd
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from models import fimp_single


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
def dp_plot(dat, imp_col, model_name, report_dir,
            figsize=(7.2, 3), dpi=350,
            ylab='Relative Importance',
            xlab='Positions',
            title_fontsize=10,
            xlab_fontsize=8,
            ylab_fontsize=8,
            xtick_fontsize=6,
            ytick_fontsize=4
            ):
    pl_title = str("Important Positions - " + model_name)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.vlines(x=dat['feature'], ymin=0, ymax=dat[imp_col], color=dat['color'], linewidth=.7, alpha=0.8)

    plt.title(pl_title, loc='center', fontsize=title_fontsize)
    plt.xlabel(xlab, fontsize=xlab_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.ylabel(ylab, fontsize=ylab_fontsize)
    plt.ylim(0, 1.1)
    plt.yticks(fontsize=ytick_fontsize)
    plt.grid(True, linewidth=.3)

    black_patch = mpatches.Patch(color='black', label='No group', linewidth=0.1)
    plt.legend(handles=[black_patch], loc='upper left', fontsize=4)

    ##annotating top 4 positions
    features = dat.sort_values(by=imp_col, ascending=False).head(4)['feature'].tolist()
    texts = []
    xtext_list = []
    ytext_list = []
    for n, ft in enumerate(features):
        x = int(ft)
        y = dat.loc[dat['feature'] == ft, imp_col].tolist()[0]
        text = str(ft)
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
                     xytext=(xtext, ytext), fontsize=4,
                     arrowprops=dict(arrowstyle='->',
                                     color='green',
                                     lw=1,
                                     ls='-'))

    plt.savefig(str(report_dir + '/' + model_name + '_' + str(dpi) + '.png'), bbox_inches='tight')
    # show the graph
    #     plt.show()
    return print(str(report_dir + '/' + model_name + '_' + str(dpi) + '.png'))


### top 4 important features for each model
def plot_imp_model(dat, trained_model, model_name, train_cols, grouped_features,
                   meta_var, n_positions, model_type, report_dir):
    tmp2 = fimp_single(trained_model=trained_model, model_name=model_name,
                       train_cols=train_cols, grouped_features=grouped_features,
                       n_positions=n_positions, report_dir=report_dir, write=False)

    temp = tmp2[['feature', 'standard_value']]
    features = temp.sort_values(by='standard_value', ascending=False).head(4)['feature'].tolist()
    features = ['p' + str(f) for f in features]

    if model_type == 'reg':

        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=350)
        fig.suptitle(meta_var + ' VS important positions (' + model_name + ')', fontsize=10)
        for nm, cl in enumerate(features):
            k, p = stats.kruskal(*[group[meta_var].values for name, group in dat.groupby(cl)])

            ax = plt.subplot(2, 2, nm + 1)
            # sns.violinplot(x=cl, y= args.metavar, data=df)
            ax = sns.boxplot(x=cl, y=meta_var, data=dat, linewidth=1)
            ax = sns.stripplot(x=cl, y=meta_var, data=dat, size=5, alpha=0.3, linewidth=1)

            ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize=8)
            ax.set_xlabel('')

        plt.savefig(str(report_dir + '/' + model_name + '_positions_box_' + str(350) + '.png'), bbox_inches='tight')

    else:
        color_dic = {'A': '#3837f7', 'C': '#f60002', 'G': '#009901', 'T': '#fed000'}

        plt.figure(figsize=(7.5, 7.5), dpi=350)
        plt.suptitle(meta_var + ' VS important positions (' + model_name + ')', fontsize=10)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        for nm, cl in enumerate(features):
            ax = plt.subplot(2, 2, nm + 1)
            # Creating crosstab
            crosstb = pd.crosstab(dat[meta_var], dat[cl])

            # chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(crosstb)

            colors = [color_dic[let.upper()] for let in crosstb.columns.tolist()]
            crosstb.plot(kind="bar", stacked=True, rot=0, ax=ax, color=colors)
            # ax.title.set_text(cl + ', P-value of Chi-square test: ' + str(round(p, 3)))
            ax.set_title(cl + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize=8)
            plt.xlabel('')
            plt.xticks(fontsize=6, rotation=90)
            plt.ylabel('Counts', fontsize=8)
            plt.yticks(fontsize=6)
            plt.legend(title=None)
        plt.savefig(str(report_dir + '/' + model_name + '_positions_box_' + str(350) + '.png'), bbox_inches='tight')
    return print(model_name, ' Done')


# All significant positions - individual box/bar plots
def plot_imp_all(trained_models, dat, train_cols, grouped_features, meta_var, model_type, n_positions, report_dir):
    plot_dir = str(report_dir + '/significant_positions_plots')

    if os.path.exists(plot_dir):
        cn = 1
        plot_dir_temp = plot_dir
        while os.path.exists(plot_dir_temp):
            plot_dir_temp = plot_dir + '_' + str(cn)
            cn += 1
        plot_dir = plot_dir_temp

    os.makedirs(plot_dir)
    feature_list = []

    for i in range(len(trained_models)):
        tmp2 = fimp_single(trained_model=trained_models[i], model_name='AAA',
                           train_cols=train_cols, grouped_features=grouped_features,
                           n_positions=n_positions, report_dir=report_dir, write=False)

        temp = tmp2[['feature', 'standard_value']]
        features = temp.sort_values(by='standard_value', ascending=False)['feature'].tolist()
        features = ['p' + str(f) for f in features]

        p = 0
        cn = 0

        while p < 0.1:
            cl = features[cn]
            cn += 1
            if cl not in feature_list:
                if model_type == 'reg':
                    try:
                        k, p = stats.kruskal(*[group[meta_var].values for name, group in dat.groupby(cl)])
                        if p < 0.05:
                            feature_list.append(cl)
                            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=350)
                            ax = sns.boxplot(x=cl, y=meta_var, data=dat, linewidth=1)
                            ax = sns.stripplot(x=cl, y=meta_var, data=dat, size=5, alpha=0.3, linewidth=1)

                            ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize=8)
                            ax.set_xlabel('')
                            plt.savefig(str(plot_dir + '/' + cl + '_boxplot_' + str(350) + '.png'), bbox_inches='tight')
                    except:
                        pass

                else:
                    try:
                        color_dic = {'A': '#3837f7', 'C': '#f60002', 'G': '#009901', 'T': '#fed000'}
                        # Creating crosstab
                        crosstb = pd.crosstab(dat[meta_var], dat[cl])
                        # chi-square test
                        chi2, p, dof, expected = stats.chi2_contingency(crosstb)

                        if p < 0.05:
                            feature_list.append(cl)
                            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=350)
                            colors = [color_dic[let.upper()] for let in crosstb.columns.tolist()]
                            crosstb.plot(kind="bar", stacked=True, rot=0, ax=ax, color=colors)
                            ax.set_title(cl + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize=8)
                            ax.set_xlabel('')
                            plt.xticks(fontsize=6, rotation=90)
                            plt.ylabel('Counts', fontsize=8)
                            plt.yticks(fontsize=6)
                            plt.savefig(str(plot_dir + '/' + cl + '_stackedbarplot_' + str(350) + '.png'),
                                        bbox_inches='tight')
                    except:
                        pass
        print('Model ', i, ' Done')
