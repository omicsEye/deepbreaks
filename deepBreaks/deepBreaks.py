#importing libraries

import os
import datetime
import argparse
import sys
import warnings
from zipfile import ZipFile
from preprocessing import *
from models import *
from visualization import *


warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqfile', '-sf', help="files contains the sequences", type= str, required=True)
    parser.add_argument('--seqtype', '-st', help="type of sequence: nuc, amino-acid", type= str, required=True, )
    parser.add_argument('--metadata', '-md', help="files contains the meta data", type= str, required=True)
    parser.add_argument('--metavar', '-mv', help="name of the meta variable (response variable)", type= str, required=True)
    ####parser.add_argument('--key', '-k', help="name of the key column", type= str, required=True)
    parser.add_argument('--anatype', '-a', help="type of analysis", choices=['reg', 'cl'], type= str, required=True)
    parser.add_argument('--fraction', '-fr', help="fraction of main data to run", type= float, required=False)

    return parser.parse_args()

args = parse_arguments()
print(len(vars(args).values()))
print(args)  # Namespace(bar=0, foo='pouet')
print(args.seqfile) 
print(args.metadata)
print(args.anatype)

if args.seqtype not in ['nu', 'amino-acid']:
    print('For sequence data type, please enter "nu" or "amino-acid" only')
    exit()
# making directory
print('direcory preparation')
dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
sqeqFileName = args.seqfile.split('.')[0]

report_dir = str(sqeqFileName +'_' + args.metavar + '_' + dt_label)
os.makedirs(report_dir)


print('reading meta-data')
# metaData = pd.read_csv(args.metadata, sep='\t', index_col=0)
metaData = read_data(args.metadata, seq_type = None, is_main=False)
print('metaData:', metaData.shape)

#importing seq data
print('reading fasta file')
df = read_data(args.seqfile, seq_type = args.seqtype, is_main=True)

positions = df.shape[1]
print('Done')
print('Shape of data is: ', df.shape)


#selecting only more frequent classes
if args.anatype == 'cl':
    df = balanced_classes(dat = df, meta_dat = metaData, feature = args.metavar)

# df = df.merge(metaData[args.metavar], left_index=True, right_index=True)
# print(df[args.metavar].value_counts())
# exit()

#taking care of missing data
print('Shape of data before missing/constant care: ', df.shape)
df_cleaned = missing_constant_care(df)
print('Shape of data after missing/constant care: ', df_cleaned.shape)

print('Shape of data before imbalanced care: ', df_cleaned.shape)
df_cleaned = imb_care(dat=df_cleaned, imbalance_treshold=0.025)
print('Shape of data after imbalanced care: ', df_cleaned.shape)


if args.fraction is not None:
    print('number of columns of main data befor: ', df_cleaned.shape[1])
    df_cleaned = col_sampler(dat=df_cleaned, sample_frac=args.fraction)
    print('number of columns of main data after: ', df_cleaned.shape[1])


print('correlation analysis')
cr = cor_cal(df_cleaned) 
print(cr.shape)


print('grouping features')
dc = group_features(cr)

dc_df = group_extend(dc)

print('dropping correlated features')
print('Shape of data before colinearity care: ', df_cleaned.shape)
df_cleaned = cor_remove(df_cleaned, dc)
print('Shape of data after colinearity care: ', df_cleaned.shape)

#merge with meta data
df = df.merge(metaData[args.metavar], left_index=True, right_index=True)
df_cleaned = df_cleaned.merge(metaData[args.metavar], left_index=True, right_index=True)


#model
print('preparing env')
if args.anatype == 'reg':
    
    from pycaret.regression import *

    exp_reg101 = setup(data = df_cleaned, target = str(args.metavar), silent = True,
                    session_id=123,
                    ignore_low_variance=True,
                    feature_selection=True
                    # , remove_multicollinearity=True
                    ) 
else:

    from pycaret.classification import *
    exp_reg101 = setup(data = df_cleaned, target = str(args.metavar), silent = True,
                   session_id=123,
                   feature_selection= True,
                   ignore_low_variance=True)


# number of top models and metric to choose the model

print('model training')
modelsToSelect = 5
if args.anatype == 'reg':
    metric = 'MAE'
    top_models = compare_models(n_select=modelsToSelect, sort=metric, verbose=False, 
                                include = ['lr', 'ridge', 'lar', 'br', 'par', 'huber', 'lasso', 'et', 'lr', 'xgboost', 'lightgbm', 'rf', 'gbc', 'dt', 'ada'])
    
else:
    metric = 'F1'
    top_models = compare_models(n_select=modelsToSelect, sort=metric, verbose=False, 
                                include = ['et', 'lr', 'xgboost', 'lightgbm', 'rf', 'gbc', 'dt', 'ada', 'ridge', 'svm'])

results_df = pull()
results_df.to_csv(str(report_dir + '/models_summary'+'.csv'))
print('report is created')


print('making plots')
for i in range(modelsToSelect):
    dp_plot(model = top_models[i], n_positions= positions, metric=metric, xlab =  "Amino acid positions")


########
# Overal Plot #
######## 
cn=0
cl = []
fileNames = os.listdir(report_dir)
for file in fileNames:
    if file.endswith('feature_importance.csv'):
        cn += 1
        temp = pd.read_csv(str(report_dir+ '/' + file), index_col=0)
        temp = temp[['feature', 'standard_value']]
        cl.append(file.split('_')[0])
        temp.rename(columns={"standard_value": cl[-1]}, inplace=True)
        if cn == 1:
            final_importance = temp
        else:
            final_importance = final_importance.merge(temp)

final_importance['mean_imp'] = final_importance[cl].mean(axis = 1)

fname = str(report_dir + '/feature_importance_mean_top_models.csv')
    
if os.path.isfile(fname):
    pass
else:
    final_importance.to_csv(fname)

final_importance = final_importance.merge(dc_df, how = 'left')
final_importance['group'].fillna('No_gr', inplace = True)

final_importance['color'] = None

for name, group in final_importance.groupby('group'):

    if name != 'No_gr':
        final_importance.loc[final_importance['group']==name, 'color'] = '#%06X' % random.randint(0, 0xFFFFFF)
    else:
        final_importance.loc[final_importance['group']==name, 'color'] = '#000000'
            
pl_title = str("Important Positions - Avg top models" )
    
plt.figure(figsize= (7.2,3), dpi = 350)
plt.vlines(x=final_importance['feature'], ymin=0, ymax=final_importance['mean_imp'], color=final_importance['color'], linewidth=.7, alpha=0.8)
# plt.scatter(x = final_importance['feature'], y = final_importance['mean_imp'], color=my_color, s=my_size, alpha=1)

# # Atmp2 title and axis names
plt.title(pl_title, loc='center', fontsize = 10)
# plt.xlabel('Base Pair Positions', fontsize = 8)amino acid positions
plt.xlabel('Amino acid positions', fontsize = 8)
plt.xticks(fontsize = 6)
plt.ylabel('Relative Importance', fontsize = 8)
plt.ylim(0, 1.1)
plt.yticks(fontsize = 6)

black_patch = mpatches.Patch(color='black', label='No group', linewidth=0.5)
plt.legend(handles=[black_patch], loc = 'upper left', fontsize = 6)

plt.grid(True, linewidth=.3)
    

features = final_importance.sort_values(by='mean_imp', ascending=False).head(4)['feature'].tolist()
texts = []
xtext_list = []
ytext_list = []
for n, ft in enumerate(features):

    x = int(ft)
    y = final_importance.loc[final_importance['feature']==ft, 'mean_imp'].tolist()[0]*1.05
    text = str(ft)
    
    if n>0:
        xtext, ytext =  best_position(x, y+.1, xlist = xtext_list, ylist= ytext_list)
    else:
        xtext = x
        ytext = y+.1
    
    if y == 1:
        ytext = y + .05
        xtext = x*1.1

    xtext_list.append(xtext)
    ytext_list.append(ytext)

    plt.annotate(text = x , 
            xy = (x, y),
            xytext = (xtext, ytext),fontsize=6,
            arrowprops = dict(arrowstyle= '->',
                            color='green',
                            lw=1,
                            ls='-'))

#save
plt.savefig(str(report_dir + '/' + 'feature_importance_mean_top_models_'+ str(350) +'.png'), bbox_inches='tight')


#feature importance of the mean values of top three models
features = final_importance.sort_values(by='mean_imp', ascending=False).head(4)['feature'].tolist()
features = ['p' + str(f) for f in features]

if args.anatype == 'reg':

    fig,ax = plt.subplots(figsize=(7.5, 7.5), dpi = 350)
    fig.suptitle(args.metavar + ' VS important positions', fontsize=10)
    for nm , cl in enumerate(features):
        k, p = stats.kruskal(*[group[args.metavar].values for name, group in df.groupby(cl)])

        ax = plt.subplot(2,2,nm+1)
        # sns.violinplot(x=cl, y= args.metavar, data=df)
        ax = sns.boxplot(x=cl, y= args.metavar, data=df, linewidth=1)
        ax = sns.stripplot(x=cl, y= args.metavar, data=df, size = 5, alpha = 0.3, linewidth=1)
        
        ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize = 8)
        ax.set_xlabel('')

    plt.savefig(str(report_dir + '/' + 'positions_box_'+ str(350) +'.png'), bbox_inches='tight')

else:
    color_dic = {'A':'#3837f7', 'C': '#f60002', 'G': '#009901', 'T': '#fed000'}

    plt.figure(figsize=(7.5, 7.5), dpi = 350)
    plt.suptitle(args.metavar + ' VS important positions', fontsize=10)
    plt.subplots_adjust(hspace=0.5, wspace = 0.5)
    for nm , cl in enumerate(features):
        ax = plt.subplot(2,2,nm+1)
        # Creating crosstab
        crosstb = pd.crosstab(df[args.metavar], df[cl])

        #chi-square test    
        chi2, p, dof, expected = stats.chi2_contingency(crosstb)

        colors = [color_dic[let.upper()] for let in crosstb.columns.tolist()]
        crosstb.plot(kind="bar", stacked=True, rot=0 , ax = ax, color=colors)
        # ax.title.set_text(cl + ', P-value of Chi-square test: ' + str(round(p, 3)))
        ax.set_title(cl + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize = 8)
        plt.xlabel('')
        plt.xticks(fontsize = 6, rotation=90)
        plt.ylabel('Counts', fontsize = 8)
        plt.yticks(fontsize = 6)
        plt.legend(title = None)
    plt.savefig(str(report_dir + '/' + 'positions_stacked_bar_'+ str(350) +'.png'), bbox_inches='tight')

### testing feature importance of each model 
fileNames = os.listdir(report_dir)
for file in fileNames:
    if file.endswith('feature_importance.csv'):
        model_name = file.split('_')[0]
        temp = pd.read_csv(str(report_dir+ '/' + file), index_col=0)
        temp = temp[['feature', 'standard_value']]
        features = temp.sort_values(by='standard_value', ascending=False).head(4)['feature'].tolist()
        features = ['p' + str(f) for f in features]

        if args.anatype == 'reg':

            fig,ax = plt.subplots(figsize=(7.5, 7.5), dpi = 350)
            fig.suptitle(args.metavar + ' VS important positions ('+ model_name+ ')', fontsize=10)
            for nm , cl in enumerate(features):
                k, p = stats.kruskal(*[group[args.metavar].values for name, group in df.groupby(cl)])

                ax = plt.subplot(2,2,nm+1)
                # sns.violinplot(x=cl, y= args.metavar, data=df)
                ax = sns.boxplot(x=cl, y= args.metavar, data=df, linewidth=1)
                ax = sns.stripplot(x=cl, y= args.metavar, data=df, size = 5, alpha = 0.3, linewidth=1)
                
                ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize = 8)
                ax.set_xlabel('')

            plt.savefig(str(report_dir + '/' + model_name +'_positions_box_'+ str(350) +'.png'), bbox_inches='tight')

        else:
            color_dic = {'A':'#3837f7', 'C': '#f60002', 'G': '#009901', 'T': '#fed000'}

            plt.figure(figsize=(7.5, 7.5), dpi = 350)
            plt.suptitle(args.metavar + ' VS important positions ('+ model_name+ ')', fontsize=10)
            plt.subplots_adjust(hspace=0.5, wspace = 0.5)
            for nm , cl in enumerate(features):
                ax = plt.subplot(2,2,nm+1)
                # Creating crosstab
                crosstb = pd.crosstab(df[args.metavar], df[cl])

                #chi-square test    
                chi2, p, dof, expected = stats.chi2_contingency(crosstb)

                colors = [color_dic[let.upper()] for let in crosstb.columns.tolist()]
                crosstb.plot(kind="bar", stacked=True, rot=0 , ax = ax, color=colors)
                # ax.title.set_text(cl + ', P-value of Chi-square test: ' + str(round(p, 3)))
                ax.set_title(cl + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize = 8)
                plt.xlabel('')
                plt.xticks(fontsize = 6, rotation=90)
                plt.ylabel('Counts', fontsize = 8)
                plt.yticks(fontsize = 6)
                plt.legend(title = None)
            plt.savefig(str(report_dir + '/' + model_name +'_positions_box_'+ str(350) +'.png'), bbox_inches='tight')

print('\n\n\t plots for significant features')
### testing feature importance of each model (all features - top to bottom and only report significant ones)
plot_dir = str(report_dir + '/significant_positions_plots')
os.makedirs(plot_dir)
fileNames = os.listdir(report_dir)
for file in fileNames:
    if file.endswith('feature_importance.csv'):
        model_name = file.split('_')[0]
        temp = pd.read_csv(str(report_dir+ '/' + file), index_col=0)
        temp = temp[['feature', 'standard_value']]
        features = temp.sort_values(by='standard_value', ascending=False)['feature'].tolist()
        features = ['p' + str(f) for f in features]

        p = 0
        cn = 0
        feature_list = []
        while p < 0.1:
            cl = features[cn]
            cn += 1
            if cl not in feature_list:
                if args.anatype == 'reg':
                    try:
                        k, p = stats.kruskal(*[group[args.metavar].values for name, group in df.groupby(cl)])
                        if p < 0.05:
                            feature_list.append(cl)
                            fig,ax = plt.subplots(figsize=(7.5, 7.5), dpi = 350)
                            ax = sns.boxplot(x=cl, y= args.metavar, data=df, linewidth=1)
                            ax = sns.stripplot(x=cl, y= args.metavar, data=df, size = 5, alpha = 0.3, linewidth=1)

                            ax.set_title(cl + ', P-value of KW test: ' + str(round(p, 3)), fontsize = 8)
                            ax.set_xlabel('')
                            plt.savefig(str(plot_dir + '/' + cl +'_boxplot_'+ str(350) +'.png'), bbox_inches='tight')
                    except:
                        pass

                else:
                    try:
                        color_dic = {'A':'#3837f7', 'C': '#f60002', 'G': '#009901', 'T': '#fed000'}
                        # Creating crosstab
                        crosstb = pd.crosstab(df[args.metavar], df[cl])
                        #chi-square test    
                        chi2, p, dof, expected = stats.chi2_contingency(crosstb)

                        if p < 0.05:
                            feature_list.append(cl)
                            fig,ax = plt.subplots(figsize=(7.5, 7.5), dpi = 350)
                            colors = [color_dic[let.upper()] for let in crosstb.columns.tolist()]
                            crosstb.plot(kind="bar", stacked=True, rot=0 , ax = ax, color=colors)
                            ax.set_title(cl + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize = 8)
                            ax.set_xlabel('')
                            plt.xticks(fontsize = 6, rotation=90)
                            plt.ylabel('Counts', fontsize = 8)
                            plt.yticks(fontsize = 6)
                            # ax.set_xlabel('')
                            # ax.set_xticklabels(ax.get_xticks(), rotation = 90, fontsize=6)
                            # ax.set_ylabel('Counts', fontsize = 8)
                            # ax.set_yticklabels(ax.get_yticks(), fontsize=6)
                            # ax.legend(title = None)
                            plt.savefig(str(plot_dir + '/' + cl +'_stackedbarplot_'+ str(350) +'.png'), bbox_inches='tight')
                    except:
                        pass

###groups of correlated variables
dc = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dc.items() ]))
dc.to_csv(str(report_dir + '/' + 'correlated_positions.csv'), index = False)

###saving correlation matrix
cr.to_csv(str(report_dir + '/' + 'correlation_matrix.csv'), index = False)
# create a zip file from the files in report
zipObj = ZipFile(str(report_dir +'/report.zip'), 'w')
fileNames = os.listdir(report_dir)
for file in fileNames:
    if not file.endswith('.zip'):
        zipObj.write(str(report_dir + '/' + file))