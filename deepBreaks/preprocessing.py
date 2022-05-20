from Bio import SeqIO
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency
from tqdm import tqdm


# sample size and classes check
def check_data(meta_dat, feature, model_type):
    if len(meta_dat) < 30:
        raise Exception('Minimum sample size for a regression analysis is 30 samples, '
                        'you provided {} samples!'.format(len(meta_dat)))
    elif model_type == 'cl':
        vl = meta_dat[feature].value_counts() / meta_dat.shape[0]
        if len(vl) == 1:
            raise Exception(
                'Minimum categories for a classification analysis is 2 classes, you provided {}'.format(len(vl)))
        elif vl[1] < 0.25:
            raise Exception('Provided sample is highly im-balanced, we need more data for the minority group.')
        else:
            message = "let's start the analysis"
    else:
        message = "let's start the analysis"

    return print(message)


# read data function
def read_data(file_path, seq_type=None, is_main=True):
    #
    if file_path.endswith('.csv'):
        dat = pd.read_csv(file_path, sep=',', index_col=0)
    elif file_path.endswith('.tsv'):
        dat = pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        dat = pd.read_excel(file_path, sep='\t', index_col=0)
    elif file_path.endswith('.fasta'):
        # importing seq data
        seq_dict = {rec.id: list(rec.seq) for rec in SeqIO.parse(file_path, "fasta")}
        dat = pd.DataFrame.from_dict(seq_dict, orient='index')
    else:
        print('For now, we can only read csv, tsv, excel, and fasta files.')
        exit()

    if is_main:
        # naming each position as p + its rank
        dat.columns = [str('p' + str(i)) for i in range(1, dat.shape[1] + 1)]
        # replacing unwanted characters with nan
        if seq_type == 'nu':
            na_values = ['-', 'r', 'y', 'k', 'm', 's', 'w', 'b', 'd', 'h', 'v', 'n']
        else:
            na_values = ['-', 'X', 'B', 'Z', 'J']
        to_replace = []
        for vl in na_values:
            to_replace.append(vl.upper())
            to_replace.append(vl.lower())
        dat.replace(to_replace, np.nan, inplace=True)
    return dat


# use top categories only:
def balanced_classes(dat, meta_dat, feature):
    tmp = dat.merge(meta_dat[feature], left_index=True, right_index=True)
    vl = tmp[feature].value_counts() / tmp.shape[0]
    categories_to_keep = vl[vl > 1 / len(vl) / 2].index.tolist()
    tmp = tmp[tmp[feature].isin(categories_to_keep)]
    dat = tmp.drop(feature, axis=1)
    return dat


# taking care of missing and constant columns
def missing_constant_care(dat, missing_threshold=0.05):
    tmp = dat.copy()
    threshold = tmp.shape[0] * missing_threshold
    cl = tmp.columns[tmp.isna().sum() > threshold]
    tmp.drop(cl, axis=1, inplace=True)
    for cl in tmp.columns:
        tmp[cl] = tmp[cl].fillna(tmp[cl].mode()[0])
    tmp = tmp.loc[:, tmp.nunique() != 1]
    return tmp

# mode = tmp.filter(tmp.columns).mode().iloc[0]  # mode of all the columns
# tmp = tmp.fillna(mode)  # replacing NA values with the mode of each column
# a function which check if one character in a certain column is below a threshold or not
# and replace that character with the mode of that column of merge the rare cases together

def colCleaner_column(dat, column, threshold=0.015):
    vl = dat[column].value_counts()
    ls = vl / dat.shape[0] < threshold
    ls = ls.index[ls == True].tolist()

    if len(ls) > 2:
        dat[column].replace(ls, ''.join(ls), inplace=True)
        ls = vl / dat.shape[0] < threshold
        ls = ls.index[ls == True].tolist()

    if len(ls) > 0:
        md = vl.index[vl == max(vl)][0]
        dat[column].replace(ls, md, inplace=True)

    return dat[column]


# taking care of rare cases in columns
def imb_care(dat, imbalance_threshold=0.05):
    tmp = dat.copy()
    n_uniq = tmp.nunique()
    cols_to_check = n_uniq.index[n_uniq > 1].tolist()
    for cl in cols_to_check:
        tmp[cl] = colCleaner_column(dat=tmp, column=cl, threshold=imbalance_threshold)
    n_uniq = tmp.nunique()
    cl_uniq = n_uniq.index[n_uniq == 1].tolist()
    if len(cl_uniq) > 0:
        tmp.drop(cl_uniq, axis=1, inplace=True)
    return tmp


# function to sample from features 
def col_sampler(dat, sample_frac=1):
    if sample_frac < 1:
        samples = int(dat.shape[1] * sample_frac)
        cl = np.random.choice(dat.columns, samples, replace=False).tolist()
        # remove the columns with the n
        dat = dat[cl]
    return dat


# function to calculate Cramer's V score
def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))  # Cross table building
    stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab)  # Number of observations
    mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
    return (stat / (obs * mini))


# function for creating correlation data frame and just report those which are above a treashold
def cor_cal(dat, report_dir, threshold=0.8):
    print('expected_calculations: ', dat.shape[1] * (dat.shape[1] - 1) / 2)
    cn = 0
    cr = pd.DataFrame(columns=['l0', 'l1', 'cor'])
    col_check = []
    for cl in combinations(dat.columns, 2):  # all paired combinations of df columns
        cn += 1
        if cn % 5000 == 0:  # show each 5000 steps
            print(cn)
        cv = cramers_V(dat[cl[0]], dat[cl[1]])
        cr.loc[len(cr)] = [cl[0], cl[1], cv]
        cr.loc[len(cr)] = [cl[1], cl[0], cv]

    cr.to_csv(str(report_dir + '/' + 'correlation_matrix.csv'), index=False)

    return cr  # [cr['cor'] >= threshold]


# a function to calculate adjusted mutual information for all the paired combinations of the dat dataset
def dist_cols(dat, score_func):
    # creating empty dataFrame to keep the results
    cn = 0
    cr = pd.DataFrame(columns=['l0', 'l1', 'cor'])

    for cl in tqdm(combinations(dat.columns, 2),
                   total=(dat.shape[1] * (dat.shape[1] - 1)) / 2):  # all paired combinations of dat columns

        score = score_func(dat[cl[0]], dat[cl[1]])
        cr.loc[len(cr)] = [cl[0], cl[1], score]
        cr.loc[len(cr)] = [cl[1], cl[0], score]

    cr.loc[cr['cor'] < 0, 'cor'] = 0
    return cr


# create dummy vars, drop those which are below a certain threshold, then calculate the similarity between these remaining columns
def ham_dist(dat, threshold=0.2):
    import gc

    # make dummy vars and keep all of them
    tmp1 = pd.get_dummies(dat, drop_first=False)

    # drop rare cases
    cl = tmp1.columns[(tmp1.sum() / dat.shape[0] > threshold)]
    tmp1 = tmp1[cl]

    # creating empty dataFrame to keep the results
    r_nam = tmp1.columns.tolist()
    dis = pd.DataFrame(columns=r_nam, index=r_nam)

    tmp1 = np.array(tmp1).T

    last = range(tmp1.shape[0])
    for i in tqdm(last):
        dist_vec = (tmp1[0, :] == tmp1).sum(axis=1)

        dis.iloc[i, i:] = dist_vec
        dis.iloc[i:, i] = dist_vec
        tmp1 = np.delete(tmp1, 0, 0)
        if i % 1000 == 0:
            gc.collect()
    gc.collect()
    gc.collect()
    dis = dis / dat.shape[0]
    print('Reset indexes')
    dis.reset_index(inplace=True)
    print('reshaping')
    dis = dis.melt(id_vars='index')
    dis['index'] = dis['index'].str.split('_').str[0]
    dis['variable'] = dis['variable'].str.split('_').str[0]
    gc.collect()
    gc.collect()
    dis = dis[dis['index'] != dis['variable']]
    print('Selecting max values')
    dis = dis.groupby(['index', 'variable']).max().reset_index()
    dis.columns = ['l0', 'l1', 'cor']
    return dis


# calculate vectorized normalized mutual information
def vec_nmi(dat):
    # get the sample size
    N = dat.shape[0]
    # create empty dataframe
    dat_temp = pd.DataFrame(columns=dat.columns, index=dat.columns)

    # transpose main dataframe
    df_dum = pd.get_dummies(dat).T.reset_index()
    df_dum[['position', 'char']] = df_dum['index'].str.split("_", expand=True)
    df_dum.drop(['index'], axis=1, inplace=True)

    # total samples within each position with specific character
    col_sum = df_dum.iloc[:, :-2].sum(axis=1)

    # array of all data
    my_array = np.array(df_dum.iloc[:, :-2])

    for name, gr in df_dum.groupby(['position']):
        mu_list = []
        char_list = list(set(df_dum.loc[df_dum['position'] == name, 'char']))
        temp = df_dum[['position', 'char']]

        for ch in char_list:
            temp['inter' + ch] = None
            temp['ui_vi'] = None
            temp['mui' + ch] = None

            intersects = my_array[(df_dum['position'] == name) & (df_dum['char'] == ch)]
            intersects = intersects + my_array
            intersects = intersects == 2

            temp['inter' + ch] = intersects.sum(axis=1)

            temp['ui_vi'] = temp.loc[(temp['position'] == name) & (df_dum['char'] == ch),
                                     'inter' + ch].values * col_sum
            temp['mui' + ch] = (temp['inter' + ch] / N) * (np.log(N * (temp['inter' + ch]) / temp['ui_vi']))

            mu_list.append('mui' + ch)

        # sum over all entropies
        temp = temp.groupby('position')[mu_list].sum().sum(axis=1)

        # insert into dataframe
        dat_temp[name] = temp

    dat_temp.fillna(0, inplace=True)

    # calculate average entropies
    entropies = np.diag(dat_temp)
    m_entropies = np.tile(entropies, reps=[len(entropies), 1])
    avg_entropies = (entropies.reshape(len(entropies), 1) + m_entropies) / 2

    # calculate normalized mutual information
    dat_temp = dat_temp / avg_entropies
    dat_temp.fillna(0, inplace=True)

    return dat_temp


# grouping features based on DBSCAN clustering algo
def db_grouped(dat, report_dir, threshold=0.8, needs_pivot=False):
    from sklearn.cluster import DBSCAN

    if needs_pivot:
        cr_mat = dat.pivot(index='l0', columns='l1', values='cor')
        cr_mat.fillna(1, inplace=True)
    else:
        cr_mat = dat
    cr_mat = (cr_mat - 1) ** 2

    db = DBSCAN(eps=(threshold - 1) ** 2, min_samples=2, metric='precomputed', n_jobs=-1)
    db.fit(cr_mat)

    dc_df = pd.DataFrame(cr_mat.index.tolist(), columns=['feature'])
    dc_df['group'] = db.labels_

    clusters = list(set(db.labels_))
    for cluster in clusters:
        if cluster == -1:
            dc_df.loc[dc_df['group'] == -1, 'group'] = 'No_gr'
        else:
            dc_df.loc[dc_df['group'] == cluster, 'group'] = 'g' + str(cluster)
    try:
        dc_df = dc_df[dc_df['group'] != 'No_gr']
    except:
        dc_df = pd.DataFrame(columns=['feature', 'group'])

    dc_df.to_csv(str(report_dir + '/' + 'correlated_positions_DBSCAN.csv'), index=False)
    return dc_df


def col_extract(dat, cl1='l0', cl2='l1'):
    ft_list = dat[cl1].tolist()
    ft_list.extend(dat[cl2].tolist())
    ft_list = list(set(ft_list))
    return ft_list


# function for grouping features in saving them in a dictionary file
def group_features(dat, report_dir):
    dc = {}
    if len(dat) > 0:
        for name, gr in dat.groupby('group'):
            tmp = dat.loc[dat['group'] == name, 'feature'].tolist()
            dc[tmp[0]] = tmp[1:]

        dc_temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dc.items()]))
        dc_temp.to_csv(str(report_dir + '/' + 'correlated_positions.csv'), index=False)
    return dc


# function that gets the grouped dictionary and returns a dataframe of grouped features
def group_extend(dic):
    dc_df = pd.DataFrame(columns=['feature', 'group'])
    for n, k in enumerate(dic.keys()):
        ft = dic[k].copy()
        ft.append(k)
        col = [str('g' + str(n))] * len(ft)
        tmp = pd.DataFrame(list(zip(ft, col)), columns=['feature', 'group'])
        dc_df = dc_df.append(tmp, ignore_index=True)
    dc_df['feature'] = dc_df['feature'].str.split('p').str[1]
    dc_df['feature'] = dc_df['feature'].astype(int)
    return dc_df


# function for removing the correlated features from the main dataframe
def cor_remove(dat, dic):
    for k in dic.keys():
        dat.drop(dic[k], axis=1, inplace=True)
    return dat
