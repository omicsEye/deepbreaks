import re
import numpy as np
import pandas
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import csv
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectFdr, chi2, f_regression


# sample size and classes check
def check_data(meta_data, feature, model_type):
    if len(meta_data) < 30:
        raise Exception('Minimum sample size for a regression analysis is 30 samples, '
                        'you provided {} samples!'.format(len(meta_data)))
    elif model_type == 'cl':
        vl = meta_data[feature].value_counts() / meta_data.shape[0]
        if len(vl) == 1:
            raise Exception(
                'Minimum categories for a classification analysis is 2 classes, you provided {}'.format(len(vl)))
        elif vl[1] < 1 / (len(vl) * 2):
            raise Exception('Provided sample is highly im-balanced, we need more data for the minority group.')
        else:
            message = "let's start the analysis"
    else:
        message = "let's start the analysis"

    return print(message)


# read fasta file
def fasta_read(f_name):
    """
    Reads a fasta file and returns a pandas dataframe with rows as IDs and columns as positions in the sequences
    :param f_name: str, path to file
    :return: pandas.DataFrame
    """
    f = open(f_name, 'r')
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    id_re = re.compile(r'>(\S+)')
    seq_re = re.compile(r'^(\S+)$')

    tmp = {}

    for line in lines:
        id_h = id_re.search(line)
        if id_h:
            seq_l = None
            seq_id = id_h.group(1)
        else:
            if seq_l is None:
                seq_l = seq_re.search(line).group(1)
            else:
                seq_l = seq_l + seq_re.search(line).group(1)
            tmp[seq_id] = list(seq_l.upper())
    return pd.DataFrame.from_dict(tmp, orient='index')


# write a pandas data frame to fasta file
def write_fasta(dat, fasta_file, report_dir, wrap=80):
    """
    Writes a Pandas.DataFrame to a fasta file.

    Parameters
    ----------
    dat : importance.loc[seq_id,:] -> seq
        Sequences saved in a pandas dataframe in rows.
    fasta_file : str
        Output FASTA file name.
    report_dir: str
        Path to the report directory.
    wrap: int
        Number of AA/NT before the line is wrapped.
    """
    file_name = report_dir + '//' + fasta_file
    with open(file_name, 'w') as f:
        for ind in dat.index:
            f.write('>{}\n'.format(ind))
            seq = ''.join(dat.loc[ind, :].fillna('-'))
            for i in range(0, len(seq), wrap):
                f.write('{}\n'.format(seq[i:i + wrap]))
        f.close()
    return print(fasta_file + ' was saved successfully')


# read data function
def read_data(file_path, seq_type=None, is_main=True, gap_threshold=0.7) -> pandas.DataFrame:
    """
    Reads data file from tsv, csv, xlsx, xls, fas, fa, and fasta formats and returns a Pandas.DataFrame
    :param file_path: str, path to file
    :param seq_type: str, 'nu' for nucleotides and 'aa' for 'amino-acid' sequences
    :param is_main: bool, True means that this is the MSA file
    :param gap_threshold: float, columns with missing values over the gap_threshold% will be dropped and
     eliminate from the analysis.
    :return: pandas.DataFrame
    """
    if file_path.endswith('.csv'):
        dat = pd.read_csv(file_path, sep=',', index_col=0)
    elif file_path.endswith('.tsv'):
        dat = pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        dat = pd.read_excel(file_path, sep='\t', index_col=0)
    elif any(file_path.endswith(s) for s in ['fasta', 'fas', 'fa']):
        # importing seq data
        dat = fasta_read(f_name=file_path)
    else:
        print('For now, we can only read csv, tsv, excel, and fasta files.')
        exit()

    if is_main:
        # replacing unwanted characters with nan
        if seq_type == 'nu':
            na_values = ['-', 'r', 'y', 'k', 'm', 's', 'w', 'b', 'd', 'h', 'v', 'n']
        else:
            na_values = ['-', 'X', 'B', 'Z', 'J']
        to_replace = []
        for vl in na_values:
            to_replace.append(vl.upper())
        dat.replace(to_replace, np.nan, inplace=True)
        # if gap_threshold > 0:
        #     col_to_drop = dat.columns[dat.isnull().sum() > (gap_threshold * dat.shape[0])]
        #     dat.drop(col_to_drop, axis=1, inplace=True)

        # raise a warning that in this version, we do not drop columns in read_data function
        if gap_threshold is not None:
            print('Columns with missing values over the gap_threshold% will be'
                  ' dropped and eliminate from the analysis during preprocessing.'
                  'Please use the missing_constant_care function to take care of missing values.')
        # naming each position as p + its rank
        dat.columns = [str('p' + str(i)) for i in range(1, dat.shape[1] + 1)]
    return dat


# use top categories only:
def balanced_classes(dat, meta_dat, feature):
    """
    Drops samples from the MSA file that are associated with rare cases. (Used only for classification cases)
        :param dat: pandas.DataFrame, MSA file
        :param meta_dat: pandas.DataFrame, meta-data
        :param feature: str, meta-variable (phenotype)
        :return: pandas.DataFrame


    Example:
    ----------
    >>> print('Shape of data is: ', df.shape)
    Shape of data is:  (1111, 49839)

    # selecting only the classes with enough number of samples
    >>> df = balanced_classes(importance=df, meta_dat=metaData, feature='mt')
    >>> print('Shape of data is: ', df.shape)
    Shape of data is:  (1006, 49839)
    """
    tmp = dat.merge(meta_dat[feature], left_index=True, right_index=True)
    vl = tmp[feature].value_counts() / tmp.shape[0]
    categories_to_keep = vl[vl > 1 / len(vl) / 2].index.tolist()
    tmp = tmp[tmp[feature].isin(categories_to_keep)]
    dat = tmp.drop(feature, axis=1)
    return dat


# taking care of missing and constant columns
def missing_constant_care(dat, missing_threshold=0.05):
    """
    Cleans the columns of a pandas.DataFrame by imputing the missing values. For columns that have missing values
    over the `missing_threshold`, it replaces them with 'GAP' and for columns that have missing values less than
    `missing_threshold`, it uses the `mode` of that column to impute the missing values.

    Parameters
    ----------
    dat : pandas.DataFrame
        MSA file
    missing_threshold : float
        threshold to impute the `GAP`s

    Returns
    -------
    pandas.DataFrame
        data frame with no missing value

    Examples
    --------
    # taking care of missing data
    >>> print('Shape of data before missing/constant care: ', df.shape)
    >>> df = missing_constant_care(df)
    >>> print('Shape of data after missing/constant care: ', df.shape)
    Shape of data before missing/constant care:  (1006, 49839)
    Shape of data after missing/constant care:  (1006, 21705)
    """

    tmp = dat.copy()
    threshold = tmp.shape[0] * missing_threshold
    cl = tmp.columns[tmp.isna().sum() > threshold]
    tmp.loc[:, cl] = tmp.loc[:, cl].fillna('GAP')

    tmp.fillna(tmp.mode().loc[0, :], inplace=True)

    tmp = tmp.loc[:, tmp.nunique() != 1]

    return tmp


# a function which check if one character in a certain column is below a threshold or not
# and replace that character with the mode of that column of merge the rare cases together
def colCleaner_column(dat, column, threshold=0.015):
    vl = dat[column].value_counts()
    ls = vl / dat.shape[0] < threshold
    ls = ls.index[ls == True].tolist()

    if len(ls) > 0:
        if (vl[ls].sum() / dat.shape[0]) < threshold:
            md = vl.index[0]
            dat[column].replace(ls, md, inplace=True)
        else:
            dat[column].replace(ls, ''.join(ls), inplace=True)
    return dat[column]


# taking care of rare cases in columns
def imb_care(dat, imbalance_threshold=0.05):
    tmp = dat.copy()
    for i in range(tmp.shape[1]):
        tmp.iloc[:, i] = colCleaner_column(dat=tmp, column=tmp.columns[i], threshold=imbalance_threshold)
    tmp = tmp.loc[:, tmp.nunique() != 1]
    return tmp


# function to sample from features
def col_sampler(dat, sample_frac=1):
    if sample_frac < 1:
        samples = int(dat.shape[1] * sample_frac)
        cl = np.random.choice(dat.columns, samples, replace=False).tolist()
        # remove the columns with the n
        dat = dat[cl]
    return dat


# statistical tests to drop redundant positions
def redundant_drop(dat, meta_dat, feature, model_type, report_dir, threshold=0.25) -> pandas.DataFrame:
    """
    This function performs statistical tests (chi-square or Kruskal–Wallis) on each column of `importance` against the
    values of `meta_dat['feature']` and based on the p-values and the given threshold, drops redundant positions. A
    report of all calculated p-values will be saved into the given report directory under name of `p_values.csv`.

    Parameters
    ----------
    dat : pandas.DataFrame, MSA file
    meta_dat : pandas.DataFrame, dataframe containing the metadata
    feature : str, the name of the feature under study (phenotype)
    model_type : str, 'reg' for regression and 'cl' for classification
    report_dir : str, path to the directory to write the `p_values.csv` file
    threshold : float, default = 0.25, threshold for p-vaule. columns that have results in
                p-values higher than this will be dropped from the analysis

    Returns
    -------
    pandas.DataFrame, a reduced dataframe with only significant positions

    Example
    --------
    # Use statistical tests to drop redundant features.
    >>> print('number of columns of main data before: ', df.shape[1])
    >>> df_cleaned = redundant_drop(importance=df, meta_dat=metaData,
    ...                                 feature='mt', model_type='reg',
    ...                                 threshold=0.25,
    ...                                 report_dir='.')
    >>> print('number of columns of main data after: ', df_cleaned.shape[1])

    number of columns of main data befor:  194
    number of columns of main data after:  102
    """

    def chisq_test(dat_main, col_list, resp_feature):
        p_val_list = []
        for cl in col_list:
            crs = pd.crosstab(dat_main[cl], dat_main[resp_feature])
            p_val_list.append(chi2_contingency(crs)[1])
        return p_val_list

    def kruskal_test(dat_main, col_list, resp_feature):
        p_val_list = []
        for cl in col_list:
            p_val_list.append(kruskal(*[group[resp_feature].values for name, group in dat_main.groupby(cl)])[1])
        return p_val_list

    tmp = dat.merge(meta_dat[feature], left_index=True, right_index=True)
    cols = dat.columns
    if model_type == 'cl':
        p_val = chisq_test(dat_main=tmp, col_list=cols, resp_feature=feature)
    elif model_type == 'reg':
        p_val = kruskal_test(dat_main=tmp, col_list=cols, resp_feature=feature)
    else:
        raise Exception('Analysis type should be either reg or cl')

    with open(report_dir + "/p_values.csv", "w", newline='') as f:
        writer = csv.writer(f)
        headers = ['position', 'p_value']
        writer.writerow(headers)
        for i in range(len(p_val)):
            content = [cols[i], p_val[i]]
            writer.writerow(content)
    if np.sum(np.array(p_val) < threshold) > 0:
        cols = cols[np.array(p_val) < threshold]
        dat = dat.loc[:, cols]
    else:
        # cols = cols[np.array(p_val) < np.median(np.array(p_val))]
        print('None of the positions meet the p-value threshold')
        dat = None

    return dat


# get dummy variables
def get_dummies(dat, drop_first=True):
    """
    Create dummy variables.
    Parameters
    ----------
    dat : pandas.DataFrame
        MSA file
    drop_first : bool
        True for dropping the first dummy variable

    Returns
    -------
    pandas.DataFrame
        a dataframe with all binary variables instead of strings
    """
    dat = pd.get_dummies(dat, drop_first=drop_first)
    return dat


# calculating distance
def distance_calc(dat, dist_method='correlation', report_dir=None) -> pd.DataFrame:
    """
    calculates the distance matrix for the columns of the provided `importance`.

    Parameters
    ----------
    dat : pandas.DataFrame
        a numeric dataframe
    dist_method : str
        method for calculating the distances. Available options are: 'correlation', 'hamming', 'jaccard',
                   'normalized_mutual_info_score', 'adjusted_mutual_info_score', 'adjusted_rand_score'
    report_dir : str
        path to directory to save the distance matrix

    Returns
    -------
    pandas.DataFrame
        a distance matrix of class pandas.DataFrame

    Example
    --------
    >>> print('calculating the distance matrix')
    >>> cr = distance_calc(importance=df_cleaned,
    ...                        dist_method='correlation',
    ...                        report_dir='.')
    >>> print(cr.shape)

    calculating the distance matrix

    (15252, 15252)

    """
    method_list = ['correlation', 'hamming', 'jaccard',
                   'normalized_mutual_info_score',
                   'adjusted_mutual_info_score', 'adjusted_rand_score']
    err_msg = "Please choose a distance metric \
    that produces values of |dist|<=1. You can choose any these metrics: {}".format(method_list)

    assert dist_method in method_list, err_msg
    cl_names = dat.columns
    if dist_method == 'correlation':
        dist = abs(abs(1 - squareform(pdist(dat.T, metric='correlation'))) - 1)
    elif dist_method in method_list[-3:]:
        exec('from sklearn.metrics import ' + dist_method)
        dist = abs(squareform(pdist(dat.T, eval(dist_method))))
        np.fill_diagonal(dist, 1)
        dist = 1 - dist
    else:
        dist = abs(squareform(pdist(dat.T, metric=dist_method)))

    dist = pd.DataFrame(dist, columns=cl_names, index=cl_names)

    if report_dir is not None:
        dist.to_csv(str(report_dir + '/' + dist_method + '.csv'), index=True)

    return dist


# grouping features based on DBSCAN clustering algo
def db_grouped(dat, report_dir=None, threshold=0.2) -> pandas.DataFrame:
    """
    This function clusters the columns of a dataframe based on a given distance matrix by using the `DBSCAN` algorithm.

    Parameters
    ----------
    dat : pandas.DataFrame
        symmetric distance matrix
    report_dir : str
        path to write the csv file of the clusters.
    threshold : float
        distance threshold for variables to be considered in clusters (eps in the DBSCAN algorithm)

    Returns
    -------
    pandas.DataFrame
        a two column dataframe. first column contains the features and the secod column contains the cluster labels

    Example
    -------
    # calculaing the distance matrix
    >>> cr = distance_calc(importance=df_cleaned,
    ...                dist_method='correlation',
    ...                report_dir='.')
    # clustering
    >>> dc_df = db_grouped(importance = cr, report_dir=report_dir, threshold=.25)
    """
    from sklearn.cluster import DBSCAN

    cr_mat = dat

    db = DBSCAN(eps=threshold, min_samples=2, metric='precomputed', n_jobs=-1)
    db.fit(cr_mat)

    dc_df = pd.DataFrame(cr_mat.index.tolist(), columns=['feature'])
    dc_df['group'] = db.labels_
    dc_df['group'] = dc_df['group'].astype(object)  # Convert to object type

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
    if report_dir is not None:
        dc_df.to_csv(str(report_dir + '/' + 'correlated_positions_DBSCAN.csv'), index=False)
    return dc_df


# function for grouping features in saving them in a dictionary file
def group_features(dat, group_dat, report_dir=None) -> dict:
    """
    Gets a two column dataframe of names and labels. Then for each label, creates a dictionary that the key is the
    representative of the group and the members are the rest of names with the same label. Representative id the closes
    to the center of the cluster

    Parameters
    ----------
    dat : pandas.DataFrame
        numeric data frame
    group_dat : pandas.DataFrame
        a two column dataframe containing names and labels of clusters
    report_dir : str/None
        path to write the dictionary file

    Returns
    -------
    dict
    a dictionary of representatives and the rest of the members of the groups

    """
    dc = {}
    if len(dat) > 0:
        for name, gr in group_dat.groupby('group'):
            tmp = group_dat.loc[group_dat['group'] == name, 'feature'].tolist()
            meds = np.array(dat.loc[:, tmp].median(axis=1)).reshape((-1, 1))
            min_s = ((dat.loc[:, tmp] - meds) ** 2).sum().argmin()

            tmp_ar = [tmp[i] for i in range(len(tmp)) if i != min_s]

            dc[tmp[min_s]] = tmp_ar
        dc_temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dc.items()]))
        if report_dir is not None:
            dc_temp.to_csv(str(report_dir + '/' + 'correlated_positions.csv'), index=False)
    return dc


def cor_remove(dat, dic) -> pandas.DataFrame:
    """
    Reduce the columns of the `importance` by dropping the correlated features and keep only one feature
    per group of correlated features.

    Parameters
    ----------
    dat : pandas.DataFrame
        MSA file
    dic : dict
        dictionary of all groups of correlated features

    Returns
    -------
    pandas.DataFrame
        a reduced `importance` dataframe
     """

    for k in dic.keys():
        dat.drop(dic[k], axis=1, inplace=True)
    return dat


# here _mis_constant_care
class MisCare(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold, gap_threshold=0.7):
        self.columns_gap_drop_ = None
        self.mode_ = None
        self.columns_with_gap_ = None
        self.n_features_in_ = None
        self.missing_threshold = missing_threshold
        self.gap_threshold = gap_threshold

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        gap_thr = self.gap_threshold * X.shape[0]
        missing_thr = self.missing_threshold * X.shape[0]
        self.columns_gap_drop_ = X.columns[X.isnull().sum() > gap_thr]
        self.columns_with_gap_ = X.columns[(X.isna().sum() > missing_thr) & (X.isna().sum() <= gap_thr)]
        self.mode_ = X.mode().loc[0, :]
        return self

    def transform(self, X, y=None):
        tmp = X.copy()
        if self.columns_gap_drop_:
            tmp.drop(self.columns_gap_drop_, axis=1, inplace=True)
        if self.columns_with_gap_:
            tmp.loc[:, self.columns_with_gap_] = tmp.loc[:, self.columns_with_gap_].fillna('GAP')
        tmp.fillna(self.mode_, inplace=True)
        return tmp

    def get_n_features_in(self):
        return self.n_features_in_

    def get_columns_with_gap(self):
        return self.columns_with_gap_

    def get_columns_mode(self):
        return self.mode_


# here _mis_constant_care
class ConstantCare(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_keep_ = None

    def fit(self, X, y=None):
        # Store the columns with non-constant values during the fitting stage
        self.columns_to_keep_ = X.loc[:, X.nunique() != 1].columns
        return self

    def transform(self, X, y=None):
        # Keep only the non-constant columns in the input data
        return X.loc[:, self.columns_to_keep_]

    def get_columns_to_keep(self):
        return self.columns_to_keep_


# here rare_case_care
class URareCare(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.replace_dict = None
        self.threshold = threshold

    def fit(self, X, y=None):
        # Store the columns with non-constant values during the fitting stage
        replace_dict = {}
        for cl in X.columns:
            vl = X.loc[:, cl].value_counts()
            ls = (vl / X.shape[0]) < self.threshold
            ls = ls.index[ls == True].tolist()
            if len(ls) > 0:
                replace_dict[cl] = {}
                if (vl[ls].sum() / X.shape[0]) < self.threshold:
                    md = vl.index[0]
                    for letter in ls:
                        replace_dict[cl][letter] = md
                else:
                    for letter in ls:
                        replace_dict[cl][letter] = ''.join(ls)

        self.replace_dict = replace_dict
        return self

    def transform(self, X, y=None):
        # Keep only the non-constant columns in the input data
        return X.replace(self.replace_dict)

    def get_replace_dict(self):
        return self.replace_dict


# oneHot
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        tmp = pd.get_dummies(X, drop_first=True)
        self.feature_names_out_ = tmp.columns
        return self

    def transform(self, X, y=None):
        transformed_X = pd.get_dummies(X, drop_first=False)
        feature_names = self.feature_names_out_
        cols_to_add = feature_names.difference(transformed_X.columns)
        if len(cols_to_add) == 0:
            transformed_X = transformed_X.loc[:, feature_names]
        else:
            tmp = np.zeros((transformed_X.shape[0], len(cols_to_add)))
            tmp = pd.DataFrame(tmp, columns=cols_to_add, index=transformed_X.index)
            transformed_X = pd.concat([transformed_X, tmp], axis=1)
            transformed_X = transformed_X.loc[:, feature_names]
        return transformed_X

    def get_feature_names_out(self):
        return self.feature_names_out_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, model_type, alpha, keep=False):
        self.p_values_ = None
        self.names_out_ = None
        self.model_type = model_type
        self.alpha = alpha
        self.keep = keep

    def fit(self, X, y=None):
        if self.model_type == 'reg':
            s_fdr = SelectFdr(f_regression, alpha=self.alpha)
        else:
            s_fdr = SelectFdr(chi2, alpha=self.alpha)
        s_fdr.fit(X, y)

        self.names_out_ = s_fdr.get_feature_names_out()
        if self.keep:
            self.p_values_ = list(zip(s_fdr.feature_names_in_, s_fdr.pvalues_, s_fdr.scores_))
        return self

    def transform(self, X, y=None):
        # Keep only the non-constant columns in the input data
        return X.loc[:, self.names_out_]

    def get_p_values(self):
        return pd.DataFrame(self.p_values_, columns=['feature', 'p_value', 'score'])

    def get_names_out(self):
        return self.names_out_


# clustering
class CollinearCare(BaseEstimator, TransformerMixin):
    def __init__(self, dist_method, threshold, keep=False):
        self.cr = None
        self.shape_out = None
        self.feature_names_out_ = None
        self.feature_grouped_dict_ = None
        self.dist_method = dist_method
        self.threshold = threshold
        self.keep = keep

    def fit(self, X, y=None):
        cr = distance_calc(dat=X, dist_method=self.dist_method)
        dc_df = db_grouped(dat=cr, report_dir=None,
                           threshold=self.threshold)
        dc = group_features(dat=X, group_dat=dc_df)

        self.feature_grouped_dict_ = dc
        if self.keep:
            self.cr = cr

        return self

    def transform(self, X, y=None):
        transfomed_X = cor_remove(X, self.feature_grouped_dict_)
        if self.keep:
            self.feature_names_out_ = transfomed_X.columns
            self.shape_out = transfomed_X.shape
        return transfomed_X

    def get_feature_grouped_dict(self):
        return self.feature_grouped_dict_

    def get_corr(self):
        return self.cr

    def get_names_out(self):
        return self.feature_names_out_

    def get_shape_out(self):
        return self.shape_out


# scaler
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = None
        self.mn = None
        self.std = None
        self.ind = None

    def fit(self, X, y=None):
        self.mn = X.mean()
        self.std = X.std()
        self.feature_names_out_ = X.columns
        self.ind = X.index
        return self

    def transform(self, X, y=None):
        transformed_X = (X - self.mn) / self.std
        return transformed_X
