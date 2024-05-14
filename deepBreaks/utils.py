import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
import joblib
from typing import List, Tuple
from sklearn import metrics


def ref_id_type(value):
    try:
        # Try parsing as an integer
        return int(value)
    except ValueError:
        # If parsing as an integer fails, return as string
        return str(value)


def df_to_dict(dat):
    return dat.apply(lambda row: ''.join(row.astype(str).replace("None", 'N')), axis=1).to_dict()


def get_models(ana_type):
    if ana_type == 'reg':
        from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, HuberRegressor, LassoLars
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor

        models = {
            'rf': RandomForestRegressor(n_jobs=-1, random_state=123),
            'Adaboost': AdaBoostRegressor(random_state=123),
            'et': ExtraTreesRegressor(n_jobs=-1, random_state=123),
            'gbc': GradientBoostingRegressor(random_state=123),
            'dt': DecisionTreeRegressor(random_state=123),
            'lr': LinearRegression(n_jobs=-1),
            'Lasso': Lasso(random_state=123),
            'LassoLars': LassoLars(random_state=123),
            'BayesianRidge': BayesianRidge(),
            'HubR': HuberRegressor(max_iter=2000, tol=1e-4),
            'lgbm': LGBMRegressor(n_jobs=-1, random_state=123),
            'xgb': XGBRegressor(n_jobs=-1, random_state=123)

        }
    else:
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        models = {
            'rf': RandomForestClassifier(n_jobs=-1, random_state=123),
            'Adaboost': AdaBoostClassifier(random_state=123),
            'et': ExtraTreesClassifier(n_jobs=-1, random_state=123),
            'lg': LogisticRegression(n_jobs=-1, random_state=123, max_iter=2000),
            'gbc': GradientBoostingClassifier(random_state=123),
            'dt': DecisionTreeClassifier(random_state=123),
            'xgb': XGBClassifier(n_jobs=-1, random_state=123),
            'lgbm': LGBMClassifier(n_jobs=-1, random_state=123)
        }

    return models


def get_scores(ana_type):
    scores = {'cl': {'Accuracy': 'accuracy',
                     'AUC': 'roc_auc_ovr',
                     'F1': 'f1_macro',
                     'Recall': 'recall_macro',
                     'Precision': 'precision_macro'
                     },
              'reg': {
                  'R2': 'r2',
                  'MAE': 'neg_mean_absolute_error',
                  'MSE': 'neg_mean_squared_error',
                  'RMSE': 'neg_root_mean_squared_error',
                  'MAPE': 'neg_mean_absolute_percentage_error'
              }
              }
    return scores[ana_type]


def calculate_metrics(model, X, y_true, ana_type):
    """
    Calculate the metrics for the given model and data.

    Parameters:
    -----------
    model : object
        The trained model.
    X : array-like
        The input data.
    y_true : array-like
        The true labels.
    ana_type : str
        The type of analysis ('reg' for regression, 'cl' for classification).

    Returns:
    --------
    metrics : dict
        A dictionary containing the calculated metrics.
    """
    if type(model) is not list:
        model = [model]
    scores = get_scores(ana_type)
    metrics_calc = {}
    for estimator in model:
        estimator_name = estimator.steps[-1][0]
        metrics_calc[estimator_name] = {}
        for score, f in scores.items():
            scorer = metrics.get_scorer(f)
            score_val = scorer(estimator, X, y_true)
            if f.startswith('neg'):
                score_val = -1 * score_val
            metrics_calc[estimator_name][score] = score_val
    return metrics_calc


def report_test_scores(model, X, y_true, ana_type, report_dir=None):
    """
    Report the test scores for the given model and data.

    Parameters:
    -----------
    model : object
        The trained model.
    X : array-like
        The input data.
    y_true : array-like
        The true labels.
    ana_type : str
        The type of analysis ('reg' for regression, 'cl' for classification).
    report_dir : str or None, optional (default=None)

    Returns:
    --------
    None
    """
    metrics_calc = calculate_metrics(model, X, y_true, ana_type)
    metrics_calc = pd.DataFrame.from_dict(metrics_calc, orient='index')
    if report_dir is not None:
        metrics_calc.to_csv(f'{report_dir}/test_scores.csv')
    return metrics_calc


def get_params():
    params = {
        'rf': {'rf__max_features': ["sqrt", "log2"]},
        'Adaboost': {'Adaboost__learning_rate': np.linspace(0.001, 0.1, num=2),
                     'Adaboost__n_estimators': [100, 200]},
        'gbc': {'gbc__max_depth': range(3, 6),
                'gbc__max_features': ['sqrt', 'log2'],
                'gbc__n_estimators': [200, 500, 800],
                'gbc__learning_rate': np.linspace(0.001, 0.1, num=2)},
        'et': {'et__max_depth': [4, 6, 8],
               'et__n_estimators': [500, 1000]},
        'dt': {'dt__max_depth': [4, 6, 8]},
        'Lasso': {'Lasso__alpha': np.linspace(0.01, 100, num=5)},
        'LassoLars': {'LassoLars__alpha': np.linspace(0.01, 100, num=5)}
    }
    return params


def get_color_palette(char_list):
    """
    Generates a dictionary of colors for each character in a given character list.

    Parameters:
    -----------
    char_list : list
        List of characters to generate color palette for.

    Returns:
    --------
    color_dic : dict
        Dictionary of color codes for each character in char_list.
    """

    # Define key and color lists
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

    # Create dictionary of character-to-color mappings
    color_dic = {}
    for n, key in enumerate(key_list):
        color_dic[key] = color_list[n]
    color_dic['U'] = color_dic['T']

    # Add gray color for mixed combinations
    for let in set(char_list):
        if let not in color_dic and let is not np.nan:
            color_dic[let.upper()] = '#808080'  # hex code for gray color

    return color_dic


def kruskal_test(data: pd.DataFrame, group_col: str, response_var: str) -> float:
    """
    Performs the Kruskal-Wallis H test on a given dataset to determine if there are significant differences between groups
    in terms of a given response variable.

    Args:
    - data (pd.DataFrame): A pandas DataFrame containing the data to be tested.
    - group_col (str): The name of the column in the DataFrame containing the grouping variable.
    - response_var (str): The name of the column in the DataFrame containing the response variable.

    Returns:
    - p (float): The p-value resulting from the Kruskal-Wallis H test.
    """
    k, p = stats.kruskal(*[group[response_var].values for name, group in data.groupby(group_col)])
    return p


def chi2_test(cross_table=None, data=None, group_col=None, response_var=None):
    """Perform a chi-square test for independence of two categorical variables.

    Args:
        cross_table (pandas.DataFrame, optional): A contingency table. Defaults to None.
        data (pandas.DataFrame, optional): Data to create contingency table from. Defaults to None.
        group_col (str, optional): Column name for grouping variable. Defaults to None.
        response_var (str, optional): Column name for response variable. Defaults to None.

    Returns:
        float: p-value of chi-square test.
    """
    if cross_table is None:
        # create contingency table from data
        cross_tab = pd.crosstab(data[response_var], data[group_col])
    else:
        cross_tab = cross_table
    # perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
    return p


def box_plot(data, group_col, response_var, figsize=(3.2, 3.2), ax=None, p=None):
    """
    Create a box plot of response variable stratified by group_col,
    optionally with an associated Kruskal-Wallis test p-value.

    Args:
    - data (pandas DataFrame): The data to be plotted.
    - group_col (str): The name of the column in `data` containing group identifiers.
    - response_var (str): The name of the column in `data` containing the response variable to be plotted.
    - ax (matplotlib Axes object, optional): The Axes object to be plotted on.
    - p (float, optional): The p-value from a Kruskal-Wallis test.

    Returns:
    - ax (matplotlib Axes object): The plotted Axes object.
    """
    tmp = data.loc[:, [group_col, response_var]]
    tmp = tmp.sort_values(by=group_col)  # sort by grouping variable
    n_groups = len(set(data.loc[:, group_col]))
    # Compute p-value if not provided
    if p is None:
        p = kruskal_test(data=tmp, group_col=group_col, response_var=response_var)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=350)
    ax.grid(color='gray', linestyle='-', linewidth=0.2, axis='y')
    ax.set_axisbelow(True)

    sns.boxplot(ax=ax, x=group_col, y=response_var, data=tmp,
                showfliers=False, dodge=False,
                width=.6, linewidth=.4,
                palette=get_color_palette(char_list=tmp.loc[:, group_col]))
    sns.despine(ax=ax)
    sns.stripplot(ax=ax, x=group_col, y=response_var, data=tmp,
                  size=5-np.log(n_groups-1), alpha=0.3, linewidth=.2, hue=group_col,
                  palette=get_color_palette(char_list=tmp.loc[:, group_col]),
                  legend=False)
    ax.set_xlabel('', fontsize=8)
    ax.set_ylabel(response_var, fontsize=8)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)

    ax.set_title(group_col + ', P-value of KW test: ' + str(round(p, 3)), fontsize=8)
    ax.set_xlabel('')

    return ax


def stacked_barplot(cross_table=None, data=None, group_col=None, response_var=None, ax=None, figsize=(3.2, 3.2)):
    """
    Generate a stacked bar plot for categorical data.

    Args:
    - cross_table (pandas.DataFrame): a contingency table of counts for categorical data.
    - data (pandas.DataFrame): the input data.
    - group_col (str): the name of the column in the input data that contains the grouping variable.
    - response_var (str): the name of the column in the input data that contains the response variable.
    - ax (matplotlib.axes.Axes): the target axes object for plotting. If None, a new figure will be created.

    Returns:
    - ax (matplotlib.axes.Axes): the generated stacked bar plot axes object.
    """
    # if cross_table is not provided, generate one from the input data
    if cross_table is None:
        cross_tb = pd.crosstab(data[response_var], data[group_col])
    else:
        cross_tb = cross_table

    # calculate chi-square test p-value
    p = chi2_test(cross_table=cross_tb)

    # get color palette for the plot
    color_dic = get_color_palette(char_list=cross_tb.columns.tolist())

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=350)

    # generate the stacked bar plot
    cross_tb.plot(kind="bar", stacked=True, rot=0, ax=ax,
                  color=color_dic, width=.3)

    # set plot title and axis labels
    ax.set_title(group_col + ', P-value of Chi-square test: ' + str(round(p, 3)), fontsize=6)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=90)
    ax.set_ylabel('Counts', fontsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(title=None, fontsize=6)

    return ax


def make_pipeline(steps: List[Tuple[str, object]], cache_dir: str = None) -> Pipeline:
    """
    Creates a scikit-learn pipeline with the given steps and memory cache.

    Parameters:
    -----------
    steps : list of tuples
        The pipeline steps as a list of tuples, where each tuple contains the step name and the corresponding estimator.
    cache_dir : str or None, optional (default=None)
        The directory to use as a memory cache for the pipeline.

    Returns:
    --------
    pipeline : Pipeline object
        The scikit-learn pipeline created with the given steps and memory cache.
    """
    # Create a scikit-learn pipeline with the given steps and memory cache
    pipeline = Pipeline(memory=cache_dir, steps=steps)
    # Return the pipeline
    return pipeline


def save_obj(obj: object, file_name: str) -> str:
    """
    Saves a Python object to a file in the pickle format.

    Parameters:
    -----------
    obj : object
        The Python object to be saved.
    file_name : str
        The name of the file to be created.

    Returns:
    --------
    str
        A string confirming that the object has been saved to the file.
    """
    # Extract the file extension from the file name
    extension = file_name.split('.')[-1]
    # Check if the file extension is '.pkl'
    assert extension == 'pkl', 'File name should be saved as a .pkl file. Please modify your file_name'
    # Save the object to the file in the pickle format using joblib.dump()
    joblib.dump(obj, file_name)
    # Return a confirmation message
    return 'Object saved'


def load_obj(file_name: str) -> object:
    """
    Loads a Python object from a file in the pickle format.

    Parameters:
    -----------
    file_name : str
        The name of the file to be loaded.

    Returns:
    --------
    object
        The Python object loaded from the file.
    """
    # Load the object from the file in the pickle format using joblib.load()
    obj = joblib.load(file_name)
    # Return the loaded object
    return obj


def print_highlighted_sequences(pos, imp, seq, start_pos, end_pos, compare_with=None):
    """
    Print the sequences with a highlighted position and position number.

    Parameters:
    - pos (int): The position to be highlighted.
    - seq (dict): A dictionary containing sequences with IDs as keys and sequences as values.
    - start_pos (int): The starting position of the highlighted segment.
    - end_pos (int): The ending position of the highlighted segment.
    - compare_with (list or None): Optional. If provided, compares sequences with another set of sequences.

    Returns:
    - None
    """
    if imp:
        print(f'Position: {pos} --- Importance: {imp}')
    else:
        print(f'Position: {pos}')

    if compare_with:
        print('\n'.join(list(seq.keys())))

    for n, seq_id in enumerate(seq):
        if n > 0 or len(seq) == 1:
            print(f'{" " * (pos - start_pos - 1)}|')
        print(''.join(seq[seq_id][start_pos:end_pos]))
    return None


def save_highlighted_sequences_to_file(pos, imp, seq, start_pos, end_pos, report_dir, compare_with=None):
    """
    Save the sequences with a highlighted position and position number to a file.

    Parameters:
    - pos (int): The position to be highlighted.
    - seq (dict): A dictionary containing sequences with IDs as keys and sequences as values.
    - start_pos (int): The starting position of the highlighted segment.
    - end_pos (int): The ending position of the highlighted segment.
    - report_dir (str): The directory where the report file will be saved.
    - compare_with (list or None): Optional. If provided, compares sequences with another set of sequences.

    Returns:
    - None
    """
    with open(f'{report_dir}/imp_seq.txt', 'a') as f:
        if imp:
            f.write(f'>Position: {pos} --- Importance: {imp}\n')
        else:
            f.write(f'Position: {pos}\n')
        if compare_with:
            f.write('\n'.join(list(seq.keys())) + '\n')
        for n, seq_id in enumerate(seq):
            if n > 0 or len(seq) == 1:
                f.write(f'{" " * (pos - start_pos - 1)}|\n')
            f.write(''.join(seq[seq_id][start_pos:end_pos]) + '\n')
    return None


def imp_print(raw_seq, position, importance=None, ref_seq_id=None,
              compare_with=None, compare_len=None, report_dir=None):
    """
    Print the sequence with the given position highlighted.

    Parameters:
    -----------
    raw_seq : str
        The raw sequences dictionary.
    position : int, or list of int
        The position to be highlighted.
    importance : float, or list of float optional (default=None)
        The importance of the position.
    ref_seq_id : str, int optional (default=None)
        The reference sequence id, or order to be compared with the raw sequence.
    compare_with : str, optional (default=None)
        The sequence to be compared with the raw sequence.
    compare_len : int, optional (default=None)
        The maximum number of characters to compare.

    Returns:
    --------
    None
    """
    # check if position is list or int
    if type(position) is not list:
        # all the positions should be integers
        assert type(position) is int, 'Position should be an integer'
        position = [position]
    else:
        for pos in position:
            assert type(pos) is int, 'Position should be an integer'
    # Get the sequences
    seq = {}
    seq_ids = list(raw_seq.keys())
    if ref_seq_id is not None:
        if type(ref_seq_id) is int:
            ref_seq_id = seq_ids[ref_seq_id]
        seq[ref_seq_id] = raw_seq[ref_seq_id]
    else:
        ref_seq_id = seq_ids[-1]
        seq[ref_seq_id] = raw_seq[ref_seq_id]
    if compare_with is not None:
        for seq_id in compare_with:
            if type(seq_id) is int:
                seq_id = seq_ids[seq_id]
            seq[seq_id] = raw_seq[seq_id]
    for n, pos in enumerate(position):
        if importance:
            if importance[n] < 0.1:
                continue
            else:
                imp = importance[n]
        else:
            imp = None

        # check if the position is within the sequence length
        if pos < 0 or pos >= len(seq[ref_seq_id]):
            print('Position is out of range')
            continue
        # make an interval around the position
        start_pos = max(0, pos - (compare_len // 2))
        end_pos = min(len(seq[ref_seq_id]), pos + (compare_len // 2))
        if report_dir is None:
            print_highlighted_sequences(pos, imp, seq, start_pos, end_pos, compare_with)
        else:
            save_highlighted_sequences_to_file(pos, imp, seq, start_pos, end_pos, report_dir, compare_with)
    return None
