import numpy as np
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid


def str_clean(x):
    x = str(x).split('(')[0]
    x = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", x)
    return x


def _importance(cross_val_obj, train_cols, n_positions, grouped_features):
    imp = []
    for i in range(len(cross_val_obj['estimator'])):
        try:
            imp.append(cross_val_obj['estimator'][i].feature_importances_)
        except:
            tmp = abs(cross_val_obj['estimator'][i].coef_)
            if len(tmp.shape) > 1:
                imp.append(tmp.sum(axis=0))
            else:
                imp.append(tmp)
    imp = np.array(imp)
    imp = imp.mean(axis=0)

    tmp = pd.DataFrame({'feature': train_cols, 'value': imp})
    for key in grouped_features:
        try:
            vl = tmp.loc[tmp.loc[:, 'feature'] == key, 'value'].values[0]
        except:
            vl = None
        if vl is not None:
            gr_tmp = pd.DataFrame(data=grouped_features[key], columns=['feature'])
            gr_tmp.loc[:, 'value'] = vl
            tmp = pd.concat([tmp, gr_tmp], axis=0, ignore_index=True)

    tmp['feature'] = tmp['feature'].str.split('_').str[0]
    tmp['feature'] = tmp['feature'].str.split('p').str[1].astype(int)
    tmp = tmp.groupby('feature')['value'].sum().reset_index()

    tmp2 = pd.DataFrame(range(1, n_positions), columns=['feature'])
    tmp2 = tmp2.merge(tmp, how='left')
    tmp2.fillna(0, inplace=True)
    tmp2['standard_value'] = tmp2['value'] / np.max(tmp2['value'])

    return tmp2.to_dict(orient='list')


def _report(cross_val_obj, scores):
    performance_metrics = []
    for key in scores.keys():
        performance_metrics.append(round(cross_val_obj['test_' + key].mean(), 4))
    return performance_metrics


def _model_report(summary, scores, sort_by):
    tm = {}
    for key in summary.keys():
        tm[key] = [str_clean(summary[key]['model'])] + summary[key]['metrics']
    tm = pd.DataFrame.from_dict(tm, orient='index',
                                columns=['Model'] + list(scores.keys()))
    tm.sort_values(by=sort_by, ascending=False, inplace=True)
    tm.iloc[:, 1:] = tm.iloc[:, 1:].abs()
    return tm


def _get_models(ana_type):

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
            'HubR': HuberRegressor(),
            'xgb': XGBRegressor(n_jobs=-1, random_state=123),
            'lgbm': LGBMRegressor(n_jobs=-1, random_state=123)
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
            'lg': LogisticRegression(n_jobs=-1, random_state=123),
            'gbc': GradientBoostingClassifier(random_state=123),
            'dt': DecisionTreeClassifier(random_state=123),
            'xgb': XGBClassifier(n_jobs=-1, random_state=123),
            'lgbm': LGBMClassifier(n_jobs=-1, random_state=123)
        }

    return models


def _get_scores(ana_type):
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


def _get_params():
    params = {
        'rf': {
            'max_depth': [4, 6, 8],
            'n_estimators': [500, 1000]
        },
        'Adaboost': {
            'learning_rate': [0.01, 0.05],
            'n_estimators': [50, 100]
        },
        'et': {
            'max_depth': [4, 6, 8],
            'n_estimators': [500, 1000]
        },
        'dt': {
            'max_depth': [4, 6, 8]
        },
        'Lasso': {
            'alpha': [0.5, 1, 3]
        },
        'LassoLars': {
            'alpha': [0.5, 1, 3]
        }
    }
    return params


def model_compare(X_train, y_train, ana_type,
                  cv=10, select_top=5,
                  models=None, scores=None,
                  params=None, sort_by=None,
                  n_positions=None,
                  grouped_features=None,
                  report_dir='.'):
    """
    Fits multiple models to the data and compare them based on the cross-validation score. Then returns the top models,
    their feature importance, and also calculates the `mean` of feature importance for all the features across the
    top models.
    Parameters
    ----------
    X_train : pandas.DataFrame
        a numeric dataframe
    y_train : 1D array
        an 1D array of values of the response variable (phenotype)
    ana_type : str
        'reg' for regression and 'cl' for classification
    cv : int
        number of folds for cross-validation
    select_top : int
        number of top models to select
    models : dict, default = None
        a dictionary of model objects. Example:
        models = {'rf': RandomForestClassifier(), 'dt': DecisionTreeClassifier()}
    scores : dict, default = None
        a dictionary of evaluation metrics. Example:
        scores = { 'R2': 'r2', 'MAE': 'neg_mean_absolute_error'}. To access the full list of the available metrics run:
        `sorted(sklearn.metrics.SCORERS.keys())`. The default metrics for classification are (in order): accuracy, AUC,
        F1, recall, precision, and for regression are: R2, MAE, MSE, RMSE, MAPE.

    params : dict, default = None
        a dictionary of parametrs for models. The keys in the dictionaries provided as `params` and `models` should be
        the same. If not, models run with only default parameters. Example:

        params = {'rf': {'max_depth': [4, 6, 8], 'n_estimators': [500, 1000]},
                  'Adaboost': {'learning_rate': [0.01, 0.05], 'n_estimators': [50, 100]}}
    sort_by : str, default = None
        the name of the evaluation metric to sort the models by it.
    n_positions : int
        the number of positions in the sequence data
    grouped_features : dict, default = None
        dictionary containing all the groups and represantives
    report_dir : str
        path to save the reports and python objects

    Returns
    -------
    dict
        It returns a nested dictionary. The first layer of the keys are the name of the models and each of them
        is a dictionary with keys ['metrics', 'importance', 'model']. For example, evaluation values can be accessed
        as: `dict['model']['metrics']`

    Examples
    --------

    >>> sm = model_compare(X_train=df_cleaned.loc[:, df_cleaned.columns != mt],
        ...        y_train=df_cleaned.loc[:, mt],
        ...        sort_by='F1',n_positions=positions,select_top=5,
        ...        grouped_features=dc, report_dir=report_dir,
        ...       ana_type='cl')

    # to get the importances
    >>> import pandas as pd
    >>> models = list(sm.keys())
    >>> dict_of_importance = sm[models[0]]['importance']
    >>> pd.DataFrame.from_dict(dict_of_importance)
    # to get the model
    >>> ml = sm[models[0]]['model']

    # to get the list performance metrics
    >>> sm[models[0]]['metrics']
    """
    if models is None:
        models = _get_models(ana_type=ana_type)
    else:
        models = models

    if scores is None:
        scores = _get_scores(ana_type)
    else:
        scores = scores

    if params is None:
        params = _get_params()
    else:
        params = params

    if n_positions is None:
        n_positions = X_train.shape[1]

    if grouped_features is None:
        grouped_features = {}

    if ana_type == 'reg':
        if sort_by is None:
            sort_by = 'MAE'
    else:
        if sort_by is None:
            sort_by = 'F1'

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    summary = {}
    for model in models.keys():
        if model in params:
            tune_params = ['default']
            tune_params.extend(ParameterGrid(params[model]))
        else:
            tune_params = ['default']

        for i in range(len(tune_params)):
            model_ind = model
            estimator = models[model]

            if tune_params[i] != 'default':
                for key in tune_params[i]:
                    model_ind = model_ind + '_' + str(key) + '=' + str(tune_params[i][key])
                    setattr(estimator, key, tune_params[i][key])
            else:
                model_ind = model_ind + '_' + 'with_default_parameters'

            summary[model_ind] = {}
            print('Fitting {}'.format(model_ind))
            crv = cross_validate(models[model],
                                 X_train, y_train,
                                 cv=cv,
                                 return_estimator=True,
                                 n_jobs=-1, scoring=scores)
            summary[model_ind]['metrics'] = _report(crv, scores)
            summary[model_ind]['importance'] = _importance(crv, train_cols=X_train.columns,
                                                           n_positions=n_positions,
                                                           grouped_features=grouped_features)
            summary[model_ind]['model'] = crv['estimator'][0]

    print('Preparing report...')
    tm = _model_report(summary=summary, scores=scores, sort_by=sort_by)
    tm.to_csv(report_dir + '/model_performance.csv')

    summary = {key: summary[key] for key in tm.iloc[:select_top, :].index}

    mean_imp = {}
    for key in summary.keys():
        mean_imp[key] = summary[key]['importance']['standard_value']
    mean_df = pd.DataFrame.from_dict(mean_imp)
    mean_df.loc[:, 'mean'] = mean_df.mean(axis=1)
    mean_df.loc[:, 'feature'] = range(1, mean_df.shape[0] + 1)
    cl = mean_df.columns
    cl = cl[::-1]
    mean_df = mean_df.loc[:, cl]
    mean_df.to_csv(report_dir + '/importance_report.csv', index=False)
    summary['mean'] = mean_df.to_dict(orient='list')
    print('Done!')
    return summary
