import numpy as np
import pandas as pd
import os


# model function
def fit_models(dat, meta_var, model_type, models_to_select, report_dir):
    if model_type == 'reg':

        import pycaret.regression as pycar

        exp_reg101 = pycar.setup(data=dat, target=meta_var, silent=True,
                                 session_id=123,
                                 ignore_low_variance=True,
                                 feature_selection=True
                                 )
        metric = 'MAE'
        top_models = pycar.compare_models(n_select=models_to_select, sort=metric, verbose=False,
                                          include=['lr', 'ridge', 'lar', 'br', 'par', 'huber',
                                                   'lasso', 'et', 'xgboost', 'lightgbm',
                                                   'rf', 'dt', 'ada'])

    else:

        import pycaret.classification as pycar
        exp_reg101 = pycar.setup(data=dat, target=meta_var, silent=True,
                                 session_id=123,
                                 feature_selection=True,
                                 ignore_low_variance=True,
                                 )
        metric = 'F1'
        top_models = pycar.compare_models(n_select=models_to_select, sort=metric, verbose=False,
                                          include=['et', 'lr', 'xgboost', 'lightgbm', 'rf', 'gbc',
                                                   'dt', 'ada', 'ridge', 'svm'])
    results_df = pycar.pull()
    results_df.to_csv(str(report_dir + '/models_summary' + '.csv'))

    return top_models, pycar.get_config('X_train').columns, results_df['Model'][:models_to_select].tolist()


# feature importance extractor
def fimp_single(trained_model, model_name, train_cols, grouped_features, n_positions, report_dir, write=True):
    try:
        imp = trained_model.feature_importances_
    except:
        imp = trained_model.coef_
        if len(imp.shape) > 1:
            imp = abs(trained_model.coef_).sum(axis=0)

    tmp = pd.DataFrame({'feature': train_cols,
                        'value': abs(imp)}).sort_values(by='value', ascending=False).reset_index(drop=True)

    tmp['feature'] = tmp['feature'].str.split('_').str[0]
    tmp['feature'] = tmp['feature'].str.split('p').str[1].astype(int)
    tmp = tmp.groupby('feature')['value'].max().reset_index()

    tmp2 = pd.DataFrame(range(n_positions), columns=['feature'])
    tmp2 = tmp2.merge(tmp, how='left')
    tmp2.fillna(0, inplace=True)
    tmp2['standard_value'] = tmp2['value'] / np.max(tmp2['value'])

    if grouped_features is not None:
        gf = grouped_features.copy()
        gf['feature'] = gf['feature'].str.split('p').str[1]
        gf['feature'] = gf['feature'].astype(int)
        tmp2 = tmp2.merge(gf, how='left')
        tmp2['group'].fillna('No_gr', inplace=True)
    else:
        tmp2['group'] = 'No_gr'

    tmp2['color'] = None

    np.random.seed(12)
    colors = np.random.randint(0, 0xFFFFFF, tmp2['group'].nunique())
    colors = ['#%06X' % i for i in colors]

    cn = 0
    for name, group in tmp2.groupby('group'):
        if name != 'No_gr':
            tmp2.loc[tmp2['group'] == name, 'color'] = colors[cn]
            mx = tmp2.loc[tmp2['group'] == name, 'standard_value'].max()
            tmp2.loc[tmp2['group'] == name, 'standard_value'] = mx
        else:
            tmp2.loc[tmp2['group'] == name, 'color'] = '#000000'
        cn += 1
    if write:
        fname = str(report_dir + '/' + model_name + '_feature_importance.csv')

        if os.path.isfile(fname):
            pass
        else:
            tmp2.to_csv(fname)

    return tmp2


# average of top model feature importance
# This function gets the models and outputs the average importance of each position based on top models
def fimp_top_models(trained_models, model_names, train_cols, grouped_features, n_positions, report_dir):
    for i in range(len(trained_models)):
        tmp2 = fimp_single(trained_model=trained_models[i], model_name=model_names[i],
                           train_cols=train_cols, grouped_features=grouped_features,
                           n_positions=n_positions, report_dir=report_dir, write=False)

        if i == 0:
            final_imp = tmp2[['feature', 'standard_value', 'group', 'color']]
            final_imp.rename(columns={"standard_value": model_names[i]}, inplace=True)
        else:
            final_imp = final_imp.merge(tmp2[['feature', 'standard_value']])
            final_imp.rename(columns={"standard_value": model_names[i]}, inplace=True)

    final_imp['mean_imp'] = final_imp[model_names].mean(axis=1)

    fname = str(report_dir + '/' + 'avg_top_models_feature_importance.csv')

    if os.path.isfile(fname):
        pass
    else:
        final_imp.to_csv(fname)

    return final_imp
