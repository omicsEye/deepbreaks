import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit


def model_cv(X, y, preprocess_pipe, model, name, scoring, cv, cache_dir=None, n_jobs=-1):
    """Train and evaluate a machine learning model using cross-validation.

    Args:
        X (array-like of shape (n_samples, n_features)):
            The training input samples.
        y (array-like of shape (n_samples,) or (n_samples, n_targets)):
            The target values.
        preprocess_pipe (sklearn.pipeline.Pipeline):
            The preprocessing pipeline to apply to the data.
        model:
            The estimator to use for the model.
        name (str):
            The name of the estimator step in the pipeline.
        scoring (str or list):
            The scoring metric(s) to use for cross-validation.
        cv (int or cross-validation generator):
            Determines the cross-validation splitting strategy.
        cache_dir (str or None, optional):
            The directory where the pipeline cache should be stored.
            If None (default), caching is disabled.
        n_jobs (int):
            Number of jobs to run in parallel. `-1` means using all processors.

    Returns:
        A tuple containing:
        - A dictionary with the average test scores for each metric in `scoring`.
        - The trained estimator from the first fold of cross-validation.
    """
    # Create the pipeline
    if preprocess_pipe is None:
        pipeline = Pipeline(memory=cache_dir,
                            steps=[
                                (name, model)
                            ])
    else:
        pipeline = Pipeline(memory=cache_dir,
                            steps=[
                                ('prep', preprocess_pipe),
                                (name, model)
                            ])

    # Perform cross-validation
    cv_results = cross_validate(estimator=pipeline, X=X, y=y,
                                cv=cv, return_estimator=True,
                                scoring=scoring, n_jobs=n_jobs)

    # Compute the mean test scores for each metric
    scores = {}
    for metric in scoring:
        if metric in cv_results:
            key = metric
        else:
            key = f'test_{metric}'
        scores[metric] = np.mean(cv_results[key])

    # Return the average scores and the trained estimator from the first fold
    return scores, cv_results['estimator'][0]


def model_compare_cv(X, y, preprocess_pipe, models_dict, scoring, cv,
                     ana_type, select_top=3, random_state=123,
                     report_dir=None, sort_by=None, cache_dir=None, n_jobs=-1):
    """Train and evaluate multiple machine learning models using cross-validation, and compare their performance.

    Args:
        X (array-like of shape (n_samples, n_features)):
            The training input samples.
        y (array-like of shape (n_samples,) or (n_samples, n_targets)):
            The target values.
        preprocess_pipe (sklearn.pipeline.Pipeline):
            The preprocessing pipeline to apply to the data.
        models_dict (dict):
            A dictionary of estimator objects, with names as keys and estimator objects as values.
        scoring (str or list):
            The scoring metric(s) to use for cross-validation.
        cv (int or cross-validation generator):
            Determines the cross-validation splitting strategy.
        ana_type (str):
            The type of analysis, either 'reg' for regression or 'class' for classification.
        select_top (int, optional):
            The number of top-performing models to select. Defaults to 3.
        random_state (int, optional):
            The random state to use for shuffling the data. Defaults to 123.
        report_dir (str, optional):
            The path to save the reports. Defaults to None.
        sort_by (str, optional):
            The name of the metric to sort the model report by. If None (default), the metric will be chosen
            based on the analysis type ('R2' for regression, 'F1' for classification).
        cache_dir (str or None, optional):
            The directory where the pipeline cache should be stored.
            If None (default), caching is disabled.
        n_jobs (int):
            Number of jobs to run in parallel. `-1` means using all processors.
    Returns:
        A tuple containing:
        - A DataFrame with the average test scores for each model and metric, sorted by the specified metric.
        - A list of the top-performing trained estimators.
    """
    # Create the KFold object with shuffling
    if isinstance(cv, (int, float)):
        if cv > 1:
            cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            if cv == 1:
                raise Exception("CV should be int/float or a cv generator. If it's an int, it should not be equal to 1")
            else:
                cv = ShuffleSplit(n_splits=1, random_state=random_state, test_size=cv)
    # Initialize dictionaries for model performance and trained estimators
    model_performance = {}
    models = {}
    top_models = []

    # Determine the metric to sort the model report by
    if sort_by is None:
        if ana_type == 'reg':
            sort_by = 'R2'
        else:
            sort_by = 'F1'

    # Train and evaluate the models using cross-validation
    for name in models_dict:
        print(f"Fitting {name}...")
        tmp, estimator = model_cv(X=X, y=y, preprocess_pipe=preprocess_pipe,
                                  model=models_dict[name], name=name, scoring=scoring, cv=cv,
                                  cache_dir=cache_dir, n_jobs=n_jobs)
        model_performance[name] = tmp
        models[name] = estimator

    # Sort the model report by the specified metric
    model_report = pd.DataFrame.from_dict(model_performance, orient='index')

    for key in scoring:
        if scoring[key].startswith('neg'):
            model_report.loc[:, key] = model_report.loc[:, key].abs()

    model_report = model_report.sort_values(by=[sort_by], ascending=False)

    # Select the top-performing trained estimators
    for model_name in model_report.index[:select_top]:
        top_models.append(models[model_name])
    if report_dir is not None:
        model_report.to_csv(str(report_dir + '/' + 'model_report.csv'), index=True)
    return model_report, top_models


def finalize_top(X, y, top_models, grid_param, cv, random_state=123, report_dir=None, n_jobs=-1):
    """
    Perform hyperparameter tuning using grid search on the top models, and return the final models.

    Args:
        top_models (list): A list of top models to be tuned.
        grid_param (dict): A dictionary containing the hyperparameters to be tuned for each model.
        cv (int): The number of cross-validation folds to use during grid search.
        random_state (int): Random seed to use for reproducibility. Defaults to 123.
        report_dir (str): Directory to save the tuned models. If None, the models will not be saved.
        n_jobs (int): Number of jobs to run in parallel. `-1` means using all processors.

    Returns:
        list: A list of the tuned final models.
    """
    final_models = []  # list to store the tuned final models
    # Create the KFold object with shuffling
    if isinstance(cv, (int, float)):
        if cv > 1:
            cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            if cv == 1:
                raise Exception("CV should be int/float or a cv generator. If it's an int, it should not be equal to 1")
            else:
                cv = ShuffleSplit(n_splits=1, random_state=random_state, test_size=cv)

    # iterate over top models to perform grid search on each of them
    for model in top_models:
        name = model.steps[-1][0]  # get the name of the model
        print(f'Tuning {name}...')

        # get the hyperparameters to be tuned for this model
        p_grid = grid_param.get(name, {})

        if len(p_grid) > 0:
            # perform grid search with cross-validation and refit the best estimator
            grid_search = GridSearchCV(model, param_grid=p_grid, cv=cv, refit=True, n_jobs=n_jobs)
            grid_search.fit(X=X, y=y)

            # save the tuned model if the report directory is provided
            if report_dir is not None:
                joblib.dump(grid_search.best_estimator_, report_dir + '/' + name + '.pkl')

            final_models.append(grid_search.best_estimator_)  # append the tuned final model to the list
        else:
            model.fit(X=X, y=y)
            if report_dir is not None:
                joblib.dump(model, report_dir + '/' + name + '.pkl')

            final_models.append(model)  # append the tuned final model to the list

    return final_models


def importance_from_pipe(model, n_positions=None, grouped_features=None):
    """
    Calculate feature importance from a pipeline containing a feature selection/preprocessing step and a model.

    Args: model (sklearn.pipeline.Pipeline): Trained pipeline containing a feature selection/preprocessing step and a
    model.

    Returns: dict: Dictionary containing two lists: 'feature' - feature indices, and 'value' - corresponding feature
    importance values.
    """
    assert 'prep' in model.named_steps or (n_positions is not None and grouped_features is not None),\
        'If the model does not have a prep step, please provide n_positions and grouped_features'
    # Get the number of features in the input dataset
    if n_positions is None:
        n_positions = model.named_steps['prep']['mc'].get_n_features_in()

    try:
        # For tree-based models, extract feature importance using feature_importances_ attribute
        imp = model[-1].feature_importances_
    except:
        # For other models, extract feature importance using the absolute value of coef_ attribute
        tmp = abs(model[-1].coef_)
        if len(tmp.shape) > 1:
            imp = tmp.sum(axis=0)
        else:
            imp = tmp

    # Get the names of the columns/features used in the model training
    try:
        train_cols = model[-1].feature_names_in_
    except:
        train_cols = model[-1].feature_name_

    # Group correlated features and assign their importance values to the group mean
    tmp = pd.DataFrame({'feature': train_cols, 'value': imp})
    if grouped_features is None:
        grouped_features = model.named_steps['prep']['collinear_care'].get_feature_grouped_dict()
    for key in grouped_features:
        try:
            vl = tmp.loc[tmp.loc[:, 'feature'] == key, 'value'].values[0]
        except:
            vl = None
        if vl is not None:
            gr_tmp = pd.DataFrame(data=grouped_features[key], columns=['feature'])
            gr_tmp.loc[:, 'value'] = vl
            tmp = pd.concat([tmp, gr_tmp], axis=0, ignore_index=True)

    # Extract the feature indices from the column names
    tmp['feature'] = tmp['feature'].str.split('_').str[0]
    tmp['feature'] = tmp['feature'].str.split('p').str[1].astype(int)

    # Aggregate feature importance values for each feature index and normalize by the maximum value
    tmp = tmp.groupby('feature')['value'].mean().reset_index()
    tmp2 = pd.DataFrame(range(1, n_positions), columns=['feature'])
    tmp2 = tmp2.merge(tmp, how='left')
    tmp2.fillna(0, inplace=True)
    tmp2['standard_value'] = tmp2['value'] / np.max(tmp2['value'])

    return tmp2.to_dict(orient='list')


def mean_importance(top_models: list, n_positions: int = None, grouped_features: dict = None, report_dir: str = None) -> pd.DataFrame:
    """
    Compute the mean feature importance of a list of models.

    Parameters:
    -----------
    top_models : list
        A list of models. Each model must be a scikit-learn Pipeline object.
    n_positions : int, Optional
        Number of positions in the initial sequence file. Only needed when the model object
         does not have a preprocessing step.
    grouped_features : dict, Optional
        a dictionary that has information of the clusters of the positions. Only needed when the model object
         does not have a preprocessing step.
    report_dir : str, optional
        The directory to save the importance report. If None, do not save the
        report (default is None).

    Returns:
    --------
    mean_imp : pandas.DataFrame
        A DataFrame with the mean feature importance of each model. The first
        column is the feature name, and the other columns are the names of the
        models. The last column is the mean feature importance across all models.
    """
    # Check if n_positions and grouped_features arguments are provided if the model has no prep step
    assert 'prep' in top_models[0].named_steps or (n_positions is not None and grouped_features is not None), \
        'If the model does not have a prep step, please provide n_positions and grouped_features'

    mean_imp = {}
    for n, model in enumerate(top_models):
        # Get the name of the model from the last step of the pipeline
        model_name = model.steps[-1][0]
        # Compute the feature importance for the model
        imp = importance_from_pipe(model, n_positions=n_positions, grouped_features=grouped_features)
        if n == 0:
            # For the first model, initialize the dictionary with the feature
            # names and the feature importance values
            mean_imp['feature'] = imp['feature']
            mean_imp[model_name] = imp['standard_value']
        else:
            # For subsequent models, add the feature importance values to the
            # dictionary using the model name as the key
            mean_imp[model_name] = imp['standard_value']
    # Convert the dictionary to a DataFrame
    mean_imp = pd.DataFrame.from_dict(mean_imp)
    # Compute the mean feature importance across all models
    mean_imp.loc[:, 'mean'] = mean_imp.iloc[:, 1:].mean(axis=1)
    if report_dir is not None:
        # Save the importance report as a CSV file
        mean_imp.to_csv(str(report_dir + '/' + 'importance_report.csv'), index=False)
    # Return the mean feature importance DataFrame
    return mean_imp


def summarize_results(top_models, grouped_features=None, p_values=None, cor_mat=None, report_dir=None):
    """
        Summarize the results of a list of models by extracting feature importance and p-values and grouping correlated features.

        Parameters:
            top_models (list): A list of models. Each model must be a scikit-learn
                Pipeline object.
            grouped_features (dict): A dictionary that maps correlated features to their respective groups.
            p_values (pandas.DataFrame): A DataFrame of p-values for each feature across all models. The first
                column must be the feature name, and the other columns are the names of the models.
            cor_mat (pandas.DataFrame): A DataFrame of the correlation matrix between all features.
            report_dir (str): The path to the directory where the summary report will be saved.

        Returns:
            p_values (pandas.DataFrame): A DataFrame of p-values for each feature across all models. The first
                column is the feature name, and the other columns are the names of the models. The last column
                is the mean p-value across all models.
    """
    assert 'prep' in top_models[0].named_steps or (p_values is not None and grouped_features is not None), \
        'If the model does not have a prep step, please provide P-values and grouped_features'
    if p_values is None:
        p_values = top_models[0].named_steps['prep']['feature_selection'].get_p_values()

    for model in top_models:
        model_name = model.steps[-1][0]
        try:
            # For tree-based models, extract feature importance using feature_importances_ attribute
            imp = model[-1].feature_importances_
        except:
            # For other models, extract feature importance using the absolute value of coef_ attribute
            tmp = abs(model[-1].coef_)
            if len(tmp.shape) > 1:
                imp = tmp.sum(axis=0)
            else:
                imp = tmp

        # Get the names of the columns/features used in the model training
        try:
            train_cols = model[-1].feature_names_in_
        except:
            train_cols = model[-1].feature_name_

        # Group correlated features and assign their importance values to the group mean
        tmp = pd.DataFrame({'feature': train_cols, model_name: imp})
        if grouped_features is None:
            grouped_features = model.named_steps['prep']['collinear_care'].get_feature_grouped_dict()
        for key in grouped_features:
            try:
                vl = tmp.loc[tmp.loc[:, 'feature'] == key, model_name].values[0]
            except:
                vl = None
            if vl is not None:
                gr_tmp = pd.DataFrame(data=grouped_features[key], columns=['feature'])
                gr_tmp.loc[:, model_name] = vl
                tmp = pd.concat([tmp, gr_tmp], axis=0, ignore_index=True)
        tmp.loc[:, model_name] = tmp.loc[:, model_name] / tmp.loc[:, model_name].max()
        p_values = p_values.merge(tmp, how='left')
        p_values.fillna(0, inplace=True)

    if p_values.shape[1] > 4:
        p_values.loc[:, 'mean'] = p_values.iloc[:, 3:].mean(axis=1)

    if report_dir is not None:
        p_values.to_csv(str(report_dir + '/' + 'pvalue_importance_report.csv'), index=False)

        with open(str(report_dir + '/grouped_features.txt'), 'w') as f:
            for key, value in grouped_features.items():
                f.write(f"{key}: {value}\n")
        if cor_mat is None:
            if 'collinear_care' in top_models[0].named_steps['prep'].named_steps:
                corr_mat = top_models[0].named_steps['prep']['collinear_care'].get_corr()
                corr_mat.to_csv(str(report_dir + '/' + 'correlation' + '.csv'), index=True)

    return p_values
