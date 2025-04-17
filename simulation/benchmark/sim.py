# Standard library imports
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")
import time
# Third-party imports
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNetCV
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    ParameterGrid,
    train_test_split
)
from sklearn.tree import DecisionTreeRegressor

# deepBreaks imports
from deepBreaks.utils import (
    get_models,
    get_scores,
    make_pipeline,
    train_test_split
)
from deepBreaks.preprocessing import (
    CollinearCare,
    ConstantCare,
    CustomOneHotEncoder,
    CustomStandardScaler,
    FeatureSelection,
    MisCare,
    URareCare
)
from deepBreaks.models import (
    importance_from_pipe,
    model_compare_cv
)


class SimulationStudy:
    def __init__(self, n_samples=1000, n_features=20, n_informative_groups=3, 
                 features_per_group=2, zero_var_features=2, random_state=42):
        """
        Initialize simulation study parameters with collinear features
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Total number of features
        n_informative_groups : int
            Number of groups of informative features
        features_per_group : int
            Number of collinear features per group
        zero_var_features : int
            Number of features with zero variance
        random_state : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative_groups = n_informative_groups
        self.features_per_group = features_per_group
        self.n_informative = n_informative_groups * features_per_group
        self.zero_var_features = zero_var_features
        self.random_state = random_state
        
        # Initialize true importance to zeros
        self.true_importance = np.zeros(n_features)
        
        # Create feature group mappings
        self.feature_groups = []
        
        # Create groups and assign equal importance within groups
        for g in range(n_informative_groups):
            # Features in this group
            group_features = list(range(g * features_per_group, (g + 1) * features_per_group))
            self.feature_groups.append(group_features)
            
            # Assign importance to this group 
            group_importance = np.random.uniform(0.1, 1.0)  # Random importance for the group
            
            # All features in the group get equal importance
            for idx in group_features:
                self.true_importance[idx] = group_importance
        
    
    def generate_data(self, collinearity_strength=0.95):
        """
        Generate synthetic data with binary features including collinear features
        
        Parameters:
        -----------
        response_type : str
            Type of response variable ('continuous' or 'binary')
        collinearity_strength : float
            How strong the collinearity is within groups (0-1)
            
        Returns:
        --------
        X : numpy array
            Feature matrix
        y : numpy array
            Response variable
        """
        np.random.seed(self.random_state)
        
        # Generate base binary features for each group
        X = np.zeros((self.n_samples, self.n_features))
        
        # Generate base features for each group
        for group in self.feature_groups:
            # Generate the first feature in the group randomly
            base_feature = np.random.randint(0, 2, size=self.n_samples)
            
            # Assign the base feature to the first position in the group
            X[:, group[0]] = base_feature
            
            # Generate collinear features
            for i in range(1, len(group)):
                # Generate a collinear feature by flipping bits with probability (1-collinearity_strength)
                collinear_feature = np.copy(base_feature)
                
                # Indices to flip
                flip_indices = np.random.random(self.n_samples) > collinearity_strength
                collinear_feature[flip_indices] = 1 - collinear_feature[flip_indices]
                
                X[:, group[i]] = collinear_feature
        
        # Generate remaining non-informative features randomly
        last_informative_idx = max([idx for group in self.feature_groups for idx in group])
        noise_features_idx = range(last_informative_idx + 1, self.n_features - self.zero_var_features)
        
        for idx in noise_features_idx:
            X[:, idx] = np.random.randint(0, 2, size=self.n_samples)
        
        # Set zero variance features
        for i in range(self.zero_var_features):
            value = np.random.choice([0, 1])
            X[:, -(i+1)] = value
        
        # Generate response variable based on informative features
        signal = np.zeros(self.n_samples)
        
        # Only use first feature from each group in the signal generation
        # (since other features in the group are collinear)
        for group in self.feature_groups:
            weight = self.true_importance[group[0]]  # All features in group have same importance
            signal += weight * X[:, group[0]]
        
        # Add random noise
        noise = np.random.normal(0, 0.5, self.n_samples)
        signal_with_noise = signal + noise
        # Generate binary response variable
        prob = 1 / (1 + np.exp(-signal_with_noise))  # logistic function
        y_cat = (prob > 0.5).astype(int)
        self.X = X
        self.y = signal_with_noise
        self.y_cat = y_cat
        self.signal = signal
        
        return X, np.exp(signal_with_noise), y_cat
    
    def transform_data(self, X):
        """
        Transform data to character matrix use A for 1 and C for 0, or G for 0 and T for 1
        """
        # Transform binary features to characters
        char_matrix = np.where(X == 1, 'A', 'C')
        
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(char_matrix, columns=[f'p{i+1}' for i in range(self.n_features)])
        
        return df
    
    def evaluate_feature_importance(self, feature_importance_dict, percentile=50):
        """
        Evaluate how well models recover true feature importance
        
        Parameters:
        -----------
        feature_importance_dict : dict
            Dictionary of feature importances from each model
            
        Returns:
        --------
        evaluation : dict
            Metrics for evaluating feature importance recovery
        """
        evaluation = {}
        true_imp_standardized = self.true_importance / np.max(self.true_importance) if np.max(self.true_importance) > 0 else self.true_importance
        print("True importance standardized: ", true_imp_standardized)
        for name, importance in feature_importance_dict.items():
            
            # Normalize feature importance to max 1
            normalized_importance = importance / np.max(importance) if np.max(importance) > 0 else importance
            print("Normalized importance: ", normalized_importance)
            # Calculate correlation between true and estimated importance
            spearman_corr = np.corrcoef(true_imp_standardized, normalized_importance)[0, 1]
            
            # Mean squared error of feature importance
            importance_mse = np.mean((true_imp_standardized - normalized_importance) ** 2)
            
            # Identify "important" features based on percentile threshold of non-zero importance
            percentile_t = np.percentile(true_imp_standardized[true_imp_standardized > 0], percentile)
            top_p_true = np.where(true_imp_standardized >= percentile_t)[0]
            
            percentile_model = np.percentile(normalized_importance[normalized_importance > 0], percentile)
            top_p_estimated = np.where(normalized_importance >= percentile_model)[0]
            
            # Create binary arrays for other metrics
            y_true = np.zeros(self.n_features)
            y_true[top_p_true] = 1
            
            y_pred = np.zeros(self.n_features)
            y_pred[top_p_estimated] = 1
            print(name)
            print("y_true: ", y_true)
            print("y_pred: ", y_pred)
            # precision, recall, f1, specificity, mcc
            mcc = matthews_corrcoef(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Create a report of all metrics
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'accuracy': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'num_important_features_true': len(top_p_true),
                'num_important_features_predicted': len(top_p_estimated)
            }



            
            # Group importance recovery - how well importance is distributed within groups
            group_recovery = {}
            
            for g_idx, group in enumerate(self.feature_groups):
                # True importance for this group (should be all equal)
                true_group_imp = self.true_importance[group[0]]
                
                # Calculate variance of estimated importance within the group
                # Lower variance means more equal distribution (better)
                group_imp_variance = np.var(normalized_importance[group])
                
                # Calculate whether the sum of importance in the group matches the expected sum
                expected_group_sum = true_group_imp * len(group)
                actual_group_sum = np.sum(normalized_importance[group])
                group_sum_error = abs(expected_group_sum - actual_group_sum)
                
                group_recovery[f'Group_{g_idx+1}'] = {
                    'importance_variance': group_imp_variance,
                    'sum_error': group_sum_error
                }
            
            # Average metrics across groups
            avg_group_variance = np.mean([metrics['importance_variance'] for metrics in group_recovery.values()])
            avg_group_sum_error = np.mean([metrics['sum_error'] for metrics in group_recovery.values()])
            
            evaluation[name] = {
                'Spearman_correlation': spearman_corr,
                'Importance_MSE': importance_mse,
                **metrics,
                'Avg_group_variance': avg_group_variance,
                'Avg_group_sum_error': avg_group_sum_error # ,'Group_details': group_recovery
            }
            
        return evaluation
    
    def fit_models(self, X_train, X_test, y_train, y_test, response_type='continuous', models=None):
        """
        Fit and evaluate models
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : training and test data
        response_type : str
            Type of response variable ('continuous' or 'binary')
        models : dict or None
            Dictionary of models to evaluate
            
        Returns:
        --------
        results : dict
            Dictionary with model performances and feature importances
        """
        # Default models if none provided
        if models is None:
            if response_type == 'continuous':
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                    'LinearModel': LinearRegression()
                }
            else:  # binary
                models = {
                    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                    'LinearModel': LogisticRegression(random_state=self.random_state, max_iter=1000)
                }
        
        results = {
            'metrics': {},
            'feature_importance': {}
        }
        
        for name, model in models.items():
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            if response_type == 'continuous':
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                results['metrics'][name] = {
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred)
                }
                
            else:  # binary
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate metrics
                results['metrics'][name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1': f1_score(y_test, y_pred),
                    'AUC': roc_auc_score(y_test, y_prob)
                }
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                if importance.ndim > 1:
                    importance = importance[0]
            else:
                importance = np.zeros(self.n_features)
                
            results['feature_importance'][name] = importance
            
        return results
    
    def run_experiment(self, response_type='continuous', models=None, collinearity_strength=0.95):
        """
        Run the complete experiment
        
        Parameters:
        -----------
        response_type : str
            Type of response variable ('continuous' or 'binary')
        models : dict or None
            Dictionary of models to evaluate
        collinearity_strength : float
            How strong the collinearity is within groups (0-1)
            
        Returns:
        --------
        full_results : dict
            Complete results of the experiment
        """
        # Generate data
        X, y, y_cat = self.generate_data(collinearity_strength)
        
        # Split data
        if response_type == 'continuous':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        else:  # binary
            X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=self.random_state)
        
        # Fit models and get performance metrics
        results = self.fit_models(X_train, X_test, y_train, y_test, response_type, models)
        
        # Evaluate feature importance
        feature_importance_eval = self.evaluate_feature_importance(results['feature_importance'])
        
        full_results = {
            'prediction_metrics': results['metrics'],
            'feature_importance': results['feature_importance'],
            'feature_importance_evaluation': feature_importance_eval,
            'true_importance': self.true_importance,
            'feature_groups': self.feature_groups
        }
        
        return full_results


# a function that concates the rows of the dataframe and write a fasta file
def write_fasta(data, filename):
    """
    Write a DataFrame to a FASTA file.
    
    Parameters:
    -----------
    daat : pd.DataFrame
        DataFrame containing the sequences to be written to the FASTA file.
    filename : str
        Name of the output FASTA file.
    """
    with open(filename, 'w') as f:
        for i in range(len(data)):
            f.write(f">Sequence_{i}\n")
            f.write("".join(data.iloc[i].astype(str)) + "\n")

def write_label(y, y_cat, filename):
    # create an ID column with Sequence_1, Sequence_2, ...
    ids = [f"Sequence_{i}" for i in range(len(y))]
    # create a dataframe with the IDs and labels
    tmp = pd.DataFrame({'ID': ids, 'label_reg': y, 'label_cat': y_cat})
    # write the dataframe to a csv file
    tmp.to_csv(filename, index=False, sep="\t")

def run_deepbreaks(df, y, ana_type):
    prep_pipeline = make_pipeline(
    steps=[
        ('mc', MisCare(missing_threshold=0.05)),
        ('cc', ConstantCare()),
        ('ur', URareCare(threshold=0.025)),
        ('cc2', ConstantCare()),
        ('one_hot', CustomOneHotEncoder()),
        ('st_sc', CustomStandardScaler()),
        ('feature_selection', FeatureSelection(model_type=ana_type, alpha=0.25, keep=False)),
        ('collinear_care', CollinearCare(dist_method='correlation', threshold=0.3, keep=False))
    ])
    models = {
                'RandomForest': RandomForestRegressor(n_estimators=500, random_state=123, n_jobs=-1),
                'Adaboost': AdaBoostRegressor(random_state=123, n_estimators=500, learning_rate=0.1),
                'AdaBoost_2': AdaBoostRegressor(random_state=123, n_estimators=500, learning_rate=0.01),
                'et': ExtraTreesRegressor(n_jobs=-1, random_state=123),
                'gbc': GradientBoostingRegressor(random_state=123),
                'gbc_2': GradientBoostingRegressor(random_state=123, n_estimators=1000, learning_rate=0.01),
                'gbc_3': GradientBoostingRegressor(random_state=123, n_estimators=1000, learning_rate=0.001),
                'dt': DecisionTreeRegressor(random_state=123),
                'lr': LinearRegression(n_jobs=-1),
                'Lasso_1': Lasso(random_state=123, alpha=0.1),
                'Lasso_2': Lasso(random_state=123, alpha=0.01),
                'Lasso_3': Lasso(random_state=123, alpha=0.05),
                'Lasso_4': Lasso(random_state=123, alpha=0.03),
                "elasticnet": ElasticNetCV(random_state=123, n_jobs=-1)
                }
    
    report, top = model_compare_cv(X=df, y=y, preprocess_pipe=prep_pipeline,
                                models_dict=models,
                                scoring=get_scores(ana_type=ana_type),
                                report_dir=None,
                                cv=10, ana_type=ana_type, cache_dir=None)
    return report, top


# a function that cleans the pyseer results
def clean_pyseer_results(file_path, n_features):
    #load the results
    pyseer_beta = pd.read_csv(file_path, sep="\t")
    pyseer_beta['variant'] = pyseer_beta['variant'].str.split("_").str[1].astype(int)
    clean_pyseer_beta = pd.DataFrame(columns=['variant'], data=range(1, n_features+1))
    clean_pyseer_beta = clean_pyseer_beta.merge(pyseer_beta, on='variant', how='left')
    clean_pyseer_beta = clean_pyseer_beta.fillna(0)
    return clean_pyseer_beta
    
param_grid = {
    'sample_size': [1000],
    'n_features': [200],
    'n_informative_groups': [10],
    'features_per_group': [int(sys.argv[1])],
    'random_state': np.arange(50)}
# create a list of all combinations of the parameters
param_combinations = list(ParameterGrid(param_grid))
print("Number of combinations: ", len(param_combinations))
simulation_results = []
cn = 0
for params in param_combinations:
    start_time = time.time()
    cn += 1
    print(f"Running combination {cn} of {len(param_combinations)}")
    sample_size = params['sample_size']
    n_features = params['n_features']
    n_informative_groups = params['n_informative_groups']
    features_per_group = params['features_per_group']
    random_state = params['random_state']

    experiment = str(sample_size) + "_" + str(n_features) + "_" + str(n_informative_groups) + "_" + str(features_per_group) + "_" + str(random_state)
    file_prefix = "simulated_" + experiment
    # Initialize simulation study
    sim = SimulationStudy(n_samples=sample_size, 
                                n_features=n_features,
                                n_informative_groups=n_informative_groups,
                                features_per_group=features_per_group,
                                random_state=random_state)
    
    X, y, y_cat = sim.generate_data(collinearity_strength=0.9)
    df = sim.transform_data(X)

    ana_type = 'reg'
    report, top = run_deepbreaks(df, y, ana_type)
    # print(report.head())
    # get the top model
    imp = importance_from_pipe(top[0])
    imp_dict = {"deepbreaks":imp['standard_value']}

    # write the dataframe to a fasta file
    write_fasta(data=df, filename=f"{file_prefix}.fasta")
    # write the labels to a csv file
    write_label(y=y, y_cat=y_cat, filename=f"{file_prefix}_labels.txt")

    # run the command line tool
    subprocess.run(["snp-sites", "-v", "-o", f"{file_prefix}.vcf", f"{file_prefix}.fasta"])
    # check if the file was created
    if os.path.exists(f"{file_prefix}.vcf"):
        print("VCF file created successfully.")
    else:
        print("Error creating VCF file.")

    # pyseer --vcf sample.vcf --phenotypes ~/Desktop/labels.txt --wg enet --phenotype-column label_reg> reg.txt
    process = subprocess.run(["pyseer", "--vcf", f"{file_prefix}.vcf",
                    "--phenotypes", f"{file_prefix}_labels.txt",
                    "--wg", "enet", "--phenotype-column", "label_reg"],
                    stdout=open(f"{file_prefix}_labels_reg.txt", "w"),
                    stderr=subprocess.PIPE,
                    text=True)

    pyseer_rsquared = 0
    print("pyseer stderr: ", process.stderr)
    for line in process.stderr.splitlines():
        if line.startswith("Best R^2"):
            line = line.split()
            pyseer_rsquared= float(line[-1])
    # print("pyseer rsquared: ", pyseer_rsquared)
    clean_pyseer_beta = clean_pyseer_results(f"{file_prefix}_labels_reg.txt", n_features)
    imp_dict['pyseer'] = abs(clean_pyseer_beta['beta'].values)
    imp_results = sim.evaluate_feature_importance(imp_dict)

    # write the results to a file
    for key, value in imp_results.items():
        imp_results[key]['n_features'] = n_features
        imp_results[key]['sample_size'] = sample_size
        imp_results[key]['n_informative_groups'] = n_informative_groups
        imp_results[key]['features_per_group'] = features_per_group
        imp_results[key]['random_state'] = random_state
        imp_results[key]['approach'] = key
        if key == 'deepbreaks':
            imp_results[key]['r_squared'] = report['R2'].iloc[0]
            imp_results[key]['model'] = report.index[0]
        else:
            imp_results[key]['r_squared'] = pyseer_rsquared
            imp_results[key]['model'] = "lasso"
        simulation_results.append(imp_results[key])
    # delete the files
    os.remove(f"{file_prefix}.fasta")
    os.remove(f"{file_prefix}.vcf")
    os.remove(f"{file_prefix}_labels.txt")
    os.remove(f"{file_prefix}_labels_reg.txt")
 
    # print the time taken
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
# write the results to a file
with open(f"simulation_results_{sys.argv[1]}.txt", "w") as f:
    for result in simulation_results:
        f.write(f"{result}\n")
