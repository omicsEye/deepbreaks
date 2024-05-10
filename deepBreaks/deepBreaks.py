# importing libraries
from deepBreaks.utils import get_models, get_scores, get_params, make_pipeline, df_to_dict, imp_print, ref_id_type
from deepBreaks.preprocessing import MisCare, ConstantCare, URareCare, CustomOneHotEncoder, FeatureSelection, \
    CollinearCare, CustomStandardScaler
from deepBreaks.preprocessing import read_data, check_data, write_fasta, balanced_classes
from deepBreaks.models import model_compare_cv, finalize_top, importance_from_pipe, mean_importance, summarize_results
from deepBreaks.visualization import plot_scatter, dp_plot, plot_imp_model, plot_imp_all
from sklearn.preprocessing import LabelEncoder
import os
import datetime
import argparse
import warnings
import logging
from zipfile import ZipFile


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqfile', '-sf', help="files contains the sequences", type=str, required=True)
    parser.add_argument('--seqtype', '-st', help="type of sequence: 'nu' for nucleotides or 'aa' for amino-acid",
                        type=str, required=True)
    parser.add_argument('--meta_data', '-md', help="files contains the meta data", type=str, required=True)
    parser.add_argument('--metavar', '-mv', help="name of the meta var (response variable)", type=str, required=True)
    parser.add_argument('--gap', '-gp', help="Threshold to drop positions that have GAPs above this proportion."
                                             " Default value is 0.7 and it means that the positions that 70%% or more "
                                             "GAPs will be dropped from the analysis.",
                        type=float, default=0.7)
    parser.add_argument('--miss_gap', '-mgp', help="Threshold to impute missing values with GAP. Gaps"
                                                   "in positions that have missing values (gaps) above this proportion"
                                                   "are replaced with the term 'GAP'. the rest of the missing values"
                                                   "are replaced by the mode of each position.",
                        type=float, default=0.15)
    parser.add_argument('--ult_rare', '-u', help="Threshold to modify the ultra rare cases in each position.",
                        type=float, default=0.025)

    parser.add_argument('--anatype', '-a', help="type of analysis", choices=['reg', 'cl'], type=str, required=True)
    parser.add_argument('--distance_metric', '-dm',
                        help="distance metric. Default is correlation.",
                        choices=['correlation', 'hamming', 'jaccard',
                                 'normalized_mutual_info_score',
                                 'adjusted_mutual_info_score', 'adjusted_rand_score'],
                        type=str, default='correlation')
    parser.add_argument('--fraction', '-fr', help="fraction of main data to run", type=float, required=False)
    parser.add_argument('--redundant_threshold', '-rt',
                        help="threshold for the p-value of the statistical tests to drop redundant features. Default"
                             "value is 0.25",
                        type=float, default=0.25)
    parser.add_argument('--distance_threshold', '-dth',
                        help="threshold for the distance between positions to put them in clusters. "
                             "features with distances <= than the threshold will be grouped together. Default "
                             "values is 0.3",
                        type=float, default=0.3)
    parser.add_argument('--top_models', '-tm',
                        help="number of top models to consider for merging the results. Default value is 5",
                        type=int, default=3)
    parser.add_argument('--aggregate', help="the aggregate function for summarising the importance values in the"
                                            "positions. Can be a string representing a built-in aggregation "
                                            "function (e.g., 'mean', 'max', 'min', 'std', etc.)",
                        type=str, default='max', required=False)
    parser.add_argument('--cv', '-cv',
                        help="number of folds for cross validation. Default is 10. If the given number is less than 1,"
                             " then instead of CV, a train/test split approach will be used with "
                             "cv being the test size.",
                        type=float, default=10)
    parser.add_argument('--ref_id', '-r', help="ID/order of the reference sequence in the sequence file."
                                               " Default is last sequence.",
                        default=-1, type=ref_id_type, required=False)
    parser.add_argument('--ref_compare', '-c', help="ID/order of the sequences to compare with the reference sequence.",
                        default=None, type=ref_id_type, required=False)
    parser.add_argument('--compare_len', '-l', help="length of output sequences",
                        default=50, type=int, required=False)
    parser.add_argument("--tune", help="After running the 10-fold cross validations, should the top selected models be"
                                       " tuned and finalize, or finalized only?",
                        action="store_true", default=False)
    parser.add_argument("--plot", help="plot all the individual positions that are statistically significant."
                                       "Depending on your data, this process may produce many plots.",
                        action="store_true", default=False)
    parser.add_argument("--write", help="During reading the fasta file we delete the positions that have GAPs over a "
                                        "certain threshold that can be changed in the `gap_threshold` argument"
                                        "in the `read_data` function. As this may change the whole FASTA file, you may"
                                        "want to save the FASTA file after this cleaning step.",
                        action="store_true", default=False)

    return parser.parse_args()


def main():
    # Parse arguments from command line
    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore')

    args = parse_arguments()
    print(args)  # printing Namespace
    if args.seqtype not in ['nu', 'aa']:
        raise Exception('For sequence data type, please enter "nu" for nucleotide or "aa" for amino-acid sequences.')
        exit()

    if args.anatype not in ['reg', 'cl']:
        raise Exception('For analysis type, please enter "reg" for regression or "cl" for classification only')
        exit()
    if args.cv == 1:
        raise Exception('cv can not be equal to 1')
        exit()
    # making directory
    print('directory preparation')
    dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    seq_file_name = args.seqfile.split('.')[0]

    report_dir = str(seq_file_name + '_' + args.metavar + '_' + dt_label)
    os.makedirs(report_dir)

    logging.basicConfig(filename=report_dir + "/log.txt", level=logging.DEBUG,
                        format="%(asctime)s %(message)s")

    logging.info("The parameters are{}".format(args))
    print('reading meta-data')
    meta_data = read_data(args.meta_data, seq_type=None, is_main=False)
    print('meta_data:', meta_data.shape)

    print('checking the quality of the data')
    check_data(meta_data=meta_data, feature=args.metavar, model_type=args.anatype)

    # importing seq data
    print('reading fasta file')
    df = read_data(args.seqfile, seq_type=args.seqtype, gap_threshold=args.gap, is_main=True)

    # sequence data in dictionary format
    raw_seq = df_to_dict(df)

    if args.write:
        print('Writing cleaned FASTA file')
        write_fasta(dat=df, fasta_file=seq_file_name + '_clean.fasta', report_dir=report_dir)

    positions = df.shape[1]
    print('Done')
    print('Shape of data is: ', df.shape)
    logging.info("Shape of data is {}".format(df.shape))
    # selecting only more frequent classes
    if args.anatype == 'cl':
        df = balanced_classes(dat=df, meta_dat=meta_data, feature=args.metavar)

    df = df.merge(meta_data.loc[:, args.metavar], left_index=True, right_index=True)
    y = df.loc[:, args.metavar].values
    df.drop(args.metavar, axis=1, inplace=True)

    if args.anatype == 'cl':
        le = LabelEncoder()
        y = le.fit_transform(y)

    if args.cv > 1:
        args.cv = int(args.cv)
        print(f'Running a {args.cv}-fold cross-validation.')
    else:
        print(f'Performing a {args.cv*100}% train-test split.')

    prep_pipeline = make_pipeline(
        steps=[
            ('mc', MisCare(missing_threshold=args.miss_gap, gap_threshold=args.gap)),
            ('cc', ConstantCare()),
            ('ur', URareCare(threshold=args.ult_rare)),
            ('cc2', ConstantCare()),
            ('one_hot', CustomOneHotEncoder()),
            ('feature_selection', FeatureSelection(model_type=args.anatype, alpha=args.redundant_threshold)),
            ('st_sc', CustomStandardScaler()),
            ('collinear_care', CollinearCare(dist_method=args.distance_metric, threshold=args.distance_threshold))
        ])

    # fit and compare models
    report, top = model_compare_cv(X=df, y=y, preprocess_pipe=prep_pipeline,
                                   models_dict=get_models(ana_type=args.anatype),
                                   scoring=get_scores(ana_type=args.anatype),
                                   report_dir=report_dir,
                                   cv=args.cv, ana_type=args.anatype, cache_dir=None)

    prep_pipeline = make_pipeline(
        steps=[
            ('mc', MisCare(missing_threshold=args.miss_gap, gap_threshold=args.gap)),
            ('cc', ConstantCare()),
            ('ur', URareCare(threshold=args.ult_rare)),
            ('cc2', ConstantCare()),
            ('one_hot', CustomOneHotEncoder()),
            ('feature_selection', FeatureSelection(model_type=args.anatype,
                                                   alpha=args.redundant_threshold, keep=True)),
            ('st_sc', CustomStandardScaler()),
            ('collinear_care', CollinearCare(dist_method=args.distance_metric,
                                             threshold=args.distance_threshold, keep=True))
        ])

    modified_top = []
    for model in top:
        modified_top.append(make_pipeline(steps=[('prep', prep_pipeline), model.steps[-1]]))

    if args.tune:
        top = finalize_top(X=df, y=y, top_models=modified_top, grid_param=get_params(),
                           report_dir=report_dir, cv=args.cv)
    else:
        top = finalize_top(X=df, y=y, top_models=modified_top, grid_param={},
                           report_dir=report_dir, cv=args.cv)

    n_positions = None
    grouped_features = None
    p_values = None
    corr_mat = None

    sr = summarize_results(top_models=top, grouped_features=grouped_features,
                           p_values=p_values, cor_mat=corr_mat, report_dir=report_dir)
    mean_imp = mean_importance(top_models=top, n_positions=n_positions,
                               aggregate_function=args.aggregate,
                               grouped_features=grouped_features, report_dir=report_dir)

    print('Visualizing the results...')
    logging.info("Visualizing the results...")

    scatter_plot = plot_scatter(summary_result=sr, report_dir=report_dir)
    dp_plot(importance=mean_imp, imp_col='mean', model_name='mean', report_dir=report_dir)

    df = prep_pipeline[:4].fit_transform(df)
    if args.anatype == 'cl':
        y = le.inverse_transform(y)

    for model in top:
        model_name = model.steps[-1][0]
        dp_plot(importance=importance_from_pipe(model=model, grouped_features=grouped_features,
                                                n_positions=n_positions, aggregate_function=args.aggregate),
                imp_col='standard_value',
                model_name=model_name, report_dir=report_dir)

        plot_imp_model(importance=importance_from_pipe(model=model, grouped_features=grouped_features,
                                                       n_positions=n_positions, aggregate_function=args.aggregate),
                       X_train=df, y_train=y, model_name=model_name,
                       meta_var=args.metavar, model_type=args.anatype, report_dir=report_dir)

    if args.plot:
        plot_imp_all(final_models=top,
                     X_train=df, y_train=y,
                     meta_var=args.metavar, model_type=args.anatype,
                     aggregate_function=args.aggregate,
                     n_positions=n_positions, grouped_features=grouped_features,
                     report_dir=report_dir, max_plots=100,
                     figsize=(1.85, 3))

    # printing important positions with their importance values
    positions = list(mean_imp.sort_values('mean', ascending=False)['feature'][:100])
    importance = list(mean_imp.sort_values('mean', ascending=False)['mean'][:100].round(3))

    imp_print(raw_seq, position=positions, importance=importance,
              ref_seq_id=args.ref_id, compare_with=args.ref_compare,
              compare_len=args.compare_len, report_dir=report_dir)

    zip_obj = ZipFile(str(report_dir + '/report.zip'), 'w')
    file_names = os.listdir(report_dir)
    for file in file_names:
        if not file.endswith('.zip'):
            zip_obj.write(str(report_dir + '/' + file))
    logging.info("done!")
    return print('done!')


# main()
if __name__ == "__main__":
    main()
