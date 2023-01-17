# importing libraries
from deepBreaks.preprocessing import *
from deepBreaks.models import *
from deepBreaks.visualization import *
import os
import datetime
import argparse
import warnings
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
                        type=int, default=5)
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
        print('For sequence data type, please enter "nu" for nucleotide or "aa" for amino-acid sequences.')
        exit()

    if args.anatype not in ['reg', 'cl']:
        print('For analysis type, please enter "reg" for regression or "cl" for classification only')
        exit()
    # making directory
    print('directory preparation')
    dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    seq_file_name = args.seqfile.split('.')[0]

    report_dir = str(seq_file_name + '_' + args.metavar + '_' + dt_label)
    os.makedirs(report_dir)

    print('reading meta-data')
    meta_data = read_data(args.meta_data, seq_type=None, is_main=False)
    print('meta_data:', meta_data.shape)

    print('checking the quality of the data')
    check_data(meta_data=meta_data, feature=args.metavar, model_type=args.anatype)

    # importing seq data
    print('reading fasta file')
    df = read_data(args.seqfile, seq_type=args.seqtype, gap_threshold=args.gap, is_main=True)

    if args.write:
        print('Writing cleaned FASTA file')
        write_fasta(dat=df, fasta_file=seq_file_name+'_clean.fasta', report_dir=report_dir)

    positions = df.shape[1]
    print('Done')
    print('Shape of data is: ', df.shape)

    # selecting only more frequent classes
    if args.anatype == 'cl':
        df = balanced_classes(dat=df, meta_dat=meta_data, feature=args.metavar)

    # taking care of missing data
    print('Shape of data before missing/constant care: ', df.shape)
    df = missing_constant_care(df, missing_threshold=args.miss_gap)
    print('Shape of data after missing/constant care: ', df.shape)

    print('Shape of data before imbalanced care: ', df.shape)
    df = imb_care(dat=df, imbalance_threshold=0.05)
    print('Shape of data after imbalanced care: ', df.shape)

    if args.fraction is not None:
        print('number of columns of main data before: ', df.shape[1])
        df = col_sampler(dat=df, sample_frac=args.fraction)
        print('number of columns of main data after: ', df.shape[1])

    print('Statistical tests to drop redundant features')
    df_cleaned = redundant_drop(dat=df, meta_dat=meta_data, feature=args.metavar, model_type=args.anatype,
                                report_dir=report_dir, threshold=args.redundant_threshold)
    if df_cleaned is None:
        exit()

    print('Shape of data after dropping redundant columns: ', df_cleaned.shape)
    print('prepare dummy variables')
    df_cleaned = get_dummies(dat=df_cleaned, drop_first=True)
    print('correlation analysis')
    cr = distance_calc(dat=df_cleaned, dist_method=args.distance_metric, report_dir=report_dir)
    print('finding collinear groups')
    dc_df = db_grouped(dat=cr, report_dir=report_dir, threshold=args.distance_threshold)
    print('grouping features')
    dc = group_features(dat=df_cleaned, group_dat=dc_df, report_dir=report_dir)
    print('dropping correlated features')
    print('Shape of data before linearity care: ', df_cleaned.shape)
    df_cleaned = cor_remove(df_cleaned, dc)
    print('Shape of data after linearity care: ', df_cleaned.shape)

    # merge with meta data
    df = df.merge(meta_data[args.metavar], left_index=True, right_index=True)
    df_cleaned = df_cleaned.merge(meta_data[args.metavar], left_index=True, right_index=True)

    # model
    print('Training the models...')
    select_top = args.top_models
    trained_models = model_compare(X_train=df_cleaned.loc[:, df_cleaned.columns != args.metavar],
                                   y_train=df_cleaned.loc[:, args.metavar], n_positions=positions,
                                   grouped_features=dc, report_dir=report_dir, ana_type=args.anatype,
                                   select_top=select_top)

    print('Visualizing the results...')
    for key in trained_models.keys():
        if key == 'mean':
            dp_plot(importance=trained_models[key], imp_col='mean', model_name=key, annotate=2, report_dir=report_dir)
        else:
            dp_plot(importance=trained_models[key]['importance'], imp_col='standard_value',
                    model_name=key, annotate=2, report_dir=report_dir)
            plot_imp_model(importance=trained_models[key]['importance'],
                           X_train=df.loc[:, df.columns != args.metavar], y_train=df.loc[:, args.metavar],
                           meta_var=args.metavar, model_type=args.anatype, model_name=key, report_dir=report_dir)

    if args.plot:
        plot_imp_all(trained_models=trained_models,
                     X_train=df.loc[:, df.columns != args.metavar], y_train=df.loc[:, args.metavar],
                     meta_var=args.metavar, model_type=args.anatype, report_dir=report_dir, max_plots=100)

    zip_obj = ZipFile(str(report_dir + '/report.zip'), 'w')
    file_names = os.listdir(report_dir)
    for file in file_names:
        if not file.endswith('.zip'):
            zip_obj.write(str(report_dir + '/' + file))

    return print('done!')


# main()
if __name__ == "__main__":
    main()
