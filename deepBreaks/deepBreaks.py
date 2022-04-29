# importing libraries
from deepBreaks.preprocessing import *
from deepBreaks.models import *
from deepBreaks.visualization import *
from sklearn.metrics import adjusted_mutual_info_score
import os
import datetime
import argparse
import warnings
from zipfile import ZipFile


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqfile', '-sf', help="files contains the sequences", type=str, required=True)
    parser.add_argument('--seqtype', '-st', help="type of sequence: nuc, amino-acid", type=str, required=True, )
    parser.add_argument('--meta_data', '-md', help="files contains the meta data", type=str, required=True)
    parser.add_argument('--metavar', '-mv', help="name of the meta var (response variable)", type=str, required=True)
    parser.add_argument('--anatype', '-a', help="type of analysis", choices=['reg', 'cl'], type=str, required=True)
    parser.add_argument('--fraction', '-fr', help="fraction of main data to run",
                        type=float, required=False)
    return parser.parse_args()


def main():
    # Parse arguments from command line
    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore')

    args = parse_arguments()
    print(args)  # printing Namespace
    if args.seqtype not in ['nu', 'amino-acid']:
        print('For sequence data type, please enter "nu" or "amino-acid" only')
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

    # importing seq data
    print('reading fasta file')
    df = read_data(args.seqfile, seq_type=args.seqtype, is_main=True)

    positions = df.shape[1]
    print('Done')
    print('Shape of data is: ', df.shape)

    # selecting only more frequent classes
    if args.anatype == 'cl':
        df = balanced_classes(dat=df, meta_dat=meta_data, feature=args.metavar)

    # taking care of missing data
    print('Shape of data before missing/constant care: ', df.shape)
    df_cleaned = missing_constant_care(df)
    print('Shape of data after missing/constant care: ', df_cleaned.shape)

    print('Shape of data before imbalanced care: ', df_cleaned.shape)
    df_cleaned = imb_care(dat=df_cleaned, imbalance_threshold=0.05)
    print('Shape of data after imbalanced care: ', df_cleaned.shape)

    if args.fraction is not None:
        print('number of columns of main data before: ', df_cleaned.shape[1])
        df_cleaned = col_sampler(dat=df_cleaned, sample_frac=args.fraction)
        print('number of columns of main data after: ', df_cleaned.shape[1])

    print('correlation analysis')
    # cr = ham_dist(dat=df_cleaned, threshold=0.2)
    # cr = dist_cols(dat=df_cleaned, score_func=adjusted_mutual_info_score)
    cr = vec_nmi(dat=df_cleaned)
    print('finding collinear groups')
    dc_df = db_grouped(dat=cr, report_dir=report_dir, threshold=0.9, needs_pivot=False)

    print('grouping features')
    dc = group_features(dat=dc_df, report_dir=report_dir)

    print('dropping correlated features')
    print('Shape of data before linearity care: ', df_cleaned.shape)
    df_cleaned = cor_remove(df_cleaned, dc)
    print('Shape of data after linearity care: ', df_cleaned.shape)

    # merge with meta data
    df = df.merge(meta_data[args.metavar], left_index=True, right_index=True)
    df_cleaned = df_cleaned.merge(meta_data[args.metavar], left_index=True, right_index=True)

    # model
    print('preparing env')
    select_top = 5
    top_models, train_cols, model_names = fit_models(dat=df_cleaned, meta_var=args.metavar,
                                                     model_type=args.anatype, models_to_select=select_top,
                                                     report_dir=report_dir)

    for i in range(select_top):
        imp = fimp_single(trained_model=top_models[i], model_name=model_names[i],
                          train_cols=train_cols, grouped_features=dc_df,
                          n_positions=positions, report_dir=report_dir)
        dp_plot(dat=imp, model_name=model_names[i], imp_col='standard_value', report_dir=report_dir)

        plot_imp_model(dat=df, trained_model=top_models[i], model_name=model_names[i],
                       train_cols=train_cols, grouped_features=dc_df,
                       meta_var=args.metavar, n_positions=positions, model_type=args.anatype, report_dir=report_dir)

    mean_imp = fimp_top_models(trained_models=top_models, model_names=model_names, train_cols=train_cols,
                               grouped_features=dc_df,
                               n_positions=positions, report_dir=report_dir)
    dp_plot(dat=mean_imp, model_name='mean', imp_col='mean_imp', report_dir=report_dir)

    # plot_imp_all(trained_models=top_models, dat=df_cleaned, train_cols=train_cols,
    #              grouped_features=dc_df, meta_var=args.metavar, model_type=args.anatype,
    #              n_positions=positions, report_dir=report_dir)

    zip_obj = ZipFile(str(report_dir + '/report.zip'), 'w')
    file_names = os.listdir(report_dir)
    for file in file_names:
        if not file.endswith('.zip'):
            zip_obj.write(str(report_dir + '/' + file))

    return print('done!')


# main()
if __name__ == "__main__":
    main()
