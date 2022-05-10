# deepBreaks: Prioritizing important regions of sequencing data for function prediction #

**deepBreaks** , a computational method, aims to identify important 
changes in association with the phenotype of interest 
using multi-alignment sequencing data from a population.

---
**Citation:**


Mahdi Baghbanzadeh, Ali Rahnavard (2022). **Prioritizing important regions of sequencing data for function prediction**, https://github.com/omicsEye/deeBreaks/.

---
# deepBreaks user manual

## Contents ##
* [Features](#features)
* [deepBreaks](#deepBreaks)
    * [deepBreaks approach](#deepBreaks-approach)
    * [Installation](#installation)
* [Getting Started with deepBreaks](#getting-started-with-deepBreaks)
    * [Test deepBreaks](#test-omeClust)
    * [Options](#options) 
    * [Input](#input)
    * [Output](#output)
    * [Demo](#demo)
* [Tutorials for normalized mutual information calculation](#tutorials-for-normalized-mutual-information-calculation)
* [Support](#Support)
------------------------------------------------------------------------------------------------------------------------------
# Features #
1. Generic software that can handle any kind of sequencing data and phenotypes
2. One place to do all analysis and producing high-quality visualizations
3. Optimized computation
4. User-friendly software
5. Provides a predictive power of most discriminative positions in a sequencing data
# DeepBreaks #
## deepBreaks approach ##
![deepBreaks Workflow overview](img/fig1_overview.png)

## INSTALLATION ##


<span style="color:#033C5A">*If you have a working conda on your system, you can safely skip to step three*</span>.

* First install *conda*  
Go to the [Anaconda website](https://www.anaconda.com/) and download the latest version for your operating system.  
*DO NOT FORGET TO ADD CONDA TO your system PATH*
* Second is to check for conda availability  
open a terminal (or command line for Windows users) and run:
```
conda --version
```
it should out put something like:
```
conda 4.9.2
```
<span style="color:#fc0335">if not, you must make *conda* available to your system for further steps.</span>
if you have problems adding conda to PATH, you can find instructions [here](https://docs.anaconda.com/anaconda/user-guide/faq/).
  
* Third create a new conda environment (let's call it deepBreaks_env) with the following command:
```
conda create --name deepBreaks_env python=3.8
```
* Then activate your conda environment:
```commandline
conda activate deepBreaks_env 
```
* Finally, install *deepBreaks*:
  * before running the following line you should change your directory to the same directory that you have cloned the deepBreaks repo:
```commandline
python -m pip install .
```
or you can directly install if from GitHub:
```commandline
python -m pip install git+https://github.com/omicsEye/deepbreaks
```
------------------------------------------------------------------------------------------------------------------------------

# Getting Started with deepBreaks #

## Test deepBreaks ##

To test if deepBreaks is installed correctly, you may run the following command in the terminal:

```#!cmd
deepBreaks -h
```
Which yields deepBreaks command line options.
```commandline
usage: deepBreaks -h 
--seqfile SEQFILE --seqtype SEQTYPE --meta_data META_DATA --metavar METAVAR --anatype {reg,cl} [--fraction FRACTION]

optional arguments:
  -h, --help            show this help message and exit
  --seqfile SEQFILE, -sf SEQFILE
                        files contains the sequences
  --seqtype SEQTYPE, -st SEQTYPE
                        type of sequence: nuc, amino-acid
  --meta_data META_DATA, -md META_DATA
                        files contains the meta data
  --metavar METAVAR, -mv METAVAR
                        name of the meta var (response variable)
  --anatype {reg,cl}, -a {reg,cl}
                        type of analysis
  --fraction FRACTION, -fr FRACTION
                        fraction of main data to run
```


## Options ##

```
$ deepBreaks -h
```
## Input ##
1. `--seqfile` or `-sf` PATH to a sequence data file
2. `--seqtype` or `-st` sequence type, values are `amino-acid` and `nu` for nucleotides
3. `--meta_data` or `-md` PATH to metadata file
4. `--metavar` or `-mv` name of the meta variable
5. `--anatype` or `-a` analysis type, options are `reg` for regression and `cl` for classification
6. `--fraction` or `-fr` fraction of the main data (sequence positions) to run. it is optional, but you can enter a value between 0 and 1 to sample from the main data set.
## Output ##  
1. correlated positions. We group all the collinear positions together.
2. models summary. list of models and their performance metrics.
3. plot of the feature importance of the top models in *modelName_dpi.png* format.
4. csv files of feature importance based on top models containing, feature, importance, relative importance, group of the position (we group all the collinear positions together)
5. plots and csv file of average of feature importance of top models.
6. box plot (regression) or stacked bar plot (classification) for top positions of each model.

## Demo ##
```commandline
deepBreaks -sf D:/RahLab/deepBreaks/lite_mar/msa_RodOpsinLambdaMax.fasta -st amino-acid -md D:/RahLab/deepBreaks/lite_mar/meta_RodOpsinLambdaMax.tsv -mv LambdaMax -a reg
```
# Tutorials for normalized mutual information calculation ##
**vec_nmi(dat)** is the function for calculating *Normalize Mutual Information*. Rows of the `dat` file are samples an columns are positions in a sequence:
<center>

| | position_1 | position_2 | ... | position_n |
| -- | --------------- | --------------- | --------------- | ----------|
|sample 1 | A | C | ... | G |
|sample 2 | A | C | ... | G |
|sample 3 | T | C | ... | G |

</center>
and the output of the function is a symmetric dataframe with rows and columns equal to positions and the value of the intersection of each row and column is their normalazied mutual information:

<center>

| | position_1 | position_2 | ... | position_n |
| -- | --------------- | --------------- | --------------- | ----------|
|position_1 | 1 | 0.02 | ... | 0.64 |
|position_2 | 0.02 | 1 | ... | 0.02 |
|... | ... | ... | ... | ... |
|position_n | 0.64 | 0.02 | ... | 1 |

</center>

### Support ###

* Please submit your questions or issues with the software at [Issues tracker](https://github.com/omicsEye/deepBreaks/issues).
