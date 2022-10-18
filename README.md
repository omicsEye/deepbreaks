# deepBreaks #

---

**deepBreaks** , a computational method, aims to identify important
changes in association with the phenotype of interest
using multi-alignment sequencing data from a population.

**Key features:**

* Generality, deepBreaks is a generic domain software that can be used for any application that deals sequencing data.
* Interpretation: Rather than checking all possible mutations (breaks), deepBreaks prioritizes only statistically
  promising candidate mutations.
* User-friendly software: one place to do all high-quality visualization and statistical.
* Computing optimization: since sequencing data can get large all modules have been written and benchmarked for
  computing time.
* Tutorial comes with a wide range of real-world applications.

---
**Citation:**

Mahdi Baghbanzadeh, Tyson Dawson, Todd Oakley, Keith A. Crandall, Ali Rahnavard (2022+). **Prioritizing important
regions of sequencing data for function prediction**, https://github.com/omicsEye/deeBreaks/.

---

# deepBreaks user manual #

## Contents ##

* [Features](#features)
* [deepBreaks](#deepBreaks)
    * [deepBreaks approach](#deepBreaks-approach)
    * [Installation](#installation)
      * [Windows Linux Mac](#Windows-Linux-Mac)
      * [Apple M1 MAC](#apple-m1-mac)
* [Getting Started with deepBreaks](#getting-started-with-deepBreaks)
    * [Test deepBreaks](#test-omeClust)
    * [Options](#options) 
    * [Input](#input)
    * [Output](#output)
    * [Demo](#demo)
* [Applications](#applications)
  * [Opsins](#opsins)
  * [HMP](#hmp)
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
* First install *conda*  
Go to the [Anaconda website](https://www.anaconda.com/) and download the latest version for your operating system.  
* For Windows users: DO NOT FORGET TO ADD CONDA TO your system PATH*
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

### Windows Linux Mac ###
If you are **NOT** using an **Apple M1 MAC** please go to the [Apple M1 MAC](#apple-m1-mac) for installation instructions.  
<span style="color:#033C5A">*If you have a working conda on your system, you can safely skip to step three*</span>.  
If you are using windows, please make sure you have both git and Microsoft Visual C++ 14.0 or greater installed.
install [git](https://gitforwindows.org/)
[Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
In case you face issues with this step, [this link](https://github.com/pycaret/pycaret/issues/1254) may help you.
1) Create a new conda environment (let's call it deepBreaks_env) with the following command:
```
conda create --name deepBreaks_env python=3.8
```
2) Activate your conda environment:
```commandline
conda activate deepBreaks_env 
```
3) Install *deepBreaks*:
you can directly install if from GitHub:
```commandline
python -m pip install git+https://github.com/omicsEye/deepbreaks
```
### Apple M1 MAC ###
1) Update/install Xcode Command Line Tools
  ```commandline
  xcode-select --install
  ```
2) Install [Brew](https://brew.sh/index_fr)
  ```commandline
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
3) Install libraries for brew
  ```commandline
  brew install cmake libomp
  ```
4) Install miniforge
  ```commandline
  brew install miniforge
  ```
5) Close the current terminal and open a new terminal
6) Create a new conda environment (let's call it deepBreaks_env) with the following command:
  ```commandline
  conda create --name deepBreaks_env python=3.8
  ```
7) Activate the conda environment
  ```commandline
  conda activate deepBreaks_env
  ```
8) Install packages from Conda
  ```commandline
  conda install numpy scipy scikit-learn==0.23.2
  ```
  Then
  ```commandline
  conda install lightgbm
  pip install xgboost
  ```
9) Finally, install *deepBreaks*:
you can directly install if from GitHub:
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
                        name of the meta var (response variable). This is teh lable will be used as phenotype of interest to find genotypes related to it.
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
6. `--fraction` or `-fr` fraction of the main data (sequence positions) to run. it is optional, 
but you can enter a value between 0 and 1 to sample from the main data set.
7. `--redundant_threshold` or `-rt` threshold for the p-value of the statistical 
tests to drop redundant features. Default value is 0.25.
8. `--distance_threshold` or `-dth` threshold for the distance between positions to put them in clusters. 
features with distances <= than the threshold will be grouped together. Default values is 0.3.
9. `--top_models` or `-tm` number of top models to consider for merging the results. Default value is 3
10. `--plot` plot all the individual positions that are statistically significant. 
Depending on your data, this process may produce many plots.
## Output ##  
1. correlated positions. We group all the collinear positions together.
2. models summary. list of models and their performance metrics.
3. plot of the feature importance of the top models in *modelName_dpi.png* format.
4. csv files of feature importance based on top models containing, feature, importance, relative importance, 
group of the position (we group all the collinear positions together)
5. plots and csv file of average of feature importance of top models.
6. box plot (regression) or stacked bar plot (classification) for top positions of each model.

## Demo ##
```commandline
deepBreaks -sf lite_mar/msa_RodOpsinLambdaMax.fasta -st amino-acid -md lite_mar/meta_RodOpsinLambdaMax.tsv -mv
 LambdaMax -a reg  -dth 0.15 --plot
```
# Applications #
Here we try to use the **deepBreaks** on different datasets and elaborate on the results.
## Opsins ##
[Jupyter Notebook](https://github.com/omicsEye/deepbreaks/blob/master/examples/continuous_phenotype.ipynb)
## HMP ##
[Jupyter Notebook](https://github.com/omicsEye/deepbreaks/blob/master/examples/discrete_phenotype.ipynb)

# Support #

* Please submit your questions or issues with the software at [Issues tracker](https://github.com/omicsEye/deepBreaks/issues).
