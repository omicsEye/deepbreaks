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
    * [Requirements](#requirements)
    * [Installation](#installation)
* [Getting Started with omeClust](#getting-started-with-omeClust)
    * [Test omeClust](#test-omeClust)
    * [Options](#options) 
    * [Input](#Input)
    * [Output](#output)  
* [Tutorials for normalized mutual information calculation](#tutorials-for-distance-calculation)
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
<br><br>
## REQUIREMENTS ##
* [matplotlib==3.3.4](http://matplotlib.org/)
* [Python 3.*](https://www.python.org/download/releases/)
* [numpy==1.19.5](http://www.numpy.org/)
* [pandas==1.2.3](http://pandas.pydata.org/getpandas.html)
* [setuptools==52.0.0](https://setuptools.pypa.io/en/latest/index.html)
* [datetime==4.3](https://docs.python.org/3/library/datetime.html)
* [bio==1.3.3](https://biopython.org/wiki/Getting_Started)
* [scipy==1.5.4](https://scipy.org/)
* [tqdm==4.59.0](https://tqdm.github.io/)
* [seaborn==0.11.1](https://seaborn.pydata.org/)
* [scikit-learn==0.23.2](https://scikit-learn.org/stable/install.html)
<br><br>
## INSTALLATION ##


<span style="color:#033C5A">*If you have a working conda on your system, you can safely skip to step three*</span>.

* First install *conda*  
Go to the [Anaconda website]('https://www.anaconda.com/') and download the latest version for your operating system.  
*DO NOT FORGET TO ADD CONDA TO your system PATH*
* Second is to check for conda availability  
open a terminal (or command line for Windows users) and run:
```
$ conda --version
```
it should out put something like:
```
conda 4.9.2
```
<span style="color:#fc0335">if not, you must make *conda* available to your system for further steps.</span>
if you have problems adding conda to PATH, you can find instructions [here](https://docs.anaconda.com/anaconda/user-guide/faq/).
  
* Third create a new conda environment (let's call it deepBreaks_env) with the following command:
```
$ conda create --name deepBreaks_env python=3.8
```
* Then activate your conda environment:
```commandline
conda activate deepBreaks_env 
```
* Finally, install *deepBreaks*:
```commandline
python setup.py install
```
------------------------------------------------------------------------------------------------------------------------------

# Getting Started with deepBreaks #

## Test deepBreaks ##

To test if deepBreaks is installed correctly, you may run the following command in the terminal:

```
#!cmd

deepBreaks -h

```

Which yields deepBreaks command line options.


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

### Demo run using synthetic data ###

### Support ###

* Please submit your questions or issues with the software at [Issues tracker](https://github.com/omicsEye/deepBreaks/issues).