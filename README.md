
# Semisupervised Multilabel Tree Ensemble

This github includes the code for "An explainable method based on a Decision Tree Ensemble with Variance Reduction for Semi-supervised Multi-label classification".

> Code was tested on python3.7. This, or a superior version, is the recommended setting to run it.  

To run the code, the **run.py** script is the main point of execution. With the code, there are two subsets of emotions dataset (Trohidis K, Tsoumakas G et al, 2008), taken out from MULAN Databases (Multilabel Datasets, 2019).

## Datasets

A common setup for the datasets should be:
**dataset-train.csv** : csv that includes ONLY traning data, features and labels
**dataset-test.csv** : vsc that includes ONLY test data, features and labels
**label_columns.cols** : comma delimited list of the labels column names

## Packages
The code needs the following packages:
```
pip install scikit-learn
```
**MLSSVRForectPredictor.py** includes the MLSSVRForestPredictor class that extends from scikit's BaseEstimator and ClassifierMixin. This allows to include the method on scikit pipelines and other tools. As the methods fit, predict and predict_proba are implemented, scoring analysis and parameter search routines can be performed with the proposed technique.

A very important consideration is that the model needs a similarity matrix. An implementation of how to get this is already done in **run.py** along with a full setup to measure model AUC (using scikit's and a reimplementation based on ESMC matlab method) (Akbarnejad A.H., Baghshah M.S. , 2019)

In order to execute run from shell, a command like this:

```
python run.py ./ emotions 0.1 0.3 experiment_svm true
```

This means:
1. folder (./) : dataset root folder. The program will use the folder to output a series of files and find the dataset csv's
2. name of the file (emotions). The program will look for a file with this name. If parameter (6) is true, then it will look for <name>-test.csv and <name>-training.csv
3. data-test (0.1): prcentage of training data that will use for testing. This parameter is ignored if parameter (6) is true
4. data unlabeled size (0.3): prcentage of training data that will become unsupervised. Automatically all of its labels will become -1.
5. Experiment name (experiment_svm): output statistical and chart files will prefix with this name.
6. Training and Testing Divided (true): when files are already divided this should be true. When there's just a single file and training and testing sets have to be divided, parameter should be false.

Compatibility matrix is calculated on a different dataframe than the original labels. Internally, MLSSVRForectPredictor will reset the labels from the unlabeled index rows to -1.

Alternatively **explain.py** is also included within this files.  The call for the command would look like:
```
python explain.py ./ emotions 0.1 0.3 experiment_svm true
```
This script requires the files generated by run.py (<dataset name>tree_<N>_explanation.txt) in the dataset folder. This command will generate the quality measures for the different mined rules taken out from the decision trees. Can look up to 1000 trees (hardcoded limit).

## Parameters on the model

- Trees Quantity
: amount of trees to calculate, common options are 50,100. Literature reports up to 250 as ideal.
- Division op
: tests for other optimization measures, leave it in max for paper results.
- Complete ss
: left for optional implementation, leave it on True.
- Alpha
: Penalization for unbalaced nodes (|left|>>|right| or viceversa). 0 means no penalization, 1 means full penalization.
- Leaf relative instance quantity
: Amount of nodes (%) that will be the minimum acceptable for a node to be spawned
- Unlabeled index
: row index that will be considered to set all labels to -1
- Compatibility Matrix
Dataframe that references every node against each other. The cells contain the intersection over union index (jaccard) for the nodes. Nodes that are being considered as unsupervised set -1 to the cell.

### References:

K. Trohidis, G. Tsoumakas, G. Kalliris, I. Vlahavas. "Multilabel Classification of Music into Emotions". Proc. 2008 International Conference on Music Information Retrieval (ISMIR 2008), pp. 325-330, Philadelphia, PA, USA, 2008.
Multilabel Datasets. (2019). http://mulan.sourceforge.net/datasets-mlc.html
Akbarnejad, A. H., & Baghshah, M. S. (2019). An Efficient Semi-Supervised Multi-label Classifier Capable of Handling Missing Labels. IEEE Transactions on Knowledge and Data Engineering, 31(2), 229–242. https://doi.org/10.1109/TKDE.2018.2833850 with its github on https://github.com/Akbarnejad/ESMC_Implementation
