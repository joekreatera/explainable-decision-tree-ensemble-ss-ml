import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DownloadHelper import *
from pandas.plotting import scatter_matrix
from random import random
from sklearn.datasets import fetch_openml
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score

# for roc curve
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from math import isnan
import datetime
import pprint
import sys
# https://stackoverflow.com/questions/54721497/scipy-sparse-indicator-matrix-from-arrays
# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html


print(__doc__)

class Rule():
    rule_id= 0
    def __init__(self, antecedent = [], consequent = []):
        # this should generate a rule based on nodes passed .
        # this idea is to gather columns and values in different arrays.
        # gather also label vector and ones distribution
        # This rule should also have a fit method. This will test the training set agains each rule. If the rule is applied, count for the metrics (this last thing is not clear)
        # quality metric
        self.antecedent = antecedent # lists of columns and values
        self.consequent = consequent # lists of labels and distribution
        Rule.rule_id += 1
        self.id = Rule.rule_id

    def get_antecedent(self, is_left  = False):
        if( len(self.antecedent) == 0 ):
            return []

        prev_list = self.antecedent[0:len(self.antecedent)-1]
        last =  self.antecedent[-1]
        last = (last[0], last[1],'<=' if is_left else '>')

        return  prev_list + [last]
        # return self.antecedent

    def get_consequent(self):
        return self.consequent

    def set_antecedent(a):
        # anteedent is the list [(column, value, side)]
        self.antecedent = a

    def set_consequent(c):
        self.consequent = c

    def predict(self, instance , truth):
        # check if rule is applicable


        for ant in self.antecedent:
            col_name = (ant[0]).strip()
            # if it gets here, is applicable
            split_value = ant[1]
            value_to_evaluate = instance[col_name]
            op = ant[2]
            if not (   eval( f'{value_to_evaluate} {op} {split_value}' )  ):
                # if just one is not applicable, do not do aything
                return None
        # if it is, get label vector
        # print("could be!")

        # labels =  "".join([str(i) for i in self.consequent[0]])
        assigned = self.consequent[0]
        # calculate tp tn fp fn for this instance on this rule
        i = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for c in list(truth):
            if( c == str(assigned[i]) and c == '0'):
                tn += 1
            elif( c == str(assigned[i]) and c == '1'):
                tp += 1
            elif( c == '0'):
                fp += 1
            else:
                fn += 1
            i+=1
            # both strings are the same size and order

        return (assigned, tp, tn, fp, fn)
    def __str__(self):
        labels = "No labels"
        dist = "No dist"
        if( len(self.consequent) > 0 ):
            labels =  "".join([str(i) for i in self.consequent[0]])
            dist = "".join([ f'{i:5.2f}' for i in self.consequent[1]])

        return f"Rule {self.id} {self.antecedent} {labels}{dist}"

class TreeNode():
    def __init__(self,base_str, parent):
        """
        prev rules include the tuples of column, value to execute. Is understood that each rule is left(less than or equal to) and right (greater than)
        """
        self.base_str = base_str
        self.left = None
        self.right = None
        self.parent = parent
        self.leaf = (base_str.find('Decision on col') == -1)
        self.parse_rule(self.leaf)
        self.previous_rule = None
        self.this_rule = None

    def set_previous_rule(self,prev_rule, is_left = False):

        self.previous_rule = prev_rule

        if( self.leaf ):
            self.this_rule = Rule( self.previous_rule.get_antecedent(is_left) , (self.label_vector, self.ones_dist) )
        else:
            ant = self.previous_rule.get_antecedent(is_left) + [(self.column, self.value, 'nd')]
            self.this_rule = Rule( ant )


    def is_leaf(self):
        return self.leaf
    def parse_rule(self, is_leaf):
        self.label_vector =""
        self.ones_dist = "" # alike support for this label on this node, after refill
        self.column = ""
        self.value = None

        if is_leaf:
            start = self.base_str.find("pred_labels:") + 15
            end = self.base_str.find("]", start)
            self.label_vector =  [ int(x) for x in self.base_str[start:end].split(" ") ]

            start = self.base_str.find("dist:") + 6
            end = self.base_str.find("]", start)
            self.ones_dist = [float(x) for x in self.base_str[start:end-1].split(" ")]
            return

        start = self.base_str.find("col:") + 4
        end = self.base_str.find("with value", start)
        self.column = self.base_str[start:end]

        start = self.base_str.find("with value") + 10
        end = self.base_str.find(" on ", start)
        self.value = float(self.base_str[start:end])

    def add_left(self,node):
        self.left = node
    def add_right(self,node):
        self.right = node
    def add_node(self,n):
        if(self.left is None):
            self.add_left(n)
            return True

        if(self.right is None):
            self.add_right(n)
            return False

        raise Exception(f'{n}' , f'A binary tree cannot have more than 2 childs')
    def get_full_rule(self):
        """
        return the rule up to this node. If is leaf, the rule will have consequent
        """
        return self.this_rule
        # return self.base_str
    def get_parent(self):
        return self.parent
    def print(self, level = 0):

        lv=  "".join([str(i) for i in self.label_vector])
        ret = f'{self.column} {self.value} {lv} {self.ones_dist[0:3]}...'
        lev = '\t'*level
        c1 = ''
        c2 = ''
        if( self.left is not None):
            c1 = f"\n{lev}" + (self.left.print(level+1))

        if( self.right is not None):
            c2 = f"\n{lev}" + (self.right.print(level+1))
        return '\n' + lev + ret + c1 + c2

def build_rules_from_tree_str(lines_str):
    tree = None
    rules = lines_str.split('\n')
    prev_indentation = -1

    built_rules = []
    actual_rule = Rule()
    for r in rules:
        if(r.strip() == ""):
            continue

        indentation = r.count('\t')

        for j in range(indentation,prev_indentation+1): # if its the same (sibbling) the parent should be executed at least once
            if( tree.get_parent() is not None):
                tree = tree.get_parent()
            # if indentation is more than prev_identation, this should not happen


        #print(r)
        node = TreeNode(r, tree)
        is_leaf = node.is_leaf()

        if( tree is None):
            tree = node
            node.set_previous_rule(actual_rule)
        else:
            is_left = tree.add_node(node)
            node.set_previous_rule( tree.get_full_rule() , is_left )

        # last step, set actual node on last node created
        tree = node

        if(is_leaf):
            # obtained rule
            built_rules.append( tree.get_full_rule() )
            # print(tree.get_full_rule())

        prev_indentation = indentation

    for j in range(0,prev_indentation): # if its the same (sibbling) the parent should be executed at least once
        if( tree.get_parent() is not None):
            tree = tree.get_parent()
    # print( tree.print() )

    # rules = ([str(b) for b in built_rules])
    #for r in rules:
    #    print(r)
    return built_rules

def do_process(dataset_root,dataset_filename,datatest_size,unlabeled_size,experiment_name,test_train_dif, complete_ss = True ):
    """
        function to be called from outside.
        dataset_root: base folder directory
        dataset_filename: name of dataset file
        datatest_size: size of testing dataset in percentage
        unlabeled_size: size of the instances that will have labels removed, in percentage
        experiment_name: experiment name as str
        test_train_dif whether bootstrap has to divide the testing and training (false), or is already divided (true). dataset for this is named <dataset_filename>-{training|test}.csv


        Big ASSUMPTION: explanation files for trees are one folder higher than dataset
    """

    """
    Label columns are on label_columns list
    instance attributes are on instance_columns list
    """
    #print('Number of arguments:', len(sys.argv), 'arguments.')
    #print('Argument List:', str(sys.argv))
    print(f"Starting {dataset_root} {dataset_filename} : {experiment_name}")
    """
    dataset_root = sys.argv[1]
    dataset_filename = sys.argv[2]
    datatest_size = float(sys.argv[3])
    unlabeled_size = float(sys.argv[4])
    experiment_name =  sys.argv[5]
    test_train_dif = sys.argv[6]
    """

    if( test_train_dif == "true"):
        test_train_dif = True
    else:
        test_train_dif = False

    label_columns_file = open(dataset_root + '/' + 'label_columns.cols','r')
    label_columns_str = label_columns_file.read()
    label_columns_str = label_columns_str.rstrip()
    label_columns = label_columns_str.split(",")
    label_columns_size = len(label_columns) # assumes they are at the end of the end

    # print ( label_columns )

    originalTime = ( time.time() )
    dataset = None
    train_set  =None
    test_set = None

    if( not test_train_dif):
        dataset = getFromOpenML(dataset_filename,version=4,ospath=dataset_root+'/', download=False, save=True)
        train_set,test_set = train_test_split(dataset, test_size=datatest_size, random_state=43)
    else:
        train_set = getFromOpenML(dataset_filename+('-')+'train',version=4,ospath=dataset_root+'/', download=False, save=True)
        test_set = getFromOpenML(dataset_filename+('-')+'test',version=4,ospath=dataset_root+'/', download=False, save=True)

    labeled_instances, unlabeled_instances =  train_test_split(train_set, test_size=unlabeled_size, random_state=43) # simulate unlabeled instances
    # will be done internally by predictor,simulate semisupervised dataset
    #train_set.loc[unlabeled_instances.index, label_columns] = -1

    # get instance columns
    # get instance labels on Y
    instance_columns = train_set.columns.to_list()
    for col in label_columns:
        #print('col:  ' + col )
        instance_columns.remove(col)

    trees = []
    for i in range(0,1000):
        dataset_specific = dataset_root.split('/')
        ds = dataset_specific[ len(dataset_specific)-1 ]
        file_to_load = dataset_root + "/" + ds + f"explanation_tree_{i}.txt"

        try:
            file_to_load = open(file_to_load)
            trees.append( file_to_load.read() )
            # print(trees[i])
        except FileNotFoundError as fnfe:
            break

    # export rules

    i = 0
    total_rules = []
    total_indexed_rules = {}

    for tree in trees:
        total_rules = total_rules + build_rules_from_tree_str(tree)
        i += 1
        # two types of rules: leafs and internal nodes
        # if leaf, count in training labeled data, for each label that reached the leaf node
        # count true Positive
        # count false positive
        # count true negative
        # count false negative
        # return  precision, hamming accuracy, f-measure, subset_accuracy



    def do_rules(instance):
        for r in total_rules:
            total_indexed_rules['r_' + str(r.id)] = r
            prediction = r.predict(instance, instance["true_labels"] )
            if(prediction is not None):
                """
                returns as : tp, tn, fp, fn
                """
                instance['r_'+str(r.id)] = prediction[0]
                instance['r_'+str(r.id)+"tp"] = prediction[1]
                instance['r_'+str(r.id)+"tn"] = prediction[2]
                instance['r_'+str(r.id)+"fp"] = prediction[3]
                instance['r_'+str(r.id)+"fn"] = prediction[4]

        return instance

    labeled_instances["true_labels"] = ""

    for col in label_columns:
        labeled_instances["true_labels"] = labeled_instances["true_labels"] + labeled_instances[col].astype(str)


    rules_columns = []
    rules_alone = []
    predictions = labeled_instances.apply(do_rules, axis=1)

    for r in predictions.columns.to_list():
        if( r.find("tp") != -1 or r.find("tn") != -1 or r.find("fp") != -1 or r.find("fn") != -1 ):
            rules_columns.append(r)

        if(r.find("r_") != -1 and r.find("tp") == -1 and r.find("tn") == -1 and r.find("fp") == -1 and r.find("fn") == -1 ):
            rules_alone.append(r)

    rules_table = predictions[rules_columns]
    rules_table = rules_table.sum(axis = 0)
    rules_table.to_csv('pred_rule_explain.csv')
    rules_table = rules_table.to_frame().transpose()

    final_explanation = pd.DataFrame( columns={'rule','rule_details','precision','hamming_accuracy','f_measure','recall', 'tp','tn','fp','fn'})
    for i in rules_alone:
        # gather tp,fn,fp,tn and calculate on a data frame the metrics
        tp = rules_table[i+"tp"][0]
        tn = rules_table[i+"tn"][0]
        fp = rules_table[i+"fp"][0]
        fn = rules_table[i+"fn"][0]
        precision = tp*1.0/(tp+fp+ 0.0000001)
        recall = tp*1.0/(tp+fn+ 0.0000001)
        hamming = (tp+tn)*1.0/(tp+fp+tn+fn+0.0000001)
        beta = 0.5
        f_measure = (beta*beta+1)/( (beta*beta)/(recall+ 0.0000001) + (1/(precision + 0.0000001) ) + 0.000001 )
        detail = total_indexed_rules[i]
        final_explanation = final_explanation.append( {'rule':i, 'rule_details':str(detail),'precision':precision, 'recall':recall, 'hamming_accuracy':hamming, 'f_measure':f_measure , 'tp':tp,'tn':tn,'fp':fp,'fn':fn}, ignore_index = True )







    # change name for the dataset and experiment name
    final_explanation.to_csv("final_explanation.csv")





# for backwards compatibility
if __name__ == "__main__":
    dataset_root = sys.argv[1]
    dataset_filename = sys.argv[2]
    datatest_size = float(sys.argv[3])
    unlabeled_size = float(sys.argv[4])
    experiment_name =  sys.argv[5]
    test_train_dif = sys.argv[6]
    # this setting will complete semi supervised always

    do_process(dataset_root,
    dataset_filename,
    datatest_size,
    unlabeled_size,
    experiment_name,
    test_train_dif)
