import pandas as pd
from random import random
import numpy as np
class DecisionTreeNodeV2:
    def __init__(self, node, index,labels_data,level = 0, is_leaf = False ):
        self.parent = node
        self.instance_index = index
        self.is_leaf = is_leaf
        self.right = None
        self.left = None
        self.level = level
        self.ones_distribution_total = None
        self.decision_column_label = ""
        self.decision_column_value = 0
        # when index is assigned, check the different conditions.
        # Amount of nodes respect to total (hyperparam)
        # Total set labeled
        # and set itself as a leaf node. If is a leaf node, algorithm won't get next part.
    def set_decision_column(self, label, value):
        self.decision_column_label = label
        self.decision_column_value = value
    def set_left(self, node):
        self.left = node
    def set_right(self, node):
        self.right = node
    def calculate_labeled_ratio(self, labels):
        labels_on_instances = labels.loc[self.instance_index.array]
        # dataframe with the count of -1,0,1  for each of the instances
        return labels_on_instances.apply(pd.value_counts)
    def set_column_value(self, x, l, dist, dataset_ones_distribution):
        if x >= 0: # stay with the value already assigned, these are labeled instances
            return x
        #print(f'scv: {x} {l} {dist[l]}')
        # quesdtion , which is best... design experiment

        """
        r = random()*dist[l]
        if r >=  0.5 :
            return 1
        return 0

         modif, Andres formula
         lo modifuque de regreso por pruebas despues de modificar el get_compatibility
         """
        r = random()
        ones_zeros_sum = dataset_ones_distribution[l]*dist[l]+ (1-dataset_ones_distribution[l])*(1-dist[l])
        p1 = dist[l]*(1-dataset_ones_distribution[l])/(1-(ones_zeros_sum))
        if(r<p1): # if less than the extra probability assigned to 1, then is 1
            return 1
        return 0

    """
    This is getting the ones distribution
    """
    def fill_ones_distribution(self, labels, dataset_ones_distribution):
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        if not self.is_leaf:
            self.left.fill_ones_distribution(labels,dataset_ones_distribution)
            self.right.fill_ones_distribution(labels,dataset_ones_distribution)
            return
        #take the sum of all 1's of each column of the labeled instances
        # get only labels vectors of this nodes:
        instances_labels = labels.loc[self.instance_index]
        # another way to get the -1,0,1 : calculate_labeled_ratio().
        distribution = (instances_labels.replace(-1,0)).sum()
        labeled_data_count =  (instances_labels.replace({0:1,-1:0})).sum()
        ones_distribution_total = distribution/(labeled_data_count+0.00001)
        # gets one of these by each leaf node
        self.ones_distribution_total = ones_distribution_total


    """
    This is actually doing two things, getting the ones distribution and filling the -1s
    """
    def fill_semisupervised(self, labels, dataset_ones_distribution):
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        if not self.is_leaf:
            self.left.fill_semisupervised(labels,dataset_ones_distribution)
            self.right.fill_semisupervised(labels,dataset_ones_distribution)
            return
        #take the sum of all 1's of each column of the labeled instances
        # get only labels vectors of this nodes:
        instances_labels = labels.loc[self.instance_index]
        # another way to get the -1,0,1 : calculate_labeled_ratio().
        distribution = (instances_labels.replace(-1,0)).sum()
        labeled_data_count =  (instances_labels.replace({0:1,-1:0})).sum()
        ones_distribution_total = distribution/(labeled_data_count+0.00001)

        
        # on random forest, update cannot happen, at least not on the original data set. Ones distribution from each of them and locally could be updated,but
        # on self.ones_distribution_total
        # should we want to update, they instances have to be classified normally and update the ones ones_distribution_total
        
        
        # for each column, apply the function, should be faster than for each row.
        # also, this set_column_value could be a funmction passed by reference, to be able to change it
        for label, column_series in instances_labels.items() :
            instances_labels[label]=column_series.apply(lambda x: self.set_column_value(x,label,ones_distribution_total,dataset_ones_distribution) )

        # this do not updates original label dataset 
        distribution = instances_labels.sum()

        labeled_data_count =  (instances_labels.replace({0:1})).sum()

        ones_distribution_total = distribution/labeled_data_count

        # gets one of these by each leaf node
        self.ones_distribution_total = ones_distribution_total
        # original labels would still be -1, but the distribution total on this one would have been updated
        

    def get_column_prediction(self, x):
        r = random()
        """ originally
        if x >=  0.5 :
            return 1
        return 0

        """
        # another idea: calculate the relation in this nodes between the features and the labels. The average of the features or something
        if r >=  1-x : # the bigger the 1, the most probable to be 1
            return 1
        return 0

    def get_proba_column_prediction(self, x):
        return x

    def get_prediction_with_proba_column_prediction(self, x):
        r = random()

        if r >=  1-x : # the bigger the 1, the most probable to be 1
            return 1, x
        return 0, x

    def predict_with_proba(self, instance , force_prediction = False, original_labels = None, log_file = None, level = 0):
        # go on to the path
        tabs = "".join(["\t"]*level)
        if self.is_leaf or force_prediction:
            prediction = []
            probability = []

            for pr in self.ones_distribution_total:
                pred, prob = self.get_prediction_with_proba_column_prediction(pr)
                prediction.append(pred)
                probability.append(prob)

            # prediction, probability = self.ones_distribution_total.apply(self.get_prediction_with_proba_column_prediction)

            if( log_file is not None):
                scores = ["%.6f" % x for x in self.ones_distribution_total]
                vector_score = " ".join(scores)
                p = " ".join(["%.1f" % x for x in prediction])
                # log_file.write(f'{tabs} vector_scores:[{vector_score}] prediction:[{p}]\n')
                log_file.write(f'{vector_score}\n')
                #log_file.write("\n----------------------------------------------------------------------------------\n")

            #if(original_labels != None)
            # this is just to be compliant with other calls.
            return np.asarray(prediction) , np.asarray(probability)
            # could update the database to keep the ones_distribution_total, right now, it is not doing that
        #decide where to go
        is_lefty = instance[self.decision_column_label] < self.decision_column_value

        #if( log_file is not None):
        #    log_file.write(f'{tabs} col_lab:{self.decision_column_label} col_val{self.decision_column_value} inst_val:{instance[self.decision_column_label]} \n')

        if( is_lefty):
                return self.left.predict_with_proba(instance, log_file = log_file, level = level+1)
        return self.right.predict_with_proba(instance ,log_file = log_file, level = level+1)


    def predict_proba(self, instance , force_prediction = False, original_labels = None, log_file = None, level = 0):
        # go on to the path
        tabs = "".join(["\t"]*level)
        if self.is_leaf or force_prediction:
            prediction = self.ones_distribution_total.apply(self.get_proba_column_prediction)

            if( log_file is not None):
                scores = ["%.6f" % x for x in self.ones_distribution_total]
                vector_score = " ".join(scores)
                p = " ".join(["%.1f" % x for x in prediction])
                # log_file.write(f'{tabs} vector_scores:[{vector_score}] prediction:[{p}]\n')
                log_file.write(f'{vector_score}\n')
                #log_file.write("\n----------------------------------------------------------------------------------\n")

            #if(original_labels != None)

            return prediction.values
            # could update the database to keep the ones_distribution_total, right now, it is not doing that
        #decide where to go
        is_lefty = instance[self.decision_column_label] < self.decision_column_value

        #if( log_file is not None):
        #    log_file.write(f'{tabs} col_lab:{self.decision_column_label} col_val{self.decision_column_value} inst_val:{instance[self.decision_column_label]} \n')

        if( is_lefty):
                return self.left.predict_proba(instance, log_file = log_file, level = level+1)
        return self.right.predict_proba(instance ,log_file = log_file, level = level+1)

    """
        force prediction will have a purpose when a validation for the amount of labeled data is done.
        When the amount of labeled data is no enough, it will have to force the return prediction of the father (not implemented)
        This is not needed as the method on semisupervised step updates everything
    """
    def predict(self, instance , force_prediction = False, original_labels = None, log_file = None, level = 0):
        # go on to the path
        tabs = "".join(["\t"]*level)
        if self.is_leaf or force_prediction:
            prediction = self.ones_distribution_total.apply(self.get_column_prediction)

            if( log_file is not None):
                scores = ["%.6f" % x for x in self.ones_distribution_total]
                vector_score = " ".join(scores)
                p = " ".join(["%.1f" % x for x in prediction])
                # log_file.write(f'{tabs} vector_scores:[{vector_score}] prediction:[{p}]\n')
                log_file.write(f'{vector_score}\n')
                #log_file.write("\n----------------------------------------------------------------------------------\n")

            #if(original_labels != None)

            return prediction.values
            # could update the database to keep the ones_distribution_total, right now, it is not doing that
        #decide where to go
        is_lefty = instance[self.decision_column_label] < self.decision_column_value

        #if( log_file is not None):
        #    log_file.write(f'{tabs} col_lab:{self.decision_column_label} col_val{self.decision_column_value} inst_val:{instance[self.decision_column_label]} \n')

        if( is_lefty):
                return self.left.predict(instance, log_file = log_file, level = level+1)
        return self.right.predict(instance ,log_file = log_file, level = level+1)

    def __str__(self):
        tabs = "".join(["\t"]*self.level)
        return f'{tabs}({self.level}) node: ({self.is_leaf}) [({self.instance_index.size})] {self.instance_index}'

    def printRules(self, file, level = 0):
        st = "\n"
        st = st + "\t"*level

        if self.is_leaf :
            prediction = self.ones_distribution_total.apply(self.get_column_prediction)
            p = ""
            for v in prediction:
                p = p + f' {v}'
            scores = ["%.6f" % x for x in self.ones_distribution_total]
            ones_d = " ".join(scores)
            st = st + f"pred on {len(self.instance_index)} nodes is pred_labels: [{p}] dist:[{ones_d}]"
        else:
            st = st + f"Decision on col:{self.decision_column_label} with value {self.decision_column_value} on {len(self.instance_index)} nodes"
        file.write(st)
        if not ( self.left is None ):
            self.left.printRules(file, level+1)
        if not ( self.left is None ):
            str = self.right.printRules(file,level+1)

class DecisionTreeNode:
    tree_node_id = 0
    def __init__(self, col_index, col_value, par = None , label_vector = None):
        self.parent = par
        self.column_index = col_index
        self.column_value = col_value
        self.left = None
        self.right = None
        self.node_index = DecisionTreeNode.tree_node_id
        DecisionTreeNode.tree_node_id = DecisionTreeNode.tree_node_id+1
        # this one will only be filled if the node is a leaf
        self.label_vector = label_vector
        self.is_leaf = True if not (label_vector is None) else False
        self.correct = 0
        self.true_negative = 0
        self.false_positive = 0
        self.true_positive = 0
        self.false_negative = 0

    def set_left_node(self,node):
        self.left = node
    def set_right_node(self,node):
        self.right = node
    def decide(self, row):
        if( row[self.column_index] < self.column_value):
            print("left on", self.node_index)
            return self if self.left is None else self.left

        print("right on", self.node_index)
        return self if self.right is None else self.right
    def print_node(self, leftTabs):
        tabs = "" + str(("_")*leftTabs)

        res = "\n"+tabs+"Node :  ["+str(self.node_index)+"] " + str(self.column_index)+"("+str(self.column_value)+")\n"
        if( self.is_leaf):
            res = "\n" + tabs+"Node :  ["+str(self.node_index)+"] " + str(self.label_vector) +"\n"

        left_print = self.left.print_node(leftTabs+1) if not self.left is None else ""
        right_print = self.right.print_node(leftTabs+1)  if not self.right is None else ""

        res = res + left_print + "\n"
        res = res + right_print
        return res
    def check_original_assignment(self, row):
        # mini confussion matrix
        falseWhenTrue = 0
        trueWhenFalse = 0
        falseWhenFalse = 0
        trueWhenTrue = 0
        correct = 0
        for i in range(0,len(self.label_vector)):
            if( self.label_vector[i][0] > self.label_vector[i][1] ):
                if( row["label_"+str(i)] == True):
                    falseWhenTrue = falseWhenTrue+1
                else:
                    falseWhenFalse = falseWhenFalse  + 1
                    correct = correct+1
            else:
                if( row["label_"+str(i)] == False):
                    trueWhenFalse = trueWhenFalse+1
                else:
                   trueWhenTrue = trueWhenTrue+1
                   correct = correct+1
        print("Correct " , correct)
        print("InCorrect " , (falseWhenTrue+trueWhenFalse) )
        self.correct = correct
        self.true_negative = falseWhenTrue
        self.false_positive = trueWhenFalse
        self.true_positive = trueWhenTrue
        self.false_negative = falseWhenFalse


class DecisionTree:
    def add_root_node(self,node):
        self.root = node
    def print_tree(self):
        print( self.root.print_node(0) )
    def classify(self, row):
        return self.root.decide(row)
    def read_tree(self, file):
        print("when reading file rebuild tree to classify")
