# method implementation
# maximizing variance reduction method
"""
(ML)Multi label classification
(SS)Semisupervised
(VR)Variance reduction method
"""

import numpy as np
from DTNode import DecisionTree, DecisionTreeNodeV2
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
from VarianceReductionChampionProposal import VarianceReductionChampionProposal
import pandas as pd
import sys
from numpy.random import default_rng
import math

LEFT_SIMPLE_SUM = 0
RIGHT_SIMPLE_SUM = 1
LEFT_SQUARED_SUM = 2
RIGHT_SQUARED_SUM = 3
LEFT_TOTAL = 4
RIGHT_TOTAL = 5
TOTAL_VARIANCE = 6
LEFT_VARIANCE = 7
RIGHT_VARIANCE = 8
ITERATION_VALUE = 9
BEST_ITERATION_VALUE = 10
BEST_POSITION_INDEX = 11
BEST_ORIGINAL_INDEX = 12


class MLSSVRForestPredictor(BaseEstimator, ClassifierMixin):
    """
    Will not implement as an extension of MLSSVRPRedictor due to changes on main fit algorithm. They will look alike, but we need to separate some common funcionalities.
    """
    def __init__(self, confidence=0.2, leaf_relative_instance_quantity=0.05, compatibilityMatrix=None, unlabeledIndex=None, rec=5000, alpha = 0.5, bfr = './', tag='vrssml', complete_ss = True, trees_quantity = 1 , division_op = 'min'):
        self.confidence = confidence
        self.trees = []
        self.random_generator = default_rng()
        self.tree = DecisionTree()
        self.tree_init = False
        self.trees_quantity = trees_quantity # at least one! solution with 1 decision tree should be equals to MLSSVRPRedictor
        self.leaf_relative_instance_quantity = leaf_relative_instance_quantity
        self.output_file = None
        self.compatibilityMatrix = compatibilityMatrix
        self.unlabeledIndex = unlabeledIndex
        self.ready = False
        self.max_recursion = rec
        self.dataset_ones_distribution = None
        self.alpha = alpha
        self.base_file_root  = bfr
        self.tag = tag
        self.complete_ss = complete_ss
        self.division_op = division_op

    def get_leaf_instance_quantity(self):
        return self.instances.shape[0]*self.leaf_relative_instance_quantity

    def get_sum_and_sum_squared(self, data):
        squared_sum = data.mul(data.values)
        return data.sum(axis=0), squared_sum.sum()

    def get_compatibility_average(self, nodes_list, compatibility):

        total = 0.0
        # print(f'NODES LIST: {nodes_list}')
        subset = compatibility.loc[nodes_list, nodes_list]
        values_overequal_0 = subset[subset >= 0].count()

        subset = subset.replace(-1, 0)
        averages = subset.sum()

        #print(averages)
        #print(values_overequal_0)

        just_over_zero = averages/values_overequal_0

        just_over_zero = just_over_zero.replace(np.nan, 0)
        #useful_cols = [ind for m,ind in zip(averages, averages.index) if m>=0 ]

        #just_over_zero = averages [ useful_cols ]
        #print(just_over_zero.shape)
        if((just_over_zero.shape[0]) == 0):
            return 0, 1

        s = (just_over_zero.transpose()).sum()

        """
        JUST COUNT the ones that can be actually something diferent that -1
        but the average get it from all, so nodes with a lot of unlabeled, with stay as a bad option for division
        for i in nodes_list:
            row = compatibility.loc[i,nodes_list]
            just_positives_and_zeros = row[ (row>-1).any(axis=1)  ]
            total = total + (just_positives_and_zeros.mean())
        return (total/len(nodes_list))
        """

        # right now, compatibility is calculated just on the ones above 0

        t = s/just_over_zero.shape[0]
        # s/len(nodes_list) other hypothesis....
        return t, just_over_zero.shape[0]

    def heuristic_function(self, leftTotal, leftSimpleSum, leftSquaredSum, rightTotal, rightSimpleSum, rightSquaredSum, totalVariance):
        total = leftTotal + rightTotal
        if(leftTotal <= 1):
            return -np.inf
        if(rightTotal <= 1):
            return -np.inf

        tVar = totalVariance
        # second part is divided two times by left total as it is the squared!!
        lVar = (leftSquaredSum-(leftSimpleSum*leftSimpleSum)/leftTotal)/leftTotal
        rVar = (rightSquaredSum-rightSimpleSum
                * rightSimpleSum/rightTotal)/rightTotal
        # change heuristic to division rather than substraction in favor of similar scaling between two different columns.
        # careful with pseudo catregorical values as they will give (i.e. 1 and 0) variances of 0.
        # optimization opportunity, count the different values, if there are two, then just order and gather, do not calculate anything<zoo

        if(lVar <= 0 and rVar <= 0):
            return np.inf

        # change to overcome scaling issues
        return tVar/(leftTotal/total*lVar + rightTotal/total*rVar)

        #return tVar - (leftTotal/total*lVar + rightTotal/total*rVar) # original

    def variance_reduction_with_labels(self, column_series, column_name, compatibility_matrix):
        ordered_colum = column_series.sort_values()
        if(ordered_colum.iloc[0] == ordered_colum.iloc[ordered_colum.size-1]):
            return VarianceReductionChampionProposal(0,
                                                     pd.Index([]),
                                                     pd.Index([]),
                                                     0,
                                                     -np.inf,
                                                     column_name)

        left_index = []
        right_index = []

        row_position = 1
        max_heur = [-np.inf, row_position, -1, -np.inf]

        #for  index,row in ordered_colum.iteritems():
        right_index = ordered_colum.index.tolist()

        for index, row in ordered_colum.iteritems():
            actual_item = row

            left_index.append(index)
            right_index.pop(0)
            heur_left, l = self.get_compatibility_average(
                left_index, compatibility_matrix)
            heur_right, r = self.get_compatibility_average(
                right_index, compatibility_matrix)
            t = l+r
            heur = heur_left*l/t + heur_right*r/t
            #print(f'{index} {heur_left} {heur_right} {heur}')
            if(heur > max_heur[0]):
                max_heur[0] = heur
                max_heur[1] = row_position
                max_heur[2] = index
                max_heur[3] = actual_item

            row_position = row_position + 1

        left_index.clear()
        right_index.clear()

        row_position = 0
        for index, row in ordered_colum.iteritems():
            if(row_position < max_heur[1]):
                left_index.append(index)
            else:
                right_index.append(index)
            row_position = row_position + 1

        vrcp = VarianceReductionChampionProposal(max_heur[2],
                                                 pd.Index(left_index),
                                                 pd.Index(right_index),
                                                 max_heur[3],
                                                 max_heur[0],
                                                 column_name)
        # do not take into account max_heur[0] in the previous structure, is worthless as that measure is the same as compatibility average
        vrcp.set_compatibility_average(max_heur[0])
        return vrcp

    def variance_reduction(self, column_series, column_name):
        ordered_colum = column_series.sort_values()

        #print(f'{ordered_colum.iloc[0]} ==== {ordered_colum.iloc[ordered_colum.size-1]}')
        # case in which all rows have the same data, this will yield a pure node, do we like this?
        if(ordered_colum.iloc[0] == ordered_colum.iloc[ordered_colum.size-1]):
            return VarianceReductionChampionProposal(0,
                                                     pd.Index([]),
                                                     pd.Index([]),
                                                     0,
                                                     -np.inf,
                                                     column_name)

        row_position = 1
        # it is ordered!!
        left_side = 1
        right_side = ordered_colum.size - 1
        left_simple_sum = 0
        left_squared_sum = 0

        average_sum, average_sum_sq = self.get_sum_and_sum_squared(
            ordered_colum)
        variance = (average_sum_sq - average_sum*average_sum
                    / ordered_colum.size)/ordered_colum.size

        right_simple_sum = average_sum
        right_squared_sum = average_sum_sq
        max_heur = [-np.inf, row_position, -1, -np.inf]

        for index, row in ordered_colum.iteritems():
            actual_item = row

            left_simple_sum = left_simple_sum + actual_item
            right_simple_sum = right_simple_sum - actual_item
            left_squared_sum = left_squared_sum + actual_item*actual_item
            right_squared_sum = right_squared_sum - actual_item*actual_item

            heur = self.heuristic_function(
                left_side, left_simple_sum, left_squared_sum, right_side, right_simple_sum, right_squared_sum, variance)

            if(heur > max_heur[0]):
                max_heur[0] = heur
                max_heur[1] = row_position
                max_heur[2] = index
                max_heur[3] = actual_item

            row_position = row_position + 1
            left_side = left_side + 1
            right_side = right_side - 1

        left_index = []
        right_index = []
        row_position = 0
        for index, row in ordered_colum.iteritems():
            if(row_position < max_heur[1]):
                left_index.append(index)
            else:
                right_index.append(index)
            row_position = row_position + 1
        #print(ordered_colum[(ordered_colum < max_heur[3] )].index)
        #print(len(left_index))
        #print("-----" + str( max_heur[1]))
        # ordered_colum[(ordered_colum >= max_heur[3] )].index,

        # print( ordered_colum[(ordered_colum < max_heur[3] )])
        return VarianceReductionChampionProposal(max_heur[2],
                                                 pd.Index(left_index),
                                                 pd.Index(right_index),
                                                 max_heur[3],
                                                 max_heur[0],
                                                 column_name)

        """        point_index,
                                left_indexes,
                                right_indexes,
                                point_value,
                                variance_reduction_coefficient,
                                col_name
                                """

    def generate_tree(self, tree , tree_index):
        # check if root node is None

        #step one
        root_node = DecisionTreeNodeV2(None, self.instances.index, self.labels)

        tree.add_root_node(root_node)
        #print(root_node)
        self.generate_children(root_node, 0, tree_index = tree_index )

    def generate_children(self, node, recursion, node_compat_coeff=0, tree_index = 0):
        # print(
        #     f"generating children rec{recursion} on node with {node.instance_index.size} instances ({self.get_leaf_instance_quantity()})")
        #for each column generate a champion, compare champions
        # each champion should have location to the left and to the right
        #                           average from variance right and left
        #                           compatibilty average for every node in the left set and right set

        # print(node.instance_index.array)
        # print(self.instances)
        indices = node.instance_index.array  # get the indices to work
        # get specific instaces to work with
        instance_index = self.instances.loc[indices,:]

        self.tree_log.write(f"\n tree_{tree_index} ___ "+str(node))
        # not enough nodes
        if(node.is_leaf):
            # self.tree_log.write(f'\nnode is leaf  with {node.instance_index.size}')
            # should check labeled data and set semisupervised fill in step-> update: not here, it will interfere with other legs0
            return

        if(recursion > self.max_recursion):
            node.is_leaf = True
            return
        # get amount of labeled data, if there is not enough labeled data,
        # TODO: when classifying, if a node is pretty label-empty , return the prediction of the father.
        # decide that a node is a leaf by getting the amount of unlabeled data (percent or absolute value)
        # return if this is the case and mark the node as a leaf.
        # this is an idea to be tested still on 10-dic-2021

        # need to document all the decisions!!!

        # amount of labeled data are those from Y that are different to -1.

        # problem! : do we set the labels during the process or after it. If it is after, this process
        # cant happen at the same time, and will be a set of nodes that will call something as : unsupervisedFill() <- IMPLEMENTED ALREADY

        champion = None


        numpy_array = instance_index.to_numpy()

        order_index = numpy_array.argsort(axis=0)
        ordered_set = numpy_array[order_index, np.arange(order_index.shape[1])]
        summary = np.zeros((ordered_set.shape[1], 15), dtype='float')
        n = ordered_set.shape[0]

        summary[:, RIGHT_SIMPLE_SUM] = np.sum(ordered_set, axis=0)
        sum_columns_squared = np.square(summary[:, RIGHT_SIMPLE_SUM])/n
        summary[:, RIGHT_SQUARED_SUM] = np.sum(np.square(ordered_set), axis=0)

        sum_squared_columns_ = np.sum(np.square(ordered_set), axis=0)
        variances = (sum_squared_columns_ - sum_columns_squared)/(n)

        summary[:, TOTAL_VARIANCE] = variances
        if( self.division_op == 'max'):
            summary[:, BEST_ITERATION_VALUE] = -np.inf
        else:
            summary[:, BEST_ITERATION_VALUE] = np.inf
        summary[:, RIGHT_TOTAL] = n
        i = 0
        mask_array = np.zeros( (ordered_set.shape) )

        # penalty importance


        for r in ordered_set:
            mask_array[i,:] += i

            if i == ordered_set.shape[0]-1:
                i += 1
                continue


            r2 = r*r
            # print(r)
            # print(r2)
            summary[:, LEFT_SIMPLE_SUM] += r
            summary[:, RIGHT_SIMPLE_SUM] -= r
            summary[:, LEFT_SQUARED_SUM] += r2
            summary[:, RIGHT_SQUARED_SUM] -= r2
            summary[:, LEFT_TOTAL] += 1
            summary[:, RIGHT_TOTAL] -= 1

            #if i == 0:
            #    i += 1
            #    continue

            # print(r[0])

            summary[:, LEFT_VARIANCE] = (summary[:, LEFT_SQUARED_SUM]-(summary[:,
                                         LEFT_SIMPLE_SUM]*summary[:, LEFT_SIMPLE_SUM])/summary[:, LEFT_TOTAL])/summary[:, LEFT_TOTAL]

            summary[:, RIGHT_VARIANCE] = (summary[:, RIGHT_SQUARED_SUM]-(summary[:,
                                          RIGHT_SIMPLE_SUM]*summary[:, RIGHT_SIMPLE_SUM])/summary[:, RIGHT_TOTAL])/summary[:, RIGHT_TOTAL]

            divisor = ( (summary[:, LEFT_TOTAL]/n)*summary[:, LEFT_VARIANCE]   +     (summary[:, RIGHT_TOTAL]/n)*summary[:, RIGHT_VARIANCE] )

            divisor[ divisor == 0 ] = np.inf

            if( self.division_op == 'max'):
                penalty = self.alpha*( abs(2.0*i/n-1 )  ) # comes from 1 - abs ( (a-b)/(a+b) ) more penalty on more unbalance a=i, b=n-i
                summary[:, ITERATION_VALUE] = (summary[:, TOTAL_VARIANCE] / divisor)*(1-penalty)

            if( self.division_op == 'min'):
                penalty = self.alpha*( abs(2.0*i/n-1 )  ) # comes from 1 - abs ( (a-b)/(a+b) ) more penalty on more unbalance a=i, b=n-i
                summary[:, ITERATION_VALUE] = (summary[:, TOTAL_VARIANCE] / divisor)*(1+penalty)

            #print(f'{summary[:, LEFT_TOTAL]} {summary[:, RIGHT_TOTAL]} {summary[0, RIGHT_VARIANCE]} {summary[0, LEFT_SQUARED_SUM]} {summary[0, LEFT_SIMPLE_SUM]} {summary[0, RIGHT_SQUARED_SUM]} {summary[0, RIGHT_SIMPLE_SUM]} { n}' )

            #print(summary[:, RIGHT_SQUARED_SUM] )
            op = np.minimum
            if( self.division_op == 'max'):
                op = np.maximum
            best_vals = op(
                summary[:, BEST_ITERATION_VALUE], summary[:, ITERATION_VALUE])

            if( self.division_op == 'max'):
                summary[summary[:, BEST_ITERATION_VALUE] < summary[:, ITERATION_VALUE], BEST_POSITION_INDEX] = i
            else:
                summary[summary[:, BEST_ITERATION_VALUE] > summary[:, ITERATION_VALUE], BEST_POSITION_INDEX] = i

            summary[:, BEST_ITERATION_VALUE] = best_vals
            i += 1



        mask_left = mask_array >  summary[:, BEST_POSITION_INDEX]
        mask_right = mask_array <= summary[:, BEST_POSITION_INDEX]

        order_left = order_index.copy() # this has all the indice to the left for each col
        order_left[np.where(mask_left)] = -1

        order_right = order_index.copy() # this has all the indices to the right for each col
        order_right[(np.where(mask_right))] = -1


        # summary has all the data to decide the best value with best iteration
        # value. Ordering will discover the best way to search for
        # compatibility

        # should order the columns, but keeping the index, gather the first one only
        col = 0
        best_col = 0
        best_compatibility = -np.inf
        best_compatibility_left = -np.inf
        best_compatibility_left_list = None
        best_compatibility_right = -np.inf
        best_compatibility_right_list = None
        split_index = -1
        split_value = np.inf
        var_value = 0

        if( self.division_op == 'max'):
            best_variance = -np.inf
        else:
            best_variance = np.inf
        # possible optimization: right now the numpy variance reductioln is done on the entire column set while it could be just on a subset.
        # indexes of columns to be considered:



        features_count = self.instances.shape[1]

        random_columns = self.random_generator.integers(low=0, high=features_count, size= int(math.log( features_count,2 )) )
        random_counter = 0
        for left_col, right_col, variance in zip(order_left.transpose(), order_right.transpose(), summary[:, BEST_ITERATION_VALUE] ):
            # print(f'{instance_index.columns[col]} {best_variance} < {variance}')
            # this is import for 1) speed as get_compatibility_average is expensive, 2) better metric performance

            if(random_counter >= len(random_columns) or random_columns[random_counter] != col  ):

                col += 1
                continue
            # if this is a selected column, then do the process and advance random columns counter
            random_counter += 1
            if( (best_variance > variance and self.division_op == 'min')  or  (best_variance < variance and self.division_op == 'max') ):
                left_c = left_col[ left_col != -1]
                right_c = right_col[ right_col != -1]

                left_compatibility_average, _ = self.get_compatibility_average(
                    instance_index.index[left_c], self.compatibilityMatrix)
                right_compatibility_average, _ = self.get_compatibility_average(
                    instance_index.index[right_c], self.compatibilityMatrix)
                l = 0.5
                r = 0.5
                prom = left_compatibility_average*l+right_compatibility_average*r

                # print(f'{col} {best_variance} < {variance} && {prom} > {best_compatibility} {left_compatibility_average} {right_compatibility_average}')
                if prom > node_compat_coeff and prom > best_compatibility:
                    best_variance  = variance
                    best_compatibility = prom
                    best_compatibility_left = left_compatibility_average
                    best_compatibility_right = right_compatibility_average
                    best_compatibility_left_list = instance_index.index[left_c]
                    best_compatibility_right_list = instance_index.index[right_c]
                    best_col = col
                    split_index = summary[col, BEST_POSITION_INDEX] # its the index of the order set, not the actual index
                    split_index = order_index[ int(split_index), col] # this the actual item. The index works in original pandas dataset
                    # print(f'{split_index} {col}')
                    # print(summary[col, BEST_ITERATION_VALUE])
                    var_value = summary[col, BEST_ITERATION_VALUE]
                    split_value = numpy_array[ int(split_index), col]


                    champion = VarianceReductionChampionProposal(
                        split_index,
                        (best_compatibility_left_list),
                        (best_compatibility_right_list),
                        split_value,
                        var_value,
                        instance_index.columns[best_col]
                    )

                    champion.set_compatibility_average(prom)
                    champion.set_compatibility_average_left(best_compatibility_left)
                    champion.set_compatibility_average_right(best_compatibility_right)

                # print( f'****** {col}** OPT VCRP \t {left_compatibility_average} \t\t {right_compatibility_average}')
            col += 1
        """
        if( best_compatibility > node_compat_coeff):
            # zero was not finally used
            champion = VarianceReductionChampionProposal(
                split_index,
                (best_compatibility_left_list),
                (best_compatibility_right_list),
                split_value,
                var_value,
                instance_index.columns[best_col]
            )

            champion.set_compatibility_average(prom)
            champion.set_compatibility_average_left(best_compatibility_left)
            champion.set_compatibility_average_right(best_compatibility_right)
        """
        # print(champion)
        # champion = None
        # print(list(champion.left.array))
        # print(list(champion.right.array))

        if(champion != None):
            # print(f'champion with {champion.column_name} {champion.value}')
            node.set_decision_column(champion.column_name, champion.value)
            left_child = DecisionTreeNodeV2(node, champion.left,  self.labels, node.level+1, not (
                champion.left.size >= self.get_leaf_instance_quantity()))
            # print( f'V: {champion.variance_coefficient} C: {champion.compatibility_average}')
            node.set_left(left_child)
            self.generate_children(
                left_child, recursion+1, champion.compatibility_average_left)
            right_child = DecisionTreeNodeV2(node, champion.right,  self.labels, node.level+1, not (
                champion.right.size >= self.get_leaf_instance_quantity()))
            # print( f'V: {champion.variance_coefficient} C: {champion.compatibility_average}')
            node.set_right(right_child)
            self.generate_children(
                right_child, recursion+1, champion.compatibility_average_right)

        else:
            node.is_leaf = True

    def fill_semisupervised(self):
        for tree in self.trees:
            # updates internal ones distribution total calculating possible label assignment
            tree.root.fill_semisupervised(self.labels, self.dataset_ones_distribution)

    def fill_ones_distribution(self):
        for tree in self.trees:
            # do not updates internal ones distribution total
            tree.root.fill_ones_distribution(self.labels, self.dataset_ones_distribution)

    def get_params(self, deep=True):
        return {'leaf_relative_instance_quantity': self.leaf_relative_instance_quantity}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def calculate_labels_distribution(self):

        # is the same for every tree in the forest, each one will update according to its own tree
        distribution = (self.labels.replace(-1, 0)).sum()
        labeled_data_count = (self.labels.replace({0: 1, -1: 0})).sum()
        ones_distribution_total = distribution/labeled_data_count
        self.dataset_ones_distribution = ones_distribution_total
        # this won't change! all the trees need the same dataset_ones_distribution

        # print(self.dataset_ones_distribution)

    def fit(self, X, y):
        y = y.copy()
        #rint(self.unlabeledIndex)
        #print(self.compatibilityMatrix)
        #print(self.unlabeledIndex)

        toSelect = None
        cantDo = (self.unlabeledIndex is None)
        noCompat = self.compatibilityMatrix is None

        if(not cantDo):
            print(f"Doing semi superv sim {self.leaf_relative_instance_quantity} {self.alpha} {self.base_file_root} {self.complete_ss}")
            toSelect = y.index.intersection(self.unlabeledIndex)
            y.loc[toSelect, :] = -1

        if(noCompat):
            return self
        self.instances = X
        self.labels = y
        #self.compatibilityMatrix = compatibilityMatrix
        self.tree_log = open(self.base_file_root+'tree_log.txt', 'w')
        print(f"Doing tree generation {self.leaf_relative_instance_quantity} {self.alpha} {self.base_file_root} {self.trees_quantity}")
        # previous to anything all the columns should be normalized, to be able to compare variances
        # testing with heuristic as division, so this should not be done.
        #self.instances = (self.instances-self.instances.min())/(self.instances.max()-self.instances.min())

        # for the amount of trees needed, generate N decision trees and train them all. Remember to select a subset of the columns
        self.trees = []
        for i in range(0,self.trees_quantity):
            # todo: this could be parallelized
            self.trees.append(DecisionTree())
            self.generate_tree(self.trees[i], i)

        self.tree_log.close()
        print(f"Doing semisupervised step {self.leaf_relative_instance_quantity} {self.alpha} {self.base_file_root}")
        # get 1's and 0's distribution
        self.calculate_labels_distribution()
        if( self.complete_ss ):
            self.fill_semisupervised()
            # updates internal ones distribution.
        else:
            self.fill_ones_distribution()
        self.ready = True
        tree_index = 0

        # this should change to another explainable relation with the forest
        for tree in self.trees :
            file = open(self.base_file_root+f'explanation_tree_{tree_index}.txt', 'w')
            tree.root.printRules(file, 0)
            file.close()
            tree_index += 1
        return self

    def predict(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        log_file = None

        if(print_prediction_log):
            log_file = open(
                self.base_file_root+f'prediction_log_{self.tag}.txt', 'w')
        r = []
        preds = []
        #predict(instance , force_prediction = False):
        for index, row in X.iterrows():
            #print(f'-> {index}')

            if(not self.ready):
                prediction = np.zeros(shape=[1, self.labels.shape[1]])
            else:
                prediction = np.zeros(shape=[1, self.labels.shape[1]])
                for tree in self.trees:
                    prediction += tree.root.predict(row, log_file=log_file)
                prediction = np.around(prediction/self.trees_quantity) # over .5 is equal to 1, less is 0
                print(f'{prediction}')

                #print("--------")
                r.append([index, prediction])
                preds.append(prediction.flatten().tolist())

        predictions = np.array(preds)
        if(print_prediction_log):
            log_file.close()
        #print(predictions)
        return predictions

    def predict_with_proba(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        log_file = None

        if(print_prediction_log):
            log_file = open(
                self.base_file_root+f'prediction_log_{self.tag}.txt', 'w')
        r = []
        preds = []
        probs = []
        #predict(instance , force_prediction = False):
        for index, row in X.iterrows():
            #print(f'-> {index}')

            if(not self.ready):
                prediction = np.zeros(shape=[1, self.labels.shape[1]])
                probability = np.zeros(shape=[1, self.labels.shape[1]])
            else:
                prediction = np.zeros(shape=[1, self.labels.shape[1]])
                probability =   np.zeros(shape=[1, self.labels.shape[1]])
                for tree in self.trees:
                    prd, prb = tree.root.predict_with_proba(row, log_file=log_file)
                    prediction += prd
                    probability += prb

                prediction = np.around(prediction/self.trees_quantity) # over .5 is equal to 1, less is 0
                probability = (probability/self.trees_quantity) # do not round

                r.append([index, prediction])
                preds.append(prediction.flatten().tolist())
                probs.append(probability.flatten().tolist())

        predictions = np.array(preds)
        probabilities = np.array(probs)

        if(print_prediction_log):
            log_file.close()
        #print(predictions)
        return predictions, probabilities


    def predict_proba(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        log_file = None

        if(print_prediction_log):
            log_file = open(
                self.base_file_root+f'prediction_log_{self.tag}.txt', 'w')
        r = []
        preds = []
        #predict(instance , force_prediction = False):
        for index, row in X.iterrows():
            #print(f'-> {index}')

            if(not self.ready):
                prediction = np.zeros(shape=[1, self.labels.shape[1]])
            else:

                prediction = np.zeros(shape=[1,self.labels.shape[1] ])

                for tree in self.trees:
                    prb = tree.root.predict_proba(row, log_file=log_file)
                    prediction += prb


                prediction = (prediction/self.trees_quantity).flatten().tolist()

                r.append([index, prediction])
                preds.append(prediction)

        predictions = np.array(preds)
        if(print_prediction_log):
            log_file.close()
        #print(predictions)
        return predictions
