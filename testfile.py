import os
import csv
import pandas as pd

"""
Gini Impurity is a measurement of the likelihood of an incorrect classification of a new instance of a random variable, 
if that new instance were randomly classified according to the distribution of class labels from the data set.
https://bambielli.com/til/2017-10-29-gini-impurity/ 
https://datascience.stackexchange.com/questions/56013/how-does-decision-tree-with-gini-impurity-calculate-root-node
https://datascience.stackexchange.com/questions/24339/how-is-a-splitting-point-chosen-for-continuous-variables-in-decision-trees
"""


def gini_gain_calc(dataframe, feature, current_split):
    columns_list = dataframe.columns

    label_column = columns_list[len(columns_list) - 1]
    # obtain all the possible outcomes (labels)
    labels_list = set(columns_list)
    total_row = dataframe.count()

    subset_left = dataframe[dataframe[feature] < current_split]
    subset_right = dataframe[dataframe[feature] > current_split]
    subsets_list = [subset_left, subset_right]

    label_probability = 0
    gini_weighted_gain_total = 0
    for subset in subsets_list:
        gini_subset_total = 0
        # count all the row of the subset
        subset_count_row = subset.count()
        # for each label sum -> p(label)*(1-p(label))
        for label in labels_list:
            # estimate the impurity
            count_label = subset[subset[label_column] == label].count()
            label_probability = count_label / subset_count_row
            gini_subset_total += label_probability * (1 - label_probability)
        # weighted sum of both splits
        subset_weight = subset_count_row / total_row
        gini_weighted_gain_total += subset_weight * gini_subset_total

    return gini_weighted_gain_total

def gini_impurity(dataframe):
    features_list = dataframe.columns[:len(dataframe.columns) - 1]
    gini_dict = {}

    for feature in features_list:
        # create a dict for each feature with impurity and the value that bring the this impurity
        gini_dict.update({feature: {"gini_gain": 1, "split_value": ''}})
        # 1) order for the feature value
        dataframe.sort_values(by=feature)
        # 2) create the needed split - size = ( n - 1 ) unique values of the feature
        # if we have 0 5 10, we have to check each possible split -> 2.5 7.5
        unique_set_feature = set(dataframe[feature])
        split_list = []
        old_value = None

        for unique_value in unique_set_feature:
            # skip the first iteration
            if old_value:
                current_split = (old_value + unique_value) / 2
                split_list.append(current_split)
                # 3) estimate the gini impurity for the current value
                current_gain = gini_gain_calc(dataframe, feature, current_split)
                # update the dict with the minimum value of impurity
                if current_gain < gini_dict[feature]:
                    gini_dict.update({feature: {"gini_gain": current_gain, "split_value": current_split}})
            # update with the used value
            old_value = unique_value
        # 4) estimate the best impurity per split


data_path = 'Data'
file_name = 'iris_data.csv'

dataframe = pd.read_csv(os.path.join(data_path, file_name))
print(dataframe.head())

gini_impurity(dataframe)
