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

GINI_GAIN_KEY = "gini_gain"
SPLIT_VALUE_KEY = "split_value"


def gini_gain_calc(dataframe, feature, current_split):
    columns_list = list(dataframe.columns)
    label_column = columns_list[len(columns_list) - 1]

    dataframe = dataframe[[feature, label_column]]

    # obtain all the possible outcomes (labels)
    labels_list = set(dataframe[label_column])
    # remove the first line in the count
    total_row = dataframe[label_column].count()

    print(
        "inside gini gain calc ***{}*** \n columns = {}, label_col = {}, \n labels = {}, total_row = {} , current_split {}".format(
            feature, columns_list, label_column, labels_list, total_row, current_split))

    subset_left = dataframe[dataframe[feature] < current_split]
    subset_right = dataframe[dataframe[feature] > current_split]
    subsets_list = [subset_left, subset_right]

    label_probability = 0
    gini_weighted_gain_total = 0
    for subset in subsets_list:
        gini_subset_total = 0
        # count all the row of the subset
        # remove the first line in the count
        subset_count_row = subset[feature].count()
        print("subset count row {} feature {} ".format(subset_count_row, feature))
        # for each label sum -> p(label)*(1-p(label))
        # TODO
        # check in this way it goes two times -> maybe it is just because it is the same split a/b(1-a/b) = (1-a)/b(1-(1-a)/b)
        for label in labels_list:
            # estimate the impurity
            # remove the first line in the count
            count_label = subset[subset[label_column] == label][feature].count()
            label_probability = count_label / subset_count_row
            gini_subset_total += label_probability * (1 - label_probability)
            print("count_label {} label {} label prob {} GINI SUBSET TOTAL = {}".format(count_label, label, label_probability, gini_subset_total))

        # weighted sum of both splits
        subset_weight = subset_count_row / total_row
        gini_weighted_gain_total += subset_weight * gini_subset_total

    # optimal gini gain
    # final formula optimal_gain - gain_obtained
    maximum_correctness = 0
    for label in labels_list:
        # remove the first line in the count
        count_label = dataframe[dataframe[label_column] == label][feature].count()
        current_probability = count_label / total_row
        maximum_correctness += current_probability * current_probability
        print("*** maximum_impurity *** {} , {} / {}".format(maximum_correctness, count_label, total_row))
    # the max impurity is 1 - probability to correctly assign the label (p(x)^2)
    maximum_impurity = 1 - maximum_correctness

    print("maximum_impurity = {} gini_weighted_gain_total = {} ".format(maximum_impurity, gini_weighted_gain_total))
    # https://stats.stackexchange.com/questions/175087/basic-gini-impurity-derivation/339514#339514
    # the gain = max impurity - impurity we got
    gini_gain = maximum_impurity - gini_weighted_gain_total
    print("gini gain is {}".format(gini_gain))
    return gini_gain


def gini_impurity(dataframe):
    features_list = dataframe.columns[:len(dataframe.columns) - 1]
    gini_dict = {}

    # TODO
    # think about add condition like =/!= for strings and >/< for numbers

    for feature in features_list:
        # create a dict for each feature with impurity and the value that bring the this impurity
        gini_dict.update({feature: {GINI_GAIN_KEY: 0, SPLIT_VALUE_KEY: ''}})
        # 1) order for the feature value -> ìt is only needed for the split
        # dataframe.sort_values(by=feature)
        # 2) create the needed split - size = ( n - 1 ) unique values of the feature
        # if we have 0 5 10, we have to check each possible split -> 2.5 7.5
        unique_set_feature = set(dataframe[feature])
        print("not sorted: {}".format(unique_set_feature))
        unique_set_feature = sorted(unique_set_feature)
        print("sorted: {}".format(unique_set_feature))
        split_list = []
        old_value = None

        for unique_value in unique_set_feature:
            # skip the first iteration
            if old_value:
                current_split = (old_value + unique_value) / 2
                split_list.append(current_split)
                # 3) estimate the gini impurity for the current value
                current_gain = gini_gain_calc(dataframe, feature, current_split)

                # update the dict with the minimum value of impurity - max of the gain
                if current_gain > gini_dict[feature][GINI_GAIN_KEY]:
                    gini_dict.update({feature: {GINI_GAIN_KEY: current_gain, SPLIT_VALUE_KEY: current_split}})
            # update with the used value
            old_value = unique_value
        # 4) estimate the best impurity per split
        print("gini dict is {}".format(gini_dict))

        # TODO
        # take the maximum split and use it as the first split and iterate again for the next step
        # save the split as a condition
    max_gain = 0
    max_split = ''
    feature = ''
    for key, value in gini_dict.items():
        loop_dict = value
        print(value)
        if loop_dict['gini_gain'] > max_gain:
            max_gain = loop_dict['gini_gain']
            max_split = loop_dict['split_value']
            feature = key
    if type(feature) != 'string':
        print(max_gain, max_split, feature)
        dict = {'split_value': max_split, 'feature': feature}


data_path = 'Data'
file_name = 'iris_data.csv'

dataframe = pd.read_csv(os.path.join(data_path, file_name))
training_set = dataframe.head(60)
print(training_set)

gini_impurity(training_set)  # (dataframe)
