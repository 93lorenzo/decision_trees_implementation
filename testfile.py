import os
import pandas as pd
import pprint
import json
import ast

"""
Gini Impurity is a measurement of the likelihood of an incorrect classification of a new instance of a random variable, 
if that new instance were randomly classified according to the distribution of class labels from the data set.
https://bambielli.com/til/2017-10-29-gini-impurity/ 
https://datascience.stackexchange.com/questions/56013/how-does-decision-tree-with-gini-impurity-calculate-root-node
https://datascience.stackexchange.com/questions/24339/how-is-a-splitting-point-chosen-for-continuous-variables-in-decision-trees
"""

GINI_GAIN_KEY = "gini_gain"
SPLIT_VALUE_KEY = "split_value"
SONS_DICT_KEY = "sons_dict"
MORE_EQUAL = '>='
LESS = '<'
EQUAL = '=='
DIFFERENT = '!='
FEATURE_KEY = 'feature'
PRUNE_LEVEL = 10
TRAINED_TREES = 'trained_trees'
output_flow_dict = {}
features_list = []
DATA_PATH = 'Data'


def gini_gain_calc(dataframe, feature, current_split, is_numeric=True):
    columns_list = list(dataframe.columns)
    label_column = columns_list[len(columns_list) - 1]

    full_data_frame_deep_copy = dataframe.copy()
    dataframe = dataframe[[feature, label_column]]

    # obtain all the possible outcomes (labels)
    labels_list = set(dataframe[label_column])
    # remove the first line in the count
    total_row = dataframe[label_column].count()

    # print(  "inside gini gain calc ***{}*** \n columns = {}, label_col = {}, \n labels = {}, total_row = {} , current_split {}".format(
    #        feature, columns_list, label_column, labels_list, total_row, current_split))

    if is_numeric:
        subset_left = dataframe[dataframe[feature] < current_split]
        subset_right = dataframe[dataframe[feature] >= current_split]
    elif not is_numeric:
        subset_left = dataframe[dataframe[feature] == current_split]
        subset_right = dataframe[dataframe[feature] != current_split]

    subsets_list = [subset_left, subset_right]

    label_probability = 0
    gini_weighted_gain_total = 0
    for subset in subsets_list:
        gini_subset_total = 0
        # count all the row of the subset
        # remove the first line in the count
        subset_count_row = subset[feature].count()
        # print("subset count row {} feature {} ".format(subset_count_row, feature))
        # for each label sum -> p(label)*(1-p(label))
        for label in labels_list:
            # estimate the impurity
            # remove the first line in the count
            count_label = subset[subset[label_column] == label][feature].count()
            label_probability = count_label / subset_count_row
            gini_subset_total += label_probability * (1 - label_probability)
            # print("count_label {} label {} label prob {} GINI SUBSET TOTAL = {}".format(count_label, label,
            #                                                                            label_probability,
            #                                                                            gini_subset_total))

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
        # print("*** maximum_impurity *** {} , {} / {}".format(maximum_correctness, count_label, total_row))
    # the max impurity is 1 - probability to correctly assign the label (p(x)^2)
    maximum_impurity = 1 - maximum_correctness

    # print("maximum_impurity = {} gini_weighted_gain_total = {} ".format(maximum_impurity, gini_weighted_gain_total))
    # https://stats.stackexchange.com/questions/175087/basic-gini-impurity-derivation/339514#339514
    # the gain = max impurity - impurity we got
    gini_gain = maximum_impurity - gini_weighted_gain_total
    # print("gini gain is {}".format(gini_gain))
    if is_numeric:
        full_data_left = full_data_frame_deep_copy[full_data_frame_deep_copy[feature] < current_split]
        full_data_right = full_data_frame_deep_copy[full_data_frame_deep_copy[feature] >= current_split]
        sons_dict = {LESS: full_data_left, MORE_EQUAL: full_data_right}
    elif not is_numeric:
        full_data_left = full_data_frame_deep_copy[full_data_frame_deep_copy[feature] == current_split]
        full_data_right = full_data_frame_deep_copy[full_data_frame_deep_copy[feature] != current_split]
        sons_dict = {EQUAL: full_data_left, DIFFERENT: full_data_right}

    return gini_gain, sons_dict


def training(dataframe, current_level, current_tree_number=1):
    columns_list = list(dataframe.columns)
    label_column = columns_list[len(columns_list) - 1]
    # obtain all the possible outcomes (labels)
    labels_list = list(dataframe[label_column])
    ##################
    # STOP CONDITION #
    # print("### len set label list = {} and label list = {}".format(len(set(labels_list)), labels_list))
    if current_level > PRUNE_LEVEL or len(set(labels_list)) == 1:
        res = max(set(labels_list), key=labels_list.count)  # two values count are the same -> the last one in the list
        return {'output': res}

    features_list = columns_list[:len(columns_list) - 1]
    gini_dict = {}
    # TODO
    # think about add condition like =/!= for strings and >/< for numbers

    for feature in features_list:
        # create a dict for each feature with impurity and the value that bring the this impurity
        gini_dict.update({feature: {GINI_GAIN_KEY: 0, SPLIT_VALUE_KEY: ''}})
        # 1) order for the feature value -> ìt is only needed for the split
        # dataframe.sort_values(by=feature)
        unique_set_feature = set(dataframe[feature])
        # print("not sorted: {}".format(unique_set_feature))
        unique_set_feature = sorted(unique_set_feature)
        # print("sorted: {}".format(unique_set_feature))
        split_list = []
        old_value = None

        if type(unique_set_feature[0]) == str:
            is_var_numeric = False
        else:
            is_var_numeric = True

        # 2) create the needed split - size = ( n - 1 ) unique values of the feature
        # if we have 0 5 10, we have to check each possible split -> 2.5 7.5
        for unique_value in unique_set_feature:
            if is_var_numeric:
                # skip the first iteration
                if old_value:
                    current_split = (old_value + unique_value) / 2
                    split_list.append(current_split)
                    # 3) estimate the gini impurity for the current value
                    current_gain, sons_dict = gini_gain_calc(dataframe, feature, current_split, is_var_numeric)

                    # update the dict with the minimum value of impurity - max of the gain
                    if current_gain > gini_dict[feature][GINI_GAIN_KEY]:
                        gini_dict.update({feature: {GINI_GAIN_KEY: current_gain, SPLIT_VALUE_KEY: current_split,
                                                    SONS_DICT_KEY: sons_dict}})
                # update with the used value
                old_value = unique_value
            elif not is_var_numeric:
                current_split = unique_value
                split_list.append(current_split)
                # 3) estimate the gini impurity for the current value
                current_gain, sons_dict = gini_gain_calc(dataframe, feature, current_split, is_var_numeric)

                # update the dict with the minimum value of impurity - max of the gain
                if current_gain > gini_dict[feature][GINI_GAIN_KEY]:
                    gini_dict.update({feature: {GINI_GAIN_KEY: current_gain, SPLIT_VALUE_KEY: current_split,
                                                SONS_DICT_KEY: sons_dict}})
                # update with the used value
            old_value = unique_value

        # 4) estimate the best impurity per split
        # print("gini dict is {}".format(gini_dict))

        # TODO
        # take the maximum split and use it as the first split and iterate again for the next step
        # save the split as a condition
    max_gain = 0
    max_split = ''
    feature = ''
    max_sons_dict = {}
    max_is_numeric = True
    for key, value in gini_dict.items():
        loop_dict = value
        # print(value)
        if loop_dict[GINI_GAIN_KEY] > max_gain:
            max_gain = loop_dict[GINI_GAIN_KEY]
            max_split = loop_dict[SPLIT_VALUE_KEY]
            max_sons_dict = loop_dict[SONS_DICT_KEY]
            feature = key
            max_is_numeric = type(max_split) != str

    if max_is_numeric:
        # print(max_gain, max_split, feature)
        output_flow_dict = {SPLIT_VALUE_KEY: max_split, FEATURE_KEY: feature,
                            LESS: training(max_sons_dict[LESS], current_level - 1),
                            MORE_EQUAL: training(max_sons_dict[MORE_EQUAL], current_level - 1)}
        # dict = {'split_value': max_split, 'feature': feature}
    else:
        output_flow_dict = {SPLIT_VALUE_KEY: max_split, FEATURE_KEY: feature,
                            EQUAL: training(max_sons_dict[EQUAL], current_level - 1),
                            DIFFERENT: training(max_sons_dict[DIFFERENT], current_level - 1)}

    # print("output_dict = {}".format(output_flow_dict))
    print("Final dict is : ")
    pprint.pprint(output_flow_dict)

    if not os.path.exists(TRAINED_TREES):
        os.mkdir(TRAINED_TREES)
    # save the trained output
    json_trained_tree = json.dumps(output_flow_dict)
    with open(os.path.join(TRAINED_TREES, "trained_tree_dict_{}.json".format(current_tree_number)), "w") as the_file:
        the_file.write(json_trained_tree)

    return output_flow_dict


def classification(classification_dict, output_dict=None, tree_path="trained_tree_dict_1.json"):
    if not output_dict:
        # with open(os.path.join(TRAINED_TREES, "trained_tree_dict_{}.json".format(n_tree)), "r") as the_file:
        with open(os.path.join(TRAINED_TREES, tree_path), "r") as the_file:
            output_dict = ast.literal_eval(the_file.read())
            pprint.pprint(output_dict)
    # if it contains only the output
    if FEATURE_KEY not in output_dict:
        return output_dict
    # check for each feature the split and which part of the dict/tree explore
    feature_to_test = output_dict[FEATURE_KEY]
    if type (classification_dict[feature_to_test]) != str:
        if classification_dict[feature_to_test] >= output_dict[SPLIT_VALUE_KEY]:
            return classification(classification_dict, output_dict[MORE_EQUAL])
        else:
            return classification(classification_dict, output_dict[LESS])
    else:
        if classification_dict[feature_to_test] == output_dict[SPLIT_VALUE_KEY]:
            return classification(classification_dict, output_dict[EQUAL])
        else:
            return classification(classification_dict, output_dict[DIFFERENT])


def main():
    n_trees = 4
    # file_name = 'iris_data.csv'
    file_name = 'adult_data.csv'

    # training_main(file_name=file_name, n_trees=n_trees)
    classification_main()


def training_main(file_name, n_trees=3, training_set_percent=0.2):
    # clean old training
    for trained_tree in os.listdir(TRAINED_TREES):
        os.remove(os.path.join(TRAINED_TREES, trained_tree))

    for i in range(n_trees):
        dataframe = pd.read_csv(os.path.join(DATA_PATH, file_name))
        training_set = dataframe.sample(frac=training_set_percent, replace=True, random_state=1)
        current_level = 0
        trained_tree = training(training_set, current_level, i)  # (dataframe)


def classification_main():
    # test_classification_dict = {'petal length': 1.5, 'sepal length': 1.05, 'petal width': 1.63, 'sepal width': 1.0}
    test_classification_dict = {"age": 38, "workclass": 'Private', "fnlwgt": 215646, "education": 'HS-grad',
                                "education-num": 9, "marital-status": 'Divorced', "occupation": 'Handlers-cleaners',
                                "relationship": 'Not-in-family', "race": 'White', "sex": 'Male', "capital-gain": 0,
                                "capital-loss": 0, "hours-per-week": 40, "native-country": 'United-States'}

    output_labels = []
    for trained_tree in os.listdir(TRAINED_TREES):
        output_labels.append(classification(test_classification_dict, tree_path=trained_tree)['output'])

    print("output labels = {}".format(output_labels))
    output_label = res = max(set(output_labels), key=output_labels.count)
    print("output label is {}".format(output_label))


if __name__ == "__main__":
    main()
