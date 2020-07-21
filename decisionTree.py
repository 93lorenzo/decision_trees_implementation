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


class decisionTree:
    """
        A class used to represent a Decision Tree

        ...

        Attributes
        ----------
        data_path : str
            path to the csv file that contains the data
        pruning_level : str
            the number of maximum depth for the training of the tree (default 4)

        Methods
        -------
        gini_gain_calc(self, dataframe, feature, current_split, is_numeric=True)
            xxx
        def __training_second_step(self, dataframe, current_level, current_tree_number=1):
            xxx
        def training(self, n_trees=3, training_set_percent=0.2):
            xxx
        def __classification_second_step(self, classification_dict, output_dict=None, tree_path="trained_tree_dict_1.json"):
            xxx
        def classification(self, test_classification_dict):
            xxx
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

    def __init__(self, data_path, pruning_level=10):
        """
        Parameters
        ----------
        :param data_path:
            path to the csv file that contains the data
        :param pruning_level:
            the number of maximum depth for the training of the tree (default 4)
        """
        self.data_path = data_path
        self.pruning_level = pruning_level

    def gini_gain_calc(self, dataframe, feature, current_split, is_numeric=True):
        """
        :param dataframe:
            current data taken into account for the split
        :param feature:
            the feature for which the split will be calculated
        :param current_split:
            the split value for the feature to test the data
        :param is_numeric:
            boolean that describes if the feature is numeric or categorical
        :return:
            gini gain that express the quality of the split and the dict that describes the split
        """
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

        # left and right part with respect to the condition
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
                count_label = subset[subset[label_column] == label][feature].count() # how many rows that label in the subset
                label_probability = count_label / subset_count_row # how many rows that label in the subset / all the rows in the subset
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
            sons_dict = {self.LESS: full_data_left, self.MORE_EQUAL: full_data_right}
        elif not is_numeric:
            full_data_left = full_data_frame_deep_copy[full_data_frame_deep_copy[feature] == current_split]
            full_data_right = full_data_frame_deep_copy[full_data_frame_deep_copy[feature] != current_split]
            sons_dict = {self.EQUAL: full_data_left, self.DIFFERENT: full_data_right}

        return gini_gain, sons_dict

    def __training_second_step(self, dataframe, current_level, current_tree_number=1):
        """
        :param dataframe:
            current data taken into account for the split
        :param current_level:
            current level of depth in the recursive training process
        :param current_tree_number:
            the number of the current tree for the random forest creation
        :return:
            the dict that contains the classification flow
        """
        columns_list = list(dataframe.columns)
        label_column = columns_list[len(columns_list) - 1]
        # obtain all the possible outcomes (labels)
        labels_list = list(dataframe[label_column])
        ##################
        # STOP CONDITION #
        # print("### len set label list = {} and label list = {}".format(len(set(labels_list)), labels_list))
        if current_level > self.PRUNE_LEVEL or len(set(labels_list)) == 1:
            res = max(set(labels_list),key=labels_list.count)# two values count are the same->the last one in the list
            return {'output': res}

        features_list = columns_list[:len(columns_list) - 1]
        gini_dict = {}

        for feature in features_list:
            # create a dict for each feature with impurity and the value that bring the this impurity
            gini_dict.update({feature: {self.GINI_GAIN_KEY: 0, self.SPLIT_VALUE_KEY: ''}})
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
                        current_gain, sons_dict = self.gini_gain_calc(dataframe, feature, current_split, is_var_numeric)

                        # update the dict with the minimum value of impurity - max of the gain
                        if current_gain > gini_dict[feature][self.GINI_GAIN_KEY]:
                            gini_dict.update(
                                {feature: {self.GINI_GAIN_KEY: current_gain, self.SPLIT_VALUE_KEY: current_split,
                                           self.SONS_DICT_KEY: sons_dict}})
                    # update with the used value
                    old_value = unique_value
                elif not is_var_numeric:
                    current_split = unique_value
                    split_list.append(current_split)
                    # 3) estimate the gini impurity for the current value
                    current_gain, sons_dict = self.gini_gain_calc(dataframe, feature, current_split, is_var_numeric)

                    # update the dict with the minimum value of impurity - max of the gain
                    if current_gain > gini_dict[feature][self.GINI_GAIN_KEY]:
                        gini_dict.update(
                            {feature: {self.GINI_GAIN_KEY: current_gain, self.SPLIT_VALUE_KEY: current_split,
                                       self.SONS_DICT_KEY: sons_dict}})
                    # update with the used value
                old_value = unique_value

            # 4) estimate the best impurity per split
            # print("gini dict is {}".format(gini_dict))
            # save the split as a condition
        max_gain = 0
        max_split = ''
        feature = ''
        max_sons_dict = {}
        max_is_numeric = True
        for key, value in gini_dict.items():
            loop_dict = value
            # print(value)
            if loop_dict[self.GINI_GAIN_KEY] > max_gain:
                max_gain = loop_dict[self.GINI_GAIN_KEY]
                max_split = loop_dict[self.SPLIT_VALUE_KEY]
                max_sons_dict = loop_dict[self.SONS_DICT_KEY]
                feature = key
                max_is_numeric = type(max_split) != str

        if max_is_numeric:
            # print(max_gain, max_split, feature)
            output_flow_dict = {self.SPLIT_VALUE_KEY: max_split, self.FEATURE_KEY: feature,
                                self.LESS: self.__training_second_step(max_sons_dict[self.LESS], current_level - 1),
                                self.MORE_EQUAL: self.__training_second_step(max_sons_dict[self.MORE_EQUAL],
                                                                           current_level - 1)}
            # dict = {'split_value': max_split, 'feature': feature}
        else:
            output_flow_dict = {self.SPLIT_VALUE_KEY: max_split, self.FEATURE_KEY: feature,
                                self.EQUAL: self.__training_second_step(max_sons_dict[self.EQUAL], current_level - 1),
                                self.DIFFERENT: self.__training_second_step(max_sons_dict[self.DIFFERENT],
                                                                          current_level - 1)}

        # print("output_dict = {}".format(output_flow_dict))
        print("Final dict is : ")
        pprint.pprint(output_flow_dict)

        if not os.path.exists(self.TRAINED_TREES):
            os.mkdir(self.TRAINED_TREES)
        # save the trained output
        json_trained_tree = json.dumps(output_flow_dict)
        with open(os.path.join(self.TRAINED_TREES, "trained_tree_dict_{}.json".format(current_tree_number)),
                  "w") as the_file:
            the_file.write(json_trained_tree)

        return output_flow_dict

    def __classification_second_step(self, classification_dict, output_dict=None, tree_path="trained_tree_dict_1.json"):
        """

        :param classification_dict:
            dict that contains the label and their values
        :param output_dict:
            the result of the classification
        :param tree_path:
            the path of the classification dict that will be saved after the training
        :return:
            the output label
        """
        if not output_dict:
            # with open(os.path.join(TRAINED_TREES, "trained_tree_dict_{}.json".format(n_tree)), "r") as the_file:
            with open(os.path.join(self.TRAINED_TREES, tree_path), "r") as the_file:
                output_dict = ast.literal_eval(the_file.read())
                pprint.pprint(output_dict)
        # if it contains only the output
        if self.FEATURE_KEY not in output_dict:
            return output_dict
        # check for each feature the split and which part of the dict/tree explore
        feature_to_test = output_dict[self.FEATURE_KEY]
        if type(classification_dict[feature_to_test]) != str:
            if classification_dict[feature_to_test] >= output_dict[self.SPLIT_VALUE_KEY]:
                return self.__classification_second_step(classification_dict, output_dict[self.MORE_EQUAL])
            else:
                return self.__classification_second_step(classification_dict, output_dict[self.LESS])
        else:
            if classification_dict[feature_to_test] == output_dict[self.SPLIT_VALUE_KEY]:
                return self.__classification_second_step(classification_dict, output_dict[self.EQUAL])
            else:
                return self.__classification_second_step(classification_dict, output_dict[self.DIFFERENT])

    def training(self, n_trees=3, training_set_percent=0.2):
        """
        :param n_trees:
            how many trees for the random forest creation
        :param training_set_percent:
            how much percentage of the dataset use for the training process
        :return:
            it call the second training method, that after the trainig will save the classification flow on a file
        """
        file_name = self.data_path
        # clean old training
        for trained_tree in os.listdir(self.TRAINED_TREES):
            os.remove(os.path.join(self.TRAINED_TREES, trained_tree))

        for i in range(n_trees):
            dataframe = pd.read_csv(os.path.join(self.DATA_PATH, file_name))
            training_set = dataframe.sample(frac=training_set_percent, replace=True, random_state=1)
            current_level = 0
            trained_tree = self.__training_second_step(training_set, current_level, i)  # (dataframe)

    def classification(self, test_classification_dict):
        """

        :param test_classification_dict:
            dict that contains the labels and their values ( but not the label)
        :return:
            returns the classification label
        """
        # test_classification_dict = {'petal length': 1.5, 'sepal length': 1.05, 'petal width': 1.63, 'sepal width': 1.0}

        # test_classification_dict = {"age": 38, "workclass": 'Private', "fnlwgt": 215646, "education": 'HS-grad',
        #                            "education-num": 9, "marital-status": 'Divorced', "occupation": 'Handlers-cleaners',
        #                            "relationship": 'Not-in-family', "race": 'White', "sex": 'Male', "capital-gain": 0,
        #                            "capital-loss": 0, "hours-per-week": 40, "native-country": 'United-States'}

        output_labels = []
        for trained_tree in os.listdir(self.TRAINED_TREES):
            output_labels.append(self.__classification_second_step(test_classification_dict, tree_path=trained_tree)['output'])

        print("output labels = {}".format(output_labels))
        output_label = res = max(set(output_labels), key=output_labels.count)
        print("output label is {}".format(output_label))
