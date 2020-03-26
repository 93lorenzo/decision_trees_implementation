from decisionTree import decisionTree


def main():
    n_trees = 4
    training_percentage = 0.2
    file_name = 'iris_data.csv'
    #file_name = 'adult_data.csv'
    myTree = decisionTree(data_path=file_name)
    myTree.training()
    #test_classification_dict = {'petal length': 1.5, 'sepal length': 1.05, 'petal width': 1.63, 'sepal width': 1.0}
    #myTree.classification(test_classification_dict)


if __name__ == "__main__":
    main()
