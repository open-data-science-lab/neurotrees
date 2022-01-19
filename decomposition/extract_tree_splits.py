import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import _tree


class BinTree:
    """
    Binary tree data structure
    """
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def print_values(self):
        def recurse_print(tree):
            print(tree.value)
            if tree.left is not None:
                recurse_print(tree.left)
            if tree.right is not None:
                recurse_print(tree.right)
        recurse_print(self)

    def values_to_list(self):
        """
        :return: tuple ('node' or 'leaf', value of the tree)
        """
        def recurse_tree(tree, results):
            results.append(
                ('node' if tree.left is not None and tree.right is not None else 'leaf',
                 tree.value))
            if tree.left is not None:
                recurse_tree(tree.left, results)
            if tree.right is not None:
                recurse_tree(tree.right, results)
        results = []
        recurse_tree(self, results)
        return results

    def print_nodes(self):
        def recurse_tree(tree):
            if tree.left is not None and tree.right is not None:
                print(tree.value)
            if tree.left is not None:
                recurse_tree(tree.left)
            if tree.right is not None:
                recurse_tree(tree.right)
        recurse_tree(self)

    def nodes_to_list(self):
        def recurse_tree(tree, results):
            if tree.left is not None and tree.right is not None:
                results.append(tree.value)
            if tree.left is not None:
                recurse_tree(tree.left, results)
            if tree.right is not None:
                recurse_tree(tree.right, results)
        results = []
        recurse_tree(self, results)
        return results

    def print_leaves(self):
        def recurse_tree(tree):
            if tree.left is None and tree.right is None:
                print(tree.value)
            if tree.left is not None:
                recurse_tree(tree.left)
            if tree.right is not None:
                recurse_tree(tree.right)
        recurse_tree(self)

    def leaves_to_list(self):
        def recurse_tree(tree, results):
            if tree.left is None and tree.right is None:
                results.append(tree.value)
            if tree.left is not None:
                recurse_tree(tree.left, results)
            if tree.right is not None:
                recurse_tree(tree.right, results)
        results = []
        recurse_tree(self, results)
        return results


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_") for f in feature_names]
    result_code = "def predict({}):".format(", ".join(feature_names)) + "\n"

    def recurse(node, depth):
        recurse_code = ''
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse_code += ("{}if {} <= {}:".format(indent, name, np.round(threshold, 2))) + "\n"
            recurse_code += recurse(tree_.children_left[node], depth + 1) + "\n"
            recurse_code += ("{}else:  # if {} > {}".format(indent, name, np.round(threshold, 2))) + "\n"
            recurse_code += recurse(tree_.children_right[node], depth + 1)
            return recurse_code
        else:
            recurse_code += ("{}return {}".format(indent, str(tree_.value[node].tolist())))
            return recurse_code

    recurse_code = recurse(0, 1)
    return result_code + recurse_code


def tree_to_binary_tree(tree, feature_names):
    tree_ = tree.tree_
    feature_names = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node):
        recurse_tree = BinTree()
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[node]
            threshold = tree_.threshold[node]
            recurse_tree.value = (name, np.round(threshold, 2))
            recurse_tree.left = recurse(tree_.children_left[node])
            recurse_tree.right = recurse(tree_.children_right[node])
            return recurse_tree
        else:
            recurse_tree.value = tree_.value[node].tolist()
            return recurse_tree

    result_tree = recurse(0)
    return result_tree


# Example of use
dataset_name = 'blood_transfusion'
data = pd.read_csv(f"datasets/numerical/{dataset_name}.csv")
label_col = "Class"
data_X = data.drop([label_col], axis=1)
data_y = data[label_col]
dtree = DecisionTreeClassifier()
dtree.fit(data_X, data_y)
print(export_text(dtree, feature_names=data_X.columns.to_list()))  # as text
fout = open(f'results/tree_code_{dataset_name}.py', 'w')
fout.write(tree_to_code(dtree, data_X.columns.to_list()))  # as python code
fout.close()
print(tree_to_binary_tree(dtree, data_X.columns.to_list()).nodes_to_list())
