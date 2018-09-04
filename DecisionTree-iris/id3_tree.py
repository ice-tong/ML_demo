# -*- coding: utf-8 -*-

import numpy as np
import random

def load_data(iris_path='./Iris.csv', rate=0.8):
    label2vec = {'Iris-setosa': 0., 'Iris-versicolor': 1., 'Iris-virginica': 2.}
    with open(iris_path) as f:
        data = f.readlines()
    dataset = list()
    for sample in data:
        sample = sample.replace('\n', '')
        row = sample.split(',')
        label = label2vec[row[-1]]
        row = row[:-1]
        row.append(label)
        dataset.append(row)
    random.shuffle(dataset)
    train_data = np.array(dataset[:int(len(dataset)*rate)], dtype=float)
    test_data = np.array(dataset[int(len(dataset)*rate):], dtype=float)
    return np.rint(train_data), np.rint(test_data)

def get_rela_entropy(dataset, feature:int):
    def get_entropy(dataset):
        label_tags = list(set(dataset[:, -1]))
        label_length = len(dataset[:, -1])
        tmp_entropy = 0
        for label_tag in label_tags:
            tmp = sum([1 for d in dataset if d[-1]==label_tag])
            tmp_entropy += (tmp/label_length)*np.math.log(tmp/label_length, 2)
        entropy = -tmp_entropy
        return entropy
    feature_tags = list(set(dataset[:, feature]))
    sub_entropy = 0
    for feature_tag in feature_tags:
        sub_dataset = [d for d in dataset if d[feature]==feature_tag]
        sub_dataset = np.array(sub_dataset)
        tmp_entropy = get_entropy(sub_dataset)
        sub_entropy += (len(sub_dataset)/len(dataset)) * tmp_entropy
    rela_entropy = get_entropy(dataset) - sub_entropy
    return rela_entropy

def select_feature(dataset, features):
    rela_entropys = list()
    for feature in features:
        feature:int
        rela_entropy = get_rela_entropy(dataset, feature)
        rela_entropys.append(rela_entropy)
    return features[rela_entropys.index(max(rela_entropys))]

def major_label(labels):
    tags = list(set(labels))
    tag_num = [sum([1 for i in labels if i==label]) for label in tags]
    k = tag_num.index(max(tag_num))
    return tags[k]

def build_tree(dataset, features) -> dict:
    labels = dataset[:, -1]
    if len(set(labels)) == 1:
        return {'label': labels[0]}
    if not len(features):
        return {'label': major_label(labels)}
    best_feature = select_feature(dataset, features)
    tree = {'feature': best_feature, 'children': {}}
    feature_tags = list(set(dataset[:, best_feature]))
    for feature_tag in feature_tags:
        sub_dataset = [d for d in dataset if d[best_feature]==feature_tag]
        sub_dataset = np.array(sub_dataset)
        if len(sub_dataset) == 0:
            tree['children'][feature_tag] = {'label': major_label(labels)}
        else:
            sub_features = [i for i in features if i != best_feature]
            tree['children'][feature_tag] = build_tree(sub_dataset, sub_features)
    return tree

def classifier(tree:dict, features_data, default):
    def classify(tree:dict, sample):
        for k, v in tree.items():
            if k != 'feature':
                return tree['label']
            else:
                return classify(tree['children'][sample[tree['feature']]], 
                                sample)
    predict_vec = list()
    for features_sample in features_data:
        try:
            predict = classify(tree, features_sample)
        except KeyError:
            predict = default
        predict_vec.append(predict)
    return predict_vec

if __name__=="__main__":
    train_data, test_data = load_data()
    tree = build_tree(train_data, list(range(train_data.shape[1]-1)))
#    print(tree)
    test_data_labels = test_data[:, -1]
    test_data_features = test_data[:, :-1]
    default = major_label(test_data_labels)
    predict_vec = classifier(tree, test_data_features, default)
#    print(predict_vec)
    accuracy = np.mean(np.array(predict_vec==test_data_labels))
    print(accuracy)