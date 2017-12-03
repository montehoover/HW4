import graphviz
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from scipy import stats


def main():
    examples, labels = parse('data_hw4_diabetes/diabetes_train.txt')
    tst_examples, tst_labels = parse('data_hw4_diabetes/diabetes_test.txt')

    # 1.0 Get accuracies with various numbers of samplings:
    accuracies = []
    for i in range(20):
        ensemble = BaggingEnsemble(examples, labels, i+1)
        accuracies.append(1 - get_loss(ensemble, tst_examples, tst_labels))
    print(accuracies)

    # 1.1 Get bias and variance with various depths of trees:
    accuracies = []
    for i in range(15):
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=i+1)
        clf =  clf.fit(examples, labels)
        accuracies.append(1 - get_loss(clf, tst_examples, tst_labels))
    print(accuracies)




def get_loss(clf, examples, labels):
    predictions = clf.predict(examples)
    errors = 0
    for p, l in zip(predictions, labels):
        if p != l:
            errors += 1
    loss = errors/len(labels)
    return loss


def parse(file_name):
    examples = []
    labels = []
    with open(file_name) as f:
        for line in f:
            line = line.split(',')
            labels.append(int(line.pop()))
            examples.append([float(x) for x in line])
    return examples, labels


class BaggingEnsemble():
    def __init__(self, sample, labels, num_bags):
        self.orig_sample = sample
        self.orig_labels = labels
        self.classifiers = []
        self.gen = np.random.RandomState(123)

        for i in range(num_bags):
            self.classifiers.append(self.get_classifier())

    def predict(self, examples):
        votes = []
        for clf in self.classifiers:
            votes.append(clf.predict(examples))
        votes = np.array(votes)
        # Get the first item in the mode-tuple which is a 2d arr with one row, and get that row. Thus [0][0].
        predictions =  stats.mode(votes)[0][0]
        return predictions

    def get_classifier(self):
        sample, labels = self.generate_sample()
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=None)
        return clf.fit(sample, labels)

    def generate_sample(self):
        size = len(self.orig_sample)
        sample = []
        labels = []
        for i in range(size):
            x = self.gen.randint(0, size)
            sample.append(self.orig_sample[x])
            labels.append(self.orig_labels[x])
        return sample, labels

if __name__ == '__main__':
    main()