import graphviz
from sklearn.datasets import load_iris
from sklearn import tree

def main():
    with open('') as f:


    iris = load_iris()
    examples = iris.data
    labels = iris.target

    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=None)
    clf = clf.fit(examples, labels)

    print(iris.data)
    print(type(iris.data))
    print(iris.data.shape)
    print(iris.target)
    print(type(iris.target))
    print(iris.target.shape)

    print(clf)

    print(iris.feature_names)
    print(iris.target_names)
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")

if __name__ == '__main__':
    main()