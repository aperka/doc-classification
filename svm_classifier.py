from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


class SvmClassifier(object):
    def __init__(self, svm_type, *args, **kwargs):
        """


        :param type: options: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        """

        self.classifier = OneVsRestClassifier(SVC(kernel=svm_type, *args, **kwargs))

    def fit(self, docs, labels):
        self.classifier.fit(docs, labels)

    def predict(self, docs):
        return self.classifier.predict(docs)


if __name__ == '__main__':
    from dataset import get_dataset
    from tf_idf import tf_idf


    train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset('reuters_dataset.json', 1)
    vectorised_train_documents, vectorised_test_documents = tf_idf(train_docs, test_docs)

    svm = SvmClassifier('linear')
    svm.fit(vectorised_train_documents, train_bin_labels)
    predictions = svm.predict(vectorised_test_documents)
    print(predictions)

