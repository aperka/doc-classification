import sys
import nltk

nltk.download('stopwords')
nltk.download('punkt')

from dataset import get_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from project_utils import evaluate
from tf_idf import tf_idf
from svm_classifier import SvmClassifier
from neural_network import nn_run
from doc2vec import doc2vec_train, doc2vec_gen_test_data
from concatenate import concatenate
from scipy.sparse import csr_matrix


def classify(classifier, train_data, test_data, train_bin_labels, test_bin_labels):
    if classifier == 'svm':
        #converting data
        if isinstance(train_data, list):
            train_data = csr_matrix(train_data)
        if isinstance(test_data, list):
            test_data = csr_matrix(test_data)

        print('Linear SVM (SVC) Training')
        svm = SvmClassifier('linear')
        svm.fit(train_data, train_bin_labels)
        print('Linear SVM (SVC) Prediction')
        predictions = svm.predict(test_data)

        evaluate(predictions, test_bin_labels)

    elif classifier == 'nn':
        #converting data
        if not isinstance(train_data, list):
            train_data = train_data.toarray().tolist()
        if not isinstance(test_data, list):
            test_data = test_data.toarray().tolist()
        print('NN Training and prediction')
        predictions = nn_run(train_data, test_data, train_bin_labels, test_bin_labels, True)

        evaluate(predictions, test_bin_labels)



def extract_features(method, train_data, test_data):
    if method == 'tf_idf':
        print("Feature extraction: tf-idf")
        train_data_features, test_data_features = tf_idf(train_data, test_data)

    elif method == 'doc2vec':
        print("Feature extraction: doc2vec")
        train_data_features = doc2vec_train(train_data)
        test_data_features = doc2vec_gen_test_data(test_data)

    elif method == "tf_idf_doc2vec":
        print("Feature extraction: tf-idf+doc2vec")
        print("Feature extraction: 1 - tf-idf")
        tf_idf_train_data, tf_idf_test_data = tf_idf(train_data, test_data)
        print("Feature extraction: 2 - doc2vec")
        embeded_train_data = doc2vec_train(train_data)
        embeded_test_data = doc2vec_gen_test_data(test_data)

        train_data_features, test_data_features = concatenate(embeded_test_data=embeded_test_data,
                                                            embeded_train_data=embeded_train_data,
                                                            tf_idf_test_data=tf_idf_test_data,
                                                            tf_idf_train_data=tf_idf_train_data)
    else:
        print("Feature extraction: "+method+" not implemented")
        raise NotImplementedError

    return train_data_features, test_data_features


def test(classifier_type, feature_extraction, dataset_json):
    for case in range(0, 5):
        train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset(dataset_json, case)
        train_data_features, test_data_features = extract_features(feature_extraction, train_docs, test_docs)
        classify(classifier_type, train_data_features, test_data_features, train_bin_labels, test_bin_labels)

if __name__ == '__main__':

    if len(sys.argv) == 4:
        classifier_type = sys.argv[1] # 'svm' or 'nn'
        feature_extraction = sys.argv[2] # 'tf_idf' or'doc2vec' or 'tf_idf_doc2vec'
        dataset_json = sys.argv[3]
    else:
        classifier_type = 'nn'  # 'svm' or 'nn'
        feature_extraction = 'tf_idf_doc2vec' # 'tf_idf' or'doc2vec' or 'tf_idf_doc2vec'
        dataset_json = 'reuters_dataset.json'

    test(classifier_type, feature_extraction, dataset_json)

