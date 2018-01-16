from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(test_labels, predictions):
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')
    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


def tf_idf_svm(train_docs, test_docs, train_bin_labels, test_bin_labels):
    svm = SvmClassifier('linear')
    vectorised_train_documents, vectorised_test_documents = tf_idf(train_docs, test_docs)

    svm.fit(vectorised_train_documents, train_bin_labels)
    predictions = svm.predict(vectorised_test_documents)

    evaluate(test_bin_labels, predictions)

def doc2vec_svm(train_docs, test_docs, train_bin_labels, test_bin_labels):
    svm = SvmClassifier('linear')
    doc2vec_train(train_docs)
    train_data = doc2vec_train(train_docs)
    test_data = doc2vec_gen_test_data(test_docs)

    svm.fit(train_data, train_bin_labels)
    predictions = svm.predict(test_data)

    evaluate(test_bin_labels, predictions)

def doc2vec_nn(train_docs, test_docs, train_bin_labels, test_bin_labels):
    doc2vec_train(train_docs)
    train_data = doc2vec_train(train_docs)
    test_data = doc2vec_gen_test_data(test_docs)
    nn_run(train_data, test_data, train_bin_labels, test_bin_labels, True)

def tf_idf_nn(train_docs, test_docs, train_bin_labels, test_bin_labels):
    vectorised_train_documents, vectorised_test_documents = tf_idf(train_docs, test_docs)
    train_data = vectorised_train_documents.toarray().tolist()
    test_data = vectorised_test_documents.toarray().tolist()
    nn_run(train_data, test_data, train_bin_labels, test_bin_labels, True)

if __name__ == '__main__':
    import sys
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    from dataset import get_dataset
    from tf_idf import tf_idf
    from svm_classifier import SvmClassifier
    from neural_network import nn_run

    from doc2vec import doc2vec_train, doc2vec_gen_test_data


    if len(sys.argv) == 2:
        classif_type = sys.argv[1]
        dataset = sys.argv[2]
    else:
        classif_type = '3'
        dataset = 'reuters_dataset.json'


    if (classif_type == '0'):
        print("RUN tf_idf_svm")
        func_name = "tf_idf_svm"
    if (classif_type == '1'):
        print("RUN doc2vec_svm")
        func_name = "doc2vec_svm"
    if (classif_type == '2'):
        print("RUN doc2vec_nn")
        func_name = "doc2vec_nn"
    if (classif_type == '3'):
        print("RUN tf_idf_nn")
        func_name = "tf_idf_nn"


    for case in range(1, 5):
        train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset(dataset, case)
        locals()[func_name](train_docs, test_docs, train_bin_labels, test_bin_labels)
