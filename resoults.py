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




if __name__ == '__main__':
    import sys
    from dataset import get_dataset
    from tf_idf import tf_idf
    from svm_classifier import SvmClassifier

    from doc2vec import doc2vec_train, doc2vec_gen_test_data

    if len(sys.argv) == 2:
        classif_type = sys.argv[1]
    else:
        print('Run default classification')
        classif_type = '0'

    for case in range(0, 6):
        train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset('reuters_dataset.json', case)

        svm = SvmClassifier('linear')
    '''
    #----------------------------------------TF_IDF----------------------------------------------
        vectorised_train_documents, vectorised_test_documents = tf_idf(train_docs, test_docs)


        svm.fit(vectorised_train_documents, train_bin_labels)
        predictions = svm.predict(vectorised_test_documents)

        evaluate(test_bin_labels, predictions)
    '''
#-------------------------------------DOC2VEC--------------------------------------------------
    doc2vec_train(train_docs)
    train_data = doc2vec_train(train_docs)
    test_data = doc2vec_gen_test_data(test_docs)

    svm.fit(train_data, train_bin_labels)
    predictions = svm.predict(test_data)

    evaluate(test_bin_labels, predictions)