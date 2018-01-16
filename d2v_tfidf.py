from tf_idf import tf_idf
from doc2vec import doc2vec_train
from doc2vec import doc2vec_predict
from scipy.sparse import csr_matrix

doc2vec_model_location = 'doc2vec-model.bin'
doc2vec_dimensions = 300


def d2v_tf_idf(train_docs, test_docs):
    tf_idf_vectorised_docs = tf_idf(train_docs, test_docs)

    vectorised_test_documents = tf_idf_vectorised_docs[1]

    doc2vec_train(train_docs)
    d2v = doc2vec_predict(test_docs)
    combined_matrix = [0] * len(test_docs)
    d = d2v[0].tolist()
    t = vectorised_test_documents.getrow(0).toarray()[0].tolist()
    for i in range(len(test_docs)):
        combined_matrix[i] = d2v[i].tolist() + vectorised_test_documents.getrow(i).toarray()[0].tolist()
    return csr_matrix(combined_matrix)


if __name__ == '__main__':
    from dataset import get_dataset
    train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset('reuters_dataset.json', 1)

    d2v_tf_idf(train_docs, test_docs)

