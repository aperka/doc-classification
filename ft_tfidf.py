from concatenate import concatenate
from tf_idf import tf_idf
from fasttext import *


def ft_tfidf(train_docs, test_docs, train_docs_ids, corpus):
    tfidf_vec_train_docs, tfidf_vec_test_docs = tf_idf(train_docs, test_docs)

    # fasttext_train(train_docs_ids, corpus)
    ft_vec_train_docs, ft_vec_test_docs = fasttext_get_vectors(train_docs, test_docs)

    t = concatenate(ft_vec_train_docs, ft_vec_test_docs, tfidf_vec_train_docs, tfidf_vec_test_docs)
    return t

if __name__ == '__main__':
    from dataset import get_dataset
    train_docs, train_bin_labels, test_docs, test_bin_labels, labels, train_docs_ids = get_dataset('reuters_dataset.json', 1)

    ft_tfidf(train_docs, test_docs, train_docs_ids, 'reuters')

