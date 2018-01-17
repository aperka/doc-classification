from scipy.sparse import csr_matrix, hstack


def concatenate(embeded_train_data, embeded_test_data, tf_idf_train_data, tf_idf_test_data):
    embeded_train_data = csr_matrix(embeded_train_data)
    embeded_test_data = csr_matrix(embeded_test_data)

    concatenated_train = csr_matrix(hstack([embeded_train_data, tf_idf_train_data]))
    concatenated_test = csr_matrix(hstack([embeded_test_data, tf_idf_test_data]))

    return concatenated_train, concatenated_test
