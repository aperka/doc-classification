from random import randint
import json
from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer


corpuses = dict(reuters=reuters, other=None)

def save_splitted_dataset(file_path, corpus_name, k):
    """

    :param corpus: name of korpus
    :param k: number of fold groups
    :return:
    """
    corpus = corpuses[corpus_name]
    documents = corpus.fileids()
    train_docs_ids = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_ids = list(filter(lambda doc: doc.startswith("test"), documents))

    groups = list([] for x in range(k))

    while len(train_docs_ids):
        number_of_docs = len(train_docs_ids)
        group_id = len(train_docs_ids) % k

        if number_of_docs > 1:
            rand_id = randint(0, number_of_docs - 1)
        else:
            rand_id = 0

        groups[group_id].append(train_docs_ids[rand_id])
        del train_docs_ids[rand_id]

    with open(file_path, 'w') as f:
        json.dump(dict(corpus=corpus_name, groups=groups, validation=test_docs_ids), f, indent=4)


def get_dataset(file_path, cross_validation_case):
    """

    :param file_path:
    :param cross_validation_case:
    :return:
    """
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    corpus = corpuses[dataset['corpus']]

    train_docs_ids = []
    test_docs_ids = []

    # if cross_validation_case is 0 it returns default train / test docs
    if cross_validation_case == 0:
        test_docs_ids = dataset['validation']

    for count, group in enumerate(dataset['groups']):
        if count == cross_validation_case - 1:
            test_docs_ids += group
        else:
            train_docs_ids += group

    train_docs = [corpus.raw(doc_id) for doc_id in train_docs_ids]
    test_docs = [corpus.raw(doc_id) for doc_id in test_docs_ids]

    # Transform multilabel labels
    mlb = MultiLabelBinarizer()
    train_bin_labels = mlb.fit_transform([corpus.categories(doc_id) for doc_id in train_docs_ids])
    test_bin_labels = mlb.transform([corpus.categories(doc_id) for doc_id in train_docs_ids])
    labels = list(mlb.classes_)

    return train_docs, train_bin_labels, test_docs, test_bin_labels, labels

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        corpus_name = sys.argv[1]
        num_of_cross_valid_groups = sys.argv[2]
        file_path = sys.argv[3]
    else:
        print('Loading default configuration')
        corpus_name = 'reuters'
        num_of_cross_valid_groups = 4
        file_path = 'reuters_dataset.json'

    save_splitted_dataset(file_path, corpus_name, num_of_cross_valid_groups)