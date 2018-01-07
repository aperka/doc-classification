from __future__ import print_function
from random import randint
import json
from nltk.corpus import reuters
from collections import defaultdict

corpuses = dict(reuters=reuters, other=None)

def prepare_data(corpus, min_category_docs):

    # Docs ID containers
    train_docs = defaultdict(list)
    test_docs = defaultdict(list)

    # List of categories
    categories = corpus.categories();
    print(str(len(categories)) + " categories");

    # Split docs into train and test groups
    for category in categories:
        for doc_id in corpus.fileids(category):
            if doc_id.startswith("train"):
                train_docs[category].append(doc_id)
            else:
                test_docs[category].append(doc_id)

        # Print amount of train / test docs in category
        print('{} contains: train: {:04d}, test: {:04d} '.format(str(category).ljust(15), len(train_docs[category]),
                                                                 len(test_docs[category])) + 'docs.', end="")
        # Remove categories with less than min_category_docs train docs
        if len(train_docs[category]) < min_category_docs:
            print(' - Category Removed')
            del train_docs[category]
            del test_docs[category]
        else:
            print('')

    return train_docs, test_docs


def split_dataset_into_groups(train_docs, num_of_groups):
    dataset = defaultdict(lambda: defaultdict(list))

    # Randomly pick up docs to groups
    for category in train_docs.keys():
        group_iter = 0
        while len(train_docs[category]):
            docs_in_category = len(train_docs[category])
            if docs_in_category > 1:
                rand_id = randint(0, docs_in_category-1)
            else:
                rand_id = 0

            dataset[group_iter%num_of_groups][category].append(train_docs[category][rand_id])
            del train_docs[category][rand_id]
            group_iter += 1

    return dataset


def load_dataset_from_file(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset


def cross_validation_get_data(dataset, case_number):

    xvalid_group_ids = range(0, len(dataset['train_docs']))

    if case_number not in xvalid_group_ids:
        raise ValueError("Invalid cross-validation case number")

    validation_docs = defaultdict(list)
    training_docs = defaultdict(list)
    categories = dataset['train_docs']['0'].keys()

    for group_id in xvalid_group_ids:
        for category in categories:
            data = dataset['train_docs'][str(group_id)][category]
            if case_number == group_id:
                validation_docs[category] += data
            else:
                training_docs[category] += data

    return training_docs, validation_docs


if __name__ == "__main__":
    '''To create cross validation dataset run script with
    args:
        corpus_name
        min_documents_in_category - minimum documents in category, categories with less docs will not exist in datast
        num_of_cross_valid_groups - number of cross validation groups in dataset
        file_path - it save dataset at this path
        '''
    import sys

    if len(sys.argv) == 5:
        corpus_name = sys.argv[1]
        min_documents_in_category = sys.argv[2]
        num_of_cross_valid_groups = sys.argv[3]
        filename = sys.argv[4]
    else:
        corpus_name = 'reuters'
        min_documents_in_category = 30
        num_of_cross_valid_groups = 3
        filename = 'cross-validation_dataset.json'

    corpus = corpuses[corpus_name]
    train_docs, test_docs = prepare_data(reuters, 30)
    train_docs = split_dataset_into_groups(train_docs, 4)

    with open(filename, 'w') as f:
        json.dump(dict(corpus=corpus_name ,train_docs=train_docs,test_docs=test_docs), f, indent=4)
