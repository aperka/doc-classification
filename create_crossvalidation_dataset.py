from __future__ import print_function
from random import randint
import json
from nltk.corpus import reuters
from collections import defaultdict

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

            print(rand_id, end=' ')
            dataset[group_iter%num_of_groups][category].append(train_docs[category][rand_id])
            del train_docs[category][rand_id]
            group_iter += 1

        print('')

    return dataset

if __name__ == "__main__":
    #with open('pre.json', 'w') as f:
    #    json.dump(prepare_data(reuters, 30), f, indent=4)

    #with open('pre.json', 'r') as f:
    #    dataset = json.load(f)

    train_docs, test_docs = prepare_data(reuters, 30)
    train_docs = split_dataset_into_groups(train_docs, 4)

    with open('cross-validation_dataset.json', 'w') as f:
        json.dump(dict(train_docs=train_docs,test_docs=test_docs), f, indent=4)


