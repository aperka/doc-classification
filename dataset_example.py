from crossvalidation_dataset import corpuses, load_dataset_from_file, cross_validation_get_data
from collections import defaultdict
#from run import tf_idf, feature_values
import json


if __name__ == "__main__":
    import sys

    # Load dataset
    dataset = load_dataset_from_file('cross-validation_dataset.json')

    # Assign corpus nltk library
    corpus = corpuses[dataset['corpus']]

    # Cross validation over data groups:
    for validation_group_id in range(0, len(dataset['train_docs'])):
        train_docs_ids, test_docs_ids = cross_validation_get_data(dataset, validation_group_id)

        #tf_idf_all_docs = []
        train_docs = defaultdict(list)
        test_docs = defaultdict(list)
        #tf_idf_vectors = {}

        for category in train_docs_ids:
            for train_doc_id in train_docs_ids[category]:
                raw_doc = corpus.raw(train_doc_id)
                train_docs[category].append(raw_doc)
                #tf_idf_all_docs.append(raw_doc)

        for category in test_docs_ids.keys():
            for test_doc_id in test_docs_ids[category]:
                test_docs[category].append(corpus.raw(test_doc_id))

        # train_docs dict contains all raw training documents fot current cross-validation case:
        # access by train_docs['category_name']
        # list categories by train_docs.keys()
        # train_docs['category']["RAW DOC ONE IN CATEGORY blah blah blah",..,"RAW DOC TWO IN CATEGORY blah blah"]
        print(train_docs)

        # test_docs dict contains all raw test documents fot current cross-validation case:
        # access by test_docs['category_name']
        # list categories by test_docs.keys() equivalent to train_docs.key()
        # test_docs['category']["RAW DOC ONE IN CATEGORY blah blah blah",..,"RAW DOC TWO IN CATEGORY blah blah"]
        print(test_docs)


        ###
        '''
        representer = tf_idf(tf_idf_all_docs)
        for category in train_docs:
            tf_idf_vectors[category] = feature_values(train_docs[category], representer)

        with open('tf_idf.json', 'w') as f:
            json.dump(tf_idf_vectors, f, indent=4)

        raise
        '''
