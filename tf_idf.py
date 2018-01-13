
from dataset import get_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from project_utils import tokenize


def tf_idf(train_docs, test_docs):

    # Tokenisation
    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    # Learn and transform train documents
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

    return vectorised_train_documents, vectorised_test_documents

train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset('reuters_dataset.json', 1)
#
# train_docs:       train_bin_labels   labels
# [['doc o kocie'],   [[1 , 0],            ['kot' , 'pies']
#  ['doc o psie'],     [0 , 1],
#  ['o kocie  psie']]  [1 , 1]]
print(tf_idf(train_docs, test_docs))

for doc_count, doc in enumerate(train_docs):
    print(doc_count)
    for label_count, label in enumerate(labels):
        print(label + ': ' + str(train_bin_labels[doc_count][label_count]))