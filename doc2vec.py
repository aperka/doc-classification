from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from random import shuffle

doc2vec_model_location = 'doc2vec-model.bin'
doc2vec_dimensions = 150


def doc2vec_train(train_docs):
    taggedDocuments = [TaggedDocument(words=word_tokenize(train_docs[i]), tags=[i]) for i in range(len(train_docs))]
    shuffle(taggedDocuments)
    # Create and train the doc2vec model
    doc2vec = Doc2Vec(size=doc2vec_dimensions, window=8, min_count=2, iter=20, workers=4, sample=1e-3)

    # Build the word2vec model from the corpus
    doc2vec.build_vocab(taggedDocuments)

    doc2vec.train(taggedDocuments, total_examples=doc2vec.corpus_count, epochs=doc2vec.iter)
    doc2vec.save(doc2vec_model_location)

    train_data = [doc2vec.infer_vector(word_tokenize(doc)) for doc in train_docs]

    return train_data


def doc2vec_gen_test_data(test_docs):
    doc2vec = Doc2Vec.load(doc2vec_model_location)

    # Convert test_doc to doc vectors

    test_data = [doc2vec.infer_vector(word_tokenize(doc)) for doc in test_docs]
    #test_data = [doc2vec.infer_vector(tokenize(doc)) for doc in test_docs]

    return test_data






