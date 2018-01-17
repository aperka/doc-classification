import gensim
import os
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim
from nltk.corpus import reuters
import nltk
from dataset import get_dataset
from project_utils import tokenize

model_path = 'fasttext-model-movie.bin'

def get_path(doc_id, corpus):
    corpus_path = nltk.data.find("corpora/%s" % corpus)  # unzip reuters.zip first
    filename = doc_id.replace('/', '\\')
    path = corpus_path + '\\' + filename
    return path


def fasttext_train(train_docs_ids, corpus):
    model = FT_gensim(size=100, min_count=1)
    path = get_path(train_docs_ids[0], corpus)
    sentence = LineSentence(path)
    # build the vocabulary
    model.build_vocab(sentence)
    # train the model
    model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)

    for doc_id in train_docs_ids[2:]:
        sentence = LineSentence(get_path(doc_id, corpus))

        # build the vocabulary
        model.build_vocab(sentence, update=True)
        # train the model
        model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)

    model.save(model_path)



def fasttext_get_vectors(trained_data, test_data):
    model = FT_gensim.load(model_path)
    trained_vector = get_vectors(model, trained_data)
    tested_vector = get_vectors(model, test_data)
    return trained_vector, tested_vector

def get_vectors(model, data):
    vector = {}
    for i, doc in enumerate(data):
        print i
        for j, word in enumerate(tokenize(doc)):
            try:
                vector[i, j] = model.word_vec(word)
            except KeyError:
                vector[i, j] = [0]
            print vector[i, j]
    return vector