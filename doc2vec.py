from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from dataset import get_dataset

doc2vec_model_location = 'doc2vec-model.bin'
doc2vec_dimensions = 300


def doc2vec_train(train_docs):
    taggedDocuments = [TaggedDocument(words=word_tokenize(train_docs[i]), tags=[i]) for i in range(len(train_docs))]

    # Create and train the doc2vec model
    model = Doc2Vec(size=doc2vec_dimensions, min_count=2, iter=10, workers=12)

    # Build the word2vec model from the corpus
    model.build_vocab(taggedDocuments)

    model.train(taggedDocuments, total_examples=model.corpus_count, epochs=model.iter)
    model.save(doc2vec_model_location)


def doc2vec_predict(test_docs):
    doc2vec = Doc2Vec.load(doc2vec_model_location)

    # Convert test_doc to doc vectors
    test_data = [doc2vec.infer_vector(word_tokenize(doc)) for doc in test_docs]

    '''
    # Initialize the neural network
    #model = load_model(classifier_model_location)

    # Make predictions
    #predictions = model.predict(numpy.asarray(test_data))

    # Convert the prediction with gives a value between 0 and 1 to exactly 0 or 1 with a threshold
    #predictions[predictions < 0.5] = 0
    #predictions[predictions >= 0.5] = 1


    # Convert predicted classes back to category names
    labelBinarizer = MultiLabelBinarizer()
    labelBinarizer.fit([reuters.categories(fileId) for fileId in reuters.fileids()])
    predicted_labels = labelBinarizer.inverse_transform(predictions)

    for predicted_label, test_article in zip(predicted_labels, test_articles):
        print('title: {}'.format(test_article['raw'].splitlines()[0]))
        print('predicted: {} - actual: {}'.format(list(predicted_label), test_article['categories']))
    print('')
    '''


if __name__ == "__main__":
    train_docs, train_bin_labels, test_docs, test_bin_labels, labels = get_dataset('reuters_dataset.json', 1)
    doc2vec_train(train_docs)
    doc2vec_predict(test_docs)


