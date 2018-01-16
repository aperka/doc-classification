from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from resoults import evaluate

def nn_scale_data(train_data, test_data):
    scaler = StandardScaler()

    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled

def nn_run(train_data, test_data, train_bin_labels, test_bin_labels, plot_epoches=True, scale_data=False):
    hidden_layer_sizes=(500, 1200, 400, 600,)
    alpha = 1e-03
    epsilon = 1e-08
    solver = 'adam'
    max_iter = 150
    warm_start = False
    batch_size = 32
    verbose = True

    if(scale_data == True):
        train_data , test_data = nn_scale_data(train_data, test_data)

    if(plot_epoches == True):
        warm_start = True

        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, epsilon=epsilon, solver=solver,
                            max_iter=1, warm_start=warm_start, batch_size=batch_size, verbose=verbose)
        print(nn.get_params())
        macro_precision = []
        micro_precision = []

        for i in range(max_iter):
            nn.fit(train_data, train_bin_labels)
            predictions = nn.predict(test_data)
            macro_precision.append(precision_score(test_bin_labels, predictions, average='macro'))
            micro_precision.append(precision_score(test_bin_labels, predictions, average='micro'))

        print("macro: {:.4f}".format(precision_score(test_bin_labels, predictions, average='macro')))
        print("micro: {:.4f}".format(precision_score(test_bin_labels, predictions, average='micro')))

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(211)
        plt.plot(macro_precision)
        plt.ylabel('macro_precision')
        plt.subplot(212)
        plt.plot(micro_precision)
        plt.ylabel('micro_precision')
        plt.show()

    else:
        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, epsilon=epsilon, solver=solver,
                            max_iter=max_iter, warm_start=warm_start)
        print(nn.get_params())
        nn.fit(train_data, train_bin_labels)
        predictions = nn.predict(test_data)
        evaluate(test_bin_labels, predictions)



