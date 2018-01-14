from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def nn_scale_data(train_data, test_data):
    scaler = StandardScaler()

    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled

def nn_create(hidden_layer_sizes=(30,), alpha=1e-05, epsilon=1e-08):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, epsilon=epsilon )
    return mlp
