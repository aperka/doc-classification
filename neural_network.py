from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def nn_scale_data(train_data, test_data):
    scaler = StandardScaler()

    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled

def nn_create(hidden_layer_sizes=(5000,), alpha=1e-01, epsilon=1e-08, solver='adam', max_iter=200):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, epsilon=epsilon, solver=solver, max_iter=max_iter )
    return mlp
