class Neural_Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, X_train, y_train, epochs, learning_rate):
        sample = len(X_train)
        for i in range(epochs):
            err = 0
            for j in range(sample):
                output = X_train[j]

                for layer in self.layers:
                    output = layer.forward_propogation(output)

                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propogation(error, learning_rate)

            err /= sample
            print('epoch %d/%d error=%f' % (i+1, epochs, err))

    def predict(self, X_test):
        sample = len(X_test)
        result = []
        for i in range(sample):
            output = X_test[i]
            for layer in self.layers:
                output = layer.forward_propogation(output)
            result.append(output)

        return result






