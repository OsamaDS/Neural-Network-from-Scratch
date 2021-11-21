class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propogation(self, input):
        raise NotImplementedError

    def backward_propogation(self, output_error, learning_rate):
        raise NotImplementedError
