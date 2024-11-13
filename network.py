# This fork comes with weight initialisation, csv file loader and metrics
# for you to check the effects on metrics by changing hyperparameters. 
# You can implement this neural network on your csv files . But make sure to preprocess the data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self, init_method="normal"):
        # Initialize weights with the chosen method
        self.init_weights(init_method)

    def init_weights(self, method="normal"):
        if method == "xavier":
            scale = np.sqrt(1 / 2)
        elif method == "he":
            scale = np.sqrt(2 / 2)
        else:
            scale = 1  # Standard normal initialization

        # Weights
        self.w1 = np.random.normal(scale=scale)
        self.w2 = np.random.normal(scale=scale)
        self.w3 = np.random.normal(scale=scale)
        self.w4 = np.random.normal(scale=scale)
        self.w5 = np.random.normal(scale=scale)
        self.w6 = np.random.normal(scale=scale)

        # Biases
        self.b1 = np.random.normal(scale=scale)
        self.b2 = np.random.normal(scale=scale)
        self.b3 = np.random.normal(scale=scale)

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    # You can use the learn_rate as a 'dial' - a hyperparameter that is
    # multiplied when updating weights and biases (line 81 - 89) . 
    # Check the metrics for different learning rates ranging 0.01 to 0.2 
    def train(self, data, all_y_trues, learn_rate=0.05, epochs=1000):
        losses = []
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Feedforward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                y_pred = sigmoid(sum_o1)

                # Calculate gradients
                d_L_d_ypred = -2 * (y_true - y_pred)
                # Gradients for weights and biases
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # Track loss for each epoch
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} loss: {loss:.3f}")

        # Plot loss over epochs
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    def evaluate_accuracy(self, data, all_y_trues):
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        predictions = np.round(y_preds)
        accuracy = (predictions == all_y_trues).mean()
        print(f"Accuracy: {accuracy:.3f}")
        return accuracy

    @staticmethod
    def load_data_from_csv(filepath):
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values  # Assumes features are all columns except last
        y = data.iloc[:, -1].values  # Assumes labels are in the last column
        return X, y

# Example usage:
# network = OurNeuralNetwork(init_method="xavier")  # Try "normal", "xavier", "he"
# data, labels = OurNeuralNetwork.load_data_from_csv("your_data.csv")
# network.train(data, labels)
# network.evaluate_accuracy(data, labels)

