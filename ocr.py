#!/usr/bin/env python3

import numpy as np
import json
import math
import os
from sklearn import datasets

class OCRNeuralNetwork:
    """
    A feedforward neural network for optical character recognition of handwritten digits.
    Uses backpropagation algorithm for training.
    """
    
    NN_FILE_PATH = 'ocr_network.json'
    
    def __init__(self, num_hidden_nodes=15, data_matrix=None, data_labels=None, 
                 training_indices=None, use_file=True):
        """
        Initialize the neural network.
        
        Args:
            num_hidden_nodes: Number of nodes in the hidden layer
            data_matrix: Training data matrix (optional)
            data_labels: Training data labels (optional) 
            training_indices: Indices for training data (optional)
            use_file: Whether to save/load weights from file
        """
        self.num_hidden_nodes = num_hidden_nodes
        self.num_inputs = 400  # 20x20 pixel input
        self.num_outputs = 10  # digits 0-9
        self.learning_rate = 0.1
        self._use_file = use_file
        
        # Initialize weights and biases
        self.theta1 = np.array(self._rand_initialize_weights(self.num_inputs, num_hidden_nodes))
        self.theta2 = np.array(self._rand_initialize_weights(num_hidden_nodes, self.num_outputs))
        self.input_layer_bias = np.zeros((num_hidden_nodes, 1))
        self.hidden_layer_bias = np.zeros((self.num_outputs, 1))
        
        # Try to load existing weights
        if self._use_file and os.path.exists(self.NN_FILE_PATH):
            self._load()
            print("📁 Loaded existing neural network weights")
        else:
            # Train on initial dataset if no saved weights exist
            if data_matrix is None:
                print("🔄 Generating initial training data...")
                data_matrix, data_labels = self._generate_training_data()
                training_indices = list(range(len(data_matrix)))
            
            if data_matrix is not None:
                print("🎯 Training neural network on initial dataset...")
                self._train_initial(data_matrix, data_labels, training_indices)
                if self._use_file:
                    self.save()
                print("✅ Initial training completed")

    def _generate_training_data(self):
        """Generate initial training data using sklearn digits dataset"""
        try:
            # Load the digits dataset (8x8 images)
            digits = datasets.load_digits()
            
            # Resize to 20x20 by repeating pixels (simple upsampling)
            data_matrix = []
            data_labels = digits.target.tolist()
            
            for img in digits.images:
                # Normalize to 0-1 range
                img_normalized = img / 16.0
                
                # Resize from 8x8 to 20x20 using nearest neighbor
                resized = np.zeros((20, 20))
                for i in range(20):
                    for j in range(20):
                        orig_i = min(int(i * 8 / 20), 7)
                        orig_j = min(int(j * 8 / 20), 7)
                        resized[i, j] = img_normalized[orig_i, orig_j]
                
                data_matrix.append(resized.flatten().tolist())
            
            print(f"📊 Generated {len(data_matrix)} training samples")
            return data_matrix, data_labels
            
        except ImportError:
            print("⚠️  sklearn not available, starting with empty network")
            return None, None

    def _rand_initialize_weights(self, size_in, size_out):
        """Initialize random weights between -0.06 and 0.06"""
        return [[((x * 0.12) - 0.06) for x in row] for row in np.random.rand(size_out, size_in)]

    def _sigmoid_scalar(self, z):
        """Sigmoid activation function for scalar input"""
        return 1 / (1 + math.e ** -z)

    def sigmoid(self, z):
        """Vectorized sigmoid activation function"""
        if np.isscalar(z):
            return self._sigmoid_scalar(z)
        else:
            return np.vectorize(self._sigmoid_scalar)(z)

    def sigmoid_prime(self, z):
        """Derivative of sigmoid function"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def _train_initial(self, data_matrix, data_labels, training_indices):
        """Train the network on initial dataset"""
        # Convert to format expected by train method
        train_data = []
        for i in training_indices:
            train_data.append({
                'y0': data_matrix[i],
                'label': data_labels[i]
            })
        
        # Train for multiple epochs
        for epoch in range(3):
            print(f"  Epoch {epoch + 1}/3...")
            self.train(train_data)

    def train(self, training_data_array):
        """
        Train the neural network using backpropagation algorithm.
        
        Args:
            training_data_array: List of dictionaries with 'y0' (input) and 'label' (expected output)
        """
        for data in training_data_array:
            # Forward propagation
            y1 = np.dot(self.theta1, np.array(data['y0']).reshape(-1, 1))
            sum1 = y1 + self.input_layer_bias  # Add bias
            y1 = self.sigmoid(sum1)

            y2 = np.dot(self.theta2, y1)
            y2 = y2 + self.hidden_layer_bias  # Add bias
            y2 = self.sigmoid(y2)

            # Backpropagation
            # Create expected output vector (one-hot encoding)
            actual_vals = [0] * 10
            actual_vals[data['label']] = 1
            
            # Calculate output layer errors
            output_errors = np.array(actual_vals).reshape(-1, 1) - y2
            
            # Calculate hidden layer errors
            hidden_errors = np.multiply(
                np.dot(self.theta2.T, output_errors), 
                self.sigmoid_prime(sum1)
            )

            # Update weights
            self.theta1 += self.learning_rate * np.dot(hidden_errors, np.array(data['y0']).reshape(1, -1))
            self.theta2 += self.learning_rate * np.dot(output_errors, y1.T)
            
            # Update biases
            self.hidden_layer_bias += self.learning_rate * output_errors
            self.input_layer_bias += self.learning_rate * hidden_errors

    def predict(self, test):
        """
        Make a prediction for given input.
        
        Args:
            test: Input data (list of 400 values representing 20x20 pixels)
            
        Returns:
            Predicted digit (0-9)
        """
        if isinstance(test, str):
            test = json.loads(test)
        
        # Forward propagation
        y1 = np.dot(self.theta1, np.array(test).reshape(-1, 1))
        y1 = y1 + self.input_layer_bias  # Add bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(self.theta2, y1)
        y2 = y2 + self.hidden_layer_bias  # Add bias
        y2 = self.sigmoid(y2)

        # Return index of maximum value (predicted digit)
        results = y2.flatten().tolist()
        return results.index(max(results))

    def save(self):
        """Save neural network weights to file"""
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        
        try:
            with open(self.NN_FILE_PATH, 'w') as nnFile:
                json.dump(json_neural_network, nnFile)
            print("💾 Network weights saved successfully")
        except Exception as e:
            print(f"❌ Failed to save weights: {e}")

    def _load(self):
        """Load neural network weights from file"""
        if not self._use_file:
            return

        try:
            with open(self.NN_FILE_PATH) as nnFile:
                nn = json.load(nnFile)
            
            self.theta1 = np.array(nn['theta1'])
            self.theta2 = np.array(nn['theta2'])
            self.input_layer_bias = np.array(nn['b1'])
            self.hidden_layer_bias = np.array(nn['b2'])
            
        except Exception as e:
            print(f"⚠️  Failed to load weights: {e}")
            print("Starting with random weights...")

    def get_accuracy(self, data_matrix, data_labels, test_indices):
        """
        Calculate accuracy of the network on test data.
        
        Args:
            data_matrix: Test data matrix
            data_labels: Test data labels
            test_indices: Indices of test data
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        correct_predictions = 0
        total_predictions = len(test_indices)
        
        for i in test_indices:
            prediction = self.predict(data_matrix[i])
            if data_labels[i] == prediction:
                correct_predictions += 1
        
        return correct_predictions / float(total_predictions)


def main():
    """Test the OCR neural network"""
    print("🧠 Testing OCR Neural Network")
    
    # Create and test network
    nn = OCRNeuralNetwork()
    
    # Simple test with a pattern
    test_pattern = [0] * 400
    # Create a simple pattern (like digit 1)
    for i in range(8, 12):  # vertical line
        for j in range(5, 15):
            idx = j * 20 + i
            if idx < 400:
                test_pattern[idx] = 1
    
    prediction = nn.predict(test_pattern)
    print(f"🔍 Test prediction: {prediction}")


if __name__ == "__main__":
    main()