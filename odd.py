import numpy as np
from sklearn.linear_model import Perceptron

# Helper function to convert ASCII numbers to binary representation
def ascii_to_binary(num):
    return [int(bit) for bit in format(num, '08b')]

# Training data
X_train = np.array([ascii_to_binary(i) for i in range(10)])  # 0 to 9 in binary
y_train = np.array([i % 2 for i in range(10)])  # 0 for even, 1 for odd

# Create and train the perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Test the perceptron
numbers_to_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for num in numbers_to_test:
    binary_representation = np.array([ascii_to_binary(num)])
    prediction = perceptron.predict(binary_representation)
    print(f"Number {num} is {'even' if prediction == 0 else 'odd'}")
