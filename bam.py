import numpy as np

class BidirectionalAssociativeMemory:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weight_matrix = np.zeros((pattern_size, pattern_size))

    def train(self, pattern1, pattern2):
        self.weight_matrix += np.outer(pattern1, pattern2)

    def recall(self, input_pattern, max_iterations=10):
        for _ in range(max_iterations):
            output_pattern = np.sign(np.dot(self.weight_matrix, input_pattern))
            input_pattern = np.sign(np.dot(self.weight_matrix.T, output_pattern))
        
        return output_pattern

if __name__ == "__main__":
    pattern_size = 4
    bam = BidirectionalAssociativeMemory(pattern_size)
    
    # Training patterns
    pattern1 = np.array([1, -1, 1, -1])
    pattern2 = np.array([-1, 1, -1, 1])
    pattern3 = np.array([1, 1, -1, -1])
    pattern4 = np.array([-1, -1, 1, 1])

    bam.train(pattern1, pattern2)
    bam.train(pattern3, pattern4)

    # Test recall with one of the trained patterns
    test_pattern = np.array([-1, 1, -1, 1])
    recalled_pattern = bam.recall(test_pattern)

    # Display results
    print("Input Pattern:", test_pattern)
    print("Recalled Pattern:", recalled_pattern)
