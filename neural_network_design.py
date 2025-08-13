#!/usr/bin/env python3

import numpy as np
from ocr import OCRNeuralNetwork
from sklearn import datasets
import random

def load_and_prepare_data():
    """Load and prepare the digits dataset for training and testing"""
    try:
        # Load the digits dataset
        digits = datasets.load_digits()
        
        print(f"📊 Loaded {len(digits.images)} digit samples")
        
        # Prepare data matrix by resizing images from 8x8 to 20x20
        data_matrix = []
        data_labels = digits.target.tolist()
        
        for img in digits.images:
            # Normalize to 0-1 range
            img_normalized = img / 16.0
            
            # Resize from 8x8 to 20x20 using nearest neighbor interpolation
            resized = np.zeros((20, 20))
            for i in range(20):
                for j in range(20):
                    orig_i = min(int(i * 8 / 20), 7)
                    orig_j = min(int(j * 8 / 20), 7)
                    resized[i, j] = img_normalized[orig_i, orig_j]
            
            data_matrix.append(resized.flatten().tolist())
        
        # Split data into training and testing sets (75% train, 25% test)
        total_samples = len(data_matrix)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        split_point = int(0.75 * total_samples)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        print(f"📈 Training samples: {len(train_indices)}")
        print(f"📉 Testing samples: {len(test_indices)}")
        
        return data_matrix, data_labels, train_indices, test_indices
        
    except ImportError:
        print("❌ scikit-learn is required for this script")
        print("Install it with: pip install scikit-learn")
        return None, None, None, None

def test_network_accuracy(data_matrix, data_labels, test_indices, nn, num_trials=10):
    """
    Test the accuracy of a neural network multiple times and return average accuracy.
    
    Args:
        data_matrix: Test data matrix
        data_labels: Test data labels  
        test_indices: Indices for test data
        nn: OCRNeuralNetwork instance
        num_trials: Number of trials to average over
        
    Returns:
        Average accuracy as float
    """
    total_accuracy = 0
    
    for trial in range(num_trials):
        correct_predictions = 0
        
        for i in test_indices:
            try:
                prediction = nn.predict(data_matrix[i])
                if data_labels[i] == prediction:
                    correct_predictions += 1
            except:
                # Handle any prediction errors
                pass
        
        accuracy = correct_predictions / float(len(test_indices))
        total_accuracy += accuracy
    
    return total_accuracy / num_trials

def find_optimal_hidden_nodes():
    """
    Experiment with different numbers of hidden nodes to find optimal configuration.
    """
    print("🔬 Starting Neural Network Architecture Optimization")
    print("=" * 60)
    
    # Load and prepare data
    data_matrix, data_labels, train_indices, test_indices = load_and_prepare_data()
    
    if data_matrix is None:
        return
    
    results = []
    
    # Test different numbers of hidden nodes
    hidden_node_counts = range(5, 51, 5)  # 5, 10, 15, ..., 50
    
    print("\n🧪 Testing different hidden node configurations...")
    print("-" * 60)
    
    for num_hidden in hidden_node_counts:
        print(f"Testing {num_hidden} hidden nodes...")
        
        try:
            # Create neural network with specific number of hidden nodes
            nn = OCRNeuralNetwork(
                num_hidden_nodes=num_hidden,
                data_matrix=data_matrix,
                data_labels=data_labels,
                training_indices=train_indices,
                use_file=False  # Don't save/load from file during testing
            )
            
            # Test accuracy
            accuracy = test_network_accuracy(data_matrix, data_labels, test_indices, nn)
            results.append((num_hidden, accuracy))
            
            print(f"✅ {num_hidden} Hidden Nodes: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Error testing {num_hidden} nodes: {e}")
            results.append((num_hidden, 0.0))
    
    # Display results
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
    
    for i, (nodes, accuracy) in enumerate(results):
        status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{status} {nodes:2d} Hidden Nodes: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Find optimal configuration
    if results:
        best_nodes, best_accuracy = results[0]
        print(f"\n🎯 RECOMMENDATION")
        print("-" * 30)
        print(f"Optimal configuration: {best_nodes} hidden nodes")
        print(f"Expected accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Look for efficiency sweet spot
        print(f"\n💡 EFFICIENCY ANALYSIS")
        print("-" * 30)
        
        # Find the configuration that gives good accuracy with fewer nodes
        efficiency_results = [(nodes, acc, acc/nodes) for nodes, acc in results]
        efficiency_results.sort(key=lambda x: x[2], reverse=True)
        
        efficient_nodes, efficient_acc, efficiency = efficiency_results[0]
        print(f"Most efficient: {efficient_nodes} nodes ({efficient_acc:.4f} accuracy)")
        print(f"Efficiency ratio: {efficiency:.6f} (accuracy per node)")

def compare_activation_functions():
    """
    Compare different activation functions (for educational purposes).
    Note: This is a simplified comparison as our current implementation uses sigmoid.
    """
    print("\n🔬 ACTIVATION FUNCTION ANALYSIS")
    print("=" * 60)
    
    print("Current implementation uses Sigmoid activation function:")
    print("  ✅ Advantages:")
    print("    - Smooth gradient (differentiable)")
    print("    - Output range [0,1] good for our binary pixel inputs")
    print("    - Well-suited for binary classification problems")
    print("  ⚠️  Potential issues:")
    print("    - Vanishing gradient problem in deep networks")
    print("    - Can saturate for large inputs")
    
    print("\n🎯 Alternative activation functions to consider:")
    print("  • ReLU: f(x) = max(0,x) - Fast, prevents vanishing gradients")
    print("  • Tanh: f(x) = tanh(x) - Similar to sigmoid but range [-1,1]")
    print("  • Leaky ReLU: Prevents dying neuron problem")

def analyze_learning_parameters():
    """Analyze the impact of different learning parameters"""
    print("\n📚 LEARNING PARAMETER ANALYSIS")
    print("=" * 60)
    
    print("Current Learning Rate: 0.1")
    print("  📈 Effects of different learning rates:")
    print("    • High (>0.5): Fast learning but may overshoot optimal weights")
    print("    • Medium (0.1-0.5): Balanced learning speed and stability")
    print("    • Low (<0.1): Slow but stable learning")
    
    print("\n🎯 Recommendations:")
    print("  • Start with 0.1 (current value)")
    print("  • Reduce if training becomes unstable")
    print("  • Increase if learning is too slow")

def main():
    """Main function to run neural network design experiments"""
    print("🧠 OCR Neural Network Design & Optimization Tool")
    print("=" * 60)
    
    try:
        # Find optimal hidden node configuration
        find_optimal_hidden_nodes()
        
        # Additional analysis
        compare_activation_functions()
        analyze_learning_parameters()
        
        print("\n🎉 Analysis Complete!")
        print("Use the recommended configuration in your main OCR system.")
        
    except KeyboardInterrupt:
        print("\n⛔ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()