import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import pickle
import os

def plot_history(history, save_path=None):
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Keras history object or dictionary with training history
        save_path: Optional path to save the plots
    """
    # If history is a file path, load it
    if isinstance(history, str):
        with open(history, 'rb') as f:
            history = pickle.load(f)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.savefig('../evaluation_results/training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved to ../evaluation_results/training_history.png")
    
    plt.close()


if __name__ == "__main__":
    # Example usage: plot saved training history
    history_path = '../models/training_history.pkl'
    if os.path.exists(history_path):
        plot_history(history_path)
    else:
        print(f"Training history not found at {history_path}")
        print("Train the model first to generate training history.")
