import json
import sys

try:
    with open('compare/A-MSB-GRPO-E/checkpoints/training_metrics.json', 'r') as f:
        data = json.load(f)

    # Let's see some samples across the training
    print("Early training (first 5 steps):")
    for metrics in data[:5]:
        print(f"  Step: {metrics['step']}, Correct Ratio: {metrics.get('correct_ratio', 0):.4f}, KL: {metrics.get('kl_loss', 0):.4f}, Total Loss: {metrics.get('total_loss', 0):.4f}")

    print("\nMiddle training (middle 5 steps):")
    mid = len(data) // 2
    for metrics in data[mid:mid+5]:
        print(f"  Step: {metrics['step']}, Correct Ratio: {metrics.get('correct_ratio', 0):.4f}, KL: {metrics.get('kl_loss', 0):.4f}, Total Loss: {metrics.get('total_loss', 0):.4f}")

    print("\nEnd of training (last 10 steps):")
    for metrics in data[-10:]:
        print(f"  Step: {metrics['step']}, Correct Ratio: {metrics.get('correct_ratio', 0):.4f}, KL: {metrics.get('kl_loss', 0):.4f}, Total Loss: {metrics.get('total_loss', 0):.4f}")
except Exception as e:
    print(f"Error: {e}")
