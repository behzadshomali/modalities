import re
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# Download the GSM8K dataset from Hugging Face
print("Downloading GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main")

def extract_answer(answer_text):
    """
    Extract the numerical answer from the answer text.
    GSM8K answers are formatted with #### followed by the numerical answer.
    """
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from numbers like 1,000
        number_str = match.group(1).replace(',', '')
        try:
            # Try to convert to float first, then to int if it's a whole number
            num = float(number_str)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            return None
    return None

# Extract answers from train and test sets
print("Extracting answers from train set...")
train_answers = []
for example in dataset['train']:
    answer = extract_answer(example['answer'])
    if answer is not None:
        train_answers.append(answer)

print("Extracting answers from test set...")
test_answers = []
for example in dataset['test']:
    answer = extract_answer(example['answer'])
    if answer is not None:
        test_answers.append(answer)

# Print statistics
print(f"\nTrain set: {len(train_answers)} answers extracted")
print(f"Test set: {len(test_answers)} answers extracted")
print(f"\nTrain set statistics:")
print(f"  Min: {min(train_answers)}")
print(f"  Max: {max(train_answers)}")
print(f"  Mean: {np.mean(train_answers):.2f}")
print(f"  Median: {np.median(train_answers):.2f}")
print(f"\nTest set statistics:")
print(f"  Min: {min(test_answers)}")
print(f"  Max: {max(test_answers)}")
print(f"  Mean: {np.mean(test_answers):.2f}")
print(f"  Median: {np.median(test_answers):.2f}")

# Create histograms
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('GSM8K Answer Distributions', fontsize=16, fontweight='bold')

# Train set - full range
axes[0, 0].hist(train_answers, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Train Set - Full Range')
axes[0, 0].set_xlabel('Answer Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Test set - full range
axes[0, 1].hist(test_answers, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_title('Test Set - Full Range')
axes[0, 1].set_xlabel('Answer Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Train set - zoomed in (0-1000 range for better visibility)
train_filtered = [x for x in train_answers if 0 <= x <= 1000]
axes[1, 0].hist(train_filtered, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].set_title('Train Set - Zoomed (0-1000)')
axes[1, 0].set_xlabel('Answer Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Test set - zoomed in (0-1000 range for better visibility)
test_filtered = [x for x in test_answers if 0 <= x <= 1000]
axes[1, 1].hist(test_filtered, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].set_title('Test Set - Zoomed (0-1000)')
axes[1, 1].set_xlabel('Answer Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gsm8k_distributions.png', dpi=300, bbox_inches='tight')
print("\nHistogram saved as 'gsm8k_distributions.png'")
plt.show()

# Additional analysis: Most common answers
from collections import Counter

print("\n=== Top 10 Most Common Answers ===")
print("\nTrain Set:")
train_counter = Counter(train_answers)
for answer, count in train_counter.most_common(10):
    print(f"  {answer}: {count} times")

print("\nTest Set:")
test_counter = Counter(test_answers)
for answer, count in test_counter.most_common(10):
    print(f"  {answer}: {count} times")