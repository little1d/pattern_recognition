import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import os

# 数据集预处理

# iris
ds = load_dataset("scikit-learn/iris", split="train")
ds = ds.train_test_split(test_size=0.2, shuffle=True)
X_train_iris = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['train']]
y_train_iris = [sample['Species'] for sample in ds['train']]
y_train_iris = labelencoder.fit_transform(y_train_iris)

X_test_iris = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['test']]
y_test_iris = [sample['Species'] for sample in ds['test']]
y_test_iris = labelencoder.fit_transform(y_test_iris)

# sonar
ds = load_dataset("mstz/sonar", split="train")
ds = ds.train_test_split(test_size=0.2, shuffle=True)

X_train_sonar = [list(sample.values())[:-1] for sample in ds['train']]
y_train_sonar = [list(sample.values())[-1] for sample in ds['train']]
y_train_sonar = labelencoder.fit_transform(y_train_sonar)

X_test_sonar = [list(sample.values())[:-1] for sample in ds['test']]
y_test_sonar = [list(sample.values())[-1] for sample in ds['test']]
y_test_sonar = labelencoder.fit_transform(y_test_sonar)


# Create a directory to store the results
output_dir = "svm_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the dataset dictionary
datasets = {
    "iris": (X_train_iris, y_train_iris, X_test_iris, y_test_iris),
    "sonar": (X_train_sonar, y_train_sonar, X_test_sonar, y_test_sonar)
}

# List of kernel functions
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Store evaluation metrics
results = {dataset_name: {kernel: {} for kernel in kernels} for dataset_name in datasets}

# Iterate over each dataset
for dataset_name, (X_train, y_train, X_test, y_test) in datasets.items():
    print(f"Dataset: {dataset_name}")
    output_text = []

    for kernel in kernels:
        print(f"Using kernel: {kernel}")
        output_text.append(f"Dataset: {dataset_name}, Kernel: {kernel}\n")
        
        # Initialize the SVM model
        model = SVC(kernel=kernel, C=1.0, gamma='scale')
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[dataset_name][kernel] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        output_text.append(f"Classification Report:\n{report}\n")
        
        # Plot and save the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Dataset: {dataset_name}, Kernel: {kernel}")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{dataset_name}_{kernel}.png"))
        plt.close()
        
        print(report)
        print("-" * 50)

    # Save textual results as a .txt file
    with open(os.path.join(output_dir, f"evaluation_{dataset_name}.txt"), 'w') as f:
        f.writelines("\n".join(output_text))

# Visualize and save the evaluation metrics as bar plots
for dataset_name in datasets:
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [results[dataset_name][kernel][metric] for kernel in kernels]
        sns.barplot(x=kernels, y=values)
        plt.title(f"{metric.capitalize()} - Dataset: {dataset_name}")
        plt.xlabel("Kernel")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, f"{metric}_{dataset_name}.png"))
        plt.close()
