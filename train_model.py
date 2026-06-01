import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load data
with open('keypoint_data/data.pkl', 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])

print(f'Total samples: {len(data)}')
print(f'Classes: {set(labels)}')

# Decide on stratified split only when feasible
N = len(data)
unique, counts = np.unique(labels, return_counts=True)
n_classes = len(unique)
test_frac = 0.2
test_size = int(test_frac * N)

use_stratify = True
if np.min(counts) < 2:
    print('Warning: at least one class has fewer than 2 samples; disabling stratified split.')
    use_stratify = False
elif test_size < n_classes:
    print('Warning: test set too small to include one sample per class; disabling stratified split.')
    use_stratify = False

if use_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_frac, shuffle=True, stratify=labels
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_frac, shuffle=True
    )

# Train
print('\nTraining...')
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model saved as model.pkl!')