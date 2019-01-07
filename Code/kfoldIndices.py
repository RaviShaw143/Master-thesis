from sklearn.model_selection import StratifiedKFold
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
print(skf.get_n_splits(X, y))
print(skf)  
StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    
    print(X_train)
    print(X_test)
    y_train, y_test = y[train_index], y[test_index]
