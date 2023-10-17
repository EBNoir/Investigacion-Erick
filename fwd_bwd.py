import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations
import statsmodels.api as sm

# Generate some random data for demonstration
np.random.seed(0)
X = np.random.rand(100, 5)
y = 2*X[:,0] + 3*X[:,1] + 0.5*X[:,2] + np.random.normal(0, 0.1, 100)

# Forward Selection
def forward_selection(X, y):
    num_features = X.shape[1]
    selected_features = []
    best_model = None
    best_mse = float('inf')
    
    for i in range(num_features):
        remaining_features = list(set(range(num_features)) - set(selected_features))
        for feature in remaining_features:
            model = LinearRegression()
            features = selected_features + [feature]
            X_subset = X[:, features]
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            mse = mean_squared_error(y, y_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_model = model
                selected_features = features
                
    return selected_features, best_model

selected_features, best_model = forward_selection(X, y)
print("Selected Features (Forward Selection):", selected_features)

# Backward Selection
def backward_selection(X, y):
    num_features = X.shape[1]
    selected_features = list(range(num_features))
    best_model = None
    best_mse = float('inf')
    
    for i in range(num_features):
        for feature in selected_features:
            features = list(set(selected_features) - set([feature]))
            model = LinearRegression()
            X_subset = X[:, features]
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            mse = mean_squared_error(y, y_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_model = model
                selected_features = features
                
    return selected_features, best_model

selected_features, best_model = backward_selection(X, y)
print("Selected Features (Backward Selection):", selected_features)
