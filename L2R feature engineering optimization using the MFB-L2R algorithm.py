
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.acquisition import gaussian_ei
from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt import dump, load

# Load breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature engineering functions
def add_square_feature(X):
    X_square = np.square(X)
    return np.hstack((X, X_square))

def add_interaction_feature(X):
    n_samples, n_features = X.shape
    X_interaction = np.zeros((n_samples, n_features*(n_features-1)//2))
    idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            X_interaction[:,idx] = X[:,i] * X[:,j]
            idx += 1
    return np.hstack((X, X_interaction))

# Define feature engineering method space
feature_methods = [
    ("None", lambda X: X),
    ("Square", add_square_feature),
    ("Interaction", add_interaction_feature)
]

# Define meta-feature extraction function
def extract_meta_features(X):
    meta_features = []
    n_samples, n_features = X.shape
    meta_features.append(n_features)
    meta_features.append(np.mean(X))
    meta_features.append(np.std(X))
    return np.array(meta_features)

# Extract meta-features for training and test sets
X_train_meta = np.apply_along_axis(extract_meta_features, 1, X_train)
X_test_meta = np.apply_along_axis(extract_meta_features, 1, X_test)

# Define L2R objective function
@use_named_args([
    Integer(0, len(feature_methods)-1, name='method'),
    Real(0.01, 1.0, name='n_estimators'),
    Real(0.1, 1.0, name='max_depth')
])
def objective(**params):
    method_idx = params['method']
    feature_method = feature_methods[method_idx][1]
    X_train_fe = feature_method(X_train)
    X_test_fe = feature_method(X_test)
    n_estimators = int(params['n_estimators'] * 100)
    max_depth = int(params['max_depth'] * 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_fe, y_train)
    y_pred = model.predict(X_test_fe)
    score = accuracy_score(y_test, y_pred)
    return -score

# Define L2R search space
space = [
    Integer(0, len(feature_methods)-1, name='method'),
    Real(0.01, 1.0, name='n_estimators'),
    Real(0.1, 1.0, name='max_depth')
]

# Initialize L2R model
gp = GaussianProcessRegressor(kernel=Matern(nu=2.5))

# Define MFB-L2R objective
