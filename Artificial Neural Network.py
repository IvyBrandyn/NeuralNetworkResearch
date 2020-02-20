# Ivy Brandyn Vasquez 
"""
This is representative of independent research into Neural Networks
exploring how these models are created, initialized, used, etc.
"""

# Verify GPU memory growth
import tensorflow as tf
config = tf.config.experimental.list_physical_devices('GPU')
for device in config:
    tf.config.experimental.set_memory_growth(device, True)

# Data Preprocessing
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = np.asarray(dataset.iloc[:, 3:13])
y = np.asarray(dataset.iloc[:, 13])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X = LabelEncoder()
X[:, 2] = labelEncoder_X.fit_transform(X[:, 2])

columnTransformer = ColumnTransformer([('Geography', OneHotEncoder(), [1])],
                                      remainder = 'passthrough')
X = np.array(columnTransformer.fit_transform(X))
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scalar_X = StandardScaler()
X_train = scalar_X.fit_transform(X_train)
X_test = scalar_X.transform(X_test)

# Version 1
# Create Artificial Neural Network
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

# Initializing ANN model
ANN_Model = Sequential()

# Hidden layers and Dropout
ANN_Model.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_dim = 11))
ANN_Model.add(Dropout(p = 0.1))
ANN_Model.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
ANN_Model.add(Dropout(p = 0.1))

# Output layer
ANN_Model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

# Compiling Artificial Neural Network and fitting to training sets
ANN_Model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ANN_Model.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = ANN_Model.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
new_prediction = ANN_Model.predict(scalar_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Generating Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Version 2
# Implementing K-fold Cross Validation into our ANN model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# ANN model created in a function
def build_ANN_Model():
    # Initializing ANN model
    ANN_Model = Sequential()
    
    # Hidden layers and Dropout
    ANN_Model.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_dim = 11))
    ANN_Model.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
    
    # Output layer
    ANN_Model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    
    # Compiling Artificial Neural Network and returning ANN model
    ANN_Model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ANN_Model

ANN_Model_2 = KerasClassifier(build_fn = build_ANN_Model, batch_size = 10, epochs = 100)
# Windows has problems with parallelism here.  n_jobs should equal -1 for all cpus
accuracies = cross_val_score(estimator = ANN_Model_2, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Version 3
# Tuning the ANN
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

def build_ANN_Model_2(optimizer):
    # Initializing ANN model
    ANN_Model = Sequential()
    
    # Hidden layers and Dropout
    ANN_Model.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_dim = 11))
    ANN_Model.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
    
    # Output layer
    ANN_Model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    
    # Compiling Artificial Neural Network and returning ANN model
    ANN_Model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ANN_Model

ANN_Model = KerasClassifier(build_fn = build_ANN_Model_2)
parameters = {'batch_size' : [25, 32], 'epochs' : [100, 500], 'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = ANN_Model, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_