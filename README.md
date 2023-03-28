---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Building an ML Model for Deployment

```python
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
```

```python
SEED = 42
EPOCHS = 888
MODEL_PATH="./model/full_iris_model.h5"
SCALER_PATH="./model/iris_data_norm.pkl"
```

## IRIS Dataset

> `wget https://gist.githubusercontent.com/Thanatoz-1/9e7fdfb8189f0cdf5d73a494e4a6392a/raw/aaecbd14aeaa468cd749528f291aa8a30c2ea09e/iris_dataset.csv`

```python
iris_dataset = pd.read_csv("./data/iris_dataset.csv")
iris_dataset.head()
```

```python
# separate features from labels
X = iris_dataset.drop('target', axis=1)
X.head()
```

```python
y = iris_dataset['target']
y.unique()
```

```python
# 1-hot encoding labels
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
y[0]
```

```python
# create training / testing datasets
X_train, X_test, y_train, y_test = train_test_split(
                                        X, y,
                                        test_size=0.2,
                                        random_state=SEED)
```

```python
# normalize training data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
```

## Building the Model

```python
iris_model = Sequential([
    Dense(units=4, activation='relu', input_shape=[4,]),
    Dense(units=3, activation='softmax')
])

iris_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
```

```python
# fitting the model
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0)
```

## Fitting the Model

```python
history_iris_model = iris_model.fit(x=X_train_norm,
         y=y_train,
         epochs=EPOCHS,
         validation_data=(X_test_norm, y_test),
         callbacks=[early_stop])
```

```python
# evaluate the model
iris_model.evaluate(X_test_norm, y_test, verbose=0)
# [0.334958016872406, 0.8999999761581421]
```

```python
# plot the validation accuracy
def plot_accuracy_curves(history, title):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['accuracy']))

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend();
```

```python
plot_accuracy_curves(history_iris_model, "IRIS Dataset :: Accuracy Curve")
```

![Deploying Prediction APIs](https://github.com/mpolinowski/ml-flask-web-app/blob/master/assets/IRIS_Dataset_Model_Deployment_01.png)


```python
# plot the training loss
def plot_loss_curves(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(history.history['loss']))

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend();
```

```python
plot_loss_curves(history_iris_model, "IRIS Dataset :: Loss Curve")
```

![Deploying Prediction APIs](https://github.com/mpolinowski/ml-flask-web-app/blob/master/assets/IRIS_Dataset_Model_Deployment_02.png)


## Fit all Data


After reaching a approx. 90% accuracy we can now add the testing data to our model training to increase the dataset variety the model was trained on.

```python
X_norm =scaler.fit_transform(X)
```

```python
iris_model_full = Sequential([
    Dense(units=4, activation='relu', input_shape=[4,]),
    Dense(units=3, activation='softmax')
])

iris_model_full.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
```

```python
history_iris_model_full = iris_model_full.fit(X_norm, y, epochs=EPOCHS)
```

```python
# evaluate the model
iris_model_full.evaluate(X_norm, y, verbose=0)
# [0.1931973546743393, 0.9733333587646484]
```

```python
# plot the validation and training loss
def plot_training_curves(history, title):
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(history.history['loss']))

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, loss, label='training_loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend();
```

```python
# plot accuracy and loss curves
plt.figure(figsize=(12, 6))
plot_training_curves(history_iris_model_full, "IRIS Dataset :: Training Curves")
```

![Deploying Prediction APIs](https://github.com/mpolinowski/ml-flask-web-app/blob/master/assets/IRIS_Dataset_Model_Deployment_03.png)


## Save the Trained Model

```python
# save the full model with training weights
iris_model_full.save(MODEL_PATH)
```

```python
# save data preprocessing
joblib.dump(scaler, SCALER_PATH)
```

## Run Predictions

```python
# load the saved model
loaded_iris_model = load_model(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)
```

```python
# verify predictions are the same
loaded_iris_model.evaluate(X_norm, y, verbose=0)
```

## Prediction API

```python
# simulate JSON API call
flower_example = {"sepal length (cm)": 5.1,
                  "sepal width (cm)": 3.5,
                  "petal length (cm)":1.4,
                  "petal width (cm)": 0.2}
```

```python
# API function (return class index with highest probability)
def return_prediction(model, scaler, json_request):
    s_len = json_request["sepal length (cm)"]
    s_wi = json_request["sepal width (cm)"]
    p_len = json_request["petal length (cm)"]
    p_w = json_request["petal width (cm)"]
    
    measures =[[s_len, s_wi, p_len, p_w]]
    measures_norm = scaler.transform(measures)
    
    flower_class_probabilities = model.predict(measures_norm)
    flower_class_index=np.argmax(flower_class_probabilities,axis=1)
                           
    return flower_class_index
```

```python
return_prediction(loaded_iris_model, loaded_scaler, flower_example)
# probabilities array([[9.987895e-01, 7.723020e-04, 4.383073e-04]], dtype=float32)
# index array([0])
```

```python
# API function (return class name)
def return_prediction(model, scaler, json_request):
    s_len = json_request["sepal length (cm)"]
    s_wi = json_request["sepal width (cm)"]
    p_len = json_request["petal length (cm)"]
    p_w = json_request["petal width (cm)"]
    
    classes = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    measures =[[s_len, s_wi, p_len, p_w]]
    measures_norm = scaler.transform(measures)
    
    flower_class_probabilities = model.predict(measures_norm)
    flower_class_index=np.argmax(flower_class_probabilities,axis=1)
                       
    return classes[flower_class_index]
```

```python
return_prediction(loaded_iris_model, loaded_scaler, flower_example)
# array(['Iris-setosa'], dtype='<U15')
```


### Prediction Frontend

Start the Flask server:


```bash
python Flask_Server.py
```


![Deploying Prediction APIs](https://github.com/mpolinowski/ml-flask-web-app/blob/master/assets/IRIS_Dataset_Model_Deployment_05.png)


![Deploying Prediction APIs](https://github.com/mpolinowski/ml-flask-web-app/blob/master/assets/IRIS_Dataset_Model_Deployment_06.png)
