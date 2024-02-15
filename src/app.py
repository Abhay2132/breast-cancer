import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pathlib import Path

csv_file_path = Path(Path.cwd(),"res", "dataset", "data.csv")
df=pd.read_csv(str(csv_file_path))

df.drop(columns=["id","Unnamed: 32"],inplace =True )

# print the number of  Malignant and Benign
df_diagnosis=df['diagnosis'].value_counts().reset_index()
df_diagnosis

# show the diferance between the number of  Malignant and Benign     
fig = px.pie(df_diagnosis, values='count', names='diagnosis',title='the number of  Malignant and Benign ')
fig.show()

X=df.drop("diagnosis",axis=1)
y=df['diagnosis']

# scal the featuers max value = 1 , min value = 0 
scaler = MinMaxScaler() 
X = scaler.fit_transform(X)

# convert the target from categorical to numerical 
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# split the data to 80% train & 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

# built ANN model with input layer & two hidden layer & one output layer & actvation function is relue & in output layer is sigmoid 
model = Sequential([
    Dense(32, activation='relu', input_dim=30),
    
    Dense(16, activation='relu'),
    
    Dense(8, activation='relu'),
    
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# model training with 60 epochs 
history = model.fit(X_train, y_train, epochs=60, validation_split=0.2)

tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = [i+1 for i in range(len(tr_acc))]

# show the roc and accuracy
plt.figure(figsize=(30, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, 'r', label='Train Loss')
plt.plot(epochs, val_loss, 'g', label='Valid Loss')
plt.title('Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, 'r', label='Train Accuracy')
plt.plot(epochs, val_acc, 'g', label='Valid Accuracy')
plt.title('Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# predict the test data 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# show the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

# display report 
name=['Benign','Malignant']
classification_rep = classification_report(y_test, y_pred, target_names=name)
print("\nClassification Report:")
print(classification_rep)