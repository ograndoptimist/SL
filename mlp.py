import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#------------------------------------------------- Reading the data ----------------------------------------------------------------------------:
data = pd.read_csv('iris.csv')
dataframe = pd.DataFrame(data)

#------------------------------------------------ Preprocessing the data -----------------------------------------------------------------------:
def normalize_data(column: str):
    array = np.array(dataframe[column])
    array = array.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler = scaler.fit(array)

    return scaler.transform(array)


dataframe['sepal_length'] = normalize_data('sepal_length')
dataframe['sepal_width'] = normalize_data('sepal_width')
dataframe['petal_length'] = normalize_data('petal_length')
dataframe['petal_width'] = normalize_data('petal_width')


test = set(dataframe['species'])
test = list(test)


for j, k in enumerate(dataframe['species']):
    if k == test[0]:
        dataframe['species'].iloc[j] = 0
    elif k == test[1]:
        dataframe['species'].iloc[j] = 1
    else:
        dataframe['species'].iloc[j] = 2
        

x_train = np.array(dataframe[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

y_train = dataframe['species']
y_train = to_categorical(y_train)

x_train_partial, x_test_partial, y_train_partial, y_test_partial = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

x_train, x_val, y_train, y_val = train_test_split(x_train_partial, y_train_partial, test_size = 0.2, random_state = 1)

#------------------------------------------------------------- Building the network ---------------------------------------------------------
def fit_model(neurons: int, epochs: int, batch_size: int):
    model = Sequential()

    model.add(Dense(neurons, activation = 'relu', input_dim = x_train_partial.shape[1]))
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'], shuffle = True)

    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_val, y_val))

    result = model.evaluate(x_test_partial, y_test_partial)
    print()
    print(result)

neurons = [16, 32, 64, 128]
repeats = [10, 100, 1000]
batch_size = 64

for k in neurons:
    for j in repeats:
        fit_model(neurons, epochs, batch_size)
        
#----------------------------------------------------------- Plotting the results -----------------------------------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
