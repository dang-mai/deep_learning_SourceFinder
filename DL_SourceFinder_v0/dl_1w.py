import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import LSTM, Flatten, Conv1D, Dense, Dropout


tmp = np.load('train.npz', allow_pickle=True)
train = tmp['features'][()]
train_label = train['label']
train_name = train['name-mode1']
train_topics = train['topics-mode1']
train_description = train['description-mode1']
train_readme = train['readme-mode1']
train_files = train['files-mode1']
train_features = np.concatenate([train_name, train_topics, train_files, train_readme, train_description], axis=1)


model = Sequential()
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_features, train_label, epochs=20, batch_size=32, validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# test_loss, test_acc = model.evaluate(x_test, y_test)
# print("test_loss:", test_loss)
# print("test_acc: ", test_acc)
