import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import LSTM, Flatten, Conv1D, Dense, Dropout
import tensorflow.keras.optimizers as opt
import tensorflow.keras.regularizers as R
import random


tmp = np.load('train.npz', allow_pickle=True)
train = tmp['features'][()]
train_label = tmp['label']
train_name = train['name-mode1']
train_topics = train['topics-mode1']
train_description = train['description-mode1']
train_readme = train['readme-mode1']
train_files = train['files-mode1']
train_features = np.concatenate([train_name, train_topics, train_files, train_readme, train_description], axis=1)



index = [i for i in range(len(train_features))]
random.shuffle(index)
train_features = train_features[index]
train_label = train_label[index]

# print(train_features.shape)
# print(train_label.shape)
# a = input()


model = Sequential()
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu')) #, kernel_regularizer=R.l2(1e-4)))
model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

optimizer = opt.Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_features, train_label, epochs=100, batch_size=32, validation_split=0.2, shuffle=True)

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


tmp = np.load('test.npz', allow_pickle=True)
test = tmp['features'][()]
test_label = tmp['label']
test_name = test['name-mode1']
test_topics = test['topics-mode1']
test_description = test['description-mode1']
test_readme = test['readme-mode1']
test_files = test['files-mode1']
test_features = np.concatenate([test_name, test_topics, test_files, test_readme, test_description], axis=1)

test_loss, test_acc = model.evaluate(test_features, test_label)
print("test_loss:", test_loss)
print("test_acc: ", test_acc)


tmp = np.load('real.npz', allow_pickle=True)
real = tmp['features'][()]
real_label = tmp['label']
real_name = real['name-mode1']
real_topics = real['topics-mode1']
real_description = real['description-mode1']
real_readme = real['readme-mode1']
real_files = real['files-mode1']
real_features = np.concatenate([real_name, real_topics, real_files, real_readme, real_description], axis=1)

real_loss, real_acc = model.evaluate(real_features, real_label)
print("real_loss:", real_loss)
print("real_acc: ", real_acc)
