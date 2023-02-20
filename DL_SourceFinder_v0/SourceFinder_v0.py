from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Embedding, LSTM, concatenate
from tensorflow.keras import Input
import matplotlib.pyplot as plt

import text_preprocess

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

name_train, name_test, topics_train, topics_test, files_train, files_test, \
des_train, des_test, readme_train, readme_test, label_train, label_test = text_preprocess.return_result()

name_max_len, topics_max_len, files_max_len, des_max_len, readme_max_len, \
name_max_features, topics_max_features, files_max_features, \
des_max_features, readme_max_features = text_preprocess.return_info()

# print(len(name_train), name_max_len)
# print(len(topics_train), topics_max_len)
# print(len(des_train), des_max_len)
# print(len(readme_train), readme_max_len)
# print(len(files_train), files_max_len)
#
# wait = input("wait")

name_input = Input(shape=(name_max_len,), dtype='int32', name='name')
topics_input = Input(shape=(topics_max_len,), dtype='int32', name='topics')
des_input = Input(shape=(des_max_len,), dtype='int32', name='des')
readme_input = Input(shape=(readme_max_len,), dtype='int32', name='readme')
files_input = Input(shape=(files_max_len,), dtype='int32', name='files')

name_embedded = Embedding(50, 32, input_length=name_max_len)(name_input)
topics_embedded = Embedding(50, 32, input_length=topics_max_len)(topics_input)
des_embedded = Embedding(500, 32, input_length=des_max_len)(des_input)
readme_embedded = Embedding(100, 32, input_length=readme_max_len)(readme_input)
files_embedded = Embedding(1000, 32, input_length=files_max_len)(files_input)

# name_vec = Flatten()(name_embedded)
# print(name_vec)
# topics_vec = Flatten()(topics_embedded)
# print(topics_vec)
# des_vec = Flatten()(des_embedded)
# print(des_vec)
# readme_vec = Flatten()(readme_embedded)
# print(readme_vec)
# files_vec = Flatten()(files_embedded)
# print(files_vec)

name_vec = LSTM(128)(name_embedded)
print(name_vec)
topics_vec = LSTM(128)(topics_embedded)
print(topics_vec)
des_vec = LSTM(128)(des_embedded)
print(des_vec)
readme_vec = LSTM(128)(readme_embedded)
print(readme_vec)
files_vec = LSTM(128)(files_embedded)
print(files_vec)

concatenated = concatenate([name_vec, topics_vec, des_vec, readme_vec, files_vec], axis=-1)
# concatenated = concatenate([topics_vec, des_vec, readme_vec, files_vec], axis=-1)
# print(concatenated)
# wait = input("wait")

# input_len = 8 * topics_max_len + 16 * des_max_len + 16 * readme_max_len + 16 * files_max_len
out1 = Dense(64, activation='relu')(concatenated)
out1 = Dropout(0.5)(out1)
out2 = Dense(32, activation='relu')(out1)
out3 = Dense(16, activation='relu')(out2)
final_out = Dense(1, activation='sigmoid')(out3)

model = Model([name_input, topics_input, des_input, readme_input, files_input], final_out)
# model = Model([topics_input, des_input, readme_input, files_input], final_out)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit([name_train, topics_train, des_train, readme_train, files_train], label_train, epochs=40,
                    batch_size=32, validation_split=0.2, shuffle=True)
# history = model.fit([topics_train, des_train, readme_train, files_train], label_train, epochs=10,
#                     batch_size=32, validation_split=0.2)

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

# name_model = Sequential()
# name_model.add(Embedding(name_max_features, 32, input_length=name_max_len))
# name_model.add(Flatten())
#
# topics_model = Sequential()
# topics_model.add(Embedding(1000, 128, input_length=topics_max_len))
# topics_model.add(Flatten())
# # topics_model.add(Dense(128, activation='relu'))
# topics_model.add(Dense(1, activation='sigmoid'))
# topics_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# topics_model.summary()
#
# history = topics_model.fit(topics_train, label_train, epochs=10, batch_size=32, validation_split=0.2)

# model = Sequential()
# model.add(Embedding(1000, 128, input_length=topics_max_len))
# model.add(Flatten())
# model.add(LSTM(256))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()
# history = model.fit(topics_train, label_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_split=0.2)


#
# history = topics_model.fit(topics_train, label_train, epochs=10, batch_size=32, validation_split=0.2)
#
# des_model = Sequential()
# des_model.add(Embedding(des_max_features, 256, input_length=des_max_len))
# des_model.add(Flatten())
#
# readme_model = Sequential()
# readme_model.add(Embedding(readme_max_features, 512, input_length=readme_max_len))
# readme_model.add(Flatten())
#
# files_model = Sequential()
# files_model.add(Embedding(files_max_features, 512, input_length=files_max_len))
# files_model.add(Flatten())
