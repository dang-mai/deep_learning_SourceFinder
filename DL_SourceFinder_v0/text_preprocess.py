import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing


import read_data

random.seed(0)
np.random.seed(0)

repos_name_train, repos_name_test, repos_files_train, repos_files_test, repos_topics_train, \
repos_topics_test, repos_des_train, repos_des_test, repos_readme_train, repos_readme_test, \
repos_label_train, repos_label_test = read_data.load_data()

name_tokenizer = Tokenizer(num_words=50)
name_tokenizer.fit_on_texts(repos_name_train)
name_sequences_train = name_tokenizer.texts_to_sequences(repos_name_train)
name_sequences_test = name_tokenizer.texts_to_sequences(repos_name_test)
# print(name_sequences_test)


files_tokenizer = Tokenizer(num_words=1000)
files_tokenizer.fit_on_texts(repos_files_train)
files_sequences_train = files_tokenizer.texts_to_sequences(repos_files_train)
files_sequences_test = files_tokenizer.texts_to_sequences(repos_files_test)
# word_index = files_tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

topics_tokenizer = Tokenizer(num_words=50)
topics_tokenizer.fit_on_texts(repos_topics_train)
topics_sequences_train = topics_tokenizer.texts_to_sequences(repos_topics_train)
topics_sequences_test = topics_tokenizer.texts_to_sequences(repos_topics_test)
# word_index = topics_tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

des_tokenizer = Tokenizer(num_words=500)
des_tokenizer.fit_on_texts(repos_des_train)
des_sequences_train = des_tokenizer.texts_to_sequences(repos_des_train)
des_sequences_test = des_tokenizer.texts_to_sequences(repos_des_test)
# word_index = des_tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

readme_tokenizer = Tokenizer(num_words=100)
readme_tokenizer.fit_on_texts(repos_readme_train)
readme_sequences_train = readme_tokenizer.texts_to_sequences(repos_readme_train)
readme_sequences_test = readme_tokenizer.texts_to_sequences(repos_readme_test)
# word_index = readme_tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

name_max_features = len(name_tokenizer.word_index)
topics_max_features = len(topics_tokenizer.word_index)
files_max_features = len(files_tokenizer.word_index)
des_max_features = len(des_tokenizer.word_index)
readme_max_features = len(readme_tokenizer.word_index)

name_max_len = max([len(s) for s in name_sequences_train])
topics_max_len = max([len(s) for s in topics_sequences_train])
files_max_len = max([len(s) for s in files_sequences_train])
des_max_len = max([len(s) for s in des_sequences_train])
readme_max_len = max([len(s) for s in readme_sequences_train])

# name_max_len = 16
# topics_max_len = 16
# files_max_len = 32
# des_max_len = 32
# readme_max_len = 32

# print(name_max_features)
# print(topics_max_features)
# print(files_max_features)
# print(des_max_features)
# print(readme_max_features)
# print()
# print(name_max_len)
# print(topics_max_len)
# print(files_max_len)
# print(des_max_len)
# print(readme_max_len)

name_train = preprocessing.sequence.pad_sequences(name_sequences_train, maxlen=name_max_len)
name_test = preprocessing.sequence.pad_sequences(name_sequences_test, maxlen=name_max_len)
topics_train = preprocessing.sequence.pad_sequences(topics_sequences_train, maxlen=topics_max_len)
topics_test = preprocessing.sequence.pad_sequences(topics_sequences_test, maxlen=topics_max_len)
files_train = preprocessing.sequence.pad_sequences(files_sequences_train, maxlen=files_max_len)
files_test = preprocessing.sequence.pad_sequences(files_sequences_test, maxlen=files_max_len)
des_train = preprocessing.sequence.pad_sequences(des_sequences_train, maxlen=des_max_len)
des_test = preprocessing.sequence.pad_sequences(des_sequences_test, maxlen=des_max_len)
readme_train = preprocessing.sequence.pad_sequences(readme_sequences_train, maxlen=readme_max_len)
readme_test = preprocessing.sequence.pad_sequences(readme_sequences_test, maxlen=readme_max_len)

label_train = repos_label_train
label_test = repos_label_test

name_train = np.asarray(name_train)
topics_train = np.asarray(topics_train)
des_train = np.asarray(des_train)
readme_train = np.asarray(readme_train)
files_train = np.asarray(files_train)
label_train = np.asarray(label_train)
# label_train = np.asarray(label_train).reshape(len(label_train), 1)

name_test = np.asarray(name_test)
topics_test = np.asarray(topics_test)
des_test = np.asarray(des_test)
readme_test = np.asarray(readme_test)
files_test = np.asarray(files_test)
label_test = np.asarray(label_test)
# label_test = np.asarray(label_test).reshape(len(label_test), 1)

# print(label_train)


def return_result():
    return name_train, name_test, topics_train, topics_test, files_train, files_test, \
           des_train, des_test, readme_train, readme_test, label_train, label_test


def return_info():
    return name_max_len, topics_max_len, files_max_len, des_max_len, readme_max_len, \
           name_max_features, topics_max_features, files_max_features, des_max_features, readme_max_features


def return_test():
    return topics_train, label_train


# print(type(name_train))
# print(name_train.shape)
