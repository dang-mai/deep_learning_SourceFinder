import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

import read_data

random.seed(0)
np.random.seed(0)
repos_name_train, repos_name_test, repos_files_train, repos_files_test, repos_topics_train, \
repos_topics_test, repos_des_train, repos_des_test, repos_readme_train, repos_readme_test, \
repos_label_train, repos_label_test = read_data.load_data()

name_sequences_train = repos_name_train
name_sequences_test = repos_name_test
# print(name_sequences_test)


files_tokenizer = Tokenizer(num_words=1000)
files_tokenizer.fit_on_texts(repos_files_train)
files_sequences_train = files_tokenizer.texts_to_sequences(repos_files_train)
files_sequences_test = files_tokenizer.texts_to_sequences(repos_files_test)

print(files_sequences_train)
