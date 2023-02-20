import random
import numpy as np
import jsonlines
from sklearn.model_selection import train_test_split

random.seed(0)
np.random.seed(0)


def load_data():
    items = []
    with jsonlines.open("repos_labeled.jsonl") as reader:
        for item in reader:
            # if item["label"] == "0" or item["label"] == "":
            #     continue
            # item["label"] = int(item["label"]) - 1
            temp = []
            for tmp in item["files"]:
                temp += tmp
            item["files"] = temp
            temp = []
            for tmp in item["topics"]:
                temp += tmp
            item["topics"] = temp
            temp = []
            for tmp in item["description"]:
                temp += tmp
            item["description"] = temp
            temp = []
            for tmp in item["readme"]:
                temp += tmp
            item["readme"] = temp

            item["name"] = ' '.join(item["name"])
            item["files"] = ' '.join(item["files"])
            item["topics"] = ' '.join(item["topics"])
            item["description"] = ' '.join(item["description"])
            item["readme"] = ' '.join(item["readme"])

            items.append(item)

    random.shuffle(items)
    random.shuffle(items)
    random.shuffle(items)

    repos_name = []
    repos_files = []
    repos_topics = []
    repos_des = []
    repos_readme = []
    repos_label = []
    for repo in items:
        repos_name.append(repo["name"])
        repos_topics.append(repo["topics"])
        repos_files.append(repo["files"])
        repos_des.append(repo["description"])
        repos_readme.append(repo["readme"])
        repos_label.append(repo["binary_label"])
    repos_name_train, repos_name_test = train_test_split(repos_name, test_size=0.25, random_state=0)
    repos_files_train, repos_files_test = train_test_split(repos_files, test_size=0.25, random_state=0)
    repos_topics_train, repos_topics_test = train_test_split(repos_topics, test_size=0.25, random_state=0)
    repos_des_train, repos_des_test = train_test_split(repos_des, test_size=0.25, random_state=0)
    repos_readme_train, repos_readme_test = train_test_split(repos_readme, test_size=0.25, random_state=0)
    repos_label_train, repos_label_test = train_test_split(repos_label, test_size=0.25, random_state=0)

    return repos_name_train, repos_name_test, repos_files_train, repos_files_test, repos_topics_train, \
           repos_topics_test, repos_des_train, repos_des_test, repos_readme_train, repos_readme_test, \
           repos_label_train, repos_label_test
