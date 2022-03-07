import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
import re
import pickle as pkl
import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from utils import power_iteration, track_trans
from mufasa_model import mufasa, mufasa_classifier, CNN_1d


def process_text(input_text, pretrained_model):
    review = re.sub("[^a-zA-Z]", " ", input_text)
    review = review.lower()
    review = review.split()
    review = [WordNetLemmatizer().lemmatize(word) for word in review if word not in stopwords.words("english")]

    # - Get valid tokens - #
    num_nodes = 0
    id_2_word = dict()
    word_2_id = dict()
    id_2_vec = dict()
    appeared_nodes = set()
    for word in review:
        if word not in word_2_id.keys() and word not in appeared_nodes:  # - find a new word - #
            try:
                vec = pretrained_model[word]  # - 300 dimension - #
                word_2_id[word] = num_nodes
                id_2_word[num_nodes] = word
                id_2_vec[num_nodes] = vec
                num_nodes += 1
                appeared_nodes.add(word)
            except:
                # print('============> ' + word + ' could not be found in the Google pre-trained model.')
                appeared_nodes.add(word)

    # print("============> number of words: " + str(num_nodes))

    # - Construct graph adjacency matrix - #
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for j in range(len(review) - 2):  # - size window is 3 - #
        word_x = review[j]
        word_y = review[j + 1]
        word_z = review[j + 2]
        if word_x in word_2_id.keys() and word_y in word_2_id.keys():
            adj_matrix[word_2_id[word_x]][word_2_id[word_y]] = 1
            adj_matrix[word_2_id[word_y]][word_2_id[word_x]] = 1
        if word_x in word_2_id.keys() and word_z in word_2_id.keys():
            adj_matrix[word_2_id[word_x]][word_2_id[word_z]] = 1
            adj_matrix[word_2_id[word_z]][word_2_id[word_x]] = 1
        if word_y in word_2_id.keys() and word_z in word_2_id.keys():
            adj_matrix[word_2_id[word_y]][word_2_id[word_z]] = 1
            adj_matrix[word_2_id[word_z]][word_2_id[word_y]] = 1

    # - Get graph embedding - #
    h_matrix = np.zeros((num_nodes, 300))
    p_matrix = np.zeros((num_nodes, num_nodes))
    for j in range(num_nodes):
        ppr = np.zeros((num_nodes,))
        ppr[j] = 1
        ppr = power_iteration(ppr, adj_matrix)
        p_matrix[j:] = ppr
        h_matrix[j:] = id_2_vec[j]
    z_matrix = np.dot(p_matrix, h_matrix)
    z_vec = np.sum(z_matrix, axis=0)  # pooling: sum to row

    return num_nodes, id_2_word, word_2_id, adj_matrix, h_matrix, p_matrix, z_vec


def prediction(z_vec, clf):
    prob = clf.predict_proba([z_vec])
    fake_prob = prob[0][0]
    real_prob = prob[0][1]

    label = clf.predict([z_vec])

    return label, prob


def misleading_top_n_words(label, prob, num_nodes, adj_matrix, p_matrix, h_matrix, id_2_word):
    word_2_misleading_degree = dict()

    for rm_idx in range(num_nodes):
        new_adj_matrix = adj_matrix.copy()
        new_adj_matrix[:, rm_idx] = 0
        new_adj_matrix[rm_idx, :] = 0
        new_p_matrix = track_trans(adj_matrix, new_adj_matrix, p_matrix)
        z_matrix = np.dot(new_p_matrix, h_matrix)
        z_vec = np.sum(z_matrix, axis=0)
        new_prob = clf.predict_proba([z_vec])

        misleading_degree = 0
        if label == 0:
            misleading_degree = new_prob[0][0] - prob[0][0]
        else:
            misleading_degree = new_prob[0][1] - prob[0][1]

        rm_word = id_2_word[rm_idx]
        word_2_misleading_degree[rm_word] = misleading_degree

    ranking = sorted(word_2_misleading_degree.items(), key=lambda item: item[1], reverse=True)

    return ranking


if __name__ == '__main__':
    pretrained_model = KeyedVectors.load_word2vec_format('fake-and-real-news-dataset/GoogleNews-vectors-negative300.bin', binary=True)

    # ------ Option 1: MuFasa ------ #
    # mf = load('./mufasa.joblib')

    # ------ Option 2: MLP ------ #
    clf = load('saved_mlp.joblib')

    fake_data = pd.read_csv("fake-and-real-news-dataset/Fake.csv")
    true_data = pd.read_csv("fake-and-real-news-dataset/True.csv")
    data = pd.concat([fake_data, true_data], axis=0, ignore_index = True)
    idx = 15173
    input_text = data["text"][idx]

    num_nodes, id_2_word, word_2_id, adj_matrix, h_matrix, p_matrix, z_vec = process_text(input_text, pretrained_model)
    label, prob = prediction(z_vec, clf)
    ranking = misleading_top_n_words(label, prob, num_nodes, adj_matrix, p_matrix, h_matrix, id_2_word)

    print("The decreasing misleading degree order is: ")
    for item in ranking:
        print("word: " + str(item[0]) + "\t" + "misleading degree: " + str(item[1]))