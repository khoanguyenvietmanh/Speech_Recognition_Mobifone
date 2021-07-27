from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import nltk

nltk.download('reuters')
from nltk.corpus import reuters
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'
np.random.seed(0)
random.seed(0)

def read_corpus(category="crude"):
    files = reuters.fileids(category)

    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

def distinct_words(corpus):
    corpus_words = []
    num_corpus_words = -1

    corpus_words = sorted(list(set([word for sentence in corpus for word in sentence])))
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words

def compute_co_occurence_matrix(corpus, window_size=4):

    corpus_words, num_corpus_words = distinct_words(corpus)

    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = dict([(word, index) for index, word in enumerate(words)])

    for sentence in corpus:
        current_indice = 0
        sentence_len = len(sentence)
        indices = [word2Ind[i] for i in sentence]

        while current_indice < sentence_len:
            left_margin = max(current_indice - window_size, 0)
            right_margin = min(current_indice + window_size, sentence_len)
            current_word = sentence[current_indice]
            current_word_index = word2Ind[current_word]

            result = indices[left_margin: current_indice] + indices[current_indice + 1: right_margin]

            for i in result:
                M[current_word_index, i] += 1

            current_indice += 1

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    n_iters = 10
    M_reduced = None

    TSVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = TSVD.fit_transform(M)

    return M_reduced

def plot_embeddings(M_reduced, word2Ind, words):
    for word in words:
        indice = word2Ind[word]
        embedding = M_reduced[indice]
        x, y = embedding[0], embedding[1]
        plt.scatter(x, y, marker='x', c='r')
        plt.text(x, y, word, fontsize=9)
    plt.show()


# reuters_corpus = read_corpus()
#
# M, word2Ind = compute_co_occurence_matrix(reuters_corpus)
#
# M_reduced = reduce_to_k_dim(M)
#
# M_length = np.linalg.norm(M_reduced, axis=1)
#
# M_reduced = M_reduced / M_length[:, np.newaxis]
#
# words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
#
# plot_embeddings(M_reduced, word2Ind, words)

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

wv_from_bin = load_word2vec()
M, word2Ind = get_matrix_of_vectors(wv_from_bin)
M_reduced = reduce_to_k_dim(M, k=2)
