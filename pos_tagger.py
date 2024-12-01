# CS114B Spring 2023
# Part-of-Speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from the lab 7 exercise
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        self.initial = np.array([-0.3, -0.7, 0.3])
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        # should raise an IndexError; if you come across an unknown word, you
        # should treat the emission scores for that word as 0
        self.unk_index = np.inf

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        tag_ind = 0
        word_ind = 0
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        split_line = line.split()
                        for word_tag in split_line:
                            dash_ind = word_tag.rindex("/")
                            word = word_tag[:dash_ind]
                            tag = word_tag[dash_ind+1:]
                            if word_tag != '<S>' and word_tag != '</S>' and word_tag != '<UNK>':
                                if tag not in self.tag_dict:
                                    self.tag_dict[tag] = tag_ind
                                    tag_ind += 1
                                if word not in self.word_dict:
                                    self.word_dict[word] = word_ind
                                    word_ind += 1


    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids = []
        tag_lists = dict()
        word_lists = dict()
        id_count = 0
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    for line in f:
                        line = line.strip()
                        if line != '':
                            sentence_tags = []
                            sentence_words = []
                            split_line = line.split()
                            for word_tag in split_line:
                                dash_ind = word_tag.rindex("/")
                                word = word_tag[:dash_ind]
                                tag = word_tag[dash_ind+1:]
                                if word_tag != '<S>' and word_tag != '</S>' and word_tag != '<UNK>':
                                    word = self.word_dict.get(word, self.unk_index)
                                    tag = self.tag_dict.get(tag, self.unk_index)
                                    sentence_tags.append(tag)
                                    sentence_words.append(word)
                            if len(sentence_tags) > 0 and len(sentence_words) > 0:
                                sentence_ids.append(id_count)
                                tag_lists[id_count] = sentence_tags
                                word_lists[id_count] = sentence_words
                                id_count += 1
        return sentence_ids, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=np.int32)
        # initialization step
        if sentence[0] != self.unk_index:
            v[:, 0] = self.initial + self.emission[sentence[0], :]
        else:
            v[:, 0] = self.initial
        backpointer[:, 0] = 0
        # recursion step
        for t in range(1, T):
            if sentence[t] != self.unk_index:
                prob = v[:, t - 1][:, np.newaxis] + self.transition + self.emission[sentence[t], :]
            else:
                prob = v[:, t - 1][:, np.newaxis] + self.transition + 0
            v[:, t] = np.max(prob, axis=0)
            backpointer[:, t] = np.argmax(prob, axis=0)
        # termination step
        best_tag = np.argmax(v[:, T-1])
        # tag assignment
        predicted_tags = [best_tag]
        for t in range(T-1, 0, -1):
            best_tag = backpointer[best_tag, t]
            predicted_tags.insert(0, best_tag)
        return predicted_tags

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set):
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        for i, sentence_id in enumerate(sentence_ids):
            # your code here
            tags = tag_lists[sentence_id]
            sentence = word_lists[sentence_id]
            predicted_tags = self.viterbi(sentence)
            # change initial values
            if predicted_tags[0] != tags[0]:
                self.initial[tags[0]] += 1
                self.initial[predicted_tags[0]] -= 1
                self.emission[sentence[0], tags[0]] += 1
                self.emission[sentence[0], predicted_tags[0]] -= 1
            for ind in range(1, len(tags)):
                prev_act = tags[ind-1]
                prev_pred = predicted_tags[ind-1]
                actual = tags[ind]
                pred = predicted_tags[ind]
                if pred != actual:
                    self.emission[sentence[ind], actual] += 1
                    self.emission[sentence[ind], pred] -= 1
                if ((prev_act != prev_pred) or (actual != pred)):
                    self.transition[prev_act, actual] += 1
                    self.transition[prev_pred, pred] -= 1
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)
        for i, sentence_id in enumerate(sentence_ids):
            # your code here
            tags = tag_lists[sentence_id]
            sentence = word_lists[sentence_id]
            predicted = self.viterbi(sentence)
            results[sentence_id]['correct'] = tags
            results[sentence_id]['predicted'] = predicted
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        accuracy = 0.0
        corr = 0
        total = 0
        for sen_id in results:
            correct = results[sen_id]['correct']
            predicted = results[sen_id]['predicted']
            for ind in range(len(correct)):
                if correct[ind] == predicted[ind]:
                    corr += 1
                total += 1
        accuracy = corr/total
        return accuracy

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
    # pos.train('data_small/train')
    results = pos.test('brown/dev')
    # results = pos.test('data_small/test')
    print('Accuracy:', pos.evaluate(results))
