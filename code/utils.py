import torch
import json
import os
import sys
import numpy as np 
import pickle
from collections import Counter
import random
import copy
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

import constant


def make_all_dirs(current_time):
    if not os.path.exists(constant.log_path):
        os.makedirs(constant.log_path)
    if not os.path.exists(constant.save_input_path):
        os.makedirs(constant.save_input_path)
    if not os.path.exists(os.path.join(constant.output_path, 'result_'+current_time)):
        os.makedirs(os.path.join(constant.output_path, 'result_'+current_time))
    if not os.path.exists(constant.output_path):
        os.makedirs(constant.output_path)
    if not os.path.exists(os.path.join(constant.save_model_path, 'model_'+current_time)):
        os.makedirs(os.path.join(constant.save_model_path, 'model_'+current_time))


def save_predicted_results(predicted_labels, truth_labels, current_time, test_step, mode='dev'):
    filepath = os.path.join(constant.output_path, 'result_'+current_time)
    predicted_filename = os.path.join(filepath, 'predicted_step_{}.pkl'.format(test_step) if mode == 'dev' else 'test_predicted_step_{}.pkl'.format(test_step))
    truth_filename = os.path.join(filepath, 'truth.pkl' if mode == 'dev' else 'truth_test.pkl')
    if not os.path.exists(truth_filename):
        with open(truth_filename, 'wb') as fout:
            pickle.dump(truth_labels, fout)
    if not os.path.exists(predicted_filename):
        with open(predicted_filename, 'wb') as fout:
            pickle.dump(predicted_labels, fout)


def save_or_read_input(path, rw='r', input_obj=None):
    if rw == 'w':
        with open(path, 'wb') as fout:
            pickle.dump(input_obj, fout)
        print("{} saved successfully!".format(path))
    elif rw == 'r':
        with open(path, 'rb') as fin:
            ret_val = pickle.load(fin)
        print("{} read successfully!".format(path))
        return ret_val


def build_embedding_matrix(word_dict, glove_loc=None, emb_loc=None, load_emb=False):
    print("Building word embedding matrix...")
    if load_emb:
        with open(emb_loc, 'rb') as fin:
            word_emb = pickle.load(fin)
    else:
        word_emb = np.random.uniform(-1, 1, (len(word_dict), constant.embedding_size))
        tokens = word_dict.keys()
        line_cnt = 0
        oov_cnt = 0
        with open(glove_loc) as fin:
            for line in fin:
                line_cnt += 1
                if line_cnt % 500000 == 0:
                    percent = line_cnt/2196017
                    percent = "%.2f%%" % (percent*100)
                    print("{} of glove read.".format(percent))
                splitted_line = line.strip().split(' ')
                word = splitted_line[0]
                assert len(splitted_line[-constant.embedding_size:]) == constant.embedding_size
                if word in tokens:
                    word_emb[word_dict[word]] = [float(v) for v in splitted_line[-constant.embedding_size:]]
                else:
                    oov_cnt += 1
        print("{} out of {} words are OOVs.".format(oov_cnt, len(word_dict)))
    print("Building word embeding matrix over.")
    return word_emb


def convert_utterances(utterances, word_dict):
    utterances_num = []
    utterance_sequence_length = [] # Sequence length of the batch
    for i in range(len(utterances)):
        one_instance = []
        one_uttr_sequence_length = []
        for one_uttr in utterances[i]:
            one_uttr_num = []
            one_uttr_sequence_length.append(len(one_uttr))
            for word in one_uttr:
                one_uttr_num.append(word_dict.get(word, constant.UNK_ID))
            one_instance.append(one_uttr_num)
        utterances_num.append(one_instance)
        utterance_sequence_length.append(one_uttr_sequence_length)
    return utterances_num, utterance_sequence_length


def padding_batch(utterances_num, labels, max_session_length, max_utterance_length, utterance_sequence_length):
    new_utterance_num = copy.deepcopy(utterances_num)
    new_labels = copy.deepcopy(labels)
    new_utterance_sequence_length = copy.deepcopy(utterance_sequence_length)
    # conversation_sequence_length = [len(x) for x in utterances_num]
    
    for i in range(len(new_utterance_num)):
        for j in range(len(new_utterance_num[i])):
            token_padding_num = max_utterance_length-len(new_utterance_num[i][j])
            for _ in range(token_padding_num):
                new_utterance_num[i][j].append(constant.PAD_ID)
        
        session_length_count = dict(Counter(labels[i]))
        # matrix_padding_part = []
        for j in range(constant.state_num-1):
            padding_num_j = max_session_length - session_length_count.get(j, 0)
            for _ in range(padding_num_j):
                empty_mat = [constant.PAD_ID] * max_utterance_length
                new_utterance_num[i].append(empty_mat)
                new_labels[i].append(j)
                new_utterance_sequence_length[i].append(1)

    new_utterance_num_numpy = np.asarray(new_utterance_num)
    new_utterance_sequence_length = np.asarray(new_utterance_sequence_length)
    # conversation_sequence_length = np.asarray(conversation_sequence_length)
    new_labels = np.asarray(new_labels)
    return new_utterance_num_numpy, new_labels, new_utterance_sequence_length


def build_state_transition_matrix(labels, max_conversation_length):
    batch_size = len(labels)
    matrix = []
    for i in range(batch_size):
        one_instance_label = labels[i]
        state = (np.zeros([max_conversation_length, constant.state_num], dtype=np.int)-1).tolist()
        label_dict = {i:0 for i in range(constant.state_num-1)}
        for j in range(len(one_instance_label)):
            state[j][0] = 0
            for k in range(constant.state_num-1):
                if label_dict[k] != 0:
                    state[j][k+1] = label_dict[k]
            label_dict[one_instance_label[j]] += 1
        matrix.append(state)
    matrix = np.asarray(matrix)
    return matrix


def get_session_sequence_length(labels):
    session_sequence_length = []
    for item in labels:
        one_length = [0] * (constant.state_num - 1)
        for one_label in item:
            one_length[one_label] += 1
        session_sequence_length.append(one_length)
    session_sequence_length = np.asarray(session_sequence_length)
    return session_sequence_length


def reorder_session(labels):
    # labels： [batch_size, conversation_max_length]
    reverse_dict = dict()
    cnt = 0
    batch_size, max_conversation_length = labels.shape
    for batch_index in range(batch_size):
        for j in range(constant.state_num):
            for i in range(len(labels[batch_index])):
                if labels[batch_index][i] == j:
                    reverse_dict[cnt] = batch_index*max_conversation_length+i
                    cnt += 1
    
    transpose_matrix = []
    for i in range(len(reverse_dict)):
        transpose_matrix.append(reverse_dict[i])
    assert len(transpose_matrix) == batch_size * max_conversation_length
    return transpose_matrix


def get_loss_labels(new_labels):
    labels = []
    for i in range(len(new_labels)):
        one_label = []
        for j in range(len(new_labels[i])):
            if j == 0:
                one_label.append(0)
            elif j != 0 and new_labels[i][j] != new_labels[i][j-1]:
                if new_labels[i][j] not in new_labels[i][:j]:
                    one_label.append(0)
                else:
                    one_label.append(new_labels[i][j] + 1)
            elif j != 0 and new_labels[i][j] == new_labels[i][j-1]:
                one_label.append(new_labels[i][j] + 1)
        labels.append(one_label)
    labels = np.asarray(labels)
    return labels


def add_noise_to_data(labels):
    noise_labels = []
    for batch_index in range(len(labels)):
        if random.random() < constant.total_noise_ratio:
            one_label = []
            index = [i for i in range(len(labels[batch_index]))]
            random.shuffle(index)
            index = index[:int(len(labels[batch_index])*constant.noise_ratio)]
            for j in range(len(labels[batch_index])):
                if j not in index:
                    one_label.append(labels[batch_index][j])
                else:
                    candidate_set = list(set([i for i in range(1,constant.state_num-1)]) -  set([labels[batch_index][j]]))
                    one_label.append(random.choice(candidate_set))
            noise_labels.append(one_label)
        else:
            noise_labels.append(labels[batch_index])
    return noise_labels


def build_batch(utterances, labels, word_dict, add_noise=False):
    max_conversation_length = max([len(x) for x in utterances]) # how many utterances in a session
    max_utterance_length = max([len(x) for one_uttr in utterances for x in one_uttr]) # max utterance length (how many words in a utterance)
    
    if add_noise:
        noise_labels = add_noise_to_data(labels)
        max_session_length = max([max(Counter(l).values()) for l in noise_labels])
    else:
        max_session_length = max([max(Counter(l).values()) for l in labels])

    conversation_lengths = [len(x) for x in utterances]
    loss_mask = torch.arange(max_conversation_length).expand(len(conversation_lengths), max_conversation_length) \
                                < torch.Tensor(conversation_lengths).unsqueeze(1)
    loss_mask = loss_mask.type(torch.int64)

    utterances_num, utterance_sequence_length = convert_utterances(utterances, word_dict)
    
    # session length, shape: [batch_size, 4]
    if add_noise:
        session_sequence_length = get_session_sequence_length(noise_labels) 
    else:
        session_sequence_length = get_session_sequence_length(labels) 
    
    # The state for a training case. Showing which of lstm state should be copied to the state matrix
    # [batch_size, max_conversation_length, 5]
    if add_noise:
        state_transition_matrix = build_state_transition_matrix(noise_labels, max_conversation_length)
    else:
        state_transition_matrix = build_state_transition_matrix(labels, max_conversation_length)

    # new_utterance_num_numpy: a numpy ndarray for the utterance number
    # new_labels: the label list after padding
    # new_utterance_sequence_length: utterance sequence length after padding 
    if add_noise:
        new_utterance_num_numpy, noise_new_labels, new_utterance_sequence_length = \
                        padding_batch(utterances_num, noise_labels, max_session_length, \
                                                    max_utterance_length, utterance_sequence_length)
        _, new_labels, _ = padding_batch(utterances_num, labels, max_session_length, max_utterance_length, utterance_sequence_length)
    else:
        new_utterance_num_numpy, new_labels, new_utterance_sequence_length = \
                        padding_batch(utterances_num, labels, max_session_length, \
                                                    max_utterance_length, utterance_sequence_length)

    if add_noise:   
        session_transpose_matrix = reorder_session(noise_new_labels)
    else:
        session_transpose_matrix = reorder_session(new_labels)

    label_for_loss = get_loss_labels(new_labels)

    if add_noise:
        return new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, \
                session_transpose_matrix, state_transition_matrix, session_sequence_length, \
                    max_conversation_length, loss_mask
    else:
        return new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, \
                session_transpose_matrix, state_transition_matrix, session_sequence_length, \
                    max_conversation_length, loss_mask
    

def calculate_purity_scores(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def calculate_shen_f_score(y_true, y_pred):
    def get_f_score(i, j, n_i_j, n_i, n_j):
        recall = n_i_j / n_i
        precision = n_i_j / n_j
        if recall == 0 and precision == 0:
            f_score = 0.
        else:
            f_score = 2 * recall * precision / (recall + precision)
        return f_score
    
    y_true_cnt = dict(Counter(y_true))
    y_pred_cnt = dict(Counter(y_pred))
    y_pred_dict = dict()
    for i, val in enumerate(y_pred):
        if y_pred_dict.get(val, None) == None:
            y_pred_dict[val] = dict()
        if y_pred_dict[val].get(y_true[i], None) == None:
            y_pred_dict[val][y_true[i]] = 0
        y_pred_dict[val][y_true[i]] += 1
    shen_f_score = 0.
    for i, val_i in y_true_cnt.items():
        f_list = []
        for j, val_j in y_pred_cnt.items():
            f_list.append(get_f_score(i, j, y_pred_dict[j].get(i, 0), val_i, val_j))
        shen_f_score += max(f_list) * y_true_cnt[i] / len(y_true)
    return shen_f_score


def compare(predicted_labels, truth_labels, metric):
    if metric == 'purity':
        purity_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            purity_scores.append(calculate_purity_scores(y_true, y_pred))
        return sum(purity_scores)/len(purity_scores)
    elif metric == 'NMI':
        NMI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            NMI_scores.append(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
        return sum(NMI_scores)/len(NMI_scores)
    elif metric == 'ARI':
        ARI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            ARI_scores.append(metrics.adjusted_rand_score(y_true, y_pred))
        return sum(ARI_scores)/len(ARI_scores)
    elif metric == "shen_f":
        f_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            f_scores.append(calculate_shen_f_score(y_true, y_pred))
        return sum(f_scores)/len(f_scores)


