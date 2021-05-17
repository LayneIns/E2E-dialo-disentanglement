import os
import sys
import numpy as np 
import torch 
import random 

from utils import build_batch
import constant


class TrainDataLoader(object):
    def __init__(self, all_utterances, labels, word_dict, name='train', add_noise=False, batch_size=constant.batch_size):
        self.all_utterances_batch = [all_utterances[i:i+batch_size] \
                                    for i in range(0, len(all_utterances), batch_size)]
        self.labels_batch = [labels[i:i+batch_size] \
                            for i in range(0, len(labels), batch_size)]
        self.word_dict = word_dict
        self.add_noise = add_noise
        assert len(self.all_utterances_batch) == len(self.labels_batch)
        self.batch_num = len(self.all_utterances_batch)
        print("{} batches created in {} set.".format(self.batch_num, name))

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= self.batch_num:
            raise IndexError

        utterances = self.all_utterances_batch[key]
        labels = self.labels_batch[key]
        new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, session_transpose_matrix, \
                state_transition_matrix, session_sequence_length, max_conversation_length, loss_mask \
                            = build_batch(utterances, labels, self.word_dict, add_noise=self.add_noise)
        if self.add_noise:
            _, label_for_loss, _, _, _, _, _, _, _ = build_batch(utterances, labels, self.word_dict)
        batch_size, max_length_1, max_length_2 = new_utterance_num_numpy.shape
        new_utterance_num_numpy = self.convert_to_tensors_1(new_utterance_num_numpy, batch_size, \
                                                            max_length_1, max_length_2)
        batch_size, max_length_1 = loss_mask.shape
        loss_mask = self.convert_to_tensors_2(loss_mask, batch_size, max_length_1)
        batch_size, max_length_1 = new_utterance_sequence_length.shape
        new_utterance_sequence_length = self.convert_to_tensors_2(new_utterance_sequence_length, batch_size, max_length_1)
        return new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, \
                    session_transpose_matrix, state_transition_matrix, session_sequence_length, \
                        max_conversation_length, loss_mask

    def convert_to_tensors_1(self, utterances, batch_size, max_length, h_size):
        # batch_size, max_conversation_length, max_utterance_length
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        for i in range(len(utterances)):
            for j in range(len(utterances[i])):
                new_batch[i, j] = torch.LongTensor(utterances[i][j])
        return new_batch

    def convert_to_tensors_2(self, batch, batch_size, max_length):
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        for i in range(len(batch)):
            new_batch[i] = torch.LongTensor(batch[i])
        return new_batch