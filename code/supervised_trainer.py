import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

import sys
import os
import numpy as np 
import logging 
from tqdm import tqdm

import constant
import utils


class SupervisedTrainer(object):
    def __init__(self, ensemble_model, logger=None, current_time=None, loss=F.nll_loss, optimizer=None):
        self.ensemble_model = ensemble_model
        self.logger = logger
        self.current_time = current_time
        self.loss_func = loss
        if optimizer == None:
            params = list(self.ensemble_model.parameters())
            self.optimizer = optim.Adam(params, lr=constant.learning_rate)
        else:
            self.optimizer = optimizer
    
    def calculate_loss(self, input_, target, loss_mask):
        target = target[:, :input_.size(1)]
        if torch.cuda.is_available():
            target = torch.cuda.LongTensor(target)
        else:
            target = torch.LongTensor(target)
        loss = -input_.gather(2, target.unsqueeze(2)).squeeze(2)*loss_mask
        loss = torch.sum(loss)/len(input_)
        return loss

    def _train_batch(self, batch):
        batch_utterances, label_for_loss, labels, utterance_sequence_length, \
                    session_transpose_matrix, state_transition_matrix, session_sequence_length, \
                        max_conversation_length, loss_mask = batch
        softmax_masked_scores = self.ensemble_model(batch)
        # [batch_size, max_conversation_length, 5]

        loss = self.calculate_loss(softmax_masked_scores, label_for_loss, loss_mask)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.item(), len(batch_utterances), sum(sum(session_sequence_length))

    def train(self, train_loader, dev_loader):
        step_cnt = 0
        for epoch in range(constant.epoch_num):
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                step_cnt += 1
                loss, batch_size, uttr_cnt = self._train_batch(batch)
                epoch_loss += loss
                log_msg = "Epoch : {}, batch: {}/{}, step: {}, batch avg loss: {}, uttr avg loss: {}".format(
                                        epoch, i, len(train_loader), step_cnt, round(loss, 4), round(loss*batch_size/uttr_cnt, 4))
                self.logger.info(log_msg)
                if step_cnt % constant.inference_step == 0:
                # if step_cnt % 2 == 0:
                    purity_score, nmi_score, ari_score, shen_f_score = self.evaluate(dev_loader, step_cnt)
                    log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4))
                    self.logger.info(log_msg)

                    model_name = os.path.join(constant.save_model_path, "model_{}".format(self.current_time), \
                                                    "step_{}.pkl".format(step_cnt))
                    log_msg = "Saving model for step {} at '{}'".format(step_cnt, model_name)
                    self.logger.info(log_msg)
                    torch.save(self.ensemble_model.state_dict(), model_name)

            log_msg = "Epoch average loss is: {}".format(round(epoch_loss/len(train_loader), 4))
            self.logger.info(log_msg)

    def init_state(self, batch_index, j, state, hidden_state_history, utterance_repre, predicted_batch_label, mask):
        # state : [5, constant.hidden_size]
        if j == 0:
            state[0] = self.ensemble_model.state_matrix_encoder.pooling(state[1:, :].unsqueeze(0))[0][0]
        else:
            label = predicted_batch_label[-1]
            if label == 0:
                position = mask.cpu().tolist().index(-1.)
                mask[position] = 0. 
                new_output, new_hidden = self.ensemble_model.session_encoder.encoder(utterance_repre[batch_index][j-1].unsqueeze(0).unsqueeze(0))
            else:
                position = label
                # state[label]: [hidden_size]
                new_output, new_hidden = self.ensemble_model.session_encoder.encoder(utterance_repre[batch_index][j-1].unsqueeze(0).unsqueeze(0), hidden_state_history[label])
                # new_output: [1, 1, hidden_size]
            state[position] = new_output.squeeze()
            hidden_state_history[position] = new_hidden
            state[0] = self.ensemble_model.state_matrix_encoder.pooling(state[1:, :].unsqueeze(0))[0][0]
        return state, mask, hidden_state_history
    
    def predict(self, batch_index, j, state, utterance_repre, conversation_repre, mask):
        # state: [5, hidden_size]
        current_uttr_repre_concat = torch.cat((utterance_repre[batch_index][j], conversation_repre[batch_index][j]), 0)
        # [hidden_size * 2]
        current_uttr_repre = self.ensemble_model.scores_calculator.utterance_projection(current_uttr_repre_concat)
        scores = torch.matmul(current_uttr_repre, state.permute(1, 0))
        masked_scores = scores + mask*10000
        softmax_masked_scores = nn.Softmax(dim=0)(masked_scores)
        label = softmax_masked_scores.topk(2)[1].cpu().numpy()
        # print(label)
        if label[0] == 0 and -1. not in mask:
            ret_label = label[1]
        else:
            ret_label = label[0]
        return ret_label

    def recover_label(self, state_labels):
        new_label = []
        current_new = 0
        for i in range(len(state_labels)):
            if state_labels[i] == 0:
                new_label.append(current_new)
                current_new += 1
            else:
                new_label.append(state_labels[i]-1)
        return new_label
    
    def recurrent_update(self, utterance_repre, conversation_repre, max_conversation_length, conversation_length_list):
        predicted_labels = []
        for batch_index in range(len(conversation_length_list)):
            predicted_batch_label = []
            shape = torch.Size([5, constant.hidden_size])
            if torch.cuda.is_available():
                state = torch.cuda.FloatTensor(shape).zero_()
            else:
                state = torch.FloatTensor(shape).zero_()
            # state = torch.randn([5, constant.hidden_size])
            if torch.cuda.is_available():
                mask = torch.cuda.LongTensor([0., -1., -1., -1., -1.]).type(torch.double)
            else:
                mask = torch.LongTensor([0., -1., -1., -1., -1.]).type(torch.double)
            hidden_state_history = {1: None, 2: None, 3:None, 4: None}
            for j in range(conversation_length_list[batch_index]):
                state, mask, hidden_state_history = self.init_state(batch_index, j, state, hidden_state_history, utterance_repre, predicted_batch_label, mask)
                label = self.predict(batch_index, j, state, utterance_repre, conversation_repre, mask)
                predicted_batch_label.append(label)
                # print("{}-{}: label: {}, {}".format(batch_index, j, label, mask.cpu().tolist()))
            new_label = self.recover_label(predicted_batch_label)
            predicted_labels.append(new_label)
        return predicted_labels

    def evaluate(self, test_loader, step_cnt):
        predicted_labels = []
        truth_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch_utterances, _, labels, utterance_sequence_length, \
                        _, _, session_sequence_length, max_conversation_length, _ = batch
                conversation_length_list = [sum(session_sequence_length[i]) for i in range(len(session_sequence_length))]
                utterance_repre, shape = self.ensemble_model.utterance_encoder(batch_utterances, utterance_sequence_length)
                attentive_repre = self.ensemble_model.attentive_encoder(batch_utterances, utterance_repre, shape)
                # [batch_size, max_conversation_length, hidden_size]
                conversation_repre = self.ensemble_model.conversation_encoder(attentive_repre)
                # [batch_size, max_conversation_length, hidden_size]
                batch_labels = self.recurrent_update(attentive_repre, conversation_repre, max_conversation_length, conversation_length_list)
                predicted_labels.extend(batch_labels)
                for j in range(len(conversation_length_list)):
                    truth_labels.append(labels[j][:conversation_length_list[j]].tolist())
        assert len(predicted_labels) == len(truth_labels)

        utils.save_predicted_results(predicted_labels, truth_labels, self.current_time, step_cnt)

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')

        return purity_score, nmi_score, ari_score, shen_f_score


    def test(self, test_loader, model_path):
        print("Loading model...")
        self.ensemble_model.load_state_dict(torch.load(model_path))

        predicted_labels = []
        truth_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch_utterances, _, labels, utterance_sequence_length, \
                        _, _, session_sequence_length, max_conversation_length, _ = batch
                conversation_length_list = [sum(session_sequence_length[i]) for i in range(len(session_sequence_length))]
                utterance_repre, shape = self.ensemble_model.utterance_encoder(batch_utterances, utterance_sequence_length)
                attentive_repre = self.ensemble_model.attentive_encoder(batch_utterances, utterance_repre, shape)
                # [batch_size, max_conversation_length, hidden_size]
                conversation_repre = self.ensemble_model.conversation_encoder(attentive_repre)
                # [batch_size, max_conversation_length, hidden_size]
                batch_labels = self.recurrent_update(attentive_repre, conversation_repre, max_conversation_length, conversation_length_list)
                predicted_labels.extend(batch_labels)
                for j in range(len(conversation_length_list)):
                    truth_labels.append(labels[j][:conversation_length_list[j]].tolist())
        assert len(predicted_labels) == len(truth_labels)

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')

        log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4))
        print(log_msg)
