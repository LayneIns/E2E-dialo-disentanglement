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
    def __init__(self, args, ensemble_model, teacher_model=None, logger=None, current_time=None, loss=nn.KLDivLoss, optimizer=None):
        self.args = args
        self.ensemble_model = ensemble_model
        self.teacher_model = teacher_model
        self.logger = logger
        self.current_time = current_time
        self.loss_func = loss(reduction='batchmean')
        if optimizer == None:
            if self.args.model == 'T':
                params = list(self.teacher_model.parameters())
            elif self.args.model == 'S':
                params = list(self.ensemble_model.parameters())
            elif self.args.model == 'TS':
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

    def _train_batch(self, batch, noise_batch):
        batch_utterances, label_for_loss, labels, utterance_sequence_length, \
                    session_transpose_matrix, state_transition_matrix, session_sequence_length, \
                        max_conversation_length, loss_mask = batch
        if self.args.model == 'TS':
            if self.args.add_noise:
                softmax_masked_scores = self.ensemble_model(noise_batch)
            else:
                softmax_masked_scores = self.ensemble_model(batch)
            teacher_scores, teacher_log_scores = self.teacher_model(batch)
            # [batch_size, max_conversation_length, 5]
            loss_kl = self.loss_func(softmax_masked_scores, teacher_scores)
            loss_1 = self.calculate_loss(softmax_masked_scores, label_for_loss, loss_mask)
            loss_2 = self.calculate_loss(teacher_log_scores, label_for_loss, loss_mask)
            loss = loss_1 + loss_kl
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss_1.data.item(), loss_2.data.item(), loss_kl.data.item(), len(batch_utterances), sum(sum(session_sequence_length))
        if self.args.model == 'S':
            if self.args.add_noise:
                softmax_masked_scores = self.ensemble_model(noise_batch)
            else:
                softmax_masked_scores = self.ensemble_model(batch)
            # [batch_size, max_conversation_length, 5]
            loss_1 = self.calculate_loss(softmax_masked_scores, label_for_loss, loss_mask)
            loss = loss_1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss_1.data.item(), 0, 0, len(batch_utterances), sum(sum(session_sequence_length))
        if self.args.model == 'T':
            if self.args.add_noise:
                teacher_scores, teacher_log_scores = self.teacher_model(noise_batch)
            else:
                teacher_scores, teacher_log_scores = self.teacher_model(batch)
            loss_2 = self.calculate_loss(teacher_log_scores, label_for_loss, loss_mask)
            loss = loss_2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return 0, loss_2.data.item(), 0, len(batch_utterances), sum(sum(session_sequence_length))

    def train(self, train_loader, noise_train_loader, dev_loader):
        step_cnt = 0
        for epoch in tqdm(range(constant.epoch_num)):
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_loader)):
                step_cnt += 1
                if self.args.add_noise:
                    noise_batch = noise_train_loader[i]
                else:
                    noise_batch = None
                loss_student, loss_teacher, loss_kl, batch_size, uttr_cnt = self._train_batch(batch, noise_batch)
                # loss_student, batch_size, uttr_cnt = self._train_batch(batch)
                if self.args.model == 'T':
                    epoch_loss += loss_teacher
                else:
                    epoch_loss += loss_student
                log_msg = "Epoch : {}, batch: {}/{}, step: {}, batch student loss: {}, teacher loss: {}, kl loss: {}, uttr avg loss: {}".format(
                                        epoch, i, len(train_loader), step_cnt, round(loss_student, 4), round(loss_teacher, 4), round(loss_kl, 4), round(loss_student*batch_size/uttr_cnt, 4))
                # log_msg = "Epoch : {}, batch: {}/{}, step: {}, batch student loss: {}, uttr avg loss: {}".format(
                                        # epoch, i, len(train_loader), step_cnt, round(loss_student, 4), round(loss_student*batch_size/uttr_cnt, 4))
                self.logger.info(log_msg)
                if step_cnt % constant.inference_step == 0:
                    if self.args.model == "T":
                        model_name = os.path.join(constant.save_model_path, "model_{}".format(self.current_time), \
                                                    "step_{}.pkl".format(step_cnt))
                        torch.save(self.teacher_model.state_dict(), model_name)
                    else:
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

    def init_state(self, batch_index, j, state, hidden_state_history, utterance_repre, conversation_repre, predicted_batch_label, mask):
        # state : [5, constant.hidden_size]
        if j == 0:
            one_res = self.ensemble_model.state_matrix_encoder.pooling(state[1:, :].unsqueeze(0))[0][0]
            state[0] = self.ensemble_model.state_matrix_encoder.new_state_projection(
                torch.cat([one_res, conversation_repre[batch_index][j]])
            )
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
            one_res = self.ensemble_model.state_matrix_encoder.pooling(state[1:, :].unsqueeze(0))[0][0]
            state[0] = self.ensemble_model.state_matrix_encoder.new_state_projection(
                torch.cat([one_res, conversation_repre[batch_index][j-1]])
            )
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
            shape = torch.Size([constant.state_num, constant.hidden_size])
            if torch.cuda.is_available():
                state = torch.cuda.FloatTensor(shape).zero_()
            else:
                state = torch.FloatTensor(shape).zero_()
            # state = torch.randn([5, constant.hidden_size])
            if torch.cuda.is_available():
                mask = [0.] + [-1.] * (constant.state_num-1)
                mask = torch.cuda.LongTensor(mask).type(torch.double)
            else:
                mask = [0.] + [-1.] * (constant.state_num-1)
                mask = torch.LongTensor(mask).type(torch.double)
            hidden_state_history = {i + 1: None for i in range(constant.state_num-1)}
            for j in range(conversation_length_list[batch_index]):
                state, mask, hidden_state_history = self.init_state(batch_index, j, state, hidden_state_history, utterance_repre, conversation_repre, predicted_batch_label, mask)
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


    def test(self, test_loader, model_path, step_cnt):
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

        utils.save_predicted_results(predicted_labels, truth_labels, self.current_time, step_cnt, mode='test')

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')

        log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4))
        print(log_msg)
