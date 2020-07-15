import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

import random
import numpy as np 

import constant


class EnsembleModel(nn.Module):
    def __init__(self, utterance_encoder, attentive_encoder, conversation_encoder, session_encoder, \
                    state_matrix_encoder, scores_calculator):
        super(EnsembleModel, self).__init__()
        self.utterance_encoder = utterance_encoder
        self.attentive_encoder = attentive_encoder
        self.conversation_encoder = conversation_encoder
        self.session_encoder = session_encoder
        self.state_matrix_encoder = state_matrix_encoder
        self.scores_calculator = scores_calculator
    
    def forward(self, batch):
        batch_utterances, label_for_loss, labels, utterance_sequence_length, \
                    session_transpose_matrix, state_transition_matrix, session_sequence_length, \
                        max_conversation_length, loss_mask = batch
        utterance_repre, shape = self.utterance_encoder(batch_utterances, utterance_sequence_length)
        # [batch_size, max_conversation_length, hidden_size]
        attentive_repre = self.attentive_encoder(batch_utterances, utterance_repre, shape)
        # [batch_size, max_conversation_length, hidden_size]
        conversation_repre = self.conversation_encoder(attentive_repre)
        # [batch_size, max_conversation_length, hidden_size]
        session_repre = self.session_encoder(attentive_repre, session_transpose_matrix)
        # [batch_size, 4, max_session_length, hidden_size]
        state_matrix = self.state_matrix_encoder(attentive_repre, conversation_repre, session_repre, state_transition_matrix, \
                        max_conversation_length)
        # [batch_size, max_conversation_length, 5, hidden_size] 
        softmax_masked_scores = self.scores_calculator(state_matrix, attentive_repre, conversation_repre, max_conversation_length)
        # [batch_size, max_conversation_length, 5]
        return softmax_masked_scores


class UtteranceEncoder(nn.Module):
    def __init__(self, word_dict, word_emb=None, bidirectional=False, n_layers=1, input_dropout=0, \
                        dropout=0, rnn_cell='lstm'):
        super(UtteranceEncoder, self).__init__()
        self.word_emb = word_emb
        self.word_emb_matrix = nn.Embedding(len(word_dict), constant.embedding_size)
        self.init_embedding()
        # self.bidirectional = bidirectional
        # self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout)
        
        # bi = 2 if self.bidirectional else 1
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(constant.embedding_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(constant.embedding_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)

    def init_embedding(self):
        if self.word_emb is None:
            self.word_emb_matrix.weight.data.uniform_(-0.1, 0.1)
        else:
            self.word_emb_matrix.weight.data.copy_(torch.from_numpy(self.word_emb))
    
    def forward(self, input_var, input_lens):
        shape = input_var.size() # batch_size, max_conversation_length, max_utterance_length
        input_var = input_var.view(-1, shape[2])
        input_lens = input_lens.reshape(-1)
        embeded_input = self.word_emb_matrix(input_var)
        # embeded_input = self.input_dropout(embeded_input)
        word_output, _ = self.encoder(embeded_input)
        # word_output: [batch_size * max_conversation_length, max_utterance_length, hidden_size]
        return word_output, shape


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, dropout=0.):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(constant.hidden_size, constant.hidden_size, bias=False)
        self.ws2 = nn.Linear(constant.hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = 1

    def forward(self, inp, lstm_output, shape):
        size = lstm_output.size()  # [batch_size * max_conversation, max_utterance_length, hidden_size]
        compressed_embeddings = lstm_output.contiguous().view(-1, size[2])  # [batch_size * max_conversation_length * max_utterance_length, hidden_size]
        transformed_inp = inp.view(size[0], 1, size[1])  # [batch_size * max_conversation_length, 1, max_utterance_length]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [batch_size * max_conversation_length, hop, max_utterance_length]

        hbar = self.tanh(self.ws1(compressed_embeddings)) # [batch_size * max_conversation_length * max_utterance_length, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [batch_size * max_conversation_length, max_utterance_length, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [batch_size * max_conversation_length, hop, max_utterance_length]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == constant.PAD_ID).float())
            # [batch_size * max_conversation_length, hop, max_utterance_length] + [batch_size * max_conversation_length, hop, max_utterance_length]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [batch_size * max_conversation_length * hop, max_utterance_length]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [batch_size * max_conversation_length, hop, max_utterance_length]
        # [batch_size * max_conversation_length, hop, hidden_size]
        ret_output = torch.bmm(alphas, lstm_output).squeeze().view(shape[0], shape[1], constant.hidden_size) 
        return ret_output


class ConversationEncoder(nn.Module):
    def __init__(self, bidirectional=False, n_layers=1, input_dropout=0, dropout=0, rnn_cell='lstm'):
        super(ConversationEncoder, self).__init__()
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
    
    def forward(self, input_var):
        # input_var: [batch_size, max_conversation_length, hidden_size]
        conv_output, _ = self.encoder(input_var)
        # conv_output: [batch_size, max_conversation_length, hidden_size]
        return conv_output


class SessionEncoder(nn.Module):
    def __init__(self, bidirectional=False, n_layers=1, input_dropout=0, dropout=0, rnn_cell='lstm'):
        super(SessionEncoder, self).__init__()
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
    
    def forward(self, input_var, transpose_matrix):
        # input_var: [batch_size, max_conversation_length, hidden_size]
        batch_size, max_conversation_length, _ = input_var.size()
        input_var = input_var.contiguous().view(-1, constant.hidden_size)
        input_var = input_var[transpose_matrix]
        # [batch_size * max_conversation_length, hidden_size]
        input_var = input_var.view(-1, int(max_conversation_length/4), constant.hidden_size)
        # [batch_size * 4, max_session_length, hidden_size]
        output, _ = self.encoder(input_var)
        # output: [batch_size*4, max_session_length, hidden_size]
        output = output.view(batch_size, 4, int(max_conversation_length/4), constant.hidden_size)
        return output


class StateMatrixEncoder(nn.Module):
    def __init__(self, pooling=nn.AvgPool2d):
        super(StateMatrixEncoder, self).__init__()
        self.new_state_projection = nn.Sequential(
            nn.Linear(constant.hidden_size*2, constant.hidden_size), 
            nn.ReLU()
        )
        self.pooling = pooling((4, 1))
    
    def build_state_matrix(self, state_matrix, session_repre, state_transition_matrix, \
                                    batch_size, max_conversation_length):
        # state_matrix: [batch_size, max_conversation_length, 5, hidden_size]
        # session_repre: [batch_size, 4, max_session_length, hidden_size]
        # state_transition_matrix: [batch_size, max_conversation_length, 5]
        for batch_index in range(batch_size):
            for i in range(max_conversation_length):
                one_res = []
                for j in range(5):
                    if state_transition_matrix[batch_index][i][j] != 0:
                        if state_transition_matrix[batch_index][i][j] != 0:
                            position = state_transition_matrix[batch_index][i][j] - 1
                            state_matrix[batch_index][i][j] = session_repre[batch_index][j-1][position]
                            one_res.append(session_repre[batch_index][j-1][position].unsqueeze(0))
                        else:
                            one_res.append(state_matrix[batch_index][j-1][j].unsqueeze(0))
                one_res = torch.cat(one_res, dim=0)
                state_matrix[batch_index][i][0] =self.pooling(one_res.unsqueeze(0))[0][0]
        # pooling_res = self.pooling(state_matrix[:, :, 1:, :])
        # print(pooling_res.size())
        # for batch_index in range(batch_size):
        #     for i in range(max_conversation_length):
        #         if state_transition_matrix[batch_index][i][0] == 0:
        #             state_matrix[batch_index][i][0] =self.pooling(state_matrix[:, :, 1:, :])[batch_index][i][0]
        # print(state_matrix[0][0])
        return state_matrix
    
    def forward(self, utterance_repre, conversation_repre, session_repre, \
                    state_transition_matrix, max_conversation_length):
        # utterance_repre: [batch_size, max_conversation_length, hidden_size]
        # conversation_repre: [batch_size, max_conversation_length, hidden_size]
        # session_repre: [batch_size, 4, max_session_length, hidden_size]
        batch_size, _, hidden_size = utterance_repre.size()
        # state_matrix = torch.randn([batch_size, max_conversation_length, 5, hidden_size])
        shape = torch.Size([batch_size, max_conversation_length, 5, hidden_size])
        if torch.cuda.is_available():
            state_matrix = torch.cuda.FloatTensor(shape).zero_()
        else:
            state_matrix = torch.FloatTensor(shape).zero_()
        # state_matrix = torch.randn(shape, out=state_matrix)
        state_matrix = self.build_state_matrix(state_matrix, session_repre, state_transition_matrix, \
                                                    batch_size, max_conversation_length)
        return state_matrix


class ScoresCalculator(nn.Module):
    def __init__(self, softmax_func=nn.LogSoftmax):
        super(ScoresCalculator, self).__init__()
        self.softmax_func = softmax_func(dim=2)
        self.utterance_projection = nn.Sequential(
            nn.Linear(constant.hidden_size*2, constant.hidden_size), 
            nn.ReLU()
        )

    def forward(self, state_matrix, utterance_repre, conversation_repre, max_conversation_length):
        # state_matrix: [batch_size, max_conversation_length, 5, hidden_size]
        # utterance_repre: [batch_size, max_conversation_length, hidden_size]
        # conversation_repre: [batch_size, max_conversation_length, hidden_size]
        # loss_mask: [batch_size, max_conversation_length, 5]
        utterance_concat = torch.cat((utterance_repre[:, :max_conversation_length, :], \
                                    conversation_repre[:, :max_conversation_length, :]), 2)
        utterance_projected = self.utterance_projection(utterance_concat)
        # [batch_size, max_conversation_length, hidden_size]

        # scores = torch.matmul(utterance_repre.unsqueeze(2), state_matrix.permute(0,1,3,2)).squeeze(2)
        scores = torch.matmul(state_matrix, utterance_projected.unsqueeze(3)).squeeze()
        # scores: [batch_size, max_conversation_length, 5]
        softmax_masked_scores = self.softmax_func(scores)

        return softmax_masked_scores



