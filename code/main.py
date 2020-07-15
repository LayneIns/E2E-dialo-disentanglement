import os
import sys
import argparse
from tqdm import tqdm 
import torch
import random
import logging
import ast
from time import strftime, gmtime
import pickle

import utils
from data_processing import read_data
from utils import build_embedding_matrix
from data_loader import TrainDataLoader
from supervised_trainer import SupervisedTrainer
from models import UtteranceEncoder, ConversationEncoder, SessionEncoder, StateMatrixEncoder, \
                        ScoresCalculator, SelfAttentiveEncoder, EnsembleModel
import constant


random.seed(constant.seed)
torch.manual_seed(constant.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
log_head = "Learning Rate: {}; Random Seed: {};".format(constant.learning_rate, constant.seed)


def train(args):
    utils.make_all_dirs(current_time)
    if args.load_var: 
        all_utterances, labels, word_dict = read_data(load_var=args.load_var, input_=None, mode='train')
        dev_utterances, dev_labels, _ = read_data(load_var=args.load_var, input_=None, mode='dev')
    else:
        all_utterances, labels, word_dict = read_data(load_var=args.load_var, \
                input_=os.path.join(constant.data_path, "entangled_train.json"), mode='train')
        dev_utterances, dev_labels, _ = read_data(load_var=args.load_var, \
                input_=os.path.join(constant.data_path, "entangled_dev.json"), mode='dev')
            
    word_emb = build_embedding_matrix(word_dict, glove_loc=args.glove_loc, \
                    emb_loc=os.path.join(constant.save_input_path, "word_emb.pk"), load_emb=True)
    
    if args.save_input:
        utils.save_or_read_input(os.path.join(constant.save_input_path, "train_utterances.pk"), \
                                    rw='w', input_obj=all_utterances)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "train_labels.pk"), \
                                    rw='w', input_obj=labels)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "word_dict.pk"), \
                                    rw='w', input_obj=word_dict)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "word_emb.pk"), \
                                    rw='w', input_obj=word_emb)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "dev_utterances.pk"), \
                                    rw='w', input_obj=dev_utterances)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "dev_labels.pk"), \
                                    rw='w', input_obj=dev_labels)
    
    train_dataloader = TrainDataLoader(all_utterances, labels, word_dict, add_noise=True)
    dev_dataloader = TrainDataLoader(dev_utterances, dev_labels, word_dict, name='dev')
    # dev_dataloader = TrainDataLoader(all_utterances, labels, word_dict, name='dev')
    
    logger_name = os.path.join(constant.log_path, "{}.txt".format(current_time))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()
    logger.info(log_head)
    
    utterance_encoder = UtteranceEncoder(word_dict, word_emb=word_emb, bidirectional=False, \
                                            n_layers=1, input_dropout=0, dropout=0, rnn_cell='lstm')
    attentive_encoder = SelfAttentiveEncoder()
    conversation_encoder = ConversationEncoder(bidirectional=False, n_layers=1, input_dropout=0, \
                                                    dropout=0, rnn_cell='lstm')
    session_encoder = SessionEncoder(bidirectional=False, n_layers=1, input_dropout=0, \
                                            dropout=0, rnn_cell='lstm')
    state_matrix_encoder = StateMatrixEncoder()
    scores_calculator = ScoresCalculator()
    ensemble_model = EnsembleModel(utterance_encoder, attentive_encoder, conversation_encoder, session_encoder, \
                                    state_matrix_encoder, scores_calculator)
    if torch.cuda.is_available():
        ensemble_model.cuda()

    supervised_trainer = SupervisedTrainer(ensemble_model, logger, current_time)
    
    supervised_trainer.train(train_dataloader, dev_dataloader)


def test(args):
    if args.load_var:
        test_utterances, test_labels, word_dict = read_data(load_var=args.load_var, input_=None, mode='test')
    else:
        test_utterances, test_labels, word_dict = read_data(load_var=args.load_var, \
                input_=os.path.join(constant.data_path, "entangled_test.json"), mode='test')
    
    if args.save_input:
        utils.save_or_read_input(os.path.join(constant.save_input_path, "test_utterances.pk"), \
                                    rw='w', input_obj=test_utterances)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "test_labels.pk"), \
                                    rw='w', input_obj=test_labels)

    test_dataloader = TrainDataLoader(test_utterances, test_labels, word_dict, name='test')
    
    
    utterance_encoder = UtteranceEncoder(word_dict, bidirectional=False, \
                                            n_layers=1, input_dropout=0, dropout=0, rnn_cell='lstm')
    attentive_encoder = SelfAttentiveEncoder()
    conversation_encoder = ConversationEncoder(bidirectional=False, n_layers=1, input_dropout=0, \
                                                    dropout=0, rnn_cell='lstm')
    session_encoder = SessionEncoder(bidirectional=False, n_layers=1, input_dropout=0, \
                                            dropout=0, rnn_cell='lstm')
    state_matrix_encoder = StateMatrixEncoder()
    scores_calculator = ScoresCalculator()
    ensemble_model = EnsembleModel(utterance_encoder, attentive_encoder, conversation_encoder, session_encoder, \
                                    state_matrix_encoder, scores_calculator)
    if torch.cuda.is_available():
        ensemble_model.cuda()

    supervised_trainer = SupervisedTrainer(ensemble_model)
    
    supervised_trainer.test(test_dataloader, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--save_input', action='store_true', 
                            help="save the input as a pickle object")
    parser.add_argument('--load_var', action='store_true', 
                            help='load a pickle object')
    parser.add_argument('--glove_loc', type=str, default=constant.glove_path)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError('Mode Error')



    
    
