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
log_head = "Learning Rate: {}; Random Seed: {}; ".format(constant.learning_rate, constant.seed)


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
     
    train_dataloader = TrainDataLoader(all_utterances, labels, word_dict)
    if args.add_noise:
        noise_train_dataloader = TrainDataLoader(all_utterances, labels, word_dict, add_noise=True)
    else:
        noise_train_dataloader = None
    dev_dataloader = TrainDataLoader(dev_utterances, dev_labels, word_dict, name='dev')
    
    logger_name = os.path.join(constant.log_path, "{}.txt".format(current_time))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()
    global log_head
    log_head = log_head + "Training Model: {}; ".format(args.model)
    if args.add_noise:
        log_head += "Add Noise: True; "
    logger.info(log_head)
    
    
    if args.model == 'T':
        ensemble_model_bidirectional = EnsembleModel(word_dict, word_emb=word_emb, bidirectional=True)
    elif args.model == 'TS':
        ensemble_model_bidirectional = EnsembleModel(word_dict, word_emb=None, bidirectional=True)
    else:
        ensemble_model_bidirectional = None
    if args.model == 'TS':
        ensemble_model_bidirectional.load_state_dict(torch.load(args.model_path))
    ensemble_model = EnsembleModel(word_dict, word_emb=word_emb, bidirectional=False)

    if torch.cuda.is_available():
        ensemble_model.cuda()
        if args.model == 'T' or args.model == 'TS':
            ensemble_model_bidirectional.cuda()

    supervised_trainer = SupervisedTrainer(args, ensemble_model, teacher_model=ensemble_model_bidirectional, \
                                                logger=logger, current_time=current_time)
    
    supervised_trainer.train(train_dataloader, noise_train_dataloader, dev_dataloader)


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
    
    ensemble_model = EnsembleModel(word_dict, word_emb=None, bidirectional=False)
    if torch.cuda.is_available():
        ensemble_model.cuda()

    supervised_trainer = SupervisedTrainer(args, ensemble_model)
    
    supervised_trainer.test(test_dataloader, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='', \
                            help="'T' for teacher training, 'S' for single model training,'TS' for teacher-student training")
    parser.add_argument('--add_noise', type=ast.literal_eval, default=False)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--save_input', type=ast.literal_eval, default=False)
    parser.add_argument('--load_var', type=ast.literal_eval, default=False)
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



    
    
