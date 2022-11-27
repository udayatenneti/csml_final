import numpy as np
import pandas as pd
import torch
import argparse
from utils import *
from models import *
from torch import optim
import os
import torch.utils.tensorboard as tb
import random


_BASEPATH = 'C:\\Users\\dazet\\Documents\\CSML\\final_project_data\\archive\\PKLot\\'
_WEATHER = ('Sunny', 'Cloudy', 'Rainy')
_PUCPR = 'PUCPR'
_BAD_WEATHER_PUCPR = 'Sunny'
_BAD_DATES_PUCPR_SUNNY = ('2012-10-30', '2012-11-06', '2012-11-07')
_CLASSES_MAP = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,101]

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--angle', type=str, default='PUCPR', help='model to run (TRIVIAL or DAN)')
    parser.add_argument('--chunks', type=int, default=5, help='chunks to split training data into')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.50d-relativized.txt', help='path to word embeddings to use')
    # Some common args have been pre-populated for you. Again, you can add more during development, but your code needs
    # to run with the default neural_sentiment_classifier for submission.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size; 1 by default and you do not need to batch unless you want to')
    args = parser.parse_args()
    return args

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


def load_data(dataset_path, classes, none_threshold, num_workers=0, batch_size=1):
    dataset = PKLot(dataset_path, classes, none_threshold)
    print(len(dataset))
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def divide_dates_train_valid(camera_angle, percent_valid=0.3):
    """
    :return:
    two lists of : ['C:\\Users\\dazet\\Documents\\CSML\\final_project_data\\archive\\PKLot\\PUCPR\\date',]
    one for train, one for valid
    """
    dataset_path = _BASEPATH + camera_angle + "\\"
    all_paths = []
    for weather in _WEATHER:
        newpath = dataset_path + weather + "\\"
        dates = os.listdir(newpath)
        if camera_angle == _PUCPR and weather == _BAD_WEATHER_PUCPR:
            #remove bad dates:
            dates_set = set(dates)
            bad_dates_set = set(_BAD_DATES_PUCPR_SUNNY)
            set_of_dates = dates_set.difference(bad_dates_set)
            dates = list(set_of_dates)
        these_paths = [newpath + x for x in dates]
        all_paths.extend(these_paths)
    total_days = len(all_paths)
    samples_valid = int(total_days * percent_valid)
    samples_train = total_days - samples_valid
    random.shuffle(all_paths)

    # return all_paths[:samples_train], all_paths[samples_train:]
    return all_paths, all_paths[samples_train:]



def train(camera_angle, chunks):
    """
    doing something
    :return:
    """

    train_logger, valid_logger = None, None

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')

    model = CNNClassifier(n_output_channels=len(_CLASSES_MAP)).to(device)
    # if args.continue_training:
    #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    train_logger = tb.SummaryWriter(os.path.join("./", 'train'))
    valid_logger = tb.SummaryWriter(os.path.join("./", 'valid'))

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.NLLLoss()

    #train_section, valid_section = [_SECTION + x for x in _DATES_TRAIN]
    train_section, valid_section = divide_dates_train_valid(camera_angle)

    none_threshold = 5 #TODO: How many unlabeled spots can we allow before we throw out image?

    #train_data = load_data(train_section, _CLASSES_MAP, none_threshold, batch_size=1)
    #valid_data = load_data(valid_section, _CLASSES_MAP, none_threshold, batch_size=1)

    num_epoch = 3
    global_step = 0
    train_data, valid_data = None, None
    for epoch in range(num_epoch):
        model.train()
        acc_vals = []
        startidx = 0
        chunksize = len(train_section) // chunks
        for counter in range(chunks+1):
            if (startidx + chunksize) >= len(train_section):
                train_data = load_data(train_section[startidx:], _CLASSES_MAP, none_threshold,
                                       batch_size=1)
            else:
                train_data = load_data(train_section[startidx: startidx + chunksize], _CLASSES_MAP, none_threshold,
                                       batch_size=1)
            startidx = startidx + chunksize
            for img, label, file_date in train_data:
                img, label = img.to(device), label.to(device)

                logit = model(img)
                loss_val = loss_func(logit, label)
                acc_val = accuracy(logit, label)

                if train_logger is not None:
                    train_logger.add_scalar('loss', loss_val, global_step)
                acc_vals.append(acc_val.detach().cpu().numpy())

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                global_step += 1
                print("finished " + file_date[0].split('\\')[-1] + " for epoch: " + str(epoch) +
                      " global step: " + str(global_step))
            train_data = None
            if startidx >= len(train_section):
                break


        avg_acc = sum(acc_vals) / len(acc_vals)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        startidx_valid = 0
        chunksize_valid = len(valid_section) // chunks
        for counter in range(chunks + 1):
            if (startidx_valid + chunksize) >= len(valid_section):
                valid_data = load_data(valid_section[startidx_valid:], _CLASSES_MAP, none_threshold,
                                       batch_size=1)
            else:
                valid_data = load_data(valid_section[startidx_valid: startidx_valid + chunksize_valid], _CLASSES_MAP, none_threshold,
                                       batch_size=1)
            startidx_valid = startidx_valid + chunksize_valid
            for img, label, _ in valid_data:
                img, label = img.to(device), label.to(device)
                acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
            valid_data = None
            if startidx_valid >= len(valid_section):
                break


        avg_vacc = sum(acc_vals) / len(acc_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model, name=camera_angle + "_stanford_trainall_1angle_1E-3_cnn.th")
    save_model(model, name=camera_angle + "_stanford_trainall_1angle_1E-3_cnn.th")


if __name__ == "__main__":
    args = _parse_args()
    camera_angle = args.angle
    chunks = args.chunks
    train(camera_angle, chunks)
