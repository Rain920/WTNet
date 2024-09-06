import os
import sys
import argparse
import pickle
import time
import math
import torch
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm
import pandas as pd
import datetime
import random
from scipy.interpolate import splev, splrep
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from prettytable import PrettyTable
from model import models


result_path = ''

def get_parameters():
    parser = argparse.ArgumentParser(description='CNN_RNN')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='learning rate float')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adamw', help='optimizer, default as sgd')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=66)

    parser.add_argument('--lambda1', type=float, default=0.9, help='ratio of cross-entropy loss')
    parser.add_argument('--lambda2', type=float, default=0.1, help='ratio of contrastive loss')

    args = parser.parse_args()

    if torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # use deterministic behavior, to ensure pytorch has the same output with the same para.
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device('cpu')
    
    return args, device


def load_data_ecg(base_dir, train_list, test_list, device):

    device = torch.device('cpu')

    X, y = [], []
    for name in train_list:
        with open(os.path.join(base_dir, name + '.pkl'), 'rb') as f:  # read preprocessing result
            apnea_ecg = pickle.load(f)
        X1, X2, X3, X4, X5, X6, Y = apnea_ecg['X1'], apnea_ecg['X2'], apnea_ecg['X3'], apnea_ecg['X4'], apnea_ecg['X5'], apnea_ecg['X6'], apnea_ecg['y']
        array_list = [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6)]
        concat_X = np.concatenate(array_list, axis=1)
        X.extend(concat_X)
        y.extend(Y)

    assert len(X) == len(y)

    num = [i for i in range(len(X))]
    trainlist, vallist, y_train, y_val = train_test_split(num, y, test_size=0.2, random_state=42,
                                                          stratify=y, shuffle=True)

    x_train, x_val = [], []
    for i in trainlist:
        x_train.append(X[i])
    for i in vallist:
        x_val.append(X[i])
    
    x_test, y_test = [], []
    sub_id = []
    for name in test_list:
        with open(os.path.join(base_dir, name + '.pkl'), 'rb') as f:  # read preprocessing result
            apnea_ecg = pickle.load(f)
        X1, X2, X3, X4, X5, X6, y = apnea_ecg['X1'], apnea_ecg['X2'], apnea_ecg['X3'], apnea_ecg['X4'], apnea_ecg['X5'], apnea_ecg['X6'], apnea_ecg['y']
        array_list = [np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6)]
        concat_X = np.concatenate(array_list, axis=1)
        x_test.extend(concat_X)
        y_test.extend(y)
        sub_id.extend([name] * len(y))
    
    assert len(x_train) == len(y_train)

    sample_analysis(y_train, y_val, y_test)

    x_train = np.array(x_train, dtype="float32")
    x_val = np.array(x_val, dtype="float32")
    x_test = np.array(x_test, dtype="float32")

    y_test = np.array(y_test, dtype="int")
    y_train = np.array(y_train, dtype="int")
    y_val = np.array(y_val, dtype="int")

    x_train = np.expand_dims(x_train, axis=1)
    x_val = np.expand_dims(x_val, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    return torch.Tensor(x_train).to(device), \
           torch.Tensor(y_train).to(device), \
           torch.Tensor(x_val).to(device),\
           torch.Tensor(y_val).to(device), \
           torch.Tensor(x_test).to(device), \
           torch.Tensor(y_test).to(device), \
           sub_id


def sample_analysis(y_train, y_val, y_test):
    positive_train = sum(y_train)
    negative_train = len(y_train) - positive_train

    positive_val = sum(y_val)
    negative_val = len(y_val) - positive_val

    positive_test = sum(y_test)
    negative_test = len(y_test) - positive_test

    total_train = len(y_train)
    total_val = len(y_val)
    total_test = len(y_test)

    ratio_positive_train = positive_train / total_train
    ratio_negative_train = negative_train / total_train

    ratio_positive_val = positive_val / total_val
    ratio_negative_val = negative_val / total_val

    ratio_positive_test = positive_test / total_test
    ratio_negative_test = negative_test / total_test

    table = PrettyTable()

    table.field_names = ["dataset", "pos_num", "neg_num", "pos_ratio", "neg_ratio"]

    table.add_row(["train", positive_train, negative_train, ratio_positive_train, ratio_negative_train])
    table.add_row(["val", positive_val, negative_val, ratio_positive_val, ratio_negative_val])
    table.add_row(["test", positive_test, negative_test, ratio_positive_test, ratio_negative_test])

    print(table)


def data_prepare(args, device):
    base_dir = '' # data after preprocessing
    train_list = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", 
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20", 
        "b01", "b02", "b03", "b04", "b05", 
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]
    test_list = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]

    x_train, y_train, x_val, y_val, x_test, y_test, sub_id = load_data_ecg(base_dir, train_list, test_list, device)

    print(x_train.shape)
    print(y_train.shape)
    
    print(x_val.shape)
    print(y_val.shape)

    print(x_test.shape)
    print(y_test.shape)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return train_iter, val_iter, test_iter, sub_id


def prepare_model(args, device):

    model = models.WTNet_cl().to(device)
    best_model = models.WTNet_cl().to(device)

    para = [p for p in model.parameters() if p.requires_grad]
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(para, lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(para, lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(para, lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "sgd":
        optimizer = optim.SGD(para, lr=args.lr, weight_decay=args.weight_decay_rate, momentum=0.9)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=args.batch_size, div_factor=10)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=1.1)

    loss_fun = nn.NLLLoss()

    return model, optimizer, scheduler, loss_fun, best_model


def save_result(result_data, result_file):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    result_file = formatted_time + '_' + result_file
    df = pd.DataFrame(result_data)
    df.to_csv(result_path + result_file, index=False)


def correct_torch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def train(args, optimizer, scheduler, model, train_iter, val_iter):
    start_time = time.time()
    best_acc, best_epoch = 0.0, 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(args.epochs):
        loss_sum, correct, n = 0.0, 0, 0
        print('EPOCH:', epoch + 1)
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            x = x.to(device)
            y = y.to(device)
            y_pred, ct_loss = model(x, y)
            y = y.long()
            loss_train = F.nll_loss(y_pred, y) * args.lambda1 + ct_loss * args.lambda2
            loss_sum += loss_train.data.item() * len(y)
            correct += correct_torch(y_pred, y)
            n += len(y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # lr_list.append(optimizer.param_groups[0]['lr'])
        train_acc = correct / n
        train_loss = loss_sum / n
        val_loss, val_acc = val(model, val_iter, device, args)
        scheduler.step()
        end_time = time.time()
        cost_time = end_time - start_time
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Cost time: {:.2f} | Lr: {:.20f} |Train loss: {:.6f} | Train accuracy: {:.4f}'\
              ' Val loss: {:.6f} | Val accuracy: {:.4f}| GPU occupy: {:.6f} MiB'.\
            format(epoch+1, cost_time, optimizer.param_groups[0]['lr'], train_loss, train_acc, val_loss, val_acc, gpu_mem_alloc))
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc.cpu().item())
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc.cpu().item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), result_path + 'best_model.pth')
    print('Best val acc: {:.6f} | Best epoch: {} | Seed: {}'.format(best_acc, best_epoch, args.seed))
    epoch_list = list(range(1, args.epochs + 1))
    result_data = {'Epoch': epoch_list, 'Train Loss': train_loss_list, 'Train Accuracy': train_acc_list, 'Val Loss': val_loss_list, 'Val Accuracy': val_acc_list}
    save_result(result_data, 'main_wt_sym4_train_loss_acc.csv')
    return best_epoch


@torch.no_grad()
def val(model, val_iter, device, args):
    model.eval()
    loss_sum, n, correct = 0.0, 0, 0
    for x, y in val_iter:
        x = x.to(device)
        y = y.to(device)
        y_pred, ct_loss = model(x, y)
        y = y.long()
        loss_val = F.nll_loss(y_pred, y) * args.lambda1 + ct_loss * args.lambda2
        # loss_val = F.binary_cross_entropy_with_logits(y_pred, y)
        loss_sum += loss_val.data.item() * len(y)
        n += len(y)
        correct += correct_torch(y_pred, y)
    val_loss = loss_sum / n
    val_acc = correct / n
    return val_loss, val_acc


@torch.no_grad()
def test(model, test_iter, device, args, sub_id, epoch=None):
    model.eval()
    y_true = []
    y_pred = []
    loss_sum, n = 0.0, 0
    for x, y in tqdm.tqdm(test_iter):
        x = x.to(device)
        y = y.to(device)
        y_out, _ = model(x)
        y = y.long()
        loss_test = F.nll_loss(y_out, y)
        loss_sum += loss_test.data.item() * len(y)
        n += len(y)
        y_true.extend(y.view(-1).cpu().numpy())
        y_pred.extend(y_out.max(1)[1].cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    spec = tn / (tn + fp)

    avg_loss = loss_sum / n

    print(conf_matrix)

    if epoch is None:
        epoch = args.epochs

    print(f'Dataset: ApneaECG | Test loss: {avg_loss:.6f} Accuracy: {acc:.6f} | Precision: {precision:.6f} | Recall: {recall:.6f} | F1 Score: {f1:.6f} | Specificity: {spec:.6f} | AUC: {auc:.6f} | Epoch: {args.epochs}/{epoch} ')


if __name__ == "__main__":
    print('-------------start-------------')
    args, device = get_parameters()
    train_iter, val_iter, test_iter, sub_id = data_prepare(args, device)
    result_model, optimizer, scheduler, loss_fun, best_model = prepare_model(args, device)
    print(result_model)

    print('---------------------------train--------------------------------')
    best_epoch = train(args, optimizer, scheduler, result_model, train_iter, val_iter)

    print('---------------------------test--------------------------------')
    test(result_model, test_iter, device, args, sub_id)

    print('---------------------------test best model--------------------------------')
    
    best_model.load_state_dict(torch.load(result_path + 'best_model.pth'))
    test(best_model, test_iter, device, args, sub_id, epoch=best_epoch)

