from __future__ import print_function
import torch
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from torchvision import datasets, models, transforms
import dataloader_final as dataloader
import pdb
import io
import PIL
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import json
import pandas as pd
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights, densenet161, DenseNet161_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights, swin_t, Swin_T_Weights, vit_b_32, ViT_B_32_Weights, vgg16, VGG16_Weights
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from openpyxl import load_workbook

sns.set_theme()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--add', '--add_number', default=2, type=int, help='add num')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--num_clean', default=5, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--gpuid', default=7, type=int)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--data_path', default='./dataset_CAMO', type=str, help='path to dataset')
parser.add_argument('--dataset', default='object', type=str)

parser.add_argument('--model', default='ResNet-18', type=str)
parser.add_argument('--clean_get', default=1, type=int)
parser.add_argument('--add_num', default=1, type=int)
parser.add_argument('--seed', default=37, type=int)
args, unknown = parser.parse_known_args()

gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, CUDA_VISIBLE_DEVICES: {gpu_id}")
args.batch_size = 16

random.seed(args.seed)
torch.manual_seed(args.seed)

log_file = "./result/information_%s_%s_%s_%s.txt"%(args.model, args.clean_get, args.add_num, args.seed)

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

def test_conf(losses, preds_classes, all_targets, all_name, conf):

    extracted_predictions_file = "./excel/information_mkl.xlsx"
    extracted_predictions = pd.read_excel(extracted_predictions_file)

    device1 = losses.device
    loss_mean = losses.mean()
    loss_std = losses.std()
    losses = (losses - loss_mean.to(device1)) / loss_std.to(device1)

    intervals = [(float('-inf'), -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.9), (-0.9, -0.8), (-0.8, -0.7),
                 (-0.7, -0.6), (-0.6, -0.5), (-0.5, -0.4), (-0.4, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0.0),
                 (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                 (0.5, 0.52), (0.52, 0.54),(0.54, 0.56),(0.56, 0.58),(0.58, 0.6),
                 (0.6, 0.62), (0.62, 0.64),(0.64, 0.66),(0.56, 0.68),(0.68, 0.7),
                 (0.7, 0.72), (0.72, 0.74),(0.74, 0.76),(0.56, 0.78),(0.78, 0.8),
                 (0.8, 0.82), (0.82, 0.84),(0.84, 0.86),(0.56, 0.88),(0.88, 0.9),
                 (0.9, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, float('inf'))]

    statistics = {interval: {'sample_count': 0, 'correct_count': 0, 'pred_1_count': 0, 'pred_0_count': 0}
                  for interval in intervals}

    total_correct = 0
    total_samples = len(losses)
    below_conf_count = 0
    below_conf_correct = 0
    above_conf_correct = 0 
    above_conf_samples = []  

    for i in range(total_samples):
        
        loss_value = losses[i]
        pred_class = preds_classes[i]
        target_class = all_targets[i]

        if loss_value < conf:
            below_conf_count += 1
            if pred_class == target_class:
                below_conf_correct += 1
        else:
            above_conf_samples.append(i)  
            if pred_class == target_class:
                above_conf_correct += 1

        if pred_class == target_class:
            total_correct += 1

        for interval in intervals:
            if interval[0] <= loss_value < interval[1]:
                statistics[interval]['sample_count'] += 1
                if pred_class == target_class:
                    statistics[interval]['correct_count'] += 1
                if pred_class == 1:
                    statistics[interval]['pred_1_count'] += 1
                if pred_class == 0:
                    statistics[interval]['pred_0_count'] += 1
                break

    for interval in intervals:
        sample_count = statistics[interval]['sample_count']
        if sample_count > 0:
            correct_rate = statistics[interval]['correct_count'] / sample_count
        else:
            correct_rate = 0
        statistics[interval]['correct_rate'] = correct_rate

    total_acc = total_correct / total_samples

    if below_conf_count > 0:
        below_conf_acc = below_conf_correct / below_conf_count
    else:
        below_conf_acc = 0

    if len(above_conf_samples) > 0:
        above_conf_acc = above_conf_correct / len(above_conf_samples)
    else:
        above_conf_acc = 0

    new_preds_classes = preds_classes.clone()
    for i in above_conf_samples:
        name = all_name[i]  
        if name in extracted_predictions['name'].values:
            new_value = extracted_predictions.loc[extracted_predictions['name'] == name, 'value'].values[0]
            new_preds_classes[i] = new_value

    new_correct = 0
    for i in range(total_samples):
        if new_preds_classes[i] == all_targets[i]:
            new_correct += 1
    new_acc = new_correct / total_samples

    with open(log_file, "a") as log:
        log.write(f"Total Accuracy: {total_acc:.4f}\n")

        for interval, stats in statistics.items():
            log.write(f"Interval: {interval}, Sample Count: {stats['sample_count']}, "
                    f"Correct Rate: {stats['correct_rate']:.4f}, Pred 1 Count: {stats['pred_1_count']}, "
                    f"Pred 0 Count: {stats['pred_0_count']}\n")

    return total_acc, below_conf_acc, above_conf_acc, new_acc

def eval_conf(losses, preds_classes, all_targets):

    device1 = losses.device
    loss_mean = losses.mean()
    loss_std = losses.std()
    losses = (losses - loss_mean.to(device1)) / loss_std.to(device1)

    intervals = [(float('-inf'), -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.9), (-0.9, -0.8), (-0.8, -0.7),
                 (-0.7, -0.6), (-0.6, -0.5), (-0.5, -0.4), (-0.4, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0.0),
                 (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                 (0.5, 0.52), (0.52, 0.54),(0.54, 0.56),(0.56, 0.58),(0.58, 0.6),
                 (0.6, 0.62), (0.62, 0.64),(0.64, 0.66),(0.56, 0.68),(0.68, 0.7),
                 (0.7, 0.72), (0.72, 0.74),(0.74, 0.76),(0.56, 0.78),(0.78, 0.8),
                 (0.8, 0.82), (0.82, 0.84),(0.84, 0.86),(0.56, 0.88),(0.88, 0.9),
                 (0.9, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, float('inf'))]

    statistics = {interval: {'sample_count': 0, 'correct_count': 0, 'pred_1_count': 0, 'pred_0_count': 0}
                  for interval in intervals}

    total_correct = 0 
    for i in range(len(losses)):
        loss_value = losses[i]
        pred_class = preds_classes[i]
        target_class = all_targets[i]

        if pred_class == target_class:
            total_correct += 1

        for interval in intervals:
            if interval[0] <= loss_value < interval[1]:
                statistics[interval]['sample_count'] += 1
                if pred_class == target_class:
                    statistics[interval]['correct_count'] += 1
                if pred_class == 1:
                    statistics[interval]['pred_1_count'] += 1
                if pred_class == 0:
                    statistics[interval]['pred_0_count'] += 1
                break

    total_samples = len(losses)
    acc = total_correct / total_samples if total_samples > 0 else 0

    for interval in intervals:
        sample_count = statistics[interval]['sample_count']
        if sample_count > 0:
            correct_rate = statistics[interval]['correct_count'] / sample_count
        else:
            correct_rate = 0
        statistics[interval]['correct_rate'] = correct_rate

    lowest_interval = None
    for interval in intervals:
        if statistics[interval]['sample_count'] > 0 and statistics[interval]['correct_rate'] < acc:
            lowest_interval = interval
            break

    if lowest_interval:
        if lowest_interval[0] == float('-inf'):
            mid_value = -2
        else: mid_value = (lowest_interval[0] + lowest_interval[1]) / 2
    else:
        mid_value = float('inf') 

    with open(log_file, "a") as log:
        for interval, stats in statistics.items():
            log.write(f"Interval: {interval}, Sample Count: {stats['sample_count']}, Correct Rate: {stats['correct_rate']}, "
                    f"Pred 1 Count: {stats['pred_1_count']}, Pred 0 Count: {stats['pred_0_count']}\n")

        log.write(f"Total Accuracy: {acc:.4f}\n")

        if mid_value != float('inf'):
            log.write(f"Mid Value of Lowest Interval Below Total Accuracy: {mid_value}\n")
        else:
            log.write("No interval found with correct rate below total accuracy.\n")


    return acc, mid_value

def compute_metrics(preds, targets):
        ba = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        return {
            "BA": ba,
            "F1": f1,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
        }

def evaluate_sample_groups_and_save_to_excel(pred1, preds_classes, all_targets, mode, filename="sample_metrics_train.xlsx"):

    selected_indices = np.where(pred1)[0]
    not_selected_indices = np.where(~pred1)[0]

    preds_selected = preds_classes[selected_indices]
    targets_selected = all_targets[selected_indices]

    preds_not_selected = preds_classes[not_selected_indices]
    targets_not_selected = all_targets[not_selected_indices]

    metrics_selected = compute_metrics(preds_selected, targets_selected)

    metrics_not_selected = compute_metrics(preds_not_selected, targets_not_selected)

    total_samples = len(selected_indices) + len(not_selected_indices)
    weighted_ba = (metrics_selected["BA"] * len(selected_indices) + metrics_not_selected["BA"] * len(not_selected_indices)) / total_samples
    weighted_f1 = (metrics_selected["F1"] * len(selected_indices) + metrics_not_selected["F1"] * len(not_selected_indices)) / total_samples
    weighted_tp = metrics_selected["TP"] + metrics_not_selected["TP"]
    weighted_fn = metrics_selected["FN"] + metrics_not_selected["FN"]
    weighted_fp = metrics_selected["FP"] + metrics_not_selected["FP"]
    weighted_tn = metrics_selected["TN"] + metrics_not_selected["TN"]

    data = {
        "mode": [mode] * 3,  
        "group": ["selected_group", "not_selected_group", "total"],  
        "BA": [metrics_selected["BA"], metrics_not_selected["BA"], weighted_ba],
        "F1": [metrics_selected["F1"], metrics_not_selected["F1"], weighted_f1],
        "TP": [metrics_selected["TP"], metrics_not_selected["TP"], weighted_tp],
        "FN": [metrics_selected["FN"], metrics_not_selected["FN"], weighted_fn],
        "FP": [metrics_selected["FP"], metrics_not_selected["FP"], weighted_fp],
        "TN": [metrics_selected["TN"], metrics_not_selected["TN"], weighted_tn],
        "num_samples": [len(selected_indices), len(not_selected_indices), total_samples]  
    }

    df = pd.DataFrame(data)

    try:
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)  
    except FileNotFoundError:
        pass  

    df.to_excel(filename, index=False)
    with open(log_file, "a") as log:
        log.write(f"Metrics successfully written to {filename}  (mode={mode}) \n")


def get_pred1(args, losses_conf, mid_value, preds_classes, preds_classes_s, all_targets, mode):

    if args.clean_get == 1:
        threshold = np.percentile(losses_conf, 33.33)
        pred1 = losses_conf < threshold
    elif args.clean_get == 2:
        threshold = np.percentile(losses_conf, 66.67)
        pred1 = losses_conf < threshold
    elif args.clean_get == 3:
        pred1 = losses_conf < mid_value
    elif args.clean_get == 4:
        preds_classes = preds_classes.squeeze()
        preds_classes_s = preds_classes_s.squeeze()
        pred1 = (preds_classes == all_targets) & (preds_classes_s == all_targets)
        pred1 = pred1.cpu().numpy()
        pred1 = pred1.flatten()
    elif args.clean_get == 5:
        preds_classes = preds_classes.squeeze()
        preds_classes_s = preds_classes_s.squeeze()
        pred1 = (preds_classes == all_targets) | (preds_classes_s == all_targets)
        pred1 = pred1.cpu().numpy()
        pred1 = pred1.flatten()
    else:
        raise ValueError("Invalid value for args.clean_get")

    evaluate_sample_groups_and_save_to_excel(pred1, preds_classes, all_targets, mode)
    return pred1

# Training
def train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, warm_up, savelog=False):

    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, targets_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, targets_u = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        labels_x = labels_x.long()
        targets_u = targets_u.long()
        w_x = w_x.mean().item()

        inputs_x, inputs_x2, labels_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device)
        inputs_u, inputs_u2, targets_u = inputs_u.to(device), inputs_u2.to(device), targets_u.to(device)

        optimizer.zero_grad()
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([labels_x, labels_x, targets_u, targets_u], dim=0)

        logits = net(all_inputs)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]

        Lx = CEloss(logits_x, all_targets[:batch_size*2]) 
        Lu = CEloss(logits_u, all_targets[batch_size*2:])    

        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))


        loss = Lx + Lu * linear_rampup(epoch,warm_up) + penalty


        loss.backward()
        optimizer.step()


def warmup(epoch, net, optimizer, dataloader, savelog=False):
    net.train()
    running_loss = 0
    num_iter = len(dataloader)  

    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.long().to(device)  

        optimizer.zero_grad()
        outputs = net(inputs)  
        loss = CEloss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


def test(epoch,net1, mid_value, acc):
    net1.eval()
    losses = torch.zeros(len(test_loader.dataset))
    all_targets = torch.zeros(len(test_loader.dataset))
    preds = torch.zeros(len(test_loader.dataset))
    preds_classes = torch.zeros(len(test_loader.dataset), 1)
    all_name = np.empty(len(eval_loader.dataset), dtype=object)
    flipped_losses = torch.zeros(len(test_loader.dataset))
    num_iter = len(test_loader)

    all_loss_values = []  
    all_acc_values = []   

    test_acc = 0
    total = 0  

    with torch.no_grad():
        for batch_idx, (weak, img_list, img, targets, index, name) in enumerate(test_loader):
            weak, targets = weak.to(device), targets.to(device).float()
            img = img.to(device)
            for i in range(len(img_list)):
                img_list[i] = img_list[i].to(device)
            loss = 0
            outputs_w = net1(weak)
            outputs_w = outputs_w * (args.T)

            outputs_y = net1(img)
            outputs_y = outputs_y * (args.T)

            for i in range(len(img_list)):
                outputs_s = net1(img_list[i])
                outputs_s = outputs_s * (args.T)
                loss += CE(outputs_s, outputs_w)
            loss /= len(img_list)

            pred = torch.argmax(outputs_y, dim=1)
            acc = (pred == targets).sum().item() 
            test_acc += acc
            total += targets.size(0)

            for b in range(weak.size(0)):
                losses[index[b]] = loss[b].item() 
                preds_classes[index[b]] = pred[b].item()
                all_targets[index[b]] = targets[b].item()
                all_name[index[b]] = name[b]

    test_conf(losses, preds_classes, all_targets, all_name, mid_value)

    device1 = losses.device
    loss_mean = losses.mean()
    loss_std = losses.std()
    losses = (losses - loss_mean.to(device1)) / loss_std.to(device1)

    loss_sorted, indices_sorted = torch.sort(losses)
    num_samples = len(losses)
    chunk_size = num_samples // 5

    metrics_per_chunk = []
    for i in range(5):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 4 else num_samples 

        chunk_indices = indices_sorted[start_idx:end_idx]

        preds_chunk = preds_classes[chunk_indices]
        targets_chunk = all_targets[chunk_indices]

        metrics_chunk = compute_metrics(preds_chunk, targets_chunk)
        metrics_chunk['epoch'] = epoch 
        metrics_chunk['Chunk'] = f"Chunk {i+1}"
        metrics_per_chunk.append(metrics_chunk)

    total_metrics = compute_metrics(preds_classes, all_targets)
    total_metrics['epoch'] = epoch  
    total_metrics['Chunk'] = 'Total'
    metrics_per_chunk.append(total_metrics)

    df = pd.DataFrame(metrics_per_chunk)

    filename = "sample_metrics_eval.xlsx"
    try:
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True)  
    except FileNotFoundError:
        pass  

    df.to_excel(filename, index=False)
    with open(log_file, "a") as log:
        log.write(f"Metrics successfully written to {filename}\n")

    accuracy = 100. * test_acc / total

    return accuracy

def eval_train(epoch, model, all_loss, all_preds, all_hist):
    model.eval()  
    losses_conf = torch.zeros(len(eval_loader.dataset))
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), 1)
    preds_classes_s = torch.zeros(len(eval_loader.dataset), 1)
    all_name = np.empty(len(eval_loader.dataset), dtype=object)
    all_targets = torch.zeros(len(eval_loader.dataset))
    train_acc = 0
    total = 0  
    num_iter = len(eval_loader)

    with torch.no_grad():  
        for batch_idx, (inputs1, inputs_list, targets, index, name) in enumerate(eval_loader):
            inputs1, targets = inputs1.to(device), targets.long().to(device)
            for i in range(len(inputs_list)):
                inputs_list[i] = inputs_list[i].to(device)

            outputs_w = model(inputs1)
            outputs_w = outputs_w * (args.T)

            loss = 0
            outputs_sum = 0
            for i in range(len(inputs_list)):
                output_s = model(inputs_list[i])
                output_s = output_s * (args.T)
                outputs_sum += output_s
                loss += CE(output_s, outputs_w)

            loss /= len(inputs_list)
            outputs_sum /= len(inputs_list)
            pred = torch.argmax(outputs_w, dim=1)
            pred_s = torch.argmax(outputs_sum, dim=1)
            acc = (pred == targets).sum().item()  
            train_acc += acc
            total += targets.size(0)  

            eval_preds = torch.sigmoid(outputs_w).to(device).data  

            for b in range(inputs1.size(0)):
                losses_conf[index[b]] = loss[b]  
                preds_classes[index[b]] = pred[b].item() 
                preds_classes_s[index[b]] = pred_s[b].item()
                all_targets[index[b]] = targets[b].item()
                all_name[index[b]] = name[b]

    acc_c, mid_value = eval_conf(losses_conf, preds_classes, all_targets)


    all_loss.append(losses_conf)
    all_hist.append(preds_classes)

    device1 = losses_conf.device
    loss_mean = losses_conf.mean()
    loss_std = losses_conf.std()
    losses_conf = (losses_conf - loss_mean.to(device1)) / loss_std.to(device1)

    accuracy = 100. * train_acc / total
    pred1 = get_pred1(args, losses_conf, mid_value, preds_classes, preds_classes_s, all_targets, mode = "Train")
    pred1 = pred1.cpu().numpy()
    pred1 = pred1.flatten()

    return pred1, losses_conf, all_loss, all_preds, all_hist, mid_value, accuracy

def create_model():
    if args.model == 'DenseNet-161':
        model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    elif args.model == 'VGG-16':
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    elif args.model == 'ResNet-50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    elif args.model == 'SwinT':
        model = model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        num_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    elif args.model == 'ViTB32':
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2) 
        )
        for param in model.heads.head.parameters():
            param.requires_grad = True
    model = model.to(device)
    return model


warm_up = 10
loader = dataloader.DataLoaderManager(args.dataset, batch_size=args.batch_size,num_workers=4,\
    root_dir=args.data_path, add_num = args.add_num)

test_loader = loader.run('test1')
eval_loader = loader.run('eval_train')

net1 = create_model()
cudnn.benchmark = True

if args.model == 'DenseNet-161':
    optimizer1 = optim.Adam(net1.classifier.parameters(), lr=0.001)
elif args.model == 'VGG-16':
    optimizer1 = optim.Adam(net1.classifier.parameters(), lr=0.001)
elif args.model == 'SwinT':
    optimizer1 = optim.Adam(net1.head.parameters(), lr=0.001)
elif args.model == 'ViTB32':
    optimizer1 = optim.Adam(net1.parameters(), lr=0.001)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

resume_epoch = 0

all_superclean = [[]]
all_idx_view_labeled = [[]]
all_idx_view_unlabeled = [[]]
all_preds = [[]]
hist_preds = [[]]
acc_hist = []
all_loss = [[]] 

warm_up = 10
best_acc = 0
acc_eval = 0

early_stop = False  
patience = 10  
counter = 0  
save_path1 = "./model/%s_%s_%s_%s_model1.pth"%(args.model, args.clean_get, args.add_num, args.seed)

for epoch in range(resume_epoch, args.num_epochs+1):
    with open(log_file, "a") as log:
        log.write(f"epoch: {epoch}\n")
    lr=args.lr

    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr

    if epoch<warm_up:
        warmup_trainloader = loader.run('warmup')

        warmup(epoch,net1,optimizer1,warmup_trainloader, savelog=True)

        #save histogram

        pred1, prob1, all_loss[0], all_preds[0], hist_preds[0], mid_value1, acc_1 = eval_train(epoch, net1, all_loss[0], all_preds[0], hist_preds[0])

        idx_view_labeled = (pred1).nonzero()[0]
        idx_view_unlabeled = (1-pred1).nonzero()[0]
        all_idx_view_labeled[0].append(idx_view_labeled)
        all_idx_view_unlabeled[0].append(idx_view_unlabeled)


    else:
        pred1, prob1, all_loss[0], all_preds[0], hist_preds[0], mid_value1, acc_1 = eval_train(epoch, net1, all_loss[0], all_preds[0], hist_preds[0])
        idx_view_labeled = (pred1).nonzero()[0]
        idx_view_unlabeled = (1-pred1).nonzero()[0]
        all_idx_view_labeled[0].append(idx_view_labeled)
        all_idx_view_unlabeled[0].append(idx_view_unlabeled)

        superclean = []
        nclean = args.num_clean
        for ii in range(len(eval_loader.dataset)):
            clean_lastn = True
            for h_ep in all_idx_view_labeled[0][-nclean:]:   #check last nclean epochs
                if ii not in h_ep:
                    clean_lastn = False
                    break
            if clean_lastn:
                superclean.append(ii)
        all_superclean[0].append(superclean)
        pred1 = np.array([True if p in superclean else False for p in range(len(pred1))])

        labeled_trainloader, unlabeled_trainloader, _ = loader.run('train',pred1,prob1) 
        labeled_len = len(labeled_trainloader.dataset) if hasattr(labeled_trainloader, 'dataset') else len(labeled_trainloader)
        unlabeled_len = len(unlabeled_trainloader.dataset) if hasattr(unlabeled_trainloader, 'dataset') else len(unlabeled_trainloader)

        with open(log_file, "a") as log:
            log.write(f"Labeled trainloader length: {labeled_len}, Unlabeled trainloader length: {unlabeled_len}\n")
        train(epoch,net1,optimizer1,labeled_trainloader, unlabeled_trainloader, warm_up, savelog=True) 

    mid_value = mid_value1
    acc_3 = test(epoch, net1, mid_value, best_acc)
    if epoch > 10:
        if acc_3 > best_acc:
            best_acc = acc_3 
            counter = 0  

            torch.save(net1.state_dict(), save_path1)
        else:
            counter += 1 

        if counter >= patience:
            with open(log_file, "a") as log:
                log.write(f"Early stopping trigger! Stop training at epoch {epoch}, the best accuracy is: {best_acc}\n")
            early_stop = True
            break

from sklearn.metrics import f1_score

def test_conf1(losses, preds_classes, all_targets, all_name, conf, log_file):
    extracted_predictions_file = "./excel/information_mkl.xlsx"
    extracted_predictions = pd.read_excel(extracted_predictions_file)

    device1 = losses.device
    loss_mean = losses.mean()
    loss_std = losses.std()
    losses = (losses - loss_mean.to(device1)) / loss_std.to(device1)

    threshold = torch.quantile(losses, 0.8)

    high_loss_samples = (losses >= threshold).nonzero(as_tuple=True)[0]
    low_loss_samples = (losses < threshold).nonzero(as_tuple=True)[0]  


    new_preds_classes = preds_classes.clone()
    original_high_loss_correct = 0  
    replaced_high_loss_correct = 0  

    for i in high_loss_samples:
        if preds_classes[i] == all_targets[i]:
            original_high_loss_correct += 1

        name = all_name[i]  
        if name in extracted_predictions['name'].values:
            new_value = extracted_predictions.loc[extracted_predictions['name'] == name, 'value'].values[0]
            new_preds_classes[i] = new_value
        else:
            print("not find name")


        if new_preds_classes[i] == all_targets[i]:
            replaced_high_loss_correct += 1

    high_loss_sample_count = len(high_loss_samples)
    original_high_loss_acc = original_high_loss_correct / high_loss_sample_count if high_loss_sample_count > 0 else 0
    replaced_high_loss_acc = replaced_high_loss_correct / high_loss_sample_count if high_loss_sample_count > 0 else 0

    low_loss_correct = 0
    for i in low_loss_samples:
        if preds_classes[i] == all_targets[i]:
            low_loss_correct += 1
    low_loss_acc = low_loss_correct / len(low_loss_samples) if len(low_loss_samples) > 0 else 0

    original_total_correct = 0
    for i in range(len(preds_classes)):
        if preds_classes[i] == all_targets[i]:
            original_total_correct += 1
    original_total_acc = original_total_correct / len(preds_classes)

    replaced_total_correct = 0
    for i in range(len(new_preds_classes)):
        if new_preds_classes[i] == all_targets[i]:
            replaced_total_correct += 1
    replaced_total_acc = replaced_total_correct / len(new_preds_classes)

    original_high_loss_f1 = f1_score(all_targets[high_loss_samples], preds_classes[high_loss_samples], average='binary') if len(high_loss_samples) > 0 else 0
    replaced_high_loss_f1 = f1_score(all_targets[high_loss_samples], new_preds_classes[high_loss_samples], average='binary') if len(high_loss_samples) > 0 else 0
    total_f1_original = f1_score(all_targets, preds_classes, average='binary')
    total_f1_replaced = f1_score(all_targets, new_preds_classes, average='binary')

    with open(log_file, "a") as log:
        log.write(f"High-loss samples count: {high_loss_sample_count}\n")
        log.write(f"High-loss accuracy before replacement: {original_high_loss_acc:.4f}\n")
        log.write(f"High-loss accuracy after replacement: {replaced_high_loss_acc:.4f}\n")
        log.write(f"Low-loss accuracy: {low_loss_acc:.4f}\n")
        log.write(f"Total accuracy before replacement: {original_total_acc:.4f}\n")
        log.write(f"Total accuracy after replacement: {replaced_total_acc:.4f}\n")
        log.write(f"High-loss F1 score before replacement: {original_high_loss_f1:.4f}\n")
        log.write(f"High-loss F1 score after replacement: {replaced_high_loss_f1:.4f}\n")
        log.write(f"Total F1 score before replacement: {total_f1_original:.4f}\n")
        log.write(f"Total F1 score after replacement: {total_f1_replaced:.4f}\n")

    return (original_high_loss_acc, replaced_high_loss_acc, low_loss_acc,
            original_total_acc, replaced_total_acc,total_f1_replaced)


def write_metrics_to_excel(add_num, low_loss_acc, cv_high_acc, brain_high_acc,
                           original_total_acc, replaced_total_acc, total_f1_replaced, ba, f1, auc, file_path):
    metrics_data = {
        "Model": [args.model],
        "Data enhancement range": [args.add_num],
        "Clean set distinction method": [args.clean_get],
        "CV part accuracy": [low_loss_acc],
        "CV accuracy of EEG part": [cv_high_acc],
        "Corresponding part accuracy of EEG": [brain_high_acc],
        "Test set accuracy": [original_total_acc],
        "Correctness after EEG correction": [replaced_total_acc],
        "F1 after EEG correction": [total_f1_replaced],
        "BA": [ba],
        "F1": [f1],
        "AUC": [auc]
    }

    df = pd.DataFrame(metrics_data)

    try:
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)  
    except FileNotFoundError:
        pass  

    df.to_excel(file_path, index=False)
    with open(log_file, "a") as log:
        log.write(f"Metrics successfully written to {file_path}\n")

def test1(net1, mid_value, acc):
    net1.eval()
    probs = torch.zeros(len(test_loader1.dataset))
    losses = torch.zeros(len(test_loader1.dataset))
    all_targets = torch.zeros(len(test_loader1.dataset))
    preds = torch.zeros(len(test_loader1.dataset))
    preds_classes = torch.zeros(len(test_loader1.dataset), 1)
    all_name = np.empty(len(test_loader1.dataset), dtype=object)
    flipped_losses = torch.zeros(len(test_loader1.dataset))
    num_iter = len(test_loader1)

    all_loss_values = []  
    all_acc_values = []   

    test_acc = 0
    total = 0  

    with torch.no_grad():
        for batch_idx, (weak, img_list, img, targets, index, name) in enumerate(test_loader1):
            weak, targets = weak.to(device), targets.to(device).float()
            img = img.to(device)
            for i in range(len(img_list)):
                img_list[i] = img_list[i].to(device)
            loss = 0
            outputs_w = net1(weak)
            outputs_w = outputs_w * (args.T)

            outputs_y = net1(img)
            outputs_y = outputs_y * (args.T)

            for i in range(len(img_list)):
                output_s = net1(img_list[i])
                output_s = output_s * (args.T)
                loss += CE(output_s, outputs_w)
            loss /= len(img_list)

            pred = torch.argmax(outputs_y, dim=1)
            acc = (pred == targets).sum().item() 
            test_acc += acc
            total += targets.size(0)  
            preds_probs = torch.softmax(outputs_y, dim=1)[:, 1]

            for b in range(weak.size(0)):
                losses[index[b]] = loss[b].item()  
                probs[index[b]] = preds_probs[b].item()
                preds_classes[index[b]] = pred[b].item()  
                all_targets[index[b]] = targets[b].item()
                all_name[index[b]] = name[b]

    cv_high_acc, brain_high_acc, low_loss_acc, original_total_acc, replaced_total_acc,total_f1_replaced = test_conf1(losses, preds_classes, all_targets, all_name, mid_value, log_file)

    all_targets_np = all_targets.cpu().numpy()
    preds_classes_np = preds_classes.cpu().numpy().flatten()
    probs_np = probs.cpu().numpy()

    add_num = args.add
    auc = metrics.roc_auc_score(all_targets_np, probs_np)
    f1 = metrics.f1_score(all_targets_np, preds_classes_np, average='binary')
    ba = metrics.balanced_accuracy_score(all_targets_np, preds_classes_np)
    acc = metrics.accuracy_score(all_targets_np, preds_classes_np)

    with open(log_file, "a") as log:
        log.write(f"AUC: {auc:.4f}\n")
        log.write(f"F1 Score: {f1:.4f}\n")
        log.write(f"Balanced Accuracy (BA): {ba:.4f}\n")
        log.write(f"Accuracy (ACC): {acc:.4f}\n")

    file_path = "./result/total_data.xlsx"
    write_metrics_to_excel(add_num, low_loss_acc, cv_high_acc, brain_high_acc, original_total_acc, replaced_total_acc, total_f1_replaced, ba, f1, auc, file_path)

    return replaced_total_acc

net1.load_state_dict(torch.load(save_path1))

test_loader1 = loader.run('test2')

current_ba = test1(net1, 0, 0)

def update_model_and_score(model_name, current_ba, net1, json_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(log_file, "a") as log:
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                model_scores = json.load(f)
        else:
            model_scores = {}

        if model_name not in model_scores:
            model_scores[model_name] = 0

        if current_ba > model_scores[model_name]:
            log.write(f"New BA value ({current_ba}) is higher than the existing value ({model_scores[model_name]}). Updating...\n")

            net1_path = os.path.join(save_dir, f"{model_name}_net1_best.pth")

            torch.save(net1.state_dict(), net1_path)

            log.write(f"net1 model saved to: {net1_path}\n")

            model_scores[model_name] = current_ba
            with open(json_file, "w") as f:
                json.dump(model_scores, f, indent=4)
        else:
            log.write(f"Current BA value ({current_ba}) is not higher than the existing value ({model_scores[model_name]}). No update.\n")


json_file = "./result/model_best_ba.json"
save_dir = "./model_best"
update_model_and_score(args.model, current_ba, net1, json_file, save_dir)