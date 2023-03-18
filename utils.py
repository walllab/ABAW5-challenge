from dataset_utils import videoEmotionDataset, videoChunkEmotionDataset, test_transforms
import numpy as np
from collections import defaultdict
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from scipy.signal import medfilt

def separate_params(model, weight_decay, lr):
    # Separate weight decay and other params
    wd_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            other_params.append(param)
        elif 'bn' in name:
            wd_params.append(param)
        elif 'bias' in name:
            other_params.append(param)
        else:
            wd_params.append(param)

    params = [{'params': wd_params, 'weight_decay':weight_decay, 'lr':lr},
              {'params': other_params, 'weight_decay':0. , 'lr':lr}]
    
    assert len(wd_params) + len(other_params) == len([x for x in model.parameters()])
    return params

def calculate_scores(logits, gt):
    logits = logits[(gt>=0)]
    gt = gt[(gt>=0)]
    
    if len(gt) == 0:
        return None, None, None
    
    preds = logits.argmax(1)
    acc = accuracy_score(preds, gt)
    f1 = f1_score(preds, gt, average='macro')
    cm = confusion_matrix(gt,preds,labels=np.arange(8))
    return acc, f1, cm

def test_on_affwild2(model, df, device, lpf=False, window_size=7):
    stats_dict = defaultdict(dict)
    logits_list, gt_list = [], []
    for name,vid_df in tqdm(df.groupby('videoname')):
        vid_dataset = videoEmotionDataset(vid_df, transforms=test_transforms)
        if len(vid_dataset) == 0:
            print(f'skipping {name}')
            continue
        logits, gt = predict_on_dataset(model, vid_dataset, device, batch_size=1024)
        if lpf:
            logits = medfilt(logits, kernel_size=[window_size,1])
        acc, f1, cm = calculate_scores(logits, gt)
        stats_dict['acc'][name] = acc
        stats_dict['f1'][name] = f1
        stats_dict['confusion_matrix'][name] = cm
        logits_list.append(logits)
        gt_list.append(gt)

    logits = np.concatenate(logits_list)
    gt = np.concatenate(gt_list)
    acc, f1, cm = calculate_scores(logits, gt)
    stats_dict['acc']['overall'] = acc
    stats_dict['f1']['overall'] = f1
    stats_dict['confusion_matrix']['overall'] = cm

    df = pd.DataFrame(stats_dict).dropna()
    return df

def adapt_and_test_on_affwild2(proto_model, state_dict, df, device,
                               method='adapt_bn_stats',
                               lr=1e-3,
                               max_steps=20,
                               momentum=0.005,
                               chunks=False,
                               batch_size=128,
                               lpf=False,
                               window_size=7,
                               on_logits=False,
                               entropy_multiplier=True,
):
    stats_dict = defaultdict(dict)
    logits_list, gt_list = [], []
    for name,vid_df in tqdm(df.groupby('videoname')):
        vid_dataset = videoEmotionDataset(vid_df, transforms=test_transforms)
        if len(vid_dataset) == 0:
            print(f'skipping {name}')
            continue
        
        if chunks:
            adapt_dataset = videoChunkEmotionDataset(vid_df, ignore_invalids=False, transforms=test_transforms, batch_size=batch_size)
        else:
            adapt_dataset = videoEmotionDataset(vid_df, ignore_invalids=False, transforms=test_transforms)

        model = proto_model()
        model.load_state_dict(state_dict)
        model.to(device)

        # Update only batchnorm stats
        if method == 'adapt_bn_stats':
            model = configure_model(model, momentum=momentum)
            model = bn_adapt(model, adapt_dataset, device, max_steps=max_steps)

        elif method == 'temporal_smoothness_bn_freeze_running_stats':
            model = configure_model(model, train_only_bn=True, freeze_and_use_running_stats=True)
            params, _ = collect_bn_params(model)

            opt = torch.optim.Adam(params, lr=lr)
            model = temporal_smoothness(model, opt, adapt_dataset, device,
                                        batch_size=batch_size,
                                        max_steps=max_steps,
                                        window_size=window_size,
                                        on_logits=on_logits,
                                        entropy_multiplier=entropy_multiplier)

        model.eval()
        logits, gt = predict_on_dataset(model, vid_dataset, device, batch_size=1024)

        if lpf:
            logits = medfilt(logits, kernel_size=[window_size,1])
        
        model.to('cpu')
        acc, f1, cm = calculate_scores(logits, gt)
        stats_dict['acc'][name] = acc
        stats_dict['f1'][name] = f1
        stats_dict['confusion_matrix'][name] = cm
        logits_list.append(logits)
        gt_list.append(gt)

    logits = np.concatenate(logits_list)
    gt = np.concatenate(gt_list)
    acc, f1, cm = calculate_scores(logits, gt)
    stats_dict['acc']['overall'] = acc
    stats_dict['f1']['overall'] = f1
    stats_dict['confusion_matrix']['overall'] = cm
    
    df = pd.DataFrame(stats_dict).dropna()
    return df

def bn_adapt(model, dataset, device, opt=None, num_epochs=1, batch_size=512, max_steps=3):
    # Update only batchnorm statistics
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    step_counter = 0
    break_flag = False
    while True:
        for images, _, _ in dataloader:
            images = images.to(device)
            with torch.no_grad():
                logits = model(images)
            step_counter += 1
            if step_counter == max_steps:
                break_flag = True
                break

        if break_flag:
            break
    return model

def predict_on_dataset(model, dataset, device, batch_size=256, shuffle=False, return_frame_nums=False):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6)
    batch_logits, batch_gt, batch_frame_nums = [], [], []
    for images, labels, frame_nums in dataloader:

        images = images.to(device)
        with torch.no_grad():
            logits = model(images)

        batch_logits.append(logits.detach().cpu().numpy())
        batch_gt.append(labels.numpy())
        batch_frame_nums.append(frame_nums.numpy())
    logits = np.concatenate(batch_logits)
    gt = np.concatenate(batch_gt)
    frame_nums = np.concatenate(batch_frame_nums)
    if return_frame_nums:
        return logits, gt, frame_nums
    else:
        return logits, gt

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def collect_bn_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model,
                    train_only_bn=True,
                    use_batch_stats=False,
                    freeze_and_use_running_stats=False,
                    momentum=0.01):
    """Configure model parameters"""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    if train_only_bn:
        model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            if use_batch_stats:
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif freeze_and_use_running_stats:
                m.track_running_stats = False
                m.training = False
            else:
                m.momentum = momentum
    return model

def temporal_smoothness(model, opt, dataset, device,
                        batch_size=256,
                        max_steps=10,
                        on_logits=True,
                        entropy_multiplier=True,
                        window_size=7):
    random_starts = isinstance(dataset, videoChunkEmotionDataset)
    if random_starts:
        dataloader = DataLoader(dataset, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    step = 0
    break_flag = False
    while True:
        for images, _, _ in dataloader:
            if step>=max_steps:
                break_flag = True
                break
                
            if random_starts:
                images = images.squeeze()

            opt.zero_grad()
            images = images.to(device)
            logits = model(images)
            preds = (logits).softmax(1)
            
            if on_logits:
                tar_signal = logits.clone().detach().cpu().numpy()
            else:
                tar_signal = preds.clone().detach().cpu().numpy()
            
            tar_signal = medfilt(tar_signal, kernel_size=[window_size,1])
            tar_signal = torch.tensor(tar_signal, device=device)
            
            if on_logits:
                loss = logits - tar_signal
            else:
                loss = preds - tar_signal

            loss = loss.pow(2)
            if entropy_multiplier:
                loss = loss * softmax_entropy(logits).unsqueeze(-1)
            loss = loss.mean()

            loss.backward()
            opt.step()
            step+=1
            
        if break_flag:
            break
    return model