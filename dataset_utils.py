import os
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

train_transforms = A.Compose([
    
    A.OneOf([
        A.Equalize(),
        A.ChannelDropout(),
        A.ColorJitter(),
        A.RandomBrightnessContrast(),
        A.OneOf([
        A.GaussianBlur(),
        A.MedianBlur(blur_limit=7),
        A.MotionBlur(),
        ]),    
    ], p=0.9),    
    
    
    A.HorizontalFlip(),    
    A.Normalize(max_pixel_value=255., always_apply=True),
#     A.Normalize(mean=0, std=1, max_pixel_value=255., always_apply=True),
    ToTensorV2(),
    
    ])

test_transforms = A.Compose([
    A.Equalize(),
    A.Normalize(max_pixel_value=255., always_apply=True),
#     A.Normalize(mean=0, std=1, max_pixel_value=255., always_apply=True),
    ToTensorV2()
])

class videoEmotionDataset(Dataset):
    def __init__(self, df, ignore_invalids=True, transforms=None):
        if ignore_invalids:
            df = df[df.exp!=-1].reset_index(drop=True)
        self.labels = np.array(df.exp).astype(np.int64)
        self.addresses = np.array(df.address)        
        self.transforms = transforms
        self.frame_nums = np.array(df.frame_num).astype(np.int64)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Select sample
        label = self.labels[idx] 
        address = self.addresses[idx]       
        frame_num = self.frame_nums[idx]
        
        image = Image.open(address)
        image = image.resize((112,112))
        image = np.array(image, dtype=np.uint8)[:,:,:3]
        
        assert image.max()>1
        
        if self.transforms:
            image = self.transforms(image=image)['image']
                
        return image, label, frame_num
    
class videoChunkEmotionDataset(Dataset):
    def __init__(self, df, ignore_invalids=True, transforms=None, batch_size=128):
        if ignore_invalids:
            df = df[df.exp!=-1].reset_index(drop=True)
        self.labels = np.array(df.exp).astype(np.int64)
        self.addresses = np.array(df.address)        
        self.transforms = transforms
        self.frame_nums = np.array(df.frame_num).astype(np.int64)

        jumps = np.where(np.diff(self.frame_nums)!=1)[0] + 1
        if len(jumps) == 0:
            jumps = np.array([0])
        elif jumps[0] != 0:
            jumps = np.concatenate([[0], jumps])
        jumps = np.concatenate([jumps, [len(self.frame_nums)-1]])
        valid_interval_starts = np.where(np.diff(jumps)>=batch_size)[0]
        if len(valid_interval_starts) == 0:
            valid_interval_starts = np.array([0])
        interval_starts = jumps[valid_interval_starts]
        interval_ends = jumps[valid_interval_starts+1]
        self.intervals = np.array(tuple(zip(interval_starts, interval_ends)))
        np.random.shuffle(self.intervals)
        
        self.jumps = jumps
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, interval_idx):
        # Select sample
        start, end = self.intervals[interval_idx]
        support = end - self.batch_size - start
        if support > 0:
            start = start + np.random.randint(support)
        end = min(start + self.batch_size, len(self.labels)-1)
        idx = np.arange(start,end)
        labels = self.labels[idx] 
        addresses = self.addresses[idx]       
        frame_nums = self.frame_nums[idx]
        images = []
        for address in addresses:
            image = Image.open(address)
            image = image.resize((112,112))
            image = np.array(image, dtype=np.uint8)[:,:,:3]

            assert image.max()>1

            if self.transforms:
                image = self.transforms(image=image)['image']
            images.append(image)
        
        return np.stack(images), labels, frame_nums

    
class emotionDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.labels = np.array(df.exp).astype(np.int64)
        self.addresses = np.array(df.address)        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Select sample
        label = self.labels[idx] 
        address = self.addresses[idx]        
        
        image = Image.open(address)
        image = image.resize((112,112))
        image = np.array(image, dtype=np.uint8)[:,:,:3]
        
        assert image.max()>1
        
        if self.transforms:
            image = self.transforms(image=image)['image']
                
        return image, label

def loadAffwild2():
    AFFWILD_FOLDER = '../../affwild2/'
    im_folder = AFFWILD_FOLDER+'cropped_aligned/'
    train_label_folder = AFFWILD_FOLDER+'annot/EXPR_Classification_Challenge/Train_Set/'
    val_label_folder = AFFWILD_FOLDER+'annot/EXPR_Classification_Challenge/Validation_Set/'

    train_vids = set([x.split('.txt')[0] for x in os.listdir(train_label_folder)])
    val_vids = set([x.split('.txt')[0] for x in os.listdir(val_label_folder)])

    train_tuples = []
    for vidname in tqdm(sorted(os.listdir(im_folder))):
        if vidname not in train_vids:
            continue
        with open(train_label_folder+f'{vidname}.txt') as labelsfile:
            lines = labelsfile.readlines()
        train_tuples.extend([(vidname, im_folder+vidname+f'/{x}', lines[int(x[:-4])].split('\n')[0], int(x[:-4])) for x in sorted(os.listdir(im_folder+vidname)) if x.endswith('.jpg')])

    val_tuples = []
    for vidname in tqdm(sorted(os.listdir(im_folder))):
        if vidname not in val_vids:
            continue
        with open(val_label_folder+f'{vidname}.txt') as labelsfile:
            lines = labelsfile.readlines()
        val_tuples.extend([(vidname, im_folder+vidname+f'/{x}', lines[int(x[:-4])].split('\n')[0], int(x[:-4])) for x in sorted(os.listdir(im_folder+vidname)) if x.endswith('.jpg')])

    val_df = pd.DataFrame(val_tuples, columns=['videoname','address','exp','frame_num'])
    train_df = pd.DataFrame(train_tuples, columns=['videoname','address','exp','frame_num'])
    train_df.exp = train_df.exp.astype(int)
    val_df.exp = val_df.exp.astype(int)

    print(f"      # Frames \nTrain : {len(train_df)} \nVal   : {len(val_df)} \nTotal : {len(train_df) + len(val_df)}")
    return train_df, val_df
