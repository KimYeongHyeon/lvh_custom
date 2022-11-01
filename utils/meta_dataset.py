

import os
import pandas as pd
import numpy as np
import collections
import cv2
from operator import index

import torch
from torch.utils.data import Dataset

import utils
from utils.heatmaps import *


class EchoDataset_Meta_heatmap():
    def __init__(self, root,
                 transforms=None,
                 split='train',
                 shot=5,
                 num_channels=1,
                 ):
        ''' 
        Inputs: 
            root(str): path to root including patient directories and labels (csv)
        
        '''
        assert split in ['train', 'val', 'test']

        self.root = root
        self.tasks = sorted(glob.glob(os.path.join(self.root, "*"))) # get all folders under root
        self.shot = shot
        self.task_label_list = []
        self.patient_list = [] 
        self.fname_list = [] 

        ## 데이터 목록 / 레이블 정보 불러오기
        for task in self.tasks:
            patient_list_per_task = glob.glob(os.path.join(task, split, '*'))
            self.patient_list.append(patient_list_per_task)
            
            data = pd.read_csv(os.path.join(self.root, 'labels.csv'), index_col=0)
            data = data[data['split']==split] # 폴더 구성을 어떻게 할지에 따라 달라짐
            data = data[data['FileName'].apply(lambda x: x in patient_list_per_task)]
            self.task_label_list.append(data)

            self.fname_list.append(data['FileName'].unique().tolist())

        self.transforms = transforms

        self.calc_dict = {
            'PLAX': ['IVSd','LVIDd','LVPWd'],
            'PSAX': [],
            '2CH': [],
            '4CH': []
        }

        self.num_channels = num_channels

    def sample(self):
        self.data = []
        self.label = []
        for task, patient_list_per_task, task_label_per_task, fname_per_task in \ 
            zip(self.tasks, self.patient_list, self.task_label_list, self.fname_list):
            
            data_idx = np.random.choice(range(len(patient_list_per_task)), self.shot, replace=False)
            sampled_fname_per_task = fname_per_task[data_idx]
            
            
        df = self.data[self.data['FileName'] == self.fname[idx]]
        df = df.sort_values(by=['Calc']).reset_index(drop=True)

        image = os.path.join(self.root, str(self.fname[idx]),
                             str(df.loc[0, 'Frame']).zfill(4)+'.png')
        image = cv2.imread(image)#, cv2.IMREAD_GRAYSCALE)
        
        ori_height, ori_width, _ = image.shape
        label = []
        for c in self.calc_list:
            try:
                c_df = df[df['Calc'] == c]
                coor = c_df[['X1', 'X2', 'Y1', 'Y2']].to_numpy().squeeze().astype(float)
                
                # xy순으로 묶음
                if c == 'IVSd':
                    label.append((coor[1], coor[3]))

                if c == 'LVIDd':
                    label.append((coor[1], coor[3]))
                    label.append((coor[0], coor[2]))
                    
                if c == 'LVPWd':
                    label.append((coor[0], coor[2]))

            except Exception as e:
                print(e)
                label.append((-1,-1))


        if self.transforms:
            transformed = self.transforms(image=image, keypoints=label)
            image, label = transformed['image'], transformed['keypoints']

        # 3 채널로 변경
        # image = np.array([image]*self.num_channels)
        data = torch.tensor(image)

        label = torch.tensor(label)
        
        # height, width = image.shape[1:]
        # label = torch.tensor(label).reshape(-1).to(torch.float32) # flatten 상태의 포인트를 예측하기 위해 reshape -1 를 수행
        sample = {
            'data':data.to(torch.float32),
            'label':label.to(torch.float32), 
            'id': df['FileName'][0],
            'height':ori_height,
            'width':ori_width
        }

        return sample
