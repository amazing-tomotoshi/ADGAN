import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # dataroot=./data/deepfashion/fashion_resize
        # phase = test or train
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_SP = opt.dirSem #semantic
        self.SP_input_nc = opt.SP_input_nc

        # pairLst = ./data/deepfashion/fashion-resize-pairs-test.csv or./data/deepfashion/fashion-resize-pairs-train.csv
        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            # pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pair = [pairs_file_train.iloc[i]['from1'], pairs_file_train.iloc[i]['from2'], pairs_file_train.iloc[i]['from3'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        # P1_name, P2_name = self.pairs[index]
        # P1_path = os.path.join(self.dir_P, P1_name) # person 1
        # BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1

        # # person 2 and its bone
        # P2_path = os.path.join(self.dir_P, P2_name) # person 2
        # BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2


        # P1_img = Image.open(P1_path).convert('RGB')
        # P2_img = Image.open(P2_path).convert('RGB')

        # BP1_img = np.load(BP1_path) # h, w, c
        # BP2_img = np.load(BP2_path) 

        # P1_img = Image.open(P1_path).convert('RGB')
        # P2_img = Image.open(P2_path).convert('RGB')

        # BP1_img = np.load(BP1_path) # h, w, c
        # BP2_img = np.load(BP2_path) 

        ## 3枚に対応
        P1_name, P2_name, P3_name, P4_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2

        P3_path = os.path.join(self.dir_P, P3_name) # person 3
        BP3_path = os.path.join(self.dir_K, P3_name + '.npy') # bone of person 3

        P4_path = os.path.join(self.dir_P, P4_name) # person 4
        BP4_path = os.path.join(self.dir_K, P4_name + '.npy') # bone of person 4


        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')
        P3_img = Image.open(P3_path).convert('RGB')
        P4_img = Image.open(P4_path).convert('RGB')

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path) 
        BP3_img = np.load(BP3_path) 
        BP4_img = np.load(BP4_path) 

        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            # BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            # BP1 = BP1.transpose(2, 0) #c,w,h
            # BP1 = BP1.transpose(2, 1) #c,h,w 

            # BP2 = torch.from_numpy(BP2_img).float()
            # BP2 = BP2.transpose(2, 0) #c,w,h
            # BP2 = BP2.transpose(2, 1) #c,h,w 

            # P1 = self.transform(P1_img)
            # P2 = self.transform(P2_img)

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            BP3 = torch.from_numpy(BP3_img).float()
            BP3 = BP3.transpose(2, 0) #c,w,h
            BP3 = BP3.transpose(2, 1) #c,h,w 

            BP4 = torch.from_numpy(BP4_img).float()
            BP4 = BP4.transpose(2, 0) #c,w,h
            BP4 = BP4.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)
            P3 = self.transform(P3_img)
            P4 = self.transform(P4_img)

        # # segmentation
        # SP1_name = self.split_name(P1_name, 'semantic_merge3')
        # SP1_path = os.path.join(self.dir_SP, SP1_name)
        # SP1_path = SP1_path[:-4] + '.npy'
        # SP1_data = np.load(SP1_path)
        # SP1 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
        # for id in range(self.SP_input_nc):
        #     SP1[id] = (SP1_data == id).astype('float32')

        # return {'P1': P1, 'BP1': BP1, 'SP1': SP1, 'P2': P2, 'BP2': BP2,
        #         'P1_path': P1_name, 'P2_path': P2_name}

        # segmentation
        SP1_name = self.split_name(P1_name, 'semantic_merge3')
        SP1_path = os.path.join(self.dir_SP, SP1_name)
        SP1_path = SP1_path[:-4] + '.npy'
        SP1_data = np.load(SP1_path)
        SP1 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
        for id in range(self.SP_input_nc):
            SP1[id] = (SP1_data == id).astype('float32')
            SP1_name = self.split_name(P1_name, 'semantic_merge3')
        
        SP2_name = self.split_name(P2_name, 'semantic_merge3')
        SP2_path = os.path.join(self.dir_SP, SP2_name)
        SP2_path = SP2_path[:-4] + '.npy'
        SP2_data = np.load(SP2_path)
        SP2 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
        for id in range(self.SP_input_nc):
            SP2[id] = (SP2_data == id).astype('float32')
        
        SP3_name = self.split_name(P3_name, 'semantic_merge3')
        SP3_path = os.path.join(self.dir_SP, SP3_name)
        SP3_path = SP3_path[:-4] + '.npy'
        SP3_data = np.load(SP3_path)
        SP3 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
        for id in range(self.SP_input_nc):
            SP3[id] = (SP3_data == id).astype('float32')

        # return {'P1': P1, 'BP1': BP1, 'SP1': SP1, 'P2': P2, 'BP2': BP2,
        #         'P1_path': P1_name, 'P2_path': P2_name}

        return {'P1': P1, 'BP1': BP1, 'SP1': SP1, 'P2': P2, 'BP2': BP2, 'SP2': SP2, 'P3': P3, 'BP3': BP3, 'SP3': SP3, 'P4': P4, 'BP4': BP4,
                'P1_path': P1_name, 'P2_path': P2_name, 'P3_path': P3_name, 'P4_path': P4_name}

                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'

    def split_name(self,str,type):
        list = []
        list.append(type)
        if (str[len('fashion'):len('fashion') + 2] == 'WO'):
            lenSex = 5
        else:
            lenSex = 3
        list.append(str[len('fashion'):len('fashion') + lenSex])
        idx = str.rfind('id0')
        list.append(str[len('fashion') + len(list[1]):idx])
        id = str[idx:idx + 10]
        list.append(id[:2]+'_'+id[2:])
        pose = str[idx + 10:]
        list.append(pose[:4]+'_'+pose[4:])

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head

