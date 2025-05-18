import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from utility.util import *
import json
import pandas as pd

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 8

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        if self._state('use_pb') is None:
            self.state['use_pb'] = True

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()

        # compute output
        self.state['output'] = model(feature_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        with torch.no_grad():
            # compute output
            self.state['output'] = model(feature_var)
            self.state['loss'] = criterion(self.state['output'], target_var)

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]

    def on_end_epoch(self, p, r, model_name):
        pred = p.cpu().detach().numpy()
        real = r.cpu().detach().numpy()

        a, s_id, w_id, pp = top_n_recall(pred, real, 3)

        s = np.array(s_id)
        w = np.array(w_id)

        img_list = json.load(open('../data/kfashion_{}/test_anno_{}_final.json'.format(model_name, model_name), 'r'))
        img_name_list = [x['file_name'] for x in img_list]

        l_r_s = real[s]
        l_p_s = pp[s]
        df = pd.DataFrame(np.array(img_name_list)[s])
        df['real'] = list(l_r_s)
        df['pred'] = list(l_p_s)
        df.columns = ['img', 'real', 'pred']
        df.to_csv('../output/{}_correct.csv'.format(model_name))

        l_r_w = real[w]
        l_p_w = pp[w]
        df = pd.DataFrame(np.array(img_name_list)[w])
        df['real'] = list(l_r_w)
        df['pred'] = list(l_p_w)
        df.columns = ['img', 'real', 'pred']
        df.to_csv('../output/{}_wrong.csv'.format(model_name))

        return round(a, 4)

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, val_dataset, model_name):

        self.init_learning(model, criterion)

        # define train and val transform
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'], map_location=torch.device('cpu'))
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            val_loader.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            criterion = criterion.cuda()

        if self.state['evaluate']:
            a = self.validate(val_loader, model, criterion, model_name)
            print('Top-3 recall:', a)
            return a

    def validate(self, data_loader, model, criterion, model_name):

        # switch to evaluate mode
        model.eval()

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(non_blocking = True)

            self.on_forward(False, model, criterion, data_loader)


        a = self.on_end_epoch(self.state['output'], self.state['target'], model_name)

        return  a

