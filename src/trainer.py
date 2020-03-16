import torch
import torch.nn as nn
import utils
import time
import sys
import os

from vgg import Vgg16
from coco_dataset import CocoDataset
from transfer_net import TransferNet
from PIL import Image
from torchvision import transforms


class Trainer(object):
    def __init__(self, args):
        ## Args
        self._train_dataset_path = args.train_dataset_path
        self._epochs = args.epochs
        self._batch_iterations = args.batch_iterations
        self._batch_size = args.batch_size
        self._learning_rate = args.learning_rate
        self._content_weight = args.content_weight
        self._style_weight = args.style_weight
        self._style_image_path = args.style_image_path
        self._log_interval = args.log_interval
        self._save_interval = args.save_interval
        self._trained_models_dir = args.trained_models_dir
        
        ## GPU if available, CPU otherwise
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._train_loader = self._get_train_loader()
        self._transfer_net = TransferNet()
        self._vgg = Vgg16().to(self._device)
        self._gram_style = self._compute_gram_from_style()

    def train(self):
        print('#------------#', flush=True)
        print('Training start', flush=True)
        print('#------------#', flush=True)
        self._start_time = time.time()
        self._transfer_net.to(self._device).train()
        self._optimizer = torch.optim.Adam(self._transfer_net.parameters(), self._learning_rate)
        self._mse_loss = nn.MSELoss()
        for epoch in range(self._epochs):
            self._current_epoch = epoch + 1
            self._run_one_epoch()
        runtime = time.time() - self._start_time
        print('#-----------#', flush=True)
        print('Training over', flush=True)
        print('#-----------#', flush=True)
        print('Total runtime: {:.2f} hours'.format((runtime / 60.) / 60.), flush=True)
    
    def _run_one_epoch(self):
        runtime = time.time() - self._start_time
        print('{:.2f}h Epoch {} starts'.format((runtime / 60.) / 60., self._current_epoch), flush=True)
        agg_content_loss = 0.
        agg_style_loss = 0.
        for i, x in enumerate(self._train_loader):
            self._optimizer.zero_grad()
            
            ## Put batch on selected device and feed it through the transfer
            ## network without normalization
            x = x.to(self._device) ## x = y_c = content target
            y = self._transfer_net(x) ## y_hat

            ## Normalize batch and transfer net output for vgg
            x = utils.normalize_batch(x)
            y = utils.normalize_batch(y)

            ## Extract features with vgg
            features_y = self._vgg(y)
            features_x = self._vgg(x)
            
            ## Losses
            ## todo: Compute Simple Loss Functions
            content_loss = self._content_loss(features_y, features_x)
            style_loss = self._style_loss(features_y)
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            total_loss = content_loss + style_loss
            
            ## Backward and optimization
            total_loss.backward()
            self._optimizer.step()
            if (i + 1) % self._log_interval == 0:
                self._log(i + 1, agg_content_loss, agg_style_loss)
                agg_content_loss, agg_style_loss = 0., 0.
            if (i + 1) % self._save_interval == 0:
                self._save_model('_i{}-{}'.format((i + 1) * self._batch_size, len(self._train_loader) * self._batch_size))
            if i + 1 >= self._batch_iterations:
                break
        runtime = time.time() - self._start_time
        print('{:.2f}h Epoch {} over'.format((runtime / 60.) / 60., self._current_epoch), flush=True)
        self._save_model(final=True)
        print('', flush=True)
    
    def _log(self, i, agg_content_loss, agg_style_loss):
        runtime = time.time() - self._start_time
        message = '{:.2f}h Epoch {}: [{}/{}] content: {:.4f} style: {:.4f} total: {:.4f}'.format(
            (runtime / 60.) / 60, self._current_epoch, i * self._batch_size, 
            len(self._train_loader) * self._batch_size,
            agg_content_loss / self._log_interval, agg_style_loss / self._log_interval, 
            (agg_content_loss + agg_style_loss) / self._log_interval)
        print(message, flush=True)
    
    def _save_model(self, optional='', final=False):
        self._transfer_net.eval().cpu()
        style_name = self._style_image_path.split('/')[-1].split('.')[0]
        prefix = ''
        if final:
            prefix = 'final_'
        model_filename = prefix + style_name + '_e' + str(self._current_epoch) + optional + '_cw' + str(int(self._content_weight)) + '_sw' + str(int(self._style_weight)) + '.pth'
        full_model_filename = os.path.join(self._trained_models_dir, model_filename)
        torch.save(self._transfer_net.state_dict(), full_model_filename)
        # print('\t', model_filename, 'saved in', self._trained_models_dir, flush=True)
        self._transfer_net.to(self._device).train()
        
    def _content_loss(self, features_y, features_x):
        ## Compute Perceptual Loss Functions
        ## Content loss: Feature Reconstruction Loss in the paper, is just a simple MSE loss between y_hat and y_c relu3_3
        return self._content_weight * self._mse_loss(features_y['relu3_3'], features_x['relu3_3'])
    
    def _style_loss(self, features_y):
        style_loss = 0.
        for ft_y, gm_s in zip(features_y.values(), self._gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += self._mse_loss(gm_y, gm_s)
        style_loss *= self._style_weight
        return style_loss
    
    def _get_train_loader(self):
        ## Load dataset and create train loader from it
        coco_dataset = CocoDataset(self._train_dataset_path)
        train_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=self._batch_size, shuffle=True)
        return train_loader

    def _compute_gram_from_style(self):
        ## Open the style image
        style_image = Image.open(self._style_image_path)
        
        ## Transform the style image to a full batch on the selected device
        transform_pipeline = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        style_image_batch = transform_pipeline(style_image)
        style_image_batch = style_image_batch.repeat(self._batch_size, 1, 1, 1).to(self._device)
        
        ## Normalize for vgg input and get features
        features_style = self._vgg(utils.normalize_batch(style_image_batch))
        
        ## Compute the style gram matrix: y_s
        gram_style = [utils.gram_matrix(y) for _, y in features_style.items()]
        return gram_style