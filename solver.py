import torch
import torch.nn as nn
import torch
from model import Deeplabv2, Model, _init_weight   
from loss import Loss
import os
from PIL import Image
import numpy as np
 

class Solver(object):
    def __init__(self, train_loader, val_loader, test_loader, args):
        self.current_epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.build_model()

    def build_model(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = Model(Deeplabv2(self.args.num_classes, self.args.num_blocks, self.args.atrous_rates), self.args.multi_scales).to(self.device)
        self.net.base.freeze_bn()
        self.loss = Loss(self.args).to(self.device)
        self.net.train()
        self.net.apply(_init_weight)
        self.optimizer = torch.optim.SGD([{"params": self.get_params(self.net, key="resnet_conv"),
                                            "lr": self.args.lr,
                                            "weight_decay": self.args.weight_decay},
                                          {"params": self.get_params(self.net, key="aspp_weight"),
                                            "lr": 10 * self.args.lr,
                                            "weight_decay": self.args.weight_decay},
                                          {"params": self.get_params(self.net, key="aspp_bias"),
                                            "lr": 20 * self.args.lr}], momentum=self.args.momentum)
        self.restore()
        self.print_param()
    
    def get_params(self, model, key):
        if key == "resnet_conv":
            for i in model.named_modules():
                if 'layer' in i[0]:
                    if isinstance(i[1], nn.Conv2d):
                        for p in i[1].parameters():
                            yield p
        elif key == "aspp_weight":
            for i in model.named_modules():
                if 'ASPP' in i[0]:
                    if isinstance(i[1], nn.Conv2d):
                        yield i[1].weight
        elif key == "aspp_bias":
            for i in model.named_modules():
                if 'ASPP' in i[0]:
                    if isinstance(i[1], nn.Conv2d):
                        yield i[1].bias

    def restore(self):
        if self.args.pretrain and os.path.isfile(self.args.pretrain):
            ckpt = torch.load(self.args.pretrain)
            self.net.base.load_state_dict(ckpt, strict=False)
        else:
            restore_dir = self.args.snapshot
            model_list = os.listdir(restore_dir)
            model_list = [x for x in model_list if os.path.isfile(os.path.join(restore_dir, x)) and x.endswith('.pth.tar')]
            if len(model_list) > 0:
                model_list.sort(key=lambda fn: os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
                snapshot = os.path.join(restore_dir, model_list[0])
            ckpt = torch.load(snapshot)
            self.net.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.current_epoch = ckpt['epoch'] + 1

    def print_param(self):
        num_params = 0
        for p in self.net.parameters():
            if p.requires_grad:
                num_params += p.numel()
        print('model:', self.args.model_name)
        print('The number of parameters: ', num_params)
    
    def adjust_lr(self, current_iter, max_iter):
        lr = self.args.lr * (1 - current_iter/max_iter ) **self.args.power
        for param in self.optimizer.param_groups:
            param['lr'] = lr
    
    def resize_labels(self, labels, h, w):
        new_labels = []
        for label in labels:
            label = label.float().numpy()
            label = Image.fromarray(label).resize((h, w), resample=Image.NEAREST)
            new_labels.append(np.asarray(label))
        new_labels = torch.LongTensor(new_labels)
        return new_labels

    def save_checkpoint(self, args, state, filename='checkpoint.pth.tar'):
        model_save_path = os.path.join(self.args.snapshot, filename)
        torch.save(state, model_save_path)
   

    def train(self):
        if self.args.val:
            best_val = 1.0
        else:
            best_val = None
        global_counter = self.args.global_counter
        max_iter = int(self.args.epochs * len(self.train_loader))

        for epoch in range(self.current_epoch, self.args.epochs):
            epoch = self.current_epoch
            epoch_loss = 0.0
            current_iteration = 0
            for i, (_, images, labels) in enumerate(self.train_loader):
                if (i + 1) > max_iter: break
                current_iteration += 1       
                global_counter += 1               # for calculate loss
                self.adjust_lr(global_counter, max_iter)
                images = images.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(images)
                loss = 0.0
                for logit in logits:
                    _, _, h, w = logit.shape
                    labels_ = self.resize_labels(labels, h, w)
                    loss += self.loss(logit, labels_.to(self.device))
                loss /=(self.args.batch_size * 4)  # 
                loss.backward() 
                self.optimizer.step()
                epoch_loss += loss.item()
                print("epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]" %(epoch, self.args.epochs, current_iteration, len(self.train_loader), loss.item()))
                
            if (epoch + 1) % self.args.epoch_val == 0 and self.args.val:
                val_loss = self.val()
                print('---Best MAE: %.2f, Curr MAE: %.2f ---' %(best_val, val_loss))
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(args=self.args, state={'epoch':epoch,
                                            'epoch_loss':val_loss,
                                            'state_dict':self.net.state_dict(),
                                            'optimizer':self.optimizer.state_dict()}, filename="best.pth.tar")
            if (epoch + 1) % self.args.epoch_save == 0:
                self.save_checkpoint(args=self.args, state={'epoch':epoch,
                                            'epoch_loss':epoch_loss,
                                            'state_dict':self.net.state_dict(),
                                            'optimizer':self.optimizer.state_dict()}, filename="epoch_%d.pth.tar" %(epoch+1))
        torch.save(self.net.state_dict(), "%s/final.pth" %(self.args.snapshot))


            
    def val(self):
        return 1
    def val_metric(self):
        pass

    def test(self):
        global_counter = self.args.global_counter
        with torch.no_grad():
            for i, (_, images, labels, img_path) in enumerate(self.test_loader):
            
                images = images.to(self.device)
                logits = self.net(images)
                for idx in range(self.args.batch_size):
                    img_name = img_path[idx].split('/')[-1].split('.')[0]
                    output_predictions = logits[idx].argmax(0)
                
                    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
                    colors = (colors % 255).numpy().astype("uint8")
                    output_predictions =  output_predictions.cpu().numpy()
                    r = Image.fromarray(np.uint8(output_predictions))  # I don't know the reason that it need to convert to uin8 here.
                    r = r.resize((321, 321))
                    
                    
                    r.putpalette(colors)
                    r.save(os.path.join('./save_bins', img_name + '.png'))
                    # prob_final = prob_final * 255
                    
                    global_counter += 1
                    print('Has finished %d images' %global_counter)
