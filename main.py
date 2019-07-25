import os
import argparse
from voc12 import voc_loader
from solver import Solver

def main(args):
    if args.mode == 'train':
        train_loader = voc_loader(args.root, args.split, args.ignore_label, args.mean_bgr, args.augment, 
                                  args.base_size, args.crop_size, args.scales, args.flip, args)
        if args.val:
            val_loader = voc_loader(args.root, 'val', None, args.mean_bgr, False, False, False, False, False, args)
            train = Solver(train_loader, val_loader, None, args)
        else:
            train = Solver(train_loader, None, None, args)
        train.train()   
    elif args.model == 'test':
        test_loader = voc_loader(args.root, 'test', None, args.mean_bgr, False, False, False, False, False, args)
        test = Solver(None, None, test_loader, args)
        test.test()
    else:
        raise ValueError('mode is not available!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--name', type=str, default='voc')
    parser.add_argument('--root', type=str, default='path/to/VOCdevkit')
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--scales', type=list, default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument('--split', type=str, default='train')
    
    # image
    parser.add_argument('--mean_bgr', type=tuple,default=(122.675, 116.669, 104.008))
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--crop_size', type=int, default='321')
    parser.add_argument('--base_size', type=int, default=None)
    parser.add_argument('--flip', type=bool, default=True)
    
    # dataloader
    parser.add_argument('--num_workers', type=int, default=0)

    # solver
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")       
    parser.add_argument('--pretrain', type=str , default='./model/deeplabv2_resnet101_msc-voc.pth') 
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--epoch_val', type=int, default=2)
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--snapshot', type=str, default='./snapshots/')
    parser.add_argument('--global_counter', type=int, default=0)

    # model
    parser.add_argument('--num_blocks', type=list, default=[3, 4, 23, 3])
    parser.add_argument('--atrous_rates', type=tuple, default=(6, 12, 18, 24))
    parser.add_argument('--multi_scales', type=tuple, default=(0.5, 0.75))
    parser.add_argument('--model_name', type=str, default='Deeplabv2-ResNet101')



    args = parser.parse_args()
    main(args)
