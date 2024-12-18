################
#script to augment features with CLIP

import pickle
import os
import clip
import torch
import network
import torch.nn as nn
from utils.stats import calc_mean_std
import argparse
from main import get_dataset
from torch.utils import data
import numpy as np
import random
from PIL import Image
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
prompt_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a low resolution photo of the {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a photo of my {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a pixelated photo of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'a dark photo of a {}.',
    'itap of my {}.',
    'a busy city street with {}.', 
    'an open highway with {}.', 
    'a scenic city view with {}.', 
    'a suburban neighborhood with {}.', 
    'a quiet street with {}.', 
    'a busy intersection with {}.', 
    'an industrial area with {}.', 
    'a crowded shopping district with {}.', 
     'a clear day view of the city with {}.',
     'a pedestrian crossing on a busy street with {}.',
     'a busy city street with {}.',
    'a highway filled with {}.',
    'a quiet suburban street with {}.',
    'a busy intersection crowded with {}.',
    'a bustling city center full of {}.',
    'a pedestrian street lined with {}.',
    'a commercial district bustling with {}.',
    'a residential area featuring {}.',
    'a scenic drive through {}.',
    'a vibrant marketplace with {}.',
    'a busy avenue lined with {}.',
    'a city square bustling with {}.',
    'a quiet alleyway with {}.',
    'a suburban neighborhood featuring {}.',
    'a busy road showcasing {}.',
    'a city park with people enjoying {}.',
    'a bustling shopping district with {}.',
    'a busy roundabout with {}.',
    'a coastal road offering views of {}.'
]


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to dataset")
    parser.add_argument("--save_dir", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','gta5'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default = 'RN50',
                        help= "backbone name" )
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type = int, default =100,
                        help= "total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False,
                        help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # target domain description
    parser.add_argument("--domain_desc", type=str , default = "driving at night.",
                        help = "description of the target domain")
    # loss weight
    parser.add_argument("--loss_weight", type=float, default=0.5,
                        help="Weight for image target loss in combined loss calculation (a)")

    return parser



class PIN(nn.Module):
    def __init__(self,shape,content_feat):
        super(PIN,self).__init__()
        self.shape = shape
        self.content_feat = content_feat.clone().detach()
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)
        self.size = self.content_feat.size()
        self.content_feat_norm = (self.content_feat - self.content_mean.expand(
        self.size)) / self.content_std.expand(self.size)

        self.style_mean =   self.content_mean.clone().detach() 
        self.style_std =   self.content_std.clone().detach()

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True)
        self.style_std = nn.Parameter(self.style_std, requires_grad = True)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        
        self.style_std.data.clamp_(min=0)
        target_feat =  self.content_feat_norm * self.style_std.expand(self.size) + self.style_mean.expand(self.size)
        target_feat = self.relu(target_feat)
        return target_feat

def main():

    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,data_aug=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19,BB= opts.BB,replace_stride_with_dilation=[False,False,False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56,56))
    else:
        t1 = lambda x:x

    #text
    #target text
    target = compose_text_with_templates(opts.domain_desc, imagenet_templates)

    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size,1).type(torch.float32)  # (B,1024)

    for i,(img_id, tar_id, images, labels) in enumerate(train_loader):
            print(i)    
            
            f1 = model.backbone(images.to(device),trunc1=False,trunc2=False,
            trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)  # (B,C1,H1,W1)
            
            #optimize mu and sigma of target features with CLIP
            model_pin_1 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
            model_pin_1.to(device)


            optimizer_pin_1 = torch.optim.SGD(params=[
                {'params': model_pin_1.parameters(), 'lr': 1},
            ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)

            if i == len(train_loader)-1 and f1.shape[0] < opts.batch_size :
                text_target = text_target[:f1.shape[0]]

            while cur_itrs< opts.total_it: 

                cur_itrs += 1
                if cur_itrs % opts.total_it==0:
                    print(cur_itrs)

                optimizer_pin_1.zero_grad()
            
                f1_hal = model_pin_1()
                f1_hal_trans = t1(f1_hal)

                #target_features (optimized)
                target_features_from_f1 = model.backbone(f1_hal_trans,trunc1=True,trunc2=False,trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=False)
                target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
       
                #loss
                loss_CLIP1 = (1- torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()

                writer.add_scalar("loss_CLIP_f1"+str(i),loss_CLIP1,cur_itrs)
               
                loss_CLIP1.backward(retain_graph=True)
              
                optimizer_pin_1.step()
        
            cur_itrs = 0
            
            for name, param in model_pin_1.named_parameters():
                if param.requires_grad and name == 'style_mean':
                    learnt_mu_f1 = param.data
                elif param.requires_grad and name == 'style_std':
                    learnt_std_f1 = param.data

            for k in range(learnt_mu_f1.shape[0]):
                learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
                learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

                stats = {}
                stats['mu_f1'] = learnt_mu_f1_
                stats['std_f1'] = learnt_std_f1_

                with open(opts.save_dir+'/'+img_id[k].split('/')[-1]+'.pkl', 'wb') as f:
                    pickle.dump(stats, f)
     
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)


def main_synth_add_text():
    print("PIN_AUG with synth_add_text")
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts.dataset, opts.data_root, opts.crop_size, data_aug=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19, BB=opts.BB, replace_stride_with_dilation=[False, False, False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56, 56))
    else:
        t1 = lambda x: x

    # 텍스트 타겟 준비
    target_text = compose_text_with_templates(opts.domain_desc, imagenet_templates)
    tokens = clip.tokenize(target_text).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size, 1).type(torch.float32)  # (B,1024)

    # 합성 이미지 타겟 준비
    synth_img = Image.open('/content/drive/MyDrive/Colab Notebooks/PODA/synth_snow1.jpeg')
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(768, 768)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    resize = transforms.Resize((224, 224))
    
    synth_img = train_transform(synth_img)
    synth_img = resize(synth_img)
    synth_img = synth_img.unsqueeze(0).cuda()
    img_target = clip_model.encode_image(synth_img)

    for i, (img_id, tar_id, images, labels) in enumerate(train_loader):
        print(i)    
        
        f1 = model.backbone(images.to(device), trunc1=False, trunc2=False, trunc3=False, trunc4=False, get1=True, get2=False, get3=False, get4=False)
        
        # PIN 모델 초기화
        model_pin_1 = PIN([f1.shape[0], 256, 1, 1], f1.to(device)) # mu_T (B,C1)  sigma_T(B,C1)
        model_pin_1.to(device)

        optimizer_pin_1 = torch.optim.SGD(params=[
            {'params': model_pin_1.parameters(), 'lr': 1},
        ], lr=1, momentum=0.9, weight_decay=opts.weight_decay)

        if i == len(train_loader) - 1 and f1.shape[0] < opts.batch_size:
            img_target = img_target[:f1.shape[0]]
            text_target = text_target[:f1.shape[0]]

        while cur_itrs < opts.total_it: 
            cur_itrs += 1
            if cur_itrs % opts.total_it == 0:
                print(cur_itrs)

            optimizer_pin_1.zero_grad()
        
            f1_hal = model_pin_1()
            f1_hal_trans = t1(f1_hal)

            # 타겟 특징 (최적화된)
            target_features_from_f1 = model.backbone(f1_hal_trans, trunc1=True, trunc2=False, trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=False)
            target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
            
            # 텍스트와 이미지 타겟의 결합 손실 계산
            loss_CLIP1 = (1 - torch.cosine_similarity(img_target, target_features_from_f1, dim=1)).mean()
            loss_CLIP2 = (1 - torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()
            combined_loss = 0.5 * (loss_CLIP1)+0.5*loss_CLIP2
            
            print("combined_loss is ", combined_loss.item())
            writer.add_scalar("loss_CLIP_f1" + str(i), combined_loss, cur_itrs)
           
            combined_loss.backward(retain_graph=True)
          
            optimizer_pin_1.step()
    
        cur_itrs = 0
        
        for name, param in model_pin_1.named_parameters():
            if param.requires_grad and name == 'style_mean':
                learnt_mu_f1 = param.data
            elif param.requires_grad and name == 'style_std':
                learnt_std_f1 = param.data

        for k in range(learnt_mu_f1.shape[0]):
            learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
            learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

            stats = {}
            stats['mu_f1'] = learnt_mu_f1_
            stats['std_f1'] = learnt_std_f1_

            with open(opts.save_dir + '/' + img_id[k].split('/')[-1] + '.pkl', 'wb') as f:
                pickle.dump(stats, f)
 
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)

def main_synth_add_text_prompt():
    print("PIN_AUG with synth_add_text_prompt")
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts.dataset, opts.data_root, opts.crop_size, data_aug=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19, BB=opts.BB, replace_stride_with_dilation=[False, False, False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56, 56))
    else:
        t1 = lambda x: x

    # 텍스트 타겟 준비
    target_text = compose_text_with_templates(opts.domain_desc, prompt_templates)
    tokens = clip.tokenize(target_text).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size, 1).type(torch.float32)  # (B,1024)

    # 합성 이미지 타겟 준비
    synth_img = Image.open('/content/drive/MyDrive/Colab Notebooks/PODA/synth_snow1.jpeg')
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(768, 768)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    resize = transforms.Resize((224, 224))
    
    synth_img = train_transform(synth_img)
    synth_img = resize(synth_img)
    synth_img = synth_img.unsqueeze(0).cuda()
    img_target = clip_model.encode_image(synth_img)

    for i, (img_id, tar_id, images, labels) in enumerate(train_loader):
        print(i)    
        
        f1 = model.backbone(images.to(device), trunc1=False, trunc2=False, trunc3=False, trunc4=False, get1=True, get2=False, get3=False, get4=False)
        
        # PIN 모델 초기화
        model_pin_1 = PIN([f1.shape[0], 256, 1, 1], f1.to(device)) # mu_T (B,C1)  sigma_T(B,C1)
        model_pin_1.to(device)

        optimizer_pin_1 = torch.optim.SGD(params=[
            {'params': model_pin_1.parameters(), 'lr': 1},
        ], lr=1, momentum=0.9, weight_decay=opts.weight_decay)

        if i == len(train_loader) - 1 and f1.shape[0] < opts.batch_size:
            img_target = img_target[:f1.shape[0]]
            text_target = text_target[:f1.shape[0]]

        while cur_itrs < opts.total_it: 
            cur_itrs += 1
            if cur_itrs % opts.total_it == 0:
                print(cur_itrs)

            optimizer_pin_1.zero_grad()
        
            f1_hal = model_pin_1()
            f1_hal_trans = t1(f1_hal)

            # 타겟 특징 (최적화된)
            target_features_from_f1 = model.backbone(f1_hal_trans, trunc1=True, trunc2=False, trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=False)
            target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
            
            # 텍스트와 이미지 타겟의 결합 손실 계산
            loss_CLIP1 = (1 - torch.cosine_similarity(img_target, target_features_from_f1, dim=1)).mean()
            loss_CLIP2 = (1 - torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()
            combined_loss = 0.5 * (loss_CLIP1)+0.5*loss_CLIP2
            
            print("combined_loss is ", combined_loss.item())
            writer.add_scalar("loss_CLIP_f1" + str(i), combined_loss, cur_itrs)
           
            combined_loss.backward(retain_graph=True)
          
            optimizer_pin_1.step()
    
        cur_itrs = 0
        
        for name, param in model_pin_1.named_parameters():
            if param.requires_grad and name == 'style_mean':
                learnt_mu_f1 = param.data
            elif param.requires_grad and name == 'style_std':
                learnt_std_f1 = param.data

        for k in range(learnt_mu_f1.shape[0]):
            learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
            learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

            stats = {}
            stats['mu_f1'] = learnt_mu_f1_
            stats['std_f1'] = learnt_std_f1_

            with open(opts.save_dir + '/' + img_id[k].split('/')[-1] + '.pkl', 'wb') as f:
                pickle.dump(stats, f)
 
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)

def main_synth_add_text_parameter():
    print("PIN_AUG with synth_add_text_parameter")
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts.dataset, opts.data_root, opts.crop_size, data_aug=False)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19, BB=opts.BB, replace_stride_with_dilation=[False, False, False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56, 56))
    else:
        t1 = lambda x: x

    # 텍스트 타겟 준비
    target_text = compose_text_with_templates(opts.domain_desc, imagenet_templates)
    tokens = clip.tokenize(target_text).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size, 1).type(torch.float32)  # (B,1024)

    # 합성 이미지 타겟 준비
    synth_img = Image.open('/content/drive/MyDrive/Colab Notebooks/PODA/synth_snow1.jpeg')
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(768, 768)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    resize = transforms.Resize((224, 224))
    
    synth_img = train_transform(synth_img)
    synth_img = resize(synth_img)
    synth_img = synth_img.unsqueeze(0).cuda()
    img_target = clip_model.encode_image(synth_img)

    # 학습 가능한 a 파라미터 초기화
    a = torch.tensor(opts.loss_weight, requires_grad=True, device=device)

    for i, (img_id, tar_id, images, labels) in enumerate(train_loader):
        print(i)    
        
        f1 = model.backbone(images.to(device), trunc1=False, trunc2=False, trunc3=False, trunc4=False, get1=True, get2=False, get3=False, get4=False)
        
        # PIN 모델 초기화
        model_pin_1 = PIN([f1.shape[0], 256, 1, 1], f1.to(device)) # mu_T (B,C1)  sigma_T(B,C1)
        model_pin_1.to(device)

        # a 값을 PIN 모델의 파라미터와 함께 최적화에 추가
        optimizer_pin_1 = torch.optim.SGD(params=[
            {'params': model_pin_1.parameters(), 'lr': 1},
            {'params': [a], 'lr': 1e-2}
        ], lr=1, momentum=0.9, weight_decay=opts.weight_decay)

        if i == len(train_loader) - 1 and f1.shape[0] < opts.batch_size:
            img_target = img_target[:f1.shape[0]]
            text_target = text_target[:f1.shape[0]]

        while cur_itrs < opts.total_it: 
            cur_itrs += 1
            if cur_itrs % opts.total_it == 0:
                print(cur_itrs)

            optimizer_pin_1.zero_grad()
        
            f1_hal = model_pin_1()
            f1_hal_trans = t1(f1_hal)

            # 타겟 특징 (최적화된)
            target_features_from_f1 = model.backbone(f1_hal_trans, trunc1=True, trunc2=False, trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=False)
            target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
            
            # 텍스트와 이미지 타겟의 결합 손실 계산
            loss_CLIP1 = (1 - torch.cosine_similarity(img_target, target_features_from_f1, dim=1)).mean()
            loss_CLIP2 = (1 - torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()

            # 결합 손실 계산 가중치 적용 (a와 1-a)
            combined_loss = a * loss_CLIP1 + (1 - a) * loss_CLIP2
            combined_loss = torch.clamp(combined_loss, min=0)
            print("a is = ", a.item())
            print("combined_loss is ", combined_loss.item())
            writer.add_scalar("loss_CLIP_f1" + str(i), combined_loss, cur_itrs)
           
            combined_loss.backward(retain_graph=True)
            optimizer_pin_1.step()

        cur_itrs = 0
        
        for name, param in model_pin_1.named_parameters():
            if param.requires_grad and name == 'style_mean':
                learnt_mu_f1 = param.data
            elif param.requires_grad and name == 'style_std':
                learnt_std_f1 = param.data

        for k in range(learnt_mu_f1.shape[0]):
            learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
            learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

            stats = {}
            stats['mu_f1'] = learnt_mu_f1_
            stats['std_f1'] = learnt_std_f1_

            with open(opts.save_dir + '/' + img_id[k].split('/')[-1] + '.pkl', 'wb') as f:
                pickle.dump(stats, f)
 
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)
def main_aug():

    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,data_aug=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19,BB= opts.BB,replace_stride_with_dilation=[False,False,False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56,56))
    else:
        t1 = lambda x:x

    #text
    #target text
    target = compose_text_with_templates(opts.domain_desc, imagenet_templates)

    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size,1).type(torch.float32)  # (B,1024)

    for i,(img_id, tar_id, images, labels) in enumerate(train_loader):
            print(i)    
            
            f1 = model.backbone(images.to(device),trunc1=False,trunc2=False,
            trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)  # (B,C1,H1,W1)
            
            #optimize mu and sigma of target features with CLIP
            model_pin_1 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
            model_pin_1.to(device)


            optimizer_pin_1 = torch.optim.SGD(params=[
                {'params': model_pin_1.parameters(), 'lr': 1},
            ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)

            if i == len(train_loader)-1 and f1.shape[0] < opts.batch_size :
                text_target = text_target[:f1.shape[0]]

            while cur_itrs< opts.total_it: 

                cur_itrs += 1
                if cur_itrs % opts.total_it==0:
                    print(cur_itrs)

                optimizer_pin_1.zero_grad()
            
                f1_hal = model_pin_1()
                f1_hal_trans = t1(f1_hal)

                #target_features (optimized)
                target_features_from_f1 = model.backbone(f1_hal_trans,trunc1=True,trunc2=False,trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=False)
                target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
       
                #loss
                loss_CLIP1 = (1- torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()

                writer.add_scalar("loss_CLIP_f1"+str(i),loss_CLIP1,cur_itrs)
               
                loss_CLIP1.backward(retain_graph=True)
              
                optimizer_pin_1.step()
        
            cur_itrs = 0
            
            for name, param in model_pin_1.named_parameters():
                if param.requires_grad and name == 'style_mean':
                    learnt_mu_f1 = param.data
                elif param.requires_grad and name == 'style_std':
                    learnt_std_f1 = param.data

            for k in range(learnt_mu_f1.shape[0]):
                learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
                learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

                stats = {}
                stats['mu_f1'] = learnt_mu_f1_
                stats['std_f1'] = learnt_std_f1_

                with open(opts.save_dir+'/'+img_id[k].split('/')[-1]+'.pkl', 'wb') as f:
                    pickle.dump(stats, f)
     
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)
def main_uniformaugment():
    print("PIN_AUG with uniformaugment")
    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,data_aug=True)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19,BB= opts.BB,replace_stride_with_dilation=[False,False,False])

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)
    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56,56))
    else:
        t1 = lambda x:x

    #text
    #target text
    target = compose_text_with_templates(opts.domain_desc, imagenet_templates)

    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    text_target = text_target.repeat(opts.batch_size,1).type(torch.float32)  # (B,1024)

    for i,(img_id, tar_id, images, labels) in enumerate(train_loader):
            print(i)    
            
            f1 = model.backbone(images.to(device),trunc1=False,trunc2=False,
            trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)  # (B,C1,H1,W1)
            
            #optimize mu and sigma of target features with CLIP
            model_pin_1 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
            model_pin_1.to(device)


            optimizer_pin_1 = torch.optim.SGD(params=[
                {'params': model_pin_1.parameters(), 'lr': 1},
            ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)

            if i == len(train_loader)-1 and f1.shape[0] < opts.batch_size :
                text_target = text_target[:f1.shape[0]]

            while cur_itrs< opts.total_it: 

                cur_itrs += 1
                if cur_itrs % opts.total_it==0:
                    print(cur_itrs)

                optimizer_pin_1.zero_grad()
            
                f1_hal = model_pin_1()
                f1_hal_trans = t1(f1_hal)

                #target_features (optimized)
                target_features_from_f1 = model.backbone(f1_hal_trans,trunc1=True,trunc2=False,trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=False)
                target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
       
                #loss
                loss_CLIP1 = (1- torch.cosine_similarity(text_target, target_features_from_f1, dim=1)).mean()

                writer.add_scalar("loss_CLIP_f1"+str(i),loss_CLIP1,cur_itrs)
               
                loss_CLIP1.backward(retain_graph=True)
              
                optimizer_pin_1.step()
        
            cur_itrs = 0
            
            for name, param in model_pin_1.named_parameters():
                if param.requires_grad and name == 'style_mean':
                    learnt_mu_f1 = param.data
                elif param.requires_grad and name == 'style_std':
                    learnt_std_f1 = param.data

            for k in range(learnt_mu_f1.shape[0]):
                learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
                learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

                stats = {}
                stats['mu_f1'] = learnt_mu_f1_
                stats['std_f1'] = learnt_std_f1_

                with open(opts.save_dir+'/'+img_id[k].split('/')[-1]+'.pkl', 'wb') as f:
                    pickle.dump(stats, f)
     
    print(learnt_mu_f1.shape)
    print(learnt_std_f1.shape)

# main()
#main_synth_add_text()
#main_synth_add_text_prompt()
#main_synth_add_text_parameter()
main_uniformaugment()