from re import L
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import Cityscapes, gta5
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from PIL import Image
from imgaug import augmenters as iaa  
import matplotlib
import matplotlib.pyplot as plt
import pickle
from utils.utils import denormalize
from torchvision.utils import save_image
from torchvision import transforms
import cv2
import numpy as np
import random
import numbers
from torchvision.transforms import functional as F



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
class ColorTransfer:
    def __init__(self, target_mean, target_std):
        """
        Args:
        - target_mean (list): Target domain's mean for color matching.
        - target_std (list): Target domain's std deviation for color matching.
        """
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, img, lbl):
        """
        Args:
        - img (PIL.Image or numpy.ndarray): 원본 이미지
        - lbl (numpy.ndarray): 라벨 (변경 없이 반환)

        Returns:
        - transferred_img (numpy.ndarray): 색상 전환이 완료된 이미지
        - lbl (numpy.ndarray): 라벨 (변경 없이 반환)
        """
        # PIL.Image를 numpy.ndarray로 변환
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = img.astype(np.float32) / 255.0
        mean_src, std_src = cv2.meanStdDev(img)
        mean_src = mean_src.flatten()
        std_src = std_src.flatten()

        for i in range(3):  # RGB 채널에 대해 각각 수행
            img[:, :, i] = ((img[:, :, i] - mean_src[i]) / std_src[i]) * self.target_std[i] + self.target_mean[i]

        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img, lbl  # 라벨은 그대로 반환

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img, lbl):
        """
        이미지에 가우시안 노이즈를 추가합니다.

        Args:
        - img (PIL.Image or numpy.ndarray): 원본 이미지
        - lbl (numpy.ndarray): 라벨 (변경 없이 반환)

        Returns:
        - noisy_img (numpy.ndarray): 노이즈가 추가된 이미지
        - lbl (numpy.ndarray): 라벨 (변경 없이 반환)
        """
        # PIL.Image를 numpy.ndarray로 변환
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # 이미지의 형태에 맞는 가우시안 노이즈 생성
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
        
        # 노이즈를 이미지에 추가
        noisy_img = img + noise * 255
        
        # 픽셀 값 범위를 [0, 255]로 클리핑하고 uint8로 변환
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return noisy_img, lbl  # 라벨은 그대로 반환

class GaussianBlur:
    def __init__(self, kernel_size=(5, 5), sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, lbl):
        """
        이미지에 가우시안 블러를 적용합니다.

        Args:
        - img (PIL.Image or numpy.ndarray): 원본 이미지
        - lbl (numpy.ndarray): 라벨 (변경 없이 반환)

        Returns:
        - blurred_img (numpy.ndarray): 가우시안 블러가 적용된 이미지
        - lbl (numpy.ndarray): 라벨 (변경 없이 반환)
        """
        # PIL.Image를 numpy.ndarray로 변환
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        blurred_img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
        return blurred_img, lbl  # 라벨은 그대로 반환

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','ACDC','gta5'], help='Name of dataset')
    parser.add_argument("--ACDC_sub", type=str, default="night",
                        help = "specify which subset of ACDC  to use")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=200e3,
                        help="epoch number (default: 200k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--forward_pass",action='store_true',default=False,
                        help="forward pass to update BN statistics")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--freeze_BB", action='store_true',default=False,
                        help="Freeze the backbone when training")
    parser.add_argument("--ckpts_path", type = str ,
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    #validation
    parser.add_argument("--val_results_dir", type=str,help="Folder name for validation results saving")
    #Augmented features
    parser.add_argument("--train_aug",action='store_true',default=False,
                        help="train on augmented features using CLIP")
    parser.add_argument("--path_mu_sig", type=str)
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")

    return parser
class UniformAugment:
    def __init__(self, augmentations, n_select=2):
        self.augmentations = augmentations
        self.n_select = n_select

    def __call__(self, img):
        aug_list = random.sample(self.augmentations, self.n_select)
        aug_seq = iaa.Sequential(aug_list)
        img = np.array(img)
        img = aug_seq(image=img)
        return Image.fromarray(img)

augmentations = [
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.Multiply((0.5, 1.5)),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
]

uniform_augment = UniformAugment(augmentations, n_select=3)

def get_dataset(dataset,data_root,crop_size,ACDC_sub="night",data_aug=FALSE):
    """ Dataset And Augmentation
    """
    if dataset == 'cityscapes':
      if data_aug:
          train_transform = et.ExtCompose([
              et.ExtRandomCrop(size=(crop_size, crop_size)),
              et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
              et.ExtRandomHorizontalFlip(),
              ColorTransfer(target_mean=[0.48145466, 0.4578275, 0.40821073],
                             target_std=[0.26862954, 0.26130258, 0.27577711]),  # Color Transfer
              AddGaussianNoise(mean=0, std=0.05),  # Additive Gaussian Noise
              GaussianBlur(kernel_size=(5, 5), sigma=1),  # Gaussian Blur
              et.ExtToTensor(),
              et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
            ])
      else:
          train_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize(crop_size),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
           ])

      val_transform = et.ExtCompose([
           et.ExtToTensor(),
           et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                          std=[0.26862954, 0.26130258, 0.27577711]),
      ])

      train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform)
      val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform)

    if dataset == 'ACDC':
        train_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform, ACDC_sub = ACDC_sub)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform, ACDC_sub = ACDC_sub)

    if dataset == "gta5":
        
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtCenterCrop(size=(1046, 1914)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_train.txt',transform=train_transform)
        val_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_val.txt',transform=val_transform)

    return train_dst, val_dst

def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists(opts.val_results_dir):
            os.mkdir(opts.val_results_dir)
        img_id = 0

    with torch.no_grad():

        for i, (im_id, tg_id, images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs,features = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
           
            metrics.update(targets, preds)
            
            if opts.save_val_results:
                for j in range(len(images)):

                    target = targets[j]
                    pred = preds[j]

                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(target).save(opts.val_results_dir+'/%d_target.png' % img_id)
                    Image.fromarray(pred).save(opts.val_results_dir+'/%d_pred.png' % img_id)

                    images[j] = denormalize(images[j],mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
                    save_image(images[j],opts.val_results_dir+'/%d_image.png' % img_id)

                    fig = plt.figure()
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    #plt.savefig(opts.val_results_dir+'/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score


def main():
    opts = get_argparser().parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    opts.data_aug=True
    # Setup random seed
    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
  
    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,opts.ACDC_sub,
                                    data_aug=opts.data_aug)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
    drop_last=True)  # drop_last=True to ignore single-image batches.

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=19, BB= opts.BB,replace_stride_with_dilation=[False,False,True])
    model.backbone.attnpool = nn.Identity()

    #fix the backbone
    if opts.freeze_BB:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

    # Set up metrics
    metrics = StreamSegMetrics(19)

    # Set up optimizer
    if opts.freeze_BB:
        optimizer = torch.optim.SGD(params=[
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.001 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    if not opts.test_only:
        utils.mkdir(opts.ckpts_path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model.to(device)
    
    # ==========   Train Loop   ==========#

    if opts.test_only:
       
        model.eval()

        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)

        print(metrics.to_str(val_score))
        print(val_score["Mean IoU"])
        print(val_score["Class IoU"])
        return

    interval_loss = 0

    if opts.train_aug:
        files = [f for f in os.listdir(opts.path_mu_sig+'/')]
    
    relu = nn.ReLU(inplace=True)

    while True:  # cur_itrs < opts.total_itrs:
    # =====  Train  =====
    
        if opts.freeze_BB:
            model.classifier.train()
        else:
            model.train()

        cur_epochs += 1

        for (im_id, tg_id, images, labels) in train_loader:
            
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            if opts.train_aug:
                mu_t_f1 = torch.zeros([opts.batch_size,256,1,1])
                std_t_f1 = torch.zeros([opts.batch_size,256,1,1])
        
                for k in range(opts.batch_size):
                    with open(opts.path_mu_sig+'/'+random.choice(files), 'rb') as f:
                        loaded_dict = pickle.load(f)
                        mu_t_f1[k] = loaded_dict['mu_f1']
                        std_t_f1[k] = loaded_dict['std_f1']

                outputs,features = model(images,mu_t_f1.to(device),std_t_f1.to(device),
                                    transfer=True,mix=opts.mix,activation=relu)
                
            else:
                outputs,features = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss",loss,cur_itrs)
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0 and not opts.train_aug:
                save_ckpt(opts.ckpts_path+'/latest_%s_%s.pth' %
                        (opts.model, opts.dataset))
                print("validation...")
                model.eval()
               
                val_score = validate(
                    opts=opts, model=model, loader=val_loader,device=device, metrics=metrics
                    )

                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(opts.ckpts_path+'/best_%s_%s.pth' %
                            (opts.model, opts.dataset))

                writer.add_scalar("mIoU", val_score['Mean IoU'] ,cur_itrs)

                if opts.freeze_BB:
                    model.classifier.train()
                else:
                    model.train()
                    
            if opts.train_aug and cur_itrs == opts.total_itrs:
                save_ckpt(opts.ckpts_path+'/adapted_%s_%s.pth' %
                        (opts.model, opts.dataset))
            
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return
            

if __name__ == '__main__':
    main()