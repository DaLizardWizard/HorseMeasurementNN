# !mkdir HorseData/train
# !mkdir HorseData/test
# !mkdir HorseData/train/H
# !mkdir HorseData/train/HH
# !mkdir HorseData/train/HHH
# !mkdir HorseData/train/HHHH
# !mkdir HorseData/train/HHHHH

# !mkdir HorseData/test/H
# !mkdir HorseData/test/HH
# !mkdir HorseData/test/HHH
# !mkdir HorseData/test/HHHH
# !mkdir HorseData/test/HHHHH

# !mkdir HorseData/verythin
# !mkdir HorseData/thin
# !mkdir HorseData/healthy
# !mkdir HorseData/fat
# !mkdir HorseData/veryfat

#Making directorys to hold images which contain too much or too little horses
# !mkdir HorseData/Under10
# !mkdir HorseData/Over60

#Moving all Horse images back into original Directory
! cp -r HorseData/*/HorseData ./

# #clearing files
# #!rm -r HorseData/test
# #!rm -r HorseData/train
# !rm HorseData/verythin/*
# !rm HorseData/thin/*
# !rm HorseData/healthy/*
# !rm HorseData/fat/*
# !rm HorseData/veryfat/*

#!mkdir HorseData/unsegmented
#!mkdir HorseData/train/H
#!mkdir HorseData/train/HH
#!mkdir HorseData/test/H
#!mkdir HorseData/test/HH
#!pwd
#!mkdir 
#!pip install seaborn
#!pip install sklearn
#!git clone https://github.com/Joeclinton1/google-images-download.git
#!cd google-images-download && python setup.py install
#!ls google-images-download/
#!apt-get install parallel

###Raw Data getting
# !parallel --verbose googleimagesdownload -k "very\ "{}"\ Horse" -o VeryThinHorse --limit 100 --format jpg ::: skinny thin starved malnourished emaciated
# !parallel --verbose googleimagesdownload -k "very\ "{}"\ Horse" -o VeryThinHorse --limit 100 --format png ::: skinny thin starved malnourished emaciated

# !parallel googleimagesdownload -k {}"\ Horse" -o ThinHorse --limit 100 --format jpg ::: skinny thin starved malnourished emaciated
# !parallel googleimagesdownload -k {}"\ Horse" -o ThinHorse --limit 100 --format png ::: skinny thin starved malnourished emaciated
            
# !parallel googleimagesdownload -k {}"\ Horse" -o FitHorse --limit 100 --format jpg ::: fit healthy active athletic strong
# !parallel googleimagesdownload -k {}"\ Horse" -o FitHorse --limit 100 --format png ::: fit healthy active athletic strong
            
# !parallel googleimagesdownload -k {}"\ Horse" -o FatHorse --limit 100 --format jpg ::: obese fat plump heavy overfed
# !parallel googleimagesdownload -k {}"\ Horse" -o FatHorse --limit 100 --format png ::: obese fat plump heavy overfed

# !parallel --verbose googleimagesdownload -k "very\ "{}"\ Horse" -o VeryFatHorse --limit 100 --format jpg ::: obese fat plump heavy overfed
# !parallel --verbose googleimagesdownload -k "very\ "{}"\ Horse" -o VeryFatHorse --limit 100 --format png ::: obese fat plump heavy overfed

# !cp VeryThinHorse/*/* HorseData/verythin
# !cp ThinHorse/*/* HorseData/thin
# !cp FitHorse/*/* HorseData/healthy
# !cp FatHorse/*/* HorseData/fat
# !cp VeryFatHorse/*/* HorseData/veryfat


# vtlen = !ls HorseData/verythin | wc -l | awk '{print int($1/10)}'
# tlen = !ls HorseData/thin | wc -l | awk '{print int($1/10)}'
# hlen = !ls HorseData/healthy | wc -l | awk '{print int($1/10)}'
# flen = !ls HorseData/fat | wc -l | awk '{print int($1/10)}'
# vflen = !ls HorseData/veryfat | wc -l | awk '{print int($1/10)}'


# vtlen,tlen,hlen,flen,vflen = int(vtlen[0]),int(tlen[0]),int(hlen[0]),int(flen[0]),int(vflen[0])

# print(vtlen,tlen,hlen,flen,vflen)
# print(vtlen*8,vtlen*2)
# print(tlen*8,tlen*2)
# print(hlen*8,hlen*2)
# print(flen*8,flen*2)
# print(vflen*8,vflen*2)

#why has it moved more images thyan it should?
# !mv `ls ./HorseData/verythin/* | head -664` ./HorseData/train/H/
# !mv `ls ./HorseData/verythin/* | head -166` ./HorseData/test/H/

# !mv `ls ./HorseData/thin/* | head -704` ./HorseData/train/HH
# !mv `ls ./HorseData/thin/* | head -176` ./HorseData/test/HH

# !mv `ls ./HorseData/healthy/* | head -736` ./HorseData/train/HHH
# !mv `ls ./HorseData/healthy/* | head -184` ./HorseData/test/HHH

# !mv `ls ./HorseData/fat/* | head -696` ./HorseData/train/HHHH
# !mv `ls ./HorseData/fat/* | head -174` ./HorseData/test/HHHH

# !mv `ls ./HorseData/veryfat/* | head -664` ./HorseData/train/HHHHH
# !mv `ls ./HorseData/veryfat/* | head -166` ./HorseData/test/HHHHH

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

#cleans webp images from files
#!rm HorseData/*/*/*.webp

#copy and paste horse function here

#Looks at all images and detects their "horseness"
#for image in HorseData/*/*/*
    #load image into into the function to produce RGB segmented horse
    #If the image has less than 10% "horse" pixels
        #provide modify filename extention to be .ten
    #If the image has less than 20% "horse" pixels
        #provide modify filename extention to be .twe
    #If the image has less than 10% "horse" pixels
        #provide modify filename extention to be .thi
    #else
        #leave
!ls HorseData/*/*/* 



###automatically remove things not detected as horse.
#mapping label to colour 
# Define the helper function
def decode_segmap(image, nc=21):
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (0, 0, 0),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb

from torchvision import models 
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

from skimage import color
from skimage import io
from skimage.io import imsave
import torchvision.transforms as T
from PIL import Image
import numpy as np

global GlobalHorsePc
GlobalHorsePc = []

def segment(net, path, TR, TC):
    img = Image.open(path).convert('RGB')
    #plt.imshow(img); plt.axis('off'); plt.show()
    TCH = int(1.2*TC)
    TRH = int(1.2*TR)
    # Comment the Resize and CenterCrop for better inference results
    #running without resizing and cropping exponentially increases segmentation time
    #default values for TR and TC are 256 and 224
    trf = T.Compose([T.Resize([TR,TRH]), 
                   T.CenterCrop([TC,TCH]), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    
    #gives us a resized image in a tensor format which we can then apply masks to
    imr = T.Compose([T.Resize([TR,TRH]), 
               T.CenterCrop([TC,TCH]), 
               T.ToTensor()])
    imrz = imr(img).unsqueeze(0)
    
    #passing DNN ready tensor through the net
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    #plt.imshow(rgb) #plt.axis('off'); 
    #plt.show()
    
    #saving image masks
    imrz = imrz.squeeze(0)
    img=imrz.numpy()
    img=np.swapaxes(img,0,1)
    img=np.swapaxes(img,1,2)
    mask = rgb[:,:,0]
#     PcHorse =(np.count_nonzero(mask)/(len(mask)*(len(mask)*1.2))*100)
#     PcHorse = round(PcHorse,2)
#     GlobalHorsePc.append(PcHorse)
    
#     if PcHorse < 20:
#         os.makedirs(os.path.dirname('HorseData/Under20/'+path), exist_ok=True)
#         shutil.move(path,'HorseData/Under10/'+path)
#     #break
#     elif PcHorse > 40:
#         os.makedirs(os.path.dirname('HorseData/Over40/'+path), exist_ok=True)
#         shutil.move(path,'HorseData/Over40/'+path)
    #break
    
    #https://stackoverflow.com/questions/34691128/how-to-use-mask-to-remove-the-background-in-python/38516107
    #where the mask is less than ten set colours to 0
    mask2 = np.where((mask<10),0,1).astype('uint8')
    #cim=background removed horse
    cim = img*mask2[:,:,np.newaxis]
#     plt.imshow(cim)
#     plt.show()
    
   #compare to reference horse image to determine image similarituy #rgb is the value we want to compare
    Rref = plt.imread('HorseData/RightRef.jpg')
    Lref = plt.imread('HorseData/LeftRef.jpg')

    from skimage.metrics import structural_similarity as ssim
    ssimR = ssim(rgb[:,:,0], Rref[:,:,0])
    ssimL = ssim(rgb[:,:,0], Lref[:,:,0])
    
    #flips colour image if horse is facing left.
    if ssimL > ssimR:
        cim = np.fliplr(cim)
    
    
    #only need to segment horse images and save them once
    os.makedirs(os.path.dirname('HorseData/segmented/'+path), exist_ok=True)
    #shutil.move(path,'HorseData/segmented/'+path)
    if (ssimR > 0.5) or (ssimL > 0.5):   
        plt.imsave('HorseData/segmented/'+path,cim)
    
#     fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,12))
#     ax1.axis('off'); ax2.axis('off'); ax3.axis('off');
#     ax1.set_title('Resized image')
#     ax2.set_title('Horse detection')
#     ax3.set_title('doot')
#     ax1.imshow(img);ax2.imshow(rgb); ax3.imshow(cim);
#     plt.show()
    
    #only saving the rgb file once 
    #plt.imsave('HorseData/LeftRef.jpg',rgb)
#     plt.figure()
#     plt.grid(None)
#     plt.axis('off')
#     plt.imshow(rgb)
#     plt.imsave('HorseData/LeftRef.jpg',rgb)
#     plt.show()
    
        
#     plt.figure()
#     plt.axis('off')
#     plt.grid(None)
#     plt.imshow(np.fliplr(rgb))
#     plt.imsave('HorseData/RightRef.jpg',np.fliplr(rgb))
#     plt.show()
    #swapping the axis and saving as right 
    
segment(dlab,'HorseData/Horse_Images/HorseData/Horse_Images/HorseSide/HORSE_1_BCS_4.jpg',512,512)
#print(GlobalHorsePc)

import glob
import random
listing = glob.glob('HorseData/test/*/*') + glob.glob('HorseData/train/*/*')

#bit dirty but it works

#print (len(listing))
print (len(listing))

#resetting counter
GlobalHorsePc = []
random.shuffle(listing)
print(GlobalHorsePc)

import shutil
for horsie in listing:
    segment(dlab,horsie,512,512)
#listing = glob.glob('HorseData/test/*/*') + glob.glob('HorseData/train/*/*')    
#print(GlobalHorsePc)

# print(max(GlobalHorsePc))

# plt.figure()
# plt.figsize=(12,12)
# plt.hist(GlobalHorsePc,100)
# plt.title("Abundance of horse coverage within image")
# plt.xlabel("Percentage of horse within image")
# plt.ylabel("Counts")
# plt.show()

#check all listings are openable images
import os
for image in listing:
    doot = os.path.getsize(image)
    print (image)

np.random.seed(0)
torch.manual_seed(0)
%matplotlib inline
sns.set_style('darkgrid')

#torch.cuda.is_available()
#torch.cuda.current_device()
#torch.cuda.device_count()
#torch.cuda.get_device_name(0)

image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    "test": transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
}

!ls
!pwd
root_dir = "HorseData/segmented/HorseData/"


rps_dataset = datasets.ImageFolder(root = root_dir + "train",
                                   transform = image_transforms["train"]
                                  )
rps_dataset

rps_dataset.class_to_idx

idx2class = {v: k for k, v in rps_dataset.class_to_idx.items()}
idx2class
print(rps_dataset)
print(rps_dataset_test)

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict
def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)
plt.figure(figsize=(15,8))
plot_from_dict(get_class_distribution(rps_dataset), plot_title="Entire Dataset (before train/val/test split)")

rps_dataset_size = len(rps_dataset)
rps_dataset_indices = list(range(rps_dataset_size))
np.random.shuffle(rps_dataset_indices)
val_split_index = int(np.floor(0.2 * rps_dataset_size))
train_idx, val_idx = rps_dataset_indices[val_split_index:], rps_dataset_indices[:val_split_index]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

rps_dataset_test = datasets.ImageFolder(root = root_dir + "test",
                                        transform = image_transforms["test"])
rps_dataset_test

train_loader = DataLoader(dataset=rps_dataset, shuffle=False, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset=rps_dataset, shuffle=False, batch_size=1, sampler=val_sampler)
test_loader = DataLoader(dataset=rps_dataset_test, shuffle=False, batch_size=1)

def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
plot_from_dict(get_class_distribution_loaders(train_loader, rps_dataset), plot_title="Train Set", ax=axes[0])
plot_from_dict(get_class_distribution_loaders(val_loader, rps_dataset), plot_title="Val Set", ax=axes[1])

single_batch = next(iter(train_loader))

print(single_batch[0].shape)
print("Output label tensors: ", single_batch[1])
print("\nOutput label tensor shape: ", single_batch[1].shape)

i=0
for batch in iter(train_loader):
    print(i,batch[1],batch[0])
    i = i+1



# i=0
# for batch in iter(train_loader):
#     print(i,batch[1][1])
#     i += 1
#print(single_batch)

# # Selecting the first image tensor from the batch. 
# single_image = single_batch[0][0]
# single_image.shape

# plt.imshow(single_image.permute(1, 2, 0))

# single_batch_grid = utils.make_grid(single_batch[0], nrow=3)
# plt.figure(figsize = (10,10))
# plt.imshow(single_batch_grid.permute(1, 2, 0))
# plt.axis("off")

class RpsClassifier(nn.Module):
    def __init__(self):
        super(RpsClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=5, kernel_size=75, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
#number of classes
len(rps_dataset.classes)

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.Size()

model = RpsClassifier()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
print(criterion)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc



accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

i=0
s=1
print("Begin training.")
for e in tqdm(range(1, 31)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        #print(i)
        if i  != (s*175)+s-1:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        i +=1 
    s+=1
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            val_epoch_loss += train_loss.item()
            val_epoch_acc += train_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

# Plot line charts
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

y_pred_list = [i[0][0][0] for i in y_pred_list]
y_true_list = [i[0] for i in y_true_list]
print(classification_report(y_true_list, y_pred_list))

print(confusion_matrix(y_true_list, y_pred_list))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_pred_list,y_true_list)).rename(columns=idx2class, index=idx2class)
fig, ax = plt.subplots(figsize=(7,5))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

torch.save(model.state_dict(), 'HorseModel')

import glob
image_path = "Horse_Images/HorseData/Horse_Images"
horse_dataset = datasets.ImageFolder(root = root_dir + image_path,
                                   transform = image_transforms["test"]
                                  )                                                                
                                                                
horse_loader = DataLoader(dataset=horse_dataset, shuffle=False, batch_size=1)

images = (root_dir+image_path+"/*/*")
image_list = (glob.glob(images))
image_ID = []
for image in image_list:
    image_ID.append(os.path.basename(image))  
    
print(image_ID)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
y_pred_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(horse_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        doot, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tag.cpu().numpy()[0][0][0])

print(len(image_ID))
print(len(y_pred_list))#

for image,horse in zip(image_ID,y_pred_list):
    print(image,horse)
    


