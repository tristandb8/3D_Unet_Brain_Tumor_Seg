import nibabel as nib
import numpy as np
import torch
import torchvision.transforms as trans
import glob
import random
import torchmetrics
from hd import hd
from sklearn.model_selection import StratifiedKFold
import sys


to128 = trans.Compose([
    trans.Resize((128, 128))
])
to224 = trans.Compose([
    trans.Resize((240, 240))
])

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        # Flatten both tensors
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        
        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        
        # Calculate Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Calculate Dice loss
        dice_loss = 1. - dice_coeff
        
        return dice_loss

def main(argv):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=4, out_channels=1, init_features=155, pretrained=False).cuda()
    print("model loaded", flush=True)

    paths = glob.glob("/home/ak119590/ishan_dataset/Task01_BrainTumour/imagesTr/BRATS*")
    random.shuffle(paths)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = DiceLoss()

    folds = np.array_split(paths, 5)

    val_loss_dice = {}
    val_loss_hd = {}

    for k in range(5):
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=4, out_channels=1, init_features=155, pretrained=False).cuda()
        print("model loaded", flush=True)

        train_paths = [fold for i, fold in enumerate(folds) if i != k]
        train_paths = [item for sublist in train_paths for item in sublist] # flatten list

        val_paths = folds[k]


        losses_dice = []
        losses_hd = []
        model.train()
        for file in train_paths:
            model, optimizer, criterion, losses_dice, losses_hd = loop(file, model, optimizer, criterion, losses_dice, losses_hd)
        print("(", k, ") AFTER TRAINING - Dice loss: ", np.mean(losses_dice), "Hausdorf loss: ", np.mean(losses_hd))
        torch.save(model.state_dict(), 'weights/'+str(k)+'.pt')


        losses_dice = []
        losses_hd = []
        model.eval()
        for file in val_paths:
            model, optimizer, criterion, losses_dice, losses_hd = loop(file, model, optimizer, criterion, losses_dice, losses_hd)
        print("(", k, ") VALIDATION: Dice loss: ", np.mean(losses_dice), "Hausdorf loss: ", np.mean(losses_hd))

        val_loss_dice[k] = np.mean(losses_dice)
        val_loss_hd[k] = np.mean(losses_hd)
    
    print("mean Dice score", np.array(list(val_loss_dice.values())).mean())
    print("mean Hausdorf score", np.array(list(val_loss_hd.values())).mean(), flush=True)
    
def loop(file, model, optimizer, criterion, losses_dice, losses_hd):
    img = nib.load(file)
    label = nib.load(file.replace("imagesTr", "labelsTr"))

    data = img.get_fdata()
    data_label = label.get_fdata()

    data = torch.tensor(data, dtype=torch.float32).cuda()
    data_label = torch.tensor(data_label, dtype=torch.float32).cuda()

    data = data.permute(2,3,0,1)
    data_label = data_label.permute(2,0,1)

    data_label = torch.unsqueeze(data_label, 1)
    data = torch.stack([to128(d) for d in data])

    #print(data.shape, data_label.shape)
    optimizer.zero_grad()
    output = model(data)
    output = torch.stack([to224(d) for d in output])

    loss_dice = criterion(output,data_label)
    loss_hd = hd(output.cpu().detach().numpy(),data_label.cpu().detach().numpy())

    losses_dice.append(loss_dice.item())
    losses_hd.append(loss_hd.item())

    loss = loss_dice + loss_hd

    loss.backward()
    optimizer.step()
    if (len(losses_dice) % 50 == 0):
        print("After", len(losses_dice), "iterations:")
        print("Dice loss: ", np.mean(losses_dice), "Hausdorf loss: ", np.mean(loss_hd), flush=True)
    
    return model, optimizer, criterion, losses_dice, losses_hd


if __name__ == "__main__":
   main(sys.argv[1:])