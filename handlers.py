from ast import parse
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from xyh import get_int
import torch


class MNIST_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy(), mode='L')
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class SVHN_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    def __getitem__(self, index):
        path, y = self.X[index], self.Y[index]
    
            
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class ImageNet_LT_Handler(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.SN3_PASCALVINCENT_TYPEMAP= {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


    @staticmethod
    def processing_fromUnet(x,index):
        file_path=os.path.join( 'C:\\Users\\F.F.Chopin\\project\\low-budget-al',x[index])
        picture=Image.open(file_path)
       # if picture.mode != 'L':
	        #picture = picture.convert('L')
        img_ndarray = np.asarray(picture)
        if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
        else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255
        return torch.as_tensor(img_ndarray.copy()).float().contiguous()

    @staticmethod
    def processing_fromEnden(x,index):
        with open(os.path.join( 'C:\\Users\\F.F.Chopin\\project\\low-budget-al',x[index]), 'rb') as f:
            data = f.read()
           # sample = Image.open(f).convert('RGB')
            magic = get_int(data[0:4])
            nd = magic % 256
            ty = magic // 256
            #assert 1 <= nd <= 3
           # assert 8 <= ty <= 14
            #torch_type = self.SN3_PASCALVINCENT_TYPEMAP[ty]
            s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
            parsed = torch.frombuffer(bytearray(data), dtype=torch.int16,)
           # x=parsed.view(*s)
            x=Image.fromarray(parsed.numpy(),mode='L')
            return x

    def __getitem__ (self,index):
        y=self.y[index]
        #print(type(y))
       # y=torch.as_tensor(y).float().contiguous()
        y=torch.tensor(y)

        x=ImageNet_LT_Handler.processing_fromUnet(self.x,index)
        #x=ImageNet_LT_Handler.processing_fromEnden(self.x,index)
        #x=self.transform(x)
        return x,y,index

    def __len__(self):
        return len(self.x)