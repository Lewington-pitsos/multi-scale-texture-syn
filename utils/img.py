import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import scipy.ndimage
from torchvision.utils import save_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_loader(image_path, transform, requires_grad=False):
    """load image, returns cuda tensor"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) #this is for VGG, may not be needed for ResNet

    image = image.to(device, torch.float)

    if requires_grad:
        image.requires_grad = True

    return image

def show_image(image_tensor):
    np_image = image_tensor.squeeze().cpu().detach().numpy()
    plt.figure()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))

def random_image(blur=6, size=320):
    # create random noise numpy array
    np_sample = np.random.rand(size, size, 3)

    # smooth it out (try commenting out this line and see the difference)
    np_sample = scipy.ndimage.filters.median_filter(np_sample, [blur, blur,1]) 

    # finally convert to a tensor with autograd enabled (since we're 
    # going to be performing gradient updates on this image)
    sample = torch.from_numpy(np_sample).float().permute(2, 0, 1).unsqueeze(0).to(device) 
    sample.requires_grad = True
    
    return sample

class Saver():
    
    sample_directory = "data/samples"
    
    def __init__(self, save_mode):
        self.save_mode = save_mode
    
    def prepair_save_directory(self, params):
        self.identifier = params["id"]
        
        if self.save_mode == "final":
            self.dir_path = Saver.sample_directory + "/lol"
            
        elif self.save_mode == "throughout":
            self.dir_path = Saver.sample_directory + "/" + self.identifier
        
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
    
    def save_path(self, iteration):
        if self.save_mode == "final":
            file_name = self.identifier
        elif self.save_mode == "throughout":
            file_name = iteration
        
        return "{}/{}.jpg".format(self.dir_path , file_name)
    
    def save_image(self, image, iteration):
        save_image(image, self.save_path(iteration))
        