import random
import numpy as np
from PIL import Image

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='RGB')            
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask)

class Shadows(object):
    def __init__(self):
        # load all shadows
        self.shadows = []
        for i in range(1,7+1):
            png = Image.open('shadows/{}.png'.format(i))
            self.shadows.append(png)

    def __call__(self, img):
        idx = random.randint(1,7)-1
        
        shadow = self.shadows[idx].resize(img.size)       

        shadow = self.shadows[idx].resize(img.size)
        img.paste(shadow, (0, 0), shadow)
        
        return img