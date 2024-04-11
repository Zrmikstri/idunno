import torch 

from torchvision.models import vgg16, VGG16_Weights

model = vgg16(weights='DEFAULT')
print(model)