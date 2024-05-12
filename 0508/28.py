import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True)
fc = nn.Sequential(
       nn.Linear(512 * 7 * 7, 4096),
       nn.ReLU(),
       nn.Dropout(),
       nn.Linear(4096, 4096),
       nn.ReLU(),
       nn.Dropout(),
       nn.Linear(4096, 10),
   )

model.classifier = fc
model.to(device)