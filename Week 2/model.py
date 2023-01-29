import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import torchvision
from torchvision import transforms, datasets

class NeuralNet(nn.Module):
     def __init__(self, input_size, hidden_size, num_classes):
         super(NeuralNet, self).__init__()
         self.input_size = input_size
         self.l1 = nn.Linear(input_size, hidden_size) 
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(hidden_size, num_classes)  
     def forward(self, x):
         out = self.l1(x)
         out = self.relu(out)
         out = self.l2(out)
         return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, num_classes).to(device)