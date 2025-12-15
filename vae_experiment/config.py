""" Конфиг """

import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 128
lr = 1e-3
num_epochs = 30
latent_dim = 16
beta = 1.0  # коэффициент при KL (можно менять)
class_idx = 2  # класс FashionMNIST, на котором обучаемся (0..9). По умолчанию 2 = Свитер
save_interval = 5
