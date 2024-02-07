import torch
from torch import optim
from torch import nn
import numpy as np
import mstcn
import pickle, time
import random
from sklearn import metrics
import copy
from torchvision import models, transforms
import io
import trans_SV

# import zipfile

# with zipfile.ZipFile('model/model.zip', 'r') as zip_ref:
#     zip_ref.extractall('./model/')


seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

MSTCN_PATH = "./model/TransSV Trained Model/TeCNO50_epoch_6_train_9935_val_8924_test_8603.pth"

out_features = 7
num_workers = 3
batch_size = 1
mstcn_causal_conv = True
learning_rate = 1e-3
min_epochs = 12
max_epochs = 25
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim= 2048
mstcn_stages = 2


sequence_length = 30

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

weights_train = np.asarray([1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,])

criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
criterion_phase1 = nn.CrossEntropyLoss()

model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model.load_state_dict(torch.load(MSTCN_PATH,map_location=torch.device('cpu')))

from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
        
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)




def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx


