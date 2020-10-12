import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.autograd import Variable

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def train_simplenet(model, dataset, optimizer, nepochs, batch_size):
    criterion = nn.MSELoss()
    
    pbar = tqdm(range(nepochs))

    x_train, y_train, x_test, y_test = dataset
    ndatapoints = len(x_train)

    for epoch in pbar:
        indices = np.random.randint(0, ndatapoints, batch_size)
        x_batch = x_train[indices]
        y_batch = y_train[indices]

        X = Variable(torch.from_numpy(x_batch)).to(device)
        Y = Variable(torch.from_numpy(y_batch)).to(device)

        y_pred = model(X)
        loss = criterion(y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0: 
            with torch.no_grad():
                X = Variable(torch.from_numpy(x_test)).to(device)
                Y = Variable(torch.from_numpy(y_test)).to(device)

                y_pred = model(X)
                loss = criterion(y_pred, Y).mean()
                
                pbar.set_description(f"Epoch {epoch}; Loss {np.round(loss.item(), 5)}")


def evaluate_simplenet(model, dataset):
    criterion = nn.MSELoss()
    with torch.no_grad():
        _, _, x_test, y_test = dataset
        X = Variable(torch.from_numpy(x_test)).to(device)
        Y = Variable(torch.from_numpy(y_test)).to(device)

        y_pred = model(X)
        loss = criterion(y_pred, Y).mean()

        return loss.item()