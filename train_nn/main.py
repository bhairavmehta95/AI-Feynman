import argparse
import random
import numpy as np
import torch
import torch.optim as optim

from train import train_simplenet, evaluate_simplenet
from net import SimpleNet

def load_data(filename):
    arr = np.loadtxt(filename)
    np.random.shuffle(arr)

    npoints = arr.shape[0]
    print(arr.shape)

    training = int(npoints * 0.9)

    x_train, y_train = arr[:training, :-1], arr[:training, -1]
    x_test, y_test = arr[training+1:, :-1], arr[training+1:, -1]

    return x_train, y_train[:, np.newaxis], x_test, y_test[:, np.newaxis]
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Symbolic MTL')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nepochs', type=int, default=5000)
    parser.add_argument('--filename', type=str, default='../example_data/example1.txt')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    dataset = load_data(args.filename)
    
    print([x.shape for x in dataset])

    ninputs = dataset[0].shape[1]
    model = SimpleNet(ninputs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_simplenet(model, dataset, optimizer, nepochs=args.nepochs, batch_size=128)


    print(evaluate_simplenet(model, dataset))