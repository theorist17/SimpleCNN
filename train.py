# Use one of image classification algorithm & programming tools to solve the problem
# and submit the source code
# Problem : from ImageNet data source, choose two categories of certain images,
# and build training model, and calculate the testing accuracy on the test dataset,
# from which the model had not learned.

# Since ImageNet and ILSVCR servers are under maintenance for months, I decided to
# used the images from sysnet-dependent image dataset.
# I chose one category of "animal, animate being, beast, brute, creture, fauna"
# and "person, individual, someone, somebody, mortal, soul" for the other,
# because both of them have relatively bigger size in scale, 3998 and 6978 respectively.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

import tqdm
from urllib.request import urlretrieve
import requests
from pathlib import Path
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

SEED: int = 0
DOWNLOAD_DATASET: bool = False
SHOW_IMAGE_ON_VALIDATION: bool = False
SHOW_LOG_ON_VALIDATION: bool = False

FILE_PATHS = {'person': 'urls_person.txt',
              'plant': 'urls_plant.txt'}
DIR_PATHS = {'person': "data/person",
             'plant': "data/plant"}
URLS = {'person': 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846',
        'plant': 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222'}
MODEL_DIR = "models"

def download(urls, file_paths, dir_paths):
    # download images from urls
    for k, url in urls.items():
        # create text file of urls
        r = requests.get(url, allow_redirects=True)

        url_dir = Path('urls')
        url_dir.mkdir(exist_ok=True)
        open(Path.joinpath(url_dir, file_paths[k]), 'wb').write(r.content)

        # create local data directory
        Path(dir_paths[k]).mkdir(parents=True, exist_ok=True)

        # download actual images
        print('Downloading', k)
        line_number = len(open(file_paths[k]).readlines())
        error_cnt = 0
        with tqdm.tqdm(total=line_number) as pbar:
            with open(file_paths[k], "r") as f:
                for line in f:
                    a_url = line.strip()
                    if a_url:
                        filename = a_url.rsplit('/', 1)[1]
                        if not Path(dir_paths[k] + '/' + filename).exists():
                            try:
                                urlretrieve(a_url, dir_paths[k] + '/' + filename)
                            except Exception as e:
                                error_cnt += 1
                    pbar.update()
        print('Error:', error_cnt)


def build_dataset(dir_paths):
    # build dataset by collecting url-retrieved images into the give directory paths
    # while building images, also build horizontally-flipped version of them
    # for better regualization of the model
    data = []
    label = []
    id2lbl = dict(zip(range(len(dir_paths)), dir_paths.keys()))
    lbl2id = dict(zip(dir_paths.keys(), range(len(dir_paths))))

    print('Building Dataset')
    file_num = 0
    for path in dir_paths.values():
        _, _, files = next(os.walk(path))
        file_num += len(files)

    error = 0
    with tqdm.tqdm(total=file_num) as pbar:
        for lbl, path in dir_paths.items():
            dir = Path(path)
            Path.joinpath(dir.parent, 'new' + lbl).mkdir(parents=True, exist_ok=True)
            new_dir = Path(Path.joinpath(dir.parent, 'new' + lbl))

            for file in dir.iterdir():
                try:
                    img = Image.open(file).convert('RGB')
                except Exception:
                    # file.unlink()
                    error += 1
                else:
                    img = img.resize((256, 256))

                    arr = np.array(img)
                    trans = np.transpose(arr, (2, 0, 1)) # from numpy to Tensor
                    data.append(trans)

                    img = arr.astype('uint8')
                    img = Image.fromarray(img, 'RGB')
                    img.save(new_dir.absolute().as_posix() + '/' + file.stem + '.png')

                    label.append(np.array(lbl2id[lbl]))

                    # simple data augmentation for model regulariztion purpose
                    flip = np.fliplr(arr)
                    trans = np.transpose(flip, (2, 0, 1))
                    data.append(trans)

                    flip = flip.astype('uint8')
                    flip = Image.fromarray(flip, 'RGB')
                    flip.save(new_dir.absolute().as_posix() + '/' + file.stem + '_flip.png')

                    label.append(np.array(lbl2id[lbl]))

                pbar.update()

    data, label = np.array(data), np.array(label)
    assert len(data) == len(label)

    print('Dataset Size:', len(data))
    print('Label Info:', lbl2id)
    print('Number of Error:', error)

    return data, label, id2lbl, lbl2id


def split(data, label, seed=0):
    # split dataset into 3 groups of train, validation, test
    # train : model learn from
    # validation : used for hyper parameter optimization
    # test : used in test.py at the last moment
    #
    # prevent model from learning test dataset
    # by setting the same seed on both train.py and test.py
    size = len(data)
    assert len(data) == len(label)

    np.random.seed(seed)
    ids = np.random.permutation(len(data))
    sp1, sp2 = map(int, [size * 0.7, size * 0.9])
    train, val, test = ids[:sp1], ids[sp1:sp2], ids[sp2:]

    train, val, test = train.astype(int), val.astype(int), test.astype(int)
    train, val, test = np.asarray(train), np.asarray(val), np.asarray(test)

    train_x, val_x, test_x = data[train], data[val], data[test]
    train_y, val_y, test_y = label[train], label[val], label[test]

    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)
    assert len(test_x) == len(test_y)

    print('Train Datatset Size:', len(train))
    print('Validation Dataset Size:', len(val))
    print('Test Dataset Size:', len(test))

    return train_x, val_x, test_x, train_y, val_y, test_y


def get_paths():
    # getting module parameters specifying paths
    return FILE_PATHS, DIR_PATHS, URLS, MODEL_DIR

def load_built_dataset():
    # load dataset that was built and serialized through numpy in advance
    # if no dataset was serialized before build one and save it to ./data/build
    # delete build directory to rebuild the dataset inside both in ./data/newperson
    # and ./data/newplant
    dir = Path('data/build')
    keys = ['data.npy', 'label.npy', 'id2lbl.npy', 'lbl2id.npy']
    values = []

    try:
        print('Deserializing Dataset')
        for filename in keys:
            file = Path.joinpath(dir, filename)
            with open(file, 'rb') as f:
                values.append(np.load(f, allow_pickle=True))
    except:
        print('No preprocessed dataset\n')
        data, label, id2lbl, lbl2id = build_dataset(DIR_PATHS)
        values = [data, label, id2lbl, lbl2id]

        files = dict(zip(keys, values))
        save_built_dataset(files)

    return values


def save_built_dataset(files):
    # serialize dataset into numpy file (.npy) into ./data/build
    print('Serializing Dataset')
    dir = Path('data/build')
    dir.mkdir(exist_ok=True)

    for filename, value in files.items():
        file = Path.joinpath(dir, filename)
        np.save(file, value)


class MyNet(nn.Module):
    # Custom network of my own
    # consists of six layers of convolutional network inspired by AlexNet and VGGNet
    # which were good at 256 x 256 size small images when ImageNet was small back in 15'
    # Used regualization method of batch normalization, drop-out
    # and also experimented Xavier initialization and Leaky-ReLU

    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  # combine small kernels for big receptive fields
        init.xavier_uniform_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu')) # ReLU is not immune to
                                                                                          # Gradient vanishing
        self.bn1 = nn.BatchNorm2d(num_features=16)  # The half of ReLUs will die, so bit of activation regualization
        self.relu1 = nn.ReLU(inplace=True)
        # no pooling

        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        init.xavier_uniform_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.relu2 = nn.ReLU(inplace=True)
        # pooling

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  # increase channels for bigger features
        init.xavier_uniform_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU(inplace=True)
        # no pooling

        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        init.xavier_uniform_(self.conv4.weight.data, gain=nn.init.calculate_gain('relu'))
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.relu4 = nn.ReLU(inplace=True)
        # pooling

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        init.xavier_uniform_(self.conv5.weight.data, gain=nn.init.calculate_gain('relu'))
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.ReLU(inplace=True)
        # no pooling

        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        init.xavier_uniform_(self.conv6.weight.data, gain=nn.init.calculate_gain('relu'))
        self.bn6 = nn.BatchNorm2d(num_features=64)
        self.relu6 = nn.ReLU(inplace=True)
        # pooling

        self.num_features, self.filter_size = 64, 32 # each filter has 32  * 32 pixels by now. Further pooling
                                                     # into 16 * 16 seems no good.

        self.fc1 = nn.Linear(self.num_features * self.filter_size ** 2, 50)
        init.xavier_uniform_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True) # tried Leaky ReLu
        self.drop1 = nn.Dropout() # tried drop-out instead of usual batch normalization

        self.fc2 = nn.Linear(50, 2)
        init.xavier_uniform_(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
        self.drop2 = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, self.num_features * self.filter_size ** 2)

        x = self.fc1(x)
        x = self.leaky1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


def save_model(model, model_name):
    # save torch model into model directory ./models
    Path(MODEL_DIR).mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR + '/' + model_name + '.pth')
    print('Model saved:', model_name)


def train(model_name=None, lr=0.001, seed=SEED, acc_threshold=0.9, total_epoch=100):
    # train model with different learning rate optimization
    # as the model goes thorough the epochs, perform one training from the entire
    # training set and perfom one validation for the whole validation set
    # (the dataset is determinisically divided by seed)
    # if the mean of accuracies from last 10 epoches are less than acc_threshold value,
    # terminate the process early. if not loop until it reaches total_epoch.

    # model name is either the specific one or the time of the train module started
    model_name = model_name if model_name else time.strftime("%Y-%m-%d-%H:%M")

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    # set DOWNLOAD_DATASET variable true on the top of module to download the dataset
    if DOWNLOAD_DATASET:
        download(URLS, FILE_PATHS, DIR_PATHS)
    (data, label, id2lbl, lbl2id) = load_built_dataset()
    train_x, val_x, test_x, train_y, val_y, test_y = split(data, label, seed)

    ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = lambda x: DataLoader(x, batch_size=10, shuffle=True)
    train, val, test = ds(train_x, train_y), ds(val_x, val_y), ds(test_x, test_y)
    train_loader, val_loader, test_loader = dl(train), dl(val), dl(test)

    model = MyNet()
    model.cuda()
    summary(model, input_size=(3, 256, 256))

    criterion = nn.CrossEntropyLoss() # Classic image recognition setup
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train module return both losses and accuracies where it will be plotted and logged
    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(total_epoch):
        print('Epoch:', epoch + 1)

        model.train()
        train_loss = 0.0

        for i, mini_batch in enumerate(train_loader, start=0):
            inputs, labels = mini_batch
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print('Train Loss:', train_loss)
        train_losses.append(train_loss)


        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            correct = 0
            for i, mini_batch in enumerate(val_loader):
                inputs, labels = mini_batch
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                labels = labels.to('cpu')
                values, indices = torch.max(outputs.to('cpu'), dim=1)
                correct += np.sum(labels.numpy() == indices.numpy())

        accuracy = correct / len(val_loader.dataset)

        print('Validation Loss:', val_loss)
        print('Validation Accuracy', accuracy)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        if epoch % 10 == 9: # Save model between every 10 epoches
            save_model(model, model_name)
        if np.mean(accuracies[-10:]) > acc_threshold: # Finish training depending

            break
    print('Finished', model_name)
    save_model(model, model_name)
    return train_losses, val_losses, accuracies


if __name__ == '__main__':
    # for hyper parameter optimization (tested only the learning rate, though)
    hyper_parameter = [0.0005]#, 0.001, 0.0001, 0.005, 0.01] # usually compare size of 5
    seeds = [0] # choose seed and match exact the same on test.py

    plt.figure(0) # loss plot
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

    plt.figure(1) # accuracy plot
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')

    i = 0
    for x in hyper_parameter:
        for seed in seeds:
            dir = Path('logs')
            dir.mkdir(exist_ok=True)

            # run train and validation
            i += 1
            model_name = 'MyNet'+str(i)
            print('\nModel Name:', model_name)
            train_losses, val_losses, accuracies = train(model_name=model_name,
                                                         lr=x,
                                                         seed=seed,
                                                         total_epoch=120,
                                                         acc_threshold=97)

            plt.figure(0)

            # plot train loss
            plt.plot(np.array(train_losses), label=model_name+'Train')
            plt.draw()

            # validation loss
            plt.plot(np.array(val_losses), label=model_name+'Val')
            plt.draw()

            plt.legend()
            plt.savefig(Path.joinpath(dir, model_name + 'Loss.png'))

            plt.figure(1)

            # accuracy
            plt.plot(np.array(accuracies), label=model_name + 'Acc')
            plt.draw()

            plt.legend()
            plt.savefig(Path.joinpath(dir, model_name + 'Acc.png'))
