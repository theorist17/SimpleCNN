from train import MyNet, get_paths, load_built_dataset, save_built_dataset, split

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

SEED: int = 0
SHOW_IMAGE_ON_TEST: bool = False
SHOW_LOG_ON_TEST: bool = False

def test(model_name, seed=SEED):
    # test the model specified by model_name

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    file_paths, dir_paths, urls, model_dir = get_paths()
    (data, label, id2lbl, lbl2id) = load_built_dataset()
    train_x, val_x, test_x, train_y, val_y, test_y = split(data, label, seed)

    ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = lambda x: DataLoader(x, batch_size=10, shuffle=True)
    train, val, test = ds(train_x, train_y), ds(val_x, val_y), ds(test_x, test_y)
    train_loader, val_loader, test_loader = dl(train), dl(val), dl(test)

    model = MyNet()
    model.cuda()
    summary(model, input_size=(3, 256, 256))

    pth = Path(model_dir).joinpath(model_name + '.pth')

    if pth:
        print('Test')
        model.load_state_dict(torch.load(pth))
        model.eval()

        with torch.no_grad():
            correct = 0
            for i, mini_batch in enumerate(test_loader):
                inputs, labels = mini_batch
                images = inputs.numpy()
                inputs, labels = inputs.to(device, dtype=torch.float), labels

                outputs = model(inputs).to('cpu')

                values, indices = torch.max(outputs, dim=1) # prediction 0 is human, 1 is plant

                labels = labels.tolist()
                indices = indices.tolist()

                images = np.transpose(images, (0, 2, 3, 1)) # from Tensor into numpy


                for j in range(len(labels)):
                    if SHOW_LOG_ON_TEST:
                        print(i * 10 + j + 1, '\tGround Truth:', labels[j], 'Prediction:', indices[j])

                    if indices[j] == labels[j]:
                        correct += 1

                    if SHOW_IMAGE_ON_TEST:
                        image = images[j]
                        plt.subplot(4, 4, j + 1)
                        plt.imshow(np.asarray(image, dtype=np.uint8))
                if SHOW_IMAGE_ON_TEST:
                    plt.show()
                if SHOW_LOG_ON_TEST:
                    print()

            accuracy = correct / len(test_loader.dataset)

            print('Test Accuracy', accuracy)

if __name__=='__main__':
    test('MyNet4', seed=0)