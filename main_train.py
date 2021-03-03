import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
import argparse
import numpy as np
from models import *
import utils
import clear

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='chose from cifarnet, vgg11, resnet18')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use, if no GPU then CPU will be used')
parser.add_argument('--resume', '-r', type=str, default='',
                    help='resume from a saved model with its name')  # e.g. default='saved_model/cpkt.pth'


def analysis_data(dict, dict_name, epoch):
    global analysis_similarity
    global analysis_sparsity
    global plot_fmap

    if analysis_similarity:
        for key, value in dict.items():
            similarity_all_batch = []
            for i in range(len(value))[1:-1]:  # last batch may be fragmented
                # eliminate 0 in a because 0 may incur similarity
                value_prev_no_zero = value[i - 1][value[i - 1] != 0]
                value_after_no_zero = value[i][value[i - 1] != 0]

                # compute similar data
                similar_data = value_prev_no_zero[value_prev_no_zero == value_after_no_zero]
                similarity = 100. * len(similar_data) / len(value[i])
                similarity_all_batch.append(similarity)

            # save to log
            similarity_all_batch = np.array(similarity_all_batch).reshape((1, -1))
            file_name = f'fmap_similarity_csv/{dict_name}-{key}.csv'
            a_or_w = 'a' if os.path.exists(file_name) else 'w'
            with open(file_name, f'{a_or_w}b+') as f:
                np.savetxt(f, similarity_all_batch, delimiter=',', newline='\n', fmt='%.4f')

    if analysis_sparsity:
        for key, value in dict.items():
            # compute sparsity
            tensor_flat = value[0].flatten()  # get the tensor from batch 0 for every epoch
            sparsity = (1.0 - np.count_nonzero(tensor_flat) / tensor_flat.size) * 100.

            # save sparsity for batch 0 every epoch to text.
            file_name = f'fmap_sparsity_csv/{dict_name}-{key}.csv'
            a_or_w = 'a' if os.path.exists(file_name) else 'w'
            with open(file_name, f'{a_or_w}+') as f:
                f.write(f'{sparsity:.2f}\n')

            if plot_fmap:
                # make dir and grid
                os.mkdir(f'plot_fmap/{dict_name}_{key}') if not os.path.exists(f'plot_fmap/{dict_name}_{key}') else None
                ret = torchvision.utils.make_grid(tensor=torch.Tensor(value[0])).numpy()

                # plot gray image
                gray_img_ch0 = ret[0]
                title = f"{key}-epoch{epoch}-batch0"
                plt.title(title)
                plt.xlabel(f"{value[0].shape[0]} images in a batch, plot channel is [0]/[{ret.shape[1]}]")
                plt.ylabel(f"batch average sparsity {sparsity:.2f}%")
                plt.imshow(gray_img_ch0, cmap='gray')
                plt.savefig(f'plot_fmap/{dict_name}_{key}/{title}.jpeg')
                plt.close()


def hook_forward(name):
    global act

    def hook(module, input, output):
        if name not in act:
            act[name] = []
        act[name].append(output.cpu().detach().numpy())

    return hook


def hook_backward(name):
    global grad_act

    def hook(module, grad_input, grad_output):
        """
        :param grad_input:  For conv, it stores (batch * layer input gradient, bias gradient)
                            For fc, it stores (bias gradient, batch * layer input gradient, weight gradient)
        :param grad_output: For conv, it stores (batch * layer output gradient)
                            For fc, it stores (batch * layer output gradient)
        """
        if name not in grad_act:
            grad_act[name] = []
        grad_act[name].append(grad_output[0].cpu().detach().numpy())

    return hook


def register_hook(net):
    """
    :param register: True - register hook, False - deregister hook
    """
    hook_handler = []
    for name, layer in net.named_children():
        hook_handler.append(layer.register_forward_hook(hook_forward(name)))
        hook_handler.append(layer.register_backward_hook(hook_backward(name)))

    return hook_handler


def remove_hook(hook_handler):
    for handler in hook_handler:
        handler.remove()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, 1)
        torch.nn.init.zeros_(m.bias)


def cifar10_train(epoch, net):
    global act
    global grad_act
    global analysis_similarity
    global analysis_sparsity
    global plot_fmap

    # setup hook_epoch for
    if analysis_similarity:
        hook_epoch = range(0, len(trainloader), int(len(trainloader)/20))  # save 20 batches for analyze similarity
    else:
        hook_epoch = [0]  # else only save the first batch in an epoch

    # start training
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # attach hook
        hook_handler = register_hook(net) if batch_idx in hook_epoch else None

        # network forward and backward
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # detach hook
        remove_hook(hook_handler) if batch_idx in hook_epoch else None

        # compute loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # display progress bar
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # save training results
    print('Epoch {}, Train loss {}'.format(epoch, train_loss / len(trainloader)))
    file_name = f'log/log.csv'
    with open(file_name, 'a' if os.path.exists(file_name) else 'w') as f:
        f.write('{}, {}, '.format(epoch, train_loss / len(trainloader)))

    # save act, grad_act, weight as csv
    utils.save_as_csv(act, epoch, map_name='act')
    utils.save_as_csv(grad_act, epoch, map_name='grad_act')
    utils.save_weight_as_csv(net, epoch)

    # analysis data difference and save to log
    analysis_data(act, 'act', epoch)
    analysis_data(grad_act, 'grad_act', epoch)

    # clear the global buffer
    act = {}
    grad_act = {}


def cifar10_test(epoch):
    global best_acc

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # display progress bar
            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # save testing results
    print('Epoch %d, Test Loss: %.3f | Acc: %.3f%% ' % (epoch, test_loss / len(testloader), 100. * correct / total))
    with open('log/log.csv', 'a') as f:
        f.write('{}, {}\n'.format(test_loss / len(testloader), 100. * correct / total))

    # save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        print(f'model saved to checkpoint/ckpt.pth, acc={acc}, epoch={epoch}')


if __name__ == "__main__":
    # get the args and clear the directory
    args = parser.parse_args()
    clear.clear_dir(mk_new=True)

    # initialize the training
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True if 'cuda' in device else None
    print(f"training using device: {device}")
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # get the activation
    act = {}
    grad_act = {}

    # flag for data analysis
    analysis_similarity = False
    analysis_sparsity = True
    plot_fmap = True

    # preparing data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # preparing model
    net_all = {'cifarnet': CifarNet(),
               'vgg11': VGG('VGG11'), 'vgg16': VGG('VGG16'), 'vgg19': VGG('VGG19'),
               'resnet18': ResNet18(), 'resnet34': ResNet34(), 'resnet50': ResNet50(), 'resnet101': ResNet101()
               }
    net = net_all[args.arch].to(device)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # prepare training criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # start training
    for epoch in range(start_epoch, start_epoch + 200):
        cifar10_train(epoch, net)
        cifar10_test(epoch)
        scheduler.step()

    # plot after training
    utils.plot_analysis(plot_type='sparsity', max_epoch=args.epochs)
    utils.plot_train_curve_from_csv()
    utils.plot_batch_hist_from_csv(np.arange(1, args.epochs, args.epochs / 9).astype(int))
