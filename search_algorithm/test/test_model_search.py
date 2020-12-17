import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from search_space import utils

from search_space.model_search import Network
from torch.autograd import Variable

import search_space.genotypes

def main():
    class Architect(object):
        def __init__(self, model, criterion):
            self.model = model
            self.criterion = criterion

        def step_v2(self, input_train, target_train, input_valid, target_valid, lambda_train_regularizer):
            optimizer = torch.optim.Adam(
                self.model.arch_parameters(),
                lr=3e-4, betas=(0.5, 0.999),
                weight_decay=1e-3)
            optimizer.zero_grad()

            # grads_alpha_with_train_dataset
            logits = self.model(input_train)
            print('-----Architect model train finished-----')
            loss_train = self.criterion(logits, target_train)
            print('loss_train:', loss_train)

            arch_parameters = self.model.arch_parameters()
            grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)
            print('-----Autograd finished-----')
            optimizer.zero_grad()

            # grads_alpha_with_val_dataset
            logits = self.model(input_valid)
            print('-----Architect model valid finished-----')
            loss_val = self.criterion(logits, target_valid)

            arch_parameters = self.model.arch_parameters()
            grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)
            print('-----Autograd finished-----')

            for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
                temp = g_train.data.mul(lambda_train_regularizer)
                g_val.data.add_(temp)

            arch_parameters = self.model.arch_parameters()
            for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
                if v.grad is None:
                    v.grad = Variable(g.data)
                else:
                    v.grad.data.copy_(g.data)

            optimizer.step()

    np.random.seed(1)
    torch.manual_seed(1)

    criterion = nn.CrossEntropyLoss()
    model = Network(16, 10, 8, criterion)

    arch_parameters = model.arch_parameters()
    print('arch_parameters:', arch_parameters)

    arch_params = list(map(id, arch_parameters))

    parameters = model.parameters()
    weight_params = filter(lambda p: id(p) not in arch_params,
                           parameters)
    print("old size =", utils.count_parameters_in_MB(model), 'MB')

    optimizer = torch.optim.SGD(
        weight_params,  # model.parameters(),
        0.025,
        momentum=0.9,
        weight_decay=3e-4)

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data = dset.CIFAR10(root='../data', train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.5 * num_train))  # split index

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=64,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=64,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(50), eta_min=0.001)

    architect = Architect(model, criterion)

    best_accuracy = 0
    best_accuracy_different_cnn_counts = dict()

    print("param size =", model.get_current_model_size(), 'MB')
    for epoch in range(2):
        print('-----EPOCH', epoch, 'START-----')
        scheduler.step()
        train_acc, train_obj, train_loss = train(epoch, train_queue, valid_queue, model, architect, criterion,
                                                 optimizer)
        print("param size =", model.get_current_model_size(), 'MB')
        print('##############################################')
        print('train_acc, train_obj, train_loss', train_acc, train_obj, train_loss)
        with torch.no_grad():
            valid_acc, valid_obj, valid_loss = infer(valid_queue, model, criterion)

        genotype, normal_cnn_count, reduce_cnn_count = model.module.genotype()
        mode_size = normal_cnn_count + reduce_cnn_count
        print("model_size:", mode_size, "epoch:", epoch)

        # early stopping
        # if normal_cnn_count == 6 and reduce_cnn_count == 0:
        # break

        print("normal_cnn_count, reduce_cnn_count:", normal_cnn_count, reduce_cnn_count)
        print('Softmax - alphas_normal:', F.softmax(model.alphas_normal))
        print('Softmax - alphas_reduce:', F.softmax(model.alphas_reduce))

        print("genotype:", str(genotype))

        # save the cnn architecture according to the CNN count
        cnn_count = normal_cnn_count * 10 + reduce_cnn_count
        print("searching_cnn_count:", cnn_count, valid_acc, "epoch:", epoch)
        if cnn_count not in best_accuracy_different_cnn_counts.keys():
            best_accuracy_different_cnn_counts[cnn_count] = valid_acc
            summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            print(summary_key_cnn_structure)

            summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            print(summary_key_best_cnn_structure)
        else:
            if valid_acc > best_accuracy_different_cnn_counts[cnn_count]:
                best_accuracy_different_cnn_counts[cnn_count] = valid_acc
                summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                print(summary_key_cnn_structure)

                summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                print(summary_key_best_cnn_structure)

        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            print("best_valid_accuracy:", valid_acc)
            print("epoch_of_best_accuracy:", epoch)


def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    count = 0
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))

        print('-----Architect', count + 1, 'START-----')
        architect.step_milenas(input, target, input_search, target_search, 1)

        # Update weights w by SGD, ignore the weights that gained during architecture training

        # logging.info("step %d. update weight by SGD. START" % step)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        parameters = model.arch_parameters()
        nn.utils.clip_grad_norm_(parameters, 5)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        print('train - step, objs.avg, top1.avg, top5.avg:', step, objs.avg, top1.avg, top5.avg)

        if count == 0:
            break

        count += 1

    return top1.avg, objs.avg, loss


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        print('valid - step, objs.avg, top1.avg, top5.avg', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, loss


if __name__ == '__main__':
    main()
