# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.autograd import Variable

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
    save_retrieval_results_to_csv
)
import src.resnet50 as resnet_models
import numpy as np
import csv
import random

logger = getLogger()
global_stats = {}

checkpoint_path = "checkpoint_eval_linear.pth.tar"
best_checkpoint_path = "best_checkpoint_eval_linear.pth.tar"

# TODO tune the number of prototypes -> this is equiv. to "clusters" that the algorithm needs to tune  

# new script:
# python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --results "C:\Users\tomer\researchsoftware\furniturestyle\logging.csv"   --data_path "C:\Users\tomer\Datasets" --pretrained checkpoints/swav_800ep_pretrain.pth.tar --num_classes 3 --topk 3 --lr 0.001 --wd 0.002

# Old script:
# python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py --data_path "C:\Users\tomer\Datasets" --pretrained checkpoints/swav_800ep_pretrain.pth.tar --num_classes 3

parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--num_classes", type=int, default=1000,
                    help="number of classes in the dataset")
parser.add_argument("--topk", type=int, default=3,
                    help="top K performance for retrieval")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=50, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,  
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")



#########################
#### logging parameters ###
#########################
parser.add_argument("--results", type=str, default="results_log.csv",
                    help="path to log of training results")

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def log_results(file_path,stats={"train_loss":"NA", "train_acc":"NA","val_loss":"NA","val_acc":"NA", "val_recall1":"NA","val_recallk":"NA"}):
    if not os.path.exists(file_path):
        with open(file_path,'w') as f:
            str_out = "{model_name},{num_classes},{train_loss},{train_acc},{val_loss},{val_acc},{val_recall1},{val_recallk},{epoch},{learning_rate},{momentum},{stepsize},{gamma},{misc}\n"
            str_out = str_out.format(model_name="model_name",
                        num_classes="num_classes",
                        train_loss="train_loss",
                        train_acc="train_acc",
                        val_loss="val_loss",
                        val_acc="val_acc",
                        val_recall1="val_recall1",
                        val_recallk="val_recallk",
                        epoch="epoch",
                        learning_rate="learning_rate",momentum="momentum",
                        stepsize="stepsize",gamma="gamma", misc="misc")
            f.write(str_out)
            f.close()
    with open(file_path,'a') as f:
        str_out = "{model_name},{num_classes},{train_loss},{train_acc},{val_loss},{val_acc},{val_recall1},{val_recallk},{epoch},{learning_rate},{momentum},{stepsize},{gamma},{misc}\n"
        misc_dict = dict(batch_size=args.batch_size,
                        num_workers=args.workers)
        misc_dict.update(global_stats) # TODO - fill with class names in data
        misc_dict_str = str(misc_dict).replace(',', ';') # to be compliant with CSV file 
        str_out = str_out.format(model_name="eval_linear.py",
                    num_classes=args.num_classes,
                    train_loss=stats["train_loss"],
                    train_acc=stats["train_acc"],
                    val_loss=stats["val_loss"],
                    val_acc=stats["val_acc"],
                    val_recall1="val_recall1",
                    val_recallk="val_recallk",
                    epoch=stats["epoch"],
                    learning_rate=args.lr,
                    momentum=args.momentum,
                    stepsize="MultiStepLR", # TODO check if there are specific multistepLR params 
                    gamma=args.gamma, 
                    misc=misc_dict_str)
        f.write(str_out)
        f.close()


def main():
    global args, best_acc
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec3", "loss_val", "prec1_val", "prec3_val"
    )

    # build data
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"))
    val_dataset = ImageFolderWithPaths(os.path.join(args.data_path, "val"))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done")

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)
    linear_classifier = RegLog(args.num_classes, args.arch, args.global_pooling, args.use_bn)

    # convert batch norm layers (if any)
    linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(
        linear_classifier,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        nesterov=args.nesterov,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    # set scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if False:
        restart_from_checkpoint(
            os.path.join(args.dump_path, checkpoint_path),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_loader.sampler.set_epoch(epoch)

        scores = train(model, linear_classifier, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model, linear_classifier)
        similar_images, val_recall1, val_recallk = retrieval(val_loader, model)

        best_acc_updated = scores_val[-1]
        scores_val = scores_val[:-1]
        val_loss = scores_val[0]
        val_acc =  scores_val[1] / 100.
        train_loss = scores[1] 
        train_acc = scores[2] / 100. 
        stats_log = {"epoch": epoch} 
        stats_log["val_loss"] = "%.4f" % val_loss
        stats_log["val_acc"] = "%.4f" % val_acc
        stats_log["train_loss"] = "%.4f" % train_loss
        stats_log["train_acc"] = "%.4f" % train_acc
        stats_log["val_recall1"] = "%.4f" % float(val_recall1)
        stats_log["val_recallk"] = "%.4f" % float(val_recallk)
        log_results(args.results,stats_log)

        training_stats.update(scores + scores_val)

        scheduler.step()

        # save checkpoint
        if args.rank == 0:    
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.dump_path, checkpoint_path))
            
            if best_acc_updated:
              print("Best model updated")
              save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
              }
              torch.save(save_dict, os.path.join(args.dump_path, best_checkpoint_path))

    # Save similar image paths into csv
    save_retrieval_results_to_csv(similar_images, args.results[:-4] + "_similar_images.csv")
    logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}\n"
                "Top-1 recall: {recall1:.1f}\n"
                "Top-{topk} recall: {recallk:.1f}\n".format(acc=best_acc, recall1=val_recall1, topk=args.topk, recallk=val_recallk))


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def train(model, reglog, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    topk = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.eval()
    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            _, output = model(inp)
        output = reglog(output)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        topk.update(acc3[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    return epoch, losses.avg, top1.avg.item(), topk.avg.item()


def validate_network(val_loader, model, linear_classifier):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    best_acc_updated = False
    global best_acc

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target, _) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            # forward
            _, output = model(inp)
            output = linear_classifier(output)
            loss = criterion(output, target)

            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            topk.update(acc3[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()
        best_acc_updated = True

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc))

    return losses.avg, top1.avg.item(), topk.avg.item(), best_acc_updated


def retrieval(val_loader, model):
    model.eval()

    with torch.no_grad():
        paths = []
        for i, (inp, target, path) in enumerate(val_loader):
            paths.extend(path)
            if i == 0:
                targets = target.numpy()
            else:
                targets = np.append(targets, target.numpy(), 0)

            # compute distance of query embedding to others
            inp = inp.cuda(non_blocking=True)
            embedding, _ = model(inp)
            if i == 0:
                embeddings = embedding.cpu().numpy()
            else:
                embeddings = np.append(embeddings, embedding.cpu().numpy(), 0)
        print('emnedding',len(embeddings),embeddings.shape)
        with open('eval_linear_embeddings.npy', 'wb') as f:
            np.save(f,embeddings)
        

    paths = np.array(paths)
    similar_images = dict()
    top1 = []
    topk = []
    with open('eval_linear_embeddings_image_paths.csv', 'w') as f:
        for idx1, embedding1 in enumerate(embeddings):
            current_distances = np.array([np.linalg.norm(embedding1 - embedding2) for embedding2 in embeddings])
            current_distances /= np.max(current_distances)  # so that distances are in the range 0-1
            ascending_sorted_distances = np.argsort(current_distances)
            # Top 1 result
            top_1_idx = ascending_sorted_distances[1].astype(int)  # ignore the queried image itself
            top_1_target = targets[top_1_idx]
            top1.append(int(targets[idx1] == top_1_target))
            # Top k result
            top_results_idx = ascending_sorted_distances[1:args.topk+1].astype(int)  # ignore the queried image itself
            top_results_targets = targets[top_results_idx]
            topk.append(int(targets[idx1] in top_results_targets))
            # All retrieval results
            all_sorted_results_idx = ascending_sorted_distances[1:].astype(int)  # ignore the queried image itself
            # log top similar image paths and distances
            similar_images[paths[idx1]] = [(path, dist) for path, dist in
                                        zip(paths[all_sorted_results_idx], current_distances[all_sorted_results_idx])]
            f.write("%s\n" % (paths[idx1] ))

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Recall@1 {}\t"
            "Recall@{} {}\t".format(
                np.mean(top1), args.topk, np.mean(topk)))

    """
    # Load and fw pass query image
    query_image_loaded = Image.open(os.path.join(args.data_path, query_image_path))
    query_image_loaded = val_transform(query_image_loaded).float()
    query_image_loaded = Variable(query_image_loaded, requires_grad=True)
    query_image_loaded = query_image_loaded.unsqueeze(0).cuda(non_blocking=True)
    query_embedding, _ = linear_classifier(model(query_image_loaded))
    """
    return similar_images, np.mean(top1), np.mean(topk)

if __name__ == "__main__":
    main()
