from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from logging import getLogger
import time
import os
import copy
import argparse
from torch.autograd import Variable
import csv
import random

from src.utils import (
    save_retrieval_results_to_csv
)

checkpoint_path =  "resnet50_checkpoint.pth.tar"
logger = getLogger()

# new script
# python resnet50_baseline.py --data_path "C:\Users\tomer\Datasets"  --results "C:\Users\tomer\researchsoftware\furniturestyle\logging.csv"  --num_classes 3 --topk 3 --lr 0.001 --wd 0.002

# Old script
#  python resnet50_baseline.py --data_path "C:\Users\tomer\Datasets" --num_classes 3

global_stats = {}

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

class Resnet50Baseline(nn.Module):
    def __init__(self, num_classes=3):
        super(Resnet50Baseline, self).__init__()
        model_embedding = models.resnet50(pretrained=True)
        num_ftrs = model_embedding.fc.in_features
        self.model_embedding = torch.nn.Sequential(*(list(model_embedding.children())[:-1]))
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model_embedding(x)
        x = x.squeeze(-1).squeeze(-1)
        return x, self.fc(x)

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
        misc_dict = dict(batch_size=args.batch_size,num_workers=args.num_workers)
        misc_dict.update(global_stats)
        misc_dict_str = str(misc_dict).replace(',', ';') # to be compliant with CSV file 
        str_out = str_out.format(model_name="resnet50_baseline",
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
                    stepsize=args.stepsize,
                    gamma=args.gamma, 
                    misc=misc_dict_str)
        f.write(str_out)
        f.close()

def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return
    
        logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )
    print("success in loading")

    # re load variable important for the run
    if run_variables is not None:
        print(run_variables)
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

                
def train_model(model, dataloaders, criterion, optimizer, scheduler, start_epoch = 0, num_epochs=25):
    global args, best_acc
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        
        stats_log = {"epoch": epoch}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, paths in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    embedding, outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            loss_log_str = "%s_loss" % (phase)
            acc_log_str = "%s_acc" % (phase)
            stats_log[loss_log_str] = "%.4f" % epoch_loss
            stats_log[acc_log_str] = "%.4f" % epoch_acc

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if args.rank == 0:
            log_results(args.results,stats_log)
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "init_lr": args.lr,
            }
            torch.save(save_dict, os.path.join(args.dump_path, checkpoint_path))

    print('Starting retrieval')
    print('-' * 10)
    phase = 'val'
    model.eval()  # Set model to evaluate mode

    paths_list = []
    # Iterate over data.
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(dataloaders[phase]):
            paths_list.extend(paths)
            if i == 0:
                targets = labels.numpy()
            else:
                targets = np.append(targets, labels.numpy(), 0)

            # compute distance of query embedding to others
            inputs = inputs.cuda(non_blocking=True)
            embedding, _ = model(inputs)
            if i == 0:
                embeddings = embedding.cpu().numpy()
            else:
                embeddings = np.append(embeddings, embedding.cpu().numpy(), 0)

    paths_list = np.array(paths_list)
    similar_images = dict()
    top1 = []
    topk = []
    for idx1, embedding1 in enumerate(embeddings):
        current_distances = np.array([np.linalg.norm(embedding1 - embedding2) for embedding2 in embeddings])
        current_distances /= np.max(current_distances)  # so that distances are in the range 0-1
        ascending_sorted_distances = np.argsort(current_distances)
        # Top 1 result
        top_1_idx = ascending_sorted_distances[1].astype(int)  # ignore the queried image itself
        top_1_target = targets[top_1_idx]
        top1.append(int(targets[idx1] == top_1_target))
        # Top k result
        top_results_idx = ascending_sorted_distances[1:args.topk + 1].astype(int)  # ignore the queried image itself
        top_results_targets = targets[top_results_idx]
        topk.append(int(targets[idx1] in top_results_targets))
        # All retrieval results
        all_sorted_results_idx = ascending_sorted_distances[1:].astype(int)  # ignore the queried image itself
        # log top similar image paths and distances
        similar_images[paths_list[idx1]] = [(path, dist) for path, dist in
                                       zip(paths_list[all_sorted_results_idx], current_distances[all_sorted_results_idx])]

    val_recall1 = np.mean(top1)
    val_recallk = np.mean(topk)
    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Recall@1 {}\t"
            "Recall@{} {}\t".format(
                val_recall1, args.topk, val_recallk))
        stats_log["val_recall1"] = "%.4f" % float(val_recall1)
        stats_log["val_recallk"] = "%.4f" % float(val_recallk)
        log_results(args.results,stats_log)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save similar image paths into csv
    save_retrieval_results_to_csv(similar_images, args.results[:-4] + "_similar_images.csv")
    logger.info("Test accuracies: top-1 {acc1:.1f}\n"
                "Top-1 recall: {recall1:.1f}\n"
                "Top-{topk} recall: {recallk:.1f}\n".format(
        acc1=best_acc,
        recall1=val_recall1, topk=args.topk, recallk=val_recallk))
    return model, loss, best_acc, similar_images, val_recall1, val_recallk





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Resnet")
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
    parser.add_argument("--results", type=str, default="results_log.csv",
                    help="path to log of training results")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="number of classes in the dataset")
    parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--epochs", default=50, type=int,
                    help="number of total epochs to run")
    parser.add_argument("--lr", default=0.001, type=float,
                    help="learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum")
    parser.add_argument("--stepsize", default=7, type=int,
                help="lr_scheduler decay LR by a factor of gamma every step_size epochs")
    parser.add_argument("--gamma", default=0.1, type=float,
            help="lr_scheduler decay LR by a factor of gamma every step_size epochs")
    parser.add_argument("--batch_size", default=4, type=int,
                help="DataLoader batch_size ")
    parser.add_argument("--num_workers", default=4, type=int,
                help="DataLoader num_workers ")
    parser.add_argument("--topk", type=int, default=3,
                        help="top K performance for accuracy and retrieval")
    args = parser.parse_args()
    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = args.data_path
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print("Current class list", class_names)
    global_stats["class_names"] = class_names

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.

    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #

    model_ft = Resnet50Baseline(args.num_classes)
    print(model_ft)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.stepsize, gamma=args.gamma)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    to_restore = {"epoch": 0, "best_acc": (0., 0.), "init_lr": args.lr}
    """
    restart_from_checkpoint(
        os.path.join(args.dump_path, checkpoint_path),
        run_variables=to_restore,
        state_dict=model_ft,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
    )
    """
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    model_ft, loss, best_acc, similar_images, val_recall1, val_recallk = train_model(model_ft, dataloaders,
                        criterion, optimizer_ft, exp_lr_scheduler,
                        start_epoch=start_epoch,
                        num_epochs=args.epochs)



"""
this should be in a CSV file 
model name, epoch, params_dict, trainloss, val loss, best acc seen 
Epoch 47/99
----------
train Loss: 0.7629 Acc: 0.6561
val Loss: 0.7649 Acc: 0.6550

train Loss: 0.7426 Acc: 0.6584
val Loss: 0.8138 Acc: 0.6395

Training complete in 100m 16s
Best val Acc: 0.660853


"""