import os.path
import shutil
from logging import getLogger
from pprint import pprint

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
import argparse
import pandas as pd

from modified_eval_linear import main as linear_main
from modified_eval_semisup import main as semisup_main
# python
# -m torch.distributed.launch
# --nproc_per_node = 1
# modified_eval_linear.py
# --results "logging.csv"
# --data_path "/home/fantasticapple/Data/3D-FUTURE-model"
# --pretrained "checkpoints/swav_800ep_pretrain.pth.tar"
# --num_classes 3
# --topk 3
# --lr 0.001
# --wd 0.002

momentum_list = [0.9, 0.5, 0.25, 0]
decay_list = [1e-4, 1e-3, 1e-2, 1e-1, 0]
#
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]

main_data_path = "D:/Data/3D-FUTURE-model/"
pretrained_path = "checkpoints/swav_800ep_pretrain.pth.tar"

def method1_wrap():

    root_dir = "method2"

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    directory = str(momentum_list[0]) + "_" + str(decay_list[0]) + "_" + str(lr_list[0])

    parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

    global checkpoint_path, best_checkpoint_path


    checkpoint_path = "checkpoint_eval_linear.pth.tar"
    best_checkpoint_path = "best_checkpoint_eval_linear.pth.tar"

    #########################
    #### main parameters ####
    #########################
    parser.add_argument("--dump_path", type=str, default=root_dir + "/" + directory,
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--data_path", type=str, default=main_data_path,
                        help="path to dataset repository")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="number of classes in the dataset")
    parser.add_argument("--topk", type=int, default=3,
                        help="top K performance for retrieval")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")

    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--pretrained", default=pretrained_path, type=str, help="path to pretrained weights")
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
    parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
    parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
    # for multi-step learning rate decay
    parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                        help="Epochs at which to decay learning rate.")
    parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
    # for cosine learning rate schedule
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

    parser.add_argument("--momentum", default=momentum_list[0], type=float, help="momentum")
    parser.add_argument("--lr", default=lr_list[0], type=float, help="initial learning rate")
    parser.add_argument("--wd", default=decay_list[0], type=float, help="weight decay")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="./somefile", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                                        number of processes: it is set automatically and
                                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    # parser.add_argument("--local_rank", default=0, type=int,
    #                     help="this argument is not used and should be ignored")

    #########################
    #### logging parameters ###
    #########################
    parser.add_argument("--results", type=str, default=root_dir + "/" + directory + "/" + "log.csv",
                        help="path to log of training results")

    args = parser.parse_args()
    # print(args.gpu_to_work_on)
    init_distributed_mode(args)



    for momentum in momentum_list:

        args.momentum = momentum

        for weight_decay in decay_list:

            args.wd = weight_decay

            for lr in lr_list:

                args.lr = lr

                directory = str(momentum) + "_" + str(weight_decay) + "_" + str(lr)

                args.dump_path = root_dir + "/" + directory
                args.results = root_dir + "/" + directory + "/" + "log.csv"

                if not os.path.exists(root_dir + "/" + directory):
                    os.mkdir(root_dir + "/" + directory)
                    # shutil.copy("params.pkl", root_dir + "/" + directory + "/" + "params.pkl")
                else:
                    log = pd.read_csv(root_dir + "/" + directory + "/" + "log.csv")

                    if len(log['epoch'].values) >= args.epochs - 1:
                        print("Skip trained hyperparameter combination")
                        continue

                fix_random_seeds(args.seed)
                logger, training_stats = initialize_exp(
                    args, "epoch", "loss", "prec1", "prec3", "loss_val", "prec1_val", "prec3_val"
                )

                print("Current Setting: " + directory)

                # linear_main(args, logger, training_stats)
                semisup_main(args, logger, training_stats)

def method2_wrap():


    root_dir = "method2"

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    directory = str(momentum_list[0]) + "_" + str(decay_list[0]) + "_" + str(lr_list[0])

    parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

    global checkpoint_path, best_checkpoint_path

    checkpoint_path = "checkpoint_eval_semisup.pth.tar"
    best_checkpoint_path = "best_checkpoint_eval_semisup.pth.tar"

    #########################
    #### main parameters ####
    #########################
    parser.add_argument("--labels_perc", type=str, default="100",
                        help="fine-tune on either 1% or 10% of labels")
    parser.add_argument("--dump_path", type=str, default=root_dir + "/" + directory,
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--data_path", type=str, default=main_data_path,
                        help="path to imagenet")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="number of classes in the dataset")
    parser.add_argument("--topk", type=int, default=3,
                        help="top K performance for retrieval")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")

    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--pretrained", default="", type=str, help=pretrained_path)

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=50, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate - trunk")
    parser.add_argument("--lr_last_layer", default=0.2, type=float, help="initial learning rate - head")
    parser.add_argument("--wd", default=0.0001, type=float, help="weight decay")
    parser.add_argument("--decay_epochs", type=int, nargs="+", default=[12, 16],
                        help="Epochs at which to decay learning rate.")
    parser.add_argument("--gamma", type=float, default=0.2, help="lr decay factor")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="file:///D:/Github/styleestimation-master/somefile.txt", type=str,
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
    parser.add_argument("--results", type=str, default=root_dir + "/" + directory + "/" + "log.csv",
                        help="path to log of training results")

    args = parser.parse_args()
    # print(args.gpu_to_work_on)
    init_distributed_mode(args)

    for momentum in momentum_list:

        args.momentum = momentum

        for weight_decay in decay_list:

            args.wd = weight_decay

            for lr in lr_list:

                args.lr = lr

                directory = str(momentum) + "_" + str(weight_decay) + "_" + str(lr)

                args.dump_path = root_dir + "/" + directory
                args.results = root_dir + "/" + directory + "/" + "log.csv"

                if not os.path.exists(root_dir + "/" + directory):
                    os.mkdir(root_dir + "/" + directory)
                    # shutil.copy("params.pkl", root_dir + "/" + directory + "/" + "params.pkl")
                else:
                    log = pd.read_csv(root_dir + "/" + directory + "/" + "log.csv")

                    if len(log['epoch'].values) >= args.epochs - 1:
                        print("Skip trained hyperparameter combination")
                        continue

                fix_random_seeds(args.seed)
                logger, training_stats = initialize_exp(
                    args, "epoch", "loss", "prec1", "prec3", "loss_val", "prec1_val", "prec3_val"
                )

                print("Current Setting: " + directory)

                # linear_main(args, logger, training_stats)
                semisup_main(args, logger, training_stats)
if __name__ == "__main__":
    # 0.5_0.001_0.0001
    extend_dict = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}
    os.environ.update(extend_dict)
    pprint(os.environ)
    # method1_wrap()
    method2_wrap()

    # environ({'PATH': '/home/fantasticapple/Tool/anaconda3/envs/swav/bin:/home/fantasticapple/Tool/anaconda3/condabin:/home/fantasticapple/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/fantasticapple/.dotnet/tools:/home/fantasticapple/.local/share/JetBrains/Toolbox/scripts', 'LC_MEASUREMENT': 'en_US.UTF-8', 'XAUTHORITY': '/run/user/1000/gdm/Xauthority', 'INVOCATION_ID': 'cfe99ab5c6b544fda3d6d9e25b803467', 'XMODIFIERS': '@im=fcitx', 'LC_TELEPHONE': 'en_US.UTF-8', 'XDG_DATA_DIRS': '/usr/share/pop:/usr/share/gnome:/home/fantasticapple/.local/share/flatpak/exports/share:/var/lib/flatpak/exports/share:/usr/local/share/:/usr/share/', 'GDMSESSION': 'pop', 'LC_TIME': 'en_US.UTF-8', 'CONDA_DEFAULT_ENV': 'swav', 'PAPERSIZE': 'letter', 'GTK_IM_MODULE': 'fcitx', 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus', 'XDG_CURRENT_DESKTOP': 'pop:GNOME', 'CONDA_PREFIX': '/home/fantasticapple/Tool/anaconda3/envs/swav', 'JOURNAL_STREAM': '8:36530', 'LC_PAPER': 'en_US.UTF-8', 'SESSION_MANAGER': 'local/pop-os:@/tmp/.ICE-unix/3036,unix/pop-os:/tmp/.ICE-unix/3036', 'USERNAME': 'fantasticapple', 'LOGNAME': 'fantasticapple', 'PWD': '/home/fantasticapple/Github/styleestimation-master', 'MANAGERPID': '2729', 'IM_CONFIG_PHASE': '1', 'PYCHARM_HOSTED': '1', 'LANGUAGE': 'en_US:en', 'GJS_DEBUG_TOPICS': 'JS ERROR;JS LOG', 'PYTHONPATH': '/home/fantasticapple/Github/styleestimation-master:/home/fantasticapple/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/213.7172.26/plugins/python/helpers/pycharm_matplotlib_backend:/home/fantasticapple/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/213.7172.26/plugins/python/helpers/pycharm_display', 'SHELL': '/bin/bash', 'LC_ADDRESS': 'en_US.UTF-8', 'GIO_LAUNCHED_DESKTOP_FILE': '/home/fantasticapple/.local/share/applications/jetbrains-pycharm.desktop', 'GNOME_DESKTOP_SESSION_ID': 'this-is-deprecated', 'GTK_MODULES': 'gail:atk-bridge:appmenu-gtk-module', 'DOTNET_ROOT': '/usr/lib/dotnet/dotnet6-6.0.108', 'CLUTTER_IM_MODULE': 'fcitx', 'CONDA_PROMPT_MODIFIER': '(swav) ', 'DOTNET_BUNDLE_EXTRACT_BASE_DIR': '/home/fantasticapple/.cache/dotnet_bundle_extract', 'SYSTEMD_EXEC_PID': '3059', 'XDG_SESSION_DESKTOP': 'pop', 'SSH_AGENT_LAUNCHER': 'gnome-keyring', 'SHLVL': '0', 'LC_IDENTIFICATION': 'en_US.UTF-8', 'LC_MONETARY': 'en_US.UTF-8', 'QT_IM_MODULE': 'fcitx', 'XDG_CONFIG_DIRS': '/etc/xdg/xdg-pop:/etc/xdg', 'LANG': 'en_US.UTF-8', 'XDG_SESSION_TYPE': 'x11', 'DISPLAY': ':1', 'LC_NAME': 'en_US.UTF-8', 'CONDA_SHLVL': '1', 'PYCHARM_DISPLAY_PORT': '63342', 'XDG_SESSION_CLASS': 'user', '_': '/usr/bin/dbus-update-activation-environment', 'PYTHONIOENCODING': 'UTF-8', 'GPG_AGENT_INFO': '/run/user/1000/gnupg/S.gpg-agent:0:1', 'DESKTOP_SESSION': 'pop', 'USER': 'fantasticapple', 'XDG_MENU_PREFIX': 'gnome-', 'GIO_LAUNCHED_DESKTOP_FILE_PID': '4395', 'QT_ACCESSIBILITY': '1', 'WINDOWPATH': '2', 'LC_NUMERIC': 'en_US.UTF-8', 'GJS_DEBUG_OUTPUT': 'stderr', 'SSH_AUTH_SOCK': '/run/user/1000/keyring/ssh', 'PYTHONUNBUFFERED': '1', 'GNOME_SHELL_SESSION_MODE': 'pop', 'XDG_RUNTIME_DIR': '/run/user/1000', 'HOME': '/home/fantasticapple', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'})