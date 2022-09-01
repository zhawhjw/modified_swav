import os.path
from pprint import pprint

from src.utils import (
    bool_flag,
    initialize_exp,
    fix_random_seeds,
    init_distributed_mode,
)
import argparse
import pandas as pd

from modified_eval_linear import main as linear_main

momentum_list = [0.9, 0.5, 0.25, 0]
# momentum_list = [0, 0.25, 0.5, 0.9]
decay_list = [1e-4, 1e-3, 1e-2, 1e-1, 0]
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]

main_data_path = "D:/Data/3D-FUTURE-model/"
pretrained_path = "checkpoints/swav_800ep_pretrain.pth.tar"
windows_fakefile_path = "file:///D:/Github/styleestimation-master/somefile.txt"
dump_root_path = "." + "/"


def method1_wrap():
    root_dir = dump_root_path + "method1"

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
    parser.add_argument("--dist_url", default=windows_fakefile_path, type=str,
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
                linear_main(args, logger, training_stats)


if __name__ == "__main__":
    # 0.9_0.01_0.001
    extend_dict = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'WORLD_SIZE': '1', 'RANK': '0',
                   'LOCAL_RANK': '0'}
    os.environ.update(extend_dict)
    pprint(os.environ)
    method1_wrap()
