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

momentum_list = [0]
decay_list = [1e-2]
#
lr_list = [1e-1]


model_list = []

for root, dirs, files in os.walk("saved", topdown=True):

    if dirs:
        for d in dirs:
            model_list.append(d)
# print(model_list)

# main_data_path = "/home/y/yz862/python38/3D-FUTURE-model/"
main_data_path = "/home/y/yz862/data/3D-Future-Model-Renders_v2/"
# pretrained_root = "saved"
# pretrained_path = "checkpoint.pth.tar"
pretrained_root = "."
pretrained_path = "checkpoints/swav_800ep_pretrain.pth.tar"
windows_fakefile_path = "env://"
dump_root_path = "." + "/"


def method2_wrap():


    root_dir = dump_root_path + "method2"

    if not os.path.exists( root_dir):
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
    parser.add_argument("--num_classes", type=int, default=12,
                        help="number of classes in the dataset")
    parser.add_argument("--topk", type=int, default=3,
                        help="top K performance for retrieval")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")

    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--pretrained", default=pretrained_path, type=str, help=pretrained_path)

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=150, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=128, type=int,
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
    parser.add_argument("--dist_url", default=windows_fakefile_path, type=str,
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

    for m in model_list:

        # args.pretrained = pretrained_root + "/" + m + "/" + pretrained_path
        args.pretrained = pretrained_root + "/"  + pretrained_path

        for momentum in momentum_list:

            args.momentum = momentum

            for weight_decay in decay_list:

                args.wd = weight_decay

                for lr in lr_list:

                    args.lr = lr
                    
                    
                    directory = str(momentum) + "_" + str(weight_decay) + "_" + str(lr) + "/" + str(m)


                    if not os.path.exists(root_dir + "/" + directory):
                        os.mkdir(root_dir + "/" + directory)
                        # shutil.copy("params.pkl", root_dir + "/" + directory + "/" + "params.pkl")
                    else:
                        log = pd.read_csv(root_dir + "/" + directory + "/" + "log.csv")

                        if len(log['epoch'].values) >= args.epochs - 1:
                            print("Skip trained hyperparameter combination")
                            continue
                    args.dump_path = root_dir + "/" + directory
                    args.results = root_dir + "/" + directory + "/" + "log.csv"

                    fix_random_seeds(args.seed)
                    logger, training_stats = initialize_exp(
                        args, "epoch", "loss", "prec1", "prec3", "loss_val", "prec1_val", "prec3_val"
                    )

                    print("Current Setting: " + directory)

                    # linear_main(args, logger, training_stats)
                    semisup_main(args, logger, training_stats)
if __name__ == "__main__":
    # 0.5_0_0.1
    # 0.25_0.0001_0.01
    # 0.25_0.01_0.001
    # 0_0.001_0.1
    extend_dict = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}
    os.environ.update(extend_dict)
    pprint(os.environ)
    # method1_wrap()
    method2_wrap()
