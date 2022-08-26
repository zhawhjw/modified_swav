import subprocess

"""
Specify different search params 

"""


learning_rate_eval_linear = [ ]

learning_rate_eval_semisup = [ ]


python_env = "C:/Users/tomer/research/style/Scripts/python"

if False:
    path_to_nn_file = "resnet50_baseline.py"
    optimizers =  ['SGD'] #,['SGD', 'RMSprop'] 
    momentums = [0.9]
    batch_sizes = [4]
    epochs = [150] # [4, 10, 25]
    learn_rates = [0.0002]
    #param_grid = dict(learn_rate=learn_rate, momentum=momentum, epochs=epochs, batch_size=batch_size, optimizer=optimizer)
    for momentum in momentums:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for lr in learn_rates:
                    for optimizer in optimizers:
                        print(optimizer, lr,epoch, batch_size, momentum)
                        command = "{python} {nn_script} --lr {lr} --epochs {epochs} --data_path {data_path} --results {results} --num_classes {num_classes}"
                        command = command.format(python=python_env,
                                                nn_script=path_to_nn_file,
                                                lr=lr,
                                                epochs=epoch,
                                                data_path="C:/Users/tomer/Datasets",
                                                results="C:/Users/tomer/researchsoftware/furniturestyle/logging.csv",
                                                num_classes="3")
                        print(">" + command)
                        #run_sync = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                        process = subprocess.Popen(command.split())
                        process.wait()
                        print(process)

path_to_nn_file = "eval_semisup.py"
optimizers =  ['SGD'] #,['SGD', 'RMSprop'] 
momentums = [0.9]
batch_sizes = [4]
epochs = [100] # [4, 10, 25]
learn_rates =  [0.003,0.002] # [0.01, 0.025] #, 0.05, 0.1]
# python  eval_semisup.py   --pretrained checkpoints/swav_800ep_pretrain.pth.tar --labels_perc "100" --lr 0.01 --lr_last_layer 0.2  
for momentum in momentums:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for lr in learn_rates:
                for optimizer in optimizers:
                    print(optimizer, lr,epoch, batch_size, momentum)
                    #checkpoint_name = "checkpoints/swav_800ep_pretrain_lr%.5f.pth.tar" % lr
                    pretrained_model_name = "checkpoints/swav_800ep_pretrain.pth.tar"
                    command = "{python} -m torch.distributed.launch --nproc_per_node=1 {nn_script} --pretrained {pretrained} --labels_perc {labels_perc} --lr_last_layer {lr_last_layer}  --lr {lr} --epochs {epochs} --data_path {data_path} --results {results} --num_classes {num_classes}"
                    command = command.format(python=python_env,
                                            nn_script=path_to_nn_file,
                                            lr=lr,
                                            epochs=epoch,
                                            data_path="C:/Users/tomer/Datasets",
                                            pretrained=pretrained_model_name,
                                            results="C:/Users/tomer/researchsoftware/furniturestyle/logging.csv",
                                            labels_perc="100",
                                            lr_last_layer=0.2,
                                            num_classes="3")
                    print(">" + command)
                    #run_sync = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    process = subprocess.Popen(command.split())
                    process.wait()
                    print(process)

# python resnet50_baseline.py --data_path "C:\Users\tomer\Datasets"  --results "C:\Users\tomer\researchsoftware\furniturestyle\logging.csv"  --num_classes 3


def log_results(file_path):
    with open(file_path,'w') as f:
        str_out = "{model_name},{num_classes},{train_loss},{train_acc},{val_loss},{val_acc},{epoch},{learning_rate},{momentum},{stepsize},{gamma},{misc}\n"
        str_out = str_out.format(model_name="model_name",
                    num_classes="num_classes",
                    train_loss="train_loss",
                    train_acc="train_acc",
                    val_loss="val_loss",
                    val_acc="val_acc",
                    epoch="epoch",
                    learning_rate="learning_rate",momentum="momentum",
                    stepsize="stepsize",gamma="gamma", misc="misc")
        f.write(str_out)
        f.close()


class Resnet50Baseline:
    pass
"""
    # https://pytorch.org/docs/stable/optim.html 
    public torch::optim::Adagrad (Class Adagrad)

    public torch::optim::Adam (Class Adam)

    public torch::optim::AdamW (Class AdamW)

    public torch::optim::LBFGS (Class LBFGS)

    public torch::optim::RMSprop (Class RMSprop)
    lr (float, optional) – learning rate (default: 1e-2)
    momentum (float, optional) – momentum factor (default: 0)
    alpha (float, optional) – smoothing constant (default: 0.99)
    eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
    centered (bool, optional) – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
    weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

    public torch::optim::SGD (Class SGD)
    lr (float) – learning rate
    momentum (float, optional) – momentum factor (default: 0)
    weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    dampening (float, optional) – dampening for momentum (default: 0)
    nesterov (bool, optional) – enables Nesterov momentum (default: False)
"""