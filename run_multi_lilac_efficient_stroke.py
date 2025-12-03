import sys
from torch.utils.tensorboard import SummaryWriter
import argparse
import glob
from loader_stroke import *
from model_stroke import *
from utils import *
import torch
import numpy as np
import os
import time
import datetime
import torch.nn as nn
# import logging
# import pandas as pd
# import itertools
from typing import List, Tuple, Union
from collections import defaultdict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


@torch.no_grad()
def get_wd_params(model: nn.Module):
    decay = []
    no_decay = []

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear,
                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,)):
            if hasattr(module, 'weight') and module.weight is not None:
                decay.append(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                no_decay.append(module.bias)
        else:
            for param_name, param in module.named_parameters(recurse=False):
                no_decay.append(param)

    assert len(decay) + len(no_decay) == len(tuple(model.parameters())), "Sanity check failed."
    return decay, no_decay


def get_optimizer(model, lr=0.03, weight_decay=0.01):
    # Get parameters with weight decay applied

    # Define optimizer
    if weight_decay > 0:
        weight_decay_params, no_decay_params = get_wd_params(model)
        optimizer = torch.optim.Adam([
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': weight_decay_params, 'weight_decay': weight_decay},
        ], lr=lr, betas=(0.9, 0.999), decoupled_weight_decay=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    return optimizer


def compute_intraclass_combinations_unique(
        labels: List[Union[str, int]],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Compute all intra-class (i ≠ j) feature and target differences for a batch.
    """

    labels = np.array(labels)

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    index_pairs = []

    for cls, indices in class_to_indices.items():
        n = len(indices)
        if n < 2:
            continue
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                idx_i, idx_j = indices[i], indices[j]
                # this check should hopefully sort out doubles
                if not (idx_j, idx_i) in index_pairs:
                    index_pairs.append((idx_i, idx_j))

    return np.array(index_pairs).T


def save_checkpoint(model, optimizer, save_path, current_epoch, earlystoppingcount, prev_val_loss, scaler,
                    scheduler=None):
    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_epoch': current_epoch,
        'earlystopping': earlystoppingcount,
        'prev_val_loss': prev_val_loss,
        'scaler_state_dict': scaler.state_dict()
    }
    if scheduler is not None:
        checkpoint_dict["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint_dict, save_path)


def load_checkpoint(model, optimizer, scaler, scheduler, load_path):
    dev = torch.cuda.current_device()
    checkpoint = torch.load(load_path, weights_only=False, map_location=lambda storage, loc: storage.cuda(dev))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['current_epoch']
    earlystoppingcount = checkpoint['earlystopping']
    prev_val_loss = checkpoint['prev_val_loss']
    if "scaler_state_dict" in checkpoint.keys():
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, current_epoch, earlystoppingcount, prev_val_loss, scaler, scheduler


def train(network, opt, use_checkpoint, checkpoint_savepath, mixed_precision):
    cuda = True
    parallel = True
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    network = network.to(device)
    optimizer = get_optimizer(network, lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = None
    if opt.scheduler_patience > 0:
        # todo parameters are currently hardcoded here
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5,
                                                               patience=opt.scheduler_patience)
    scaler = torch.amp.GradScaler("cuda")
    if use_checkpoint and os.path.isfile(checkpoint_savepath):
        print("loading previous checkpoint from {}".format(checkpoint_savepath))
        network, optimizer, current_epoch, earlystoppingcount, prev_val_loss, scaler, scheduler = load_checkpoint(
            network, optimizer, scaler, scheduler, checkpoint_savepath)

    else:
        prev_val_loss = 1e+100
        earlystoppingcount = 0
        current_epoch = 0

    os.makedirs(f"{opt.output_fullname}/", exist_ok=True)
    if parallel:
        network = nn.DataParallel(network).to(device)
    else:
        network = network.cuda()

    # experimental, should be safe but let's see
    network.compile()

    opt.epoch = current_epoch

    steps_per_epoch = opt.save_epoch_num
    writer = SummaryWriter(log_dir=f"{opt.output_fullname}")
    prev_time = time.time()

    loader_val = torch.utils.data.DataLoader(args.val_loader,
                                             batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers,
                                             drop_last=False)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for epoch in range(opt.epoch, opt.max_epoch):
        # done so there are new group-wise transformations each epoch
        if args.efficient:
            args.train_loader.reshuffle_dataset()

        # it is very important to set shuffle to false here
        loader_train = torch.utils.data.DataLoader(args.train_loader,
                                                   batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers,
                                                   drop_last=False)

        if earlystoppingcount > opt.earlystopping:
            break

        epoch_total_loss = []
        epoch_step_time = []
        train_loss_atributes = [[] for x in range(len(network.module.network_heads))]
        training_time = time.time()
        print('training step')
        for step, batch in enumerate(loader_train):
            step_start_time = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):

                images, attributes, meta_dict = batch
                subject_list = meta_dict["subject"]
                # use_for_loss: boolean tensor of [batchsize x num_attributes]

                input_combinations = compute_intraclass_combinations_unique(subject_list)

                if len(meta_dict.keys()) > 1:
                    meta = torch.stack([meta_dict[item] for item in meta_dict.keys() if item != 'subject'],
                                       dim=1).float()
                    predicted = network(images.type(Tensor), input_combinations, meta)
                else:
                    predicted = network(images.type(Tensor), input_combinations)

                attributes = np.array(attributes).T
                # predicted: list of len num_attributes of [batchsize x 1 ] tensors
                targetdiff = torch.Tensor(
                    attributes[input_combinations[0]] - attributes[input_combinations[1]]).unsqueeze(-1).type(Tensor)
                # flip a given amount on zero differences to also be evaluated
                # here we first sample for each target randomly whether they should be flipped 
                # and then apply this to the 0 difference values in the batch
                # a flip_percentage of one means that all combinations should be computed
                non_zero_differences = targetdiff != 0
                flip_indices = torch.Tensor(np.random.rand(*targetdiff.shape) < args.zero_flip_percentage).to(
                    torch.bool).to(device)
                non_zero_differences[~non_zero_differences] = flip_indices[~non_zero_differences]
                # add axis for indexing
                #non_zero_differences = non_zero_differences[..., np.newaxis]

                # TODO do check whether age diff and parameter diff have same sign, set flag to 0 of not
                # --> generally most cognitive scores only increase monothonally
                # target1 or 2: List of lists of size [batchsize x num_attributes]
                # if (step in [0]) and (epoch == opt.epoch):  #[1, 5, 10]:
                #     print("network output contains NaNs: {}".format(torch.isnan(predicted).any()))
                #     print("attributes.shape")
                #     print(attributes.shape)
                #     print("np.array(input_combinations).shape")
                #     print(np.array(input_combinations).shape, flush=True)
                #     print("np.array(images).shape")
                #     print(np.array(images).shape, flush=True)
                #     print("targetdiff.shape")
                #     print(targetdiff.shape, flush=True)
                #
                #     print("non_zero_differences.shape")
                #     print(non_zero_differences.shape, flush=True)
                #     print("predicted.shape")
                #     print(predicted.shape, flush=True)

                if opt.task_option == 'o':
                    targetdiff[targetdiff > 0] = 1
                    targetdiff[targetdiff == 0] = 0.5
                    targetdiff[targetdiff < 0] = 0

                # Loss
                optimizer.zero_grad()
                combined_loss = 0
                for i, _ in enumerate(network.module.network_heads):
                    attribute_diff = targetdiff[:, i]
                    use_for_loss = non_zero_differences[:, i]
                    prediction_attribute = predicted[:, i]

                    loss = args.loss(prediction_attribute[use_for_loss], attribute_diff[use_for_loss])
                    train_loss_atributes[i].append(loss.item())

                    combined_loss = combined_loss + loss

            if (step in [0]) and (epoch == opt.epoch):
                print("combined_loss.get_device()")
                print(combined_loss.get_device(), flush=True)
                print("next(network.module.parameters()).is_cuda")
                print(next(network.module.parameters()).is_cuda, flush=True)

            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_total_loss.append(combined_loss.item())

            # Log Progress
            batches_done = epoch * len(loader_train) + step
            batches_left = opt.max_epoch * len(loader_train) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f ] ETA: %s"
                % (
                    epoch,
                    opt.max_epoch,
                    step,
                    len(loader_train),
                    loss.item(),
                    time_left,
                )
            )
            epoch_step_time.append(time.time() - step_start_time)
        epoch_time = datetime.timedelta(seconds=time.time() - training_time)

        print("\ntraining time used for epoch {}: {}".format(epoch, epoch_time))

        if (epoch + 1) % steps_per_epoch == 0:  # (step != 0) &
            # print epoch info
            epoch_info = '\nValidating... Step %d/%d / Epoch %d/%d' % (
                step, len(loader_train), epoch, opt.max_epoch)
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            loss_info = 'train loss: %.4e ' % (np.mean(epoch_total_loss))
            train_loss_means = np.mean(train_loss_atributes, axis=1)
            print()
            for i, train_loss_attribute in enumerate(train_loss_means):
                print("train loss attribute {}: {}".format(i, train_loss_attribute))
                log_stats([train_loss_attribute], ['loss/train_attribute_{}'.format(i)], epoch, writer)

            log_stats([np.mean(epoch_total_loss)], ['loss/train'], epoch, writer)

            network.eval()
            valloss_total = []
            val_predictions = [np.array([]) for x in range(len(network.module.network_heads))]
            val_targets = [np.array([]) for x in range(len(network.module.network_heads))]
            val_loss_atributes = [np.array([]) for x in range(len(network.module.network_heads))]

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                    # print('valstep, batch')
                    for valstep, batch in enumerate(loader_val):
                        images, attributes, meta_dict = batch
                        # print('val step metadict', meta_dict.keys())
                        subject_list = meta_dict["subject"]
                        meta_list = [v for k, v in meta_dict.items() if k != 'subject']
                        meta_list = meta_list if len(meta_list) > 0 else None
                        # incorporated dud approach here, maybe one needs to be careful here
                        non_dud_entries = np.array(subject_list) != "dud"
                        subject_list = np.array(subject_list)[non_dud_entries]
                        images = images[non_dud_entries]
                        attributes = np.array(attributes).T[non_dud_entries]
                        meta_array = np.array(meta_list).T[non_dud_entries] if meta_list is not None else None

                        input_combinations = compute_intraclass_combinations_unique(subject_list)

                        predicted = network(images.type(Tensor), input_combinations,
                                            meta_array)  # predicted: list of len num_attributes of [batchsize x 1 ] tensors

                        targetdiff = torch.Tensor(
                            attributes[input_combinations[0]] - attributes[input_combinations[1]]).unsqueeze(-1).type(
                            Tensor)
                        # For the validation, all combinations should be evaluated                      
                        non_zero_differences = torch.ones(*targetdiff.shape).to(torch.bool)

                        if opt.task_option == 'o':
                            targetdiff[targetdiff > 0] = 1
                            targetdiff[targetdiff == 0] = 0.5
                            targetdiff[targetdiff < 0] = 0

                        combined_val_loss = 0

                        for i, _ in enumerate(network.module.network_heads):
                            attribute_diff = targetdiff[:, i]
                            # not used for validation
                            use_for_loss = non_zero_differences[:, i]
                            prediction_attribute = predicted[:, i]

                            prediction_attribute = np.array(prediction_attribute.cpu()).flatten()
                            attribute_diff = np.array(attribute_diff.cpu()).flatten()

                            val_predictions[i] = np.append(val_predictions[i], prediction_attribute)
                            val_targets[i] = np.append(val_targets[i], attribute_diff)

                            valloss = args.loss(torch.Tensor(prediction_attribute[use_for_loss.flatten()]),
                                                torch.Tensor(attribute_diff[use_for_loss.flatten()]))
                            valauc = roc_auc_score(val_targets[i], val_predictions[i])
                            valbalacc = balanced_accuracy_score(val_targets[i], (val_predictions[i] > 0.4).astype(int))
                            print(i, 'metrics-->', val_targets[i].shape, '--valauc', valauc, 'valbalacc', valbalacc)
                            val_loss_atributes[i] = np.append(val_loss_atributes[i], np.abs(prediction_attribute))
                            #  val_loss_atributes[i].append(valloss.item())
                            combined_val_loss = combined_val_loss + valloss

                        valloss_total.append(combined_val_loss.item())

            log_stats([np.mean(valloss_total)], ['loss/val_batchwise'], epoch, writer)
            val_loss_info = 'val loss batch mean: %.4e' % (np.mean(valloss_total))
            print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
            val_loss_per_attribute = []
            bal_acc_attribute = []
            for i, _ in enumerate(network.module.network_heads):
                val_loss_per_attribute.append(
                    args.loss(torch.Tensor(val_predictions[i]), torch.Tensor(val_targets[i])).item())
                print('Metrics:', val_targets[i].shape, 'bal-acc',
                      balanced_accuracy_score(val_targets[i], (val_predictions[i] > 0.4).astype(int)),
                      'roc-auc', roc_auc_score(val_targets[i], (val_predictions[i] > 0.4).astype(int)))
                print("val loss attribute {}: {}".format(i, val_loss_per_attribute[i]))
                log_stats([val_loss_per_attribute[i]], ['loss/val_attribute_{}'.format(i)], epoch, writer)
                print("val mean absolute prediction attribute {}: {}".format(i, np.mean(np.abs(val_predictions[i]))))
                log_stats([np.mean(np.abs(val_predictions[i]))], ['loss/val_prediction_abs_mean_attr_{}'.format(i)],
                          epoch, writer)
            print('balanced accuracy', bal_acc_attribute)
            log_stats([np.sum(val_loss_per_attribute)], ['loss/val_loss_total_{}'.format(i)], epoch, writer)
            curr_val_loss = np.mean(val_loss_per_attribute)
            if opt.scheduler_patience > 0:
                scheduler.step(curr_val_loss)
            if curr_val_loss < prev_val_loss:
                torch.save(network.state_dict(),
                           f"{opt.output_fullname}/model_best.pth")
                np.savetxt(f"{opt.output_fullname}/model_best.info", np.array([epoch]))
                prev_val_loss = curr_val_loss
                earlystoppingcount = 0
            else:
                earlystoppingcount += 1
                print(f'Early stopping count: {earlystoppingcount}')

            save_checkpoint(model, optimizer, checkpoint_savepath, epoch, earlystoppingcount, prev_val_loss, scaler,
                            scheduler)
            network.train()

    torch.save(network.state_dict(), f"{opt.output_fullname}/model_epoch{epoch}.pth")
    network.eval()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', default='lilac', type=str, help="name of job")

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.03, type=float)
    parser.add_argument('--zero_flip_percentage', default=1.0, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--earlystopping', default=10, type=int, help="early stopping criterion")
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_class_size', default=8, type=int)

    parser.add_argument('--max_epoch', default=300, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--save_epoch_num', default=1, type=int, help="validate and save every N epoch")

    parser.add_argument('--image_directory', default='./datasets', type=str)  # , required=True)
    parser.add_argument('--csv_file_train', default='./datasets/demo_oasis_train.csv', type=str,
                        help="csv file for training set")  # , required=True)
    parser.add_argument('--csv_file_val', default='./datasets/demo_oasis_val.csv', type=str,
                        help="csv file for validation set")  # , required=True)
    parser.add_argument('--csv_file_test', default='./datasets/demo_oasis_test.csv', type=str,
                        help="csv file for testing set")  # , required=True)
    parser.add_argument('--output_directory', default='./output', type=str,
                        help="directory path for saving model and outputs")  # , required=True)
    parser.add_argument('--image_size', default="128, 128, 128", type=str, help="w,h for 2D and w,h,d for 3D")
    parser.add_argument('--image_channel', default=1, type=int)
    parser.add_argument('--task_option', default='o', choices=['o', 't', 's'],
                        type=str, help="o: temporal 'o'rdering\n "
                                       "t: regression for 't'ime interval\n "
                                       "s: regression with optional meta for a 's'pecific target variable\n ")
    parser.add_argument('--targetname', default='ScanOrder', type=str)
    parser.add_argument('--optional_meta', default='', type=str,
                        help='list optional meta names to be used (e.g., ["AGE", "AGE_x_SEX"]). csv files should include the meta data name')
    parser.add_argument('--backbone_name', default='cnn_3D', type=str,
                        help="implemented models: cnn_3D, cnn3d_latent, cnn_2D, resnet50_2D, resnet18_2D")

    parser.add_argument('--run_mode', default='train', choices=['train', 'eval'], help="select mode")  # required=True,
    parser.add_argument('--mixed_precision', default=False, action='store_true')
    parser.add_argument('--use_checkpoint', default=False, action='store_true')
    parser.add_argument('--attribute_list', nargs='+', type=str, default=["ScanOrder"])
    # TODO this might be very wrong way to parse booleans, should be restructured

    parser.add_argument('--scheduler_patience', default=-1, type=int,
                        help="patience used by the learning rate scheduler, -1 means no scheduler is used")
    parser.add_argument('--rescale_intensity', default=False, action='store_true')
    parser.add_argument('--layer_norm', default='batchnorm', type=str)
    parser.add_argument('--output_scaling', default='non', type=str,
                        help="way to scale the output per attribute, options include max, stdev and non_zero_stdev")
    parser.add_argument('--efficient', default=True, action='store_true')
    args = parser.parse_args()

    return args


def run_setup(args):
    dict_loss = {'o': nn.BCEWithLogitsLoss(), 't': nn.MSELoss(), 's': nn.MSELoss()}
    dict_task = {'o': 'temporal_ordering', 't': 'regression', 's': 'regression'}

    args.loss = dict_loss[args.task_option]

    path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + \
                'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(
        args.batchsize)

    args.output_fullname = os.path.join(args.output_directory, path_pref)
    os.makedirs(args.output_fullname, exist_ok=True)
    # check path
    assert os.path.exists(args.image_directory), "incorrect image directory path"

    # set up seed
    set_manual_seed(args.seed)

    # set up GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        print("!! NO GPU AVAILABLE !!")

    # string to list
    image_size = [int(item) for item in args.image_size.split(',')]
    args.image_size = image_size
    if len(args.optional_meta) > 0 and ',' in args.optional_meta:
        optiona_meta_names = [item for item in args.optional_meta.split(',')]
        args.optional_meta = optiona_meta_names
    elif len(args.optional_meta) > 0 and not (',' in args.optional_meta):
        args.optional_meta = [args.optional_meta]
    else:
        args.optional_meta = []

    if args.run_mode == 'train':
        args.train_loader = single_image_multiple_target_loader3D_stroke(args, trainvaltest='train',
                                                                         extra_meta=args.optional_meta)
        args.val_loader = single_image_multiple_target_loader3D_stroke(args, trainvaltest='val',
                                                                       extra_meta=args.optional_meta)
    if args.run_mode == 'eval':
        args.test_loader = single_image_multiple_target_loader3D_stroke(args, trainvaltest='test',
                                                                        extra_meta=args.optional_meta)
    if args.run_mode == 'extract':
        args.extract_loader = single_image_multiple_target_loader3D_stroke(args, trainvaltest='test',
                                                                           extra_meta=args.optional_meta)

    print(' ----------------- Run Setup Summary -----------------')
    print(f'JOB NAME: {args.jobname}')
    print(f'TASK: {dict_task[args.task_option]}')
    print(f'Target Attribute: {args.targetname}')
    if len(args.optional_meta) > 0:
        print(f'Optional Meta: {args.optional_meta}')
    # print(f'BACKBONE: {args.backbone_name}')
    print(f'RUN MODE: {args.run_mode}')
    print(f"Num of GPUs: {torch.cuda.device_count()}")


if __name__ == "__main__":
    #attribute_list = ["age_decimal", "CDRSB"]
    args = parse_args()
    # print('mixed_precision', args.mixed_precision)

    run_setup(args)
    checkpoint_savepath = os.path.join(args.output_fullname, "training_checkpoint.pth")

    model = efficient_multi_LILAC(args, len(args.attribute_list), args.output_scaling)

    # print(model)
    print("Num of Model Parameter:", count_parameters(model))

    if args.run_mode == 'eval':
        print(' ----------------- Testing initiated -----------------')


    else:
        assert args.run_mode == 'train', "check run_mode"
        print(' ----------------- Training initiated -----------------')
        train(model, args, args.use_checkpoint, checkpoint_savepath, args.mixed_precision)
