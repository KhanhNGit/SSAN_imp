import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from functools import partial

from configs import parse_args
from datasets import data_merge
from loss import *
from networks import get_model
from optimizers import get_optimizer, GACFAS
from transformers import *
from utils import *


def main(args):
    data_bank = data_merge(args.data_dir)
    # define train loader
    if args.trans in ["o"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, transform=transformer_train(), debug_subset_size=args.debug_subset_size)
    elif args.trans in ["p"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, transform=transformer_train_pure(), debug_subset_size=args.debug_subset_size)
    elif args.trans in ["I"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, transform=transformer_train_ImageNet(), debug_subset_size=args.debug_subset_size)
    else:
        raise Exception
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    max_iter = args.num_epochs*len(train_loader)
    # define model
    model = get_model(max_iter, args.num_dataset_train).cuda()
    set_trainable(model, False, ['cls_head'], [0]) # If warming up
    # def optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
    # def scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-5)
    # model = nn.DataParallel(model).cuda()

    #def minimize
    minimizer = GACFAS(model=model, rho=args.minimizer_rho, eta=args.minimizer_eta, alpha=args.minimizer_alpha, n_domains=args.num_dataset_train)

    # make dirs
    model_root_path = os.path.join(args.result_path, args.protocol, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.protocol, "score")
    check_folder(score_root_path)


    # define loss
    binary_fuc = nn.CrossEntropyLoss()
    contra_fun = ContrastLoss()

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }

    for epoch in range(args.num_epochs):
        cls_loss_record = AverageMeter()
        constra_loss_record = AverageMeter()
        adv_loss_record = AverageMeter()
        loss_record = AverageMeter()

        ce_loss_record_0 = AverageMeter()
        ce_loss_record_1 = AverageMeter()
        ce_loss_record_2 = AverageMeter()
        
        acc_record_0 = AverageMeter()
        acc_record_1 = AverageMeter()
        acc_record_2 = AverageMeter()

        sum_ce_criterion = partial(binary_func_sep, return_sum=True, n_sources=args.num_dataset_train)
        ele_ce_criterion = partial(binary_func_sep, return_sum=False, n_sources=args.num_dataset_train)

        if epoch == args.warming_epochs: # Finish warming up
            set_trainable(model, True, [], [0])
            
        # train
        model.train()
        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            image_x, label, UUID = sample_batched["image_x"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()
            # train process
            rand_idx = torch.randperm(image_x.shape[0])
            cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])

            cls_loss = ele_ce_criterion(logits=torch.cat([cls_x1_x1, cls_x1_x1]), 
                                        label=torch.cat([label, label]), 
                                        UUID=torch.cat([UUID, UUID]), 
                                        ce_loss_record_0=ce_loss_record_0, 
                                        ce_loss_record_1=ce_loss_record_1, 
                                        ce_loss_record_2=ce_loss_record_2,
                                        acc_record_0=acc_record_0, 
                                        acc_record_1=acc_record_1, 
                                        acc_record_2=acc_record_2)


            contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
            contrast_label = torch.where(contrast_label==True, 1, -1)
            constra_loss = contra_fun(fea_x1_x1, fea_x1_x2, contrast_label)
            adv_loss = binary_fuc(domain_invariant, UUID.long()) * args.lamda_adv


            if (epoch >= args.warming_minimizer):
                model.zero_grad()
                assert isinstance(cls_loss, list) and len(cls_loss)==args.num_dataset_train, f"List loss for domain is not provided"            
                for idx_domain in range(args.num_dataset_train):
                    if cls_loss[idx_domain].item() != 0:
                        cls_loss[idx_domain].backward(retain_graph=True)
                    minimizer.get_perturb_norm(idx_domain) 

                # ascent step
                for idx_domain in range(args.num_dataset_train):
                    if cls_loss[idx_domain].item() != 0:
                        minimizer.ascent_step(idx_domain) 

                for idx_domain in range(args.num_dataset_train):
                    if cls_loss[idx_domain].item() != 0:
                        minimizer.proxy_gradients(idx_domain, input1=image_x, input2=image_x[rand_idx, :, :, :], 
                                                  labels=torch.cat([label, label]), 
                                                  loss_func=sum_ce_criterion, 
                                                  UUID=torch.cat([UUID, UUID]))
                
                minimizer.sync_grad_step(cls_loss) # Get average gradient at every top and update for main model.
                if isinstance(cls_loss, list): 
                    cls_loss = sum(cls_loss)
            else:
                model.zero_grad()
                if isinstance(cls_loss, list): 
                    cls_loss = sum(cls_loss)

            loss_all = cls_loss + constra_loss + adv_loss

            if (epoch >= args.warming_minimizer) and (minimizer is not None):
                minimizer.descent_step()

            n = image_x.shape[0]
            cls_loss_record.update(cls_loss.data, n)
            constra_loss_record.update(constra_loss.data, n)
            adv_loss_record.update(adv_loss.data, n)
            loss_record.update(loss_all.data, n)

            model.zero_grad()
            loss_all.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            if i % args.print_freq == args.print_freq - 1:
                print("Epoch:{:d}, mini-batch:{:d}, lr={:.4f}, binary_loss={:.4f}, constra_loss={:.4f}, adv_loss={:.4f}, Loss={:.4f}".format(epoch + 1, i + 1, lr, cls_loss_record.avg, constra_loss_record.avg, adv_loss_record.avg, loss_record.avg))
        
        # whole epoch average
        print("Epoch:{:d}, Train: lr={:f}, Loss={:.4f}".format(epoch + 1, lr, loss_record.avg))
        scheduler.step()

        # test
        epoch_test = 1
        if epoch % epoch_test == epoch_test-1:
            if args.trans in ["o", "p"]:
                test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size, transform=transformer_test_video(), debug_subset_size=args.debug_subset_size)
            elif args.trans in ["I"]:
                test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size, transform=transformer_test_video_ImageNet(), debug_subset_size=args.debug_subset_size)
            else:
                raise Exception
            score_path = os.path.join(score_root_path, "Epoch_{}".format(epoch+1))
            check_folder(score_path)
            for i, test_name in enumerate(test_data_dic.keys()):
                print("[{}/{}]Validating {}...".format(i+1, len(test_data_dic), test_name))
                test_set = test_data_dic[test_name]
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                HTER, auc_test = validate(model, test_loader, score_path, epoch, name=test_name)
                if auc_test-HTER>=eva["best_auc"]-eva["best_HTER"]:
                    eva["best_auc"] = auc_test
                    eva["best_HTER"] = HTER
                    eva["best_epoch"] = epoch+1
                    model_path = os.path.join(model_root_path, "{}_best.pth".format(args.protocol))
                    torch.save({
                        'epoch': epoch+1,
                        'state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler,
                        'args':args,
                    }, model_path)
                    print("Model saved to {}".format(model_path))
                print("[Best result] Epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"],  eva["best_HTER"], eva["best_auc"]))
            model_path = os.path.join(model_root_path, "{}_recent.pth".format(args.protocol))
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler,
                'args':args,
            }, model_path)
            print("Model saved to {}".format(model_path))


def validate(model, test_loader, score_root_path, epoch, name=""):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        scores_list = []
        for i, sample_batched in enumerate(test_loader):
            image_x, label = sample_batched["image_x"].cuda(), sample_batched["label"].cuda()
            # rand_idx = torch.randperm(image_x.shape[0])
            cls_x1_x1, fea_x1_x1, fea_x1_x2, _ = model(image_x, image_x)
            score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]
            for ii in range(image_x.shape[0]):
                scores_list.append("{} {}\n".format(score_norm[ii], label[ii][0]))
            
        map_score_val_filename = os.path.join(score_root_path, "{}_score.txt".format(name))
        print("score: write val scores to {}".format(map_score_val_filename))
        with open(map_score_val_filename, 'w') as file:
            file.writelines(scores_list)

        test_ACC, FPR, FRR, HTER, auc_test, test_err = performances_val(map_score_val_filename)
        print("## {} score:".format(name))
        print("Epoch:{:d}, val_result:  val_ACC={:.4f}, FPR={:.4f}, FRR={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}".format(epoch+1, test_ACC, FPR, FRR, HTER, auc_test, test_err))
        print("Validate phase cost {:.4f}s".format(time.time()-start_time))
    return HTER, auc_test

    
if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    seed_all()
    main(args=args)
