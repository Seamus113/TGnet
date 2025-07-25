import os
import time
from tqdm import tqdm

import torch
from torch import nn
# import torch._dynamo
import numpy as np
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy, 
    BinaryAccuracy, 
    BinaryF1Score, 
    BinaryJaccardIndex, 
    MulticlassJaccardIndex)
import wandb
from early_stopping import EarlyStopping
from dataprocess import MFInstSegDataset
from segmentors import TGNetSegmentor
from misc import seed_torch, init_logger, print_num_params



if __name__ == '__main__':
    torch.set_float32_matmul_precision("high") # may be faster if GPU support TF32
    os.environ["WANDB_API_KEY"] = '##################'
    os.environ["WANDB_MODE"] = "offline"
    
    # start a new wandb run to track this script
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
            # set the wandb project where this run will be logged
            project="TGNet" + "InstSeg",

            # track hyperparameters and run metadata
            config={
                "edge_attr_dim": 12,
                "node_attr_dim": 10,
                "edge_attr_emb": 64, 
                "node_attr_emb": 64, ## recommend: 64
                "edge_grid_dim": 0, 
                "node_grid_dim": 7,
                "edge_grid_emb": 0, 
                "node_grid_emb": 64, # recommend: 64
                "num_layers": 3, ## recommend: 3
                "delta": 2, # obsolete
                "mlp_ratio": 2,
                #"drop": 0.25,
                "drop": 0.25, #0.25
                "drop_path": 0.25, #0.25
                "head_hidden_dim": 64,##64
                "conv_on_edge": False,
                "use_uv_gird": True,
                "use_edge_attr": True,
                "use_face_attr": True,


                "seed": 42,
                "device": 'cuda',
                "architecture": "TGNet", # recommend: TGNet option: GCN SAGE GIN GAT  AAGNetGraphEncoder
                "dataset_type": "full",
                "dataset": "D:\\dataset\\databj-526",
                # "transformer_heads" : 4,
                # "transformer_layers": 1,


                "epochs": 100,
                #"lr": 1e-2,
                "lr": 1e-4,
                # # 添加学习率调度
                # "scheduler": "cosine",
                # "min_lr": 1e-6,
                # "warmup_epochs": 5,
                # "weight_decay": 1e-2,
                "weight_decay": 1e-4,
                #batch_size": 256,
                "batch_size": 64,
                "ema_decay_per_epoch": 0.95,
                "seg_a": 1,
                "inst_a": 3.,
                "bottom_a": 0.5,
                }
        )
    
    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = wandb.config['device']
    dataset = wandb.config['dataset']
    dataset_type = wandb.config['dataset_type']
    n_classes = MFInstSegDataset.num_classes(dataset_type)

    print('n_classes',n_classes)

    model = TGNetSegmentor(num_classes=n_classes,
                            arch=wandb.config['architecture'],
                            edge_attr_dim=wandb.config['edge_attr_dim'], 
                            node_attr_dim=wandb.config['node_attr_dim'], 
                            edge_attr_emb=wandb.config['edge_attr_emb'], 
                            node_attr_emb=wandb.config['node_attr_emb'],
                            edge_grid_dim=wandb.config['edge_grid_dim'], 
                            node_grid_dim=wandb.config['node_grid_dim'], 
                            edge_grid_emb=wandb.config['edge_grid_emb'], 
                            node_grid_emb=wandb.config['node_grid_emb'], 
                            num_layers=wandb.config['num_layers'], 
                            delta=wandb.config['delta'], 
                            mlp_ratio=wandb.config['mlp_ratio'], 
                            drop=wandb.config['drop'], 
                            drop_path=wandb.config['drop_path'], 
                            head_hidden_dim=wandb.config['head_hidden_dim'],
                            conv_on_edge=wandb.config['conv_on_edge'],
                            use_uv_gird=wandb.config['use_uv_gird'],
                            use_edge_attr=wandb.config['use_edge_attr'],
                            use_face_attr=wandb.config['use_face_attr'],)
    model = model.to(device)

    # Reset since we are using a different mode.
     #torch._dynamo.reset()
     #model = torch.compile(model, mode="reduce-overhead")

    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params


    train_dataset = MFInstSegDataset(root_dir=dataset, split='train', 
                                     center_and_scale=False, normalize=True, random_rotate=False,
                                     dataset_type=dataset_type, num_threads=8)
    graphs = train_dataset.graphs() # no need to load graphs again !
    val_dataset = MFInstSegDataset(root_dir=dataset, graphs=graphs, split='val', 
                                   center_and_scale=False, normalize=True,
                                   dataset_type=dataset_type, num_threads=8)
    train_loader = train_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)
    val_loader = val_dataset.get_dataloader(batch_size=wandb.config['batch_size'], shuffle=False, drop_last=False, pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()
    instance_loss = nn.BCEWithLogitsLoss()
    bottom_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=wandb.config['epochs'], eta_min=0)

    train_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    train_inst_acc = BinaryAccuracy().to(device)
    train_bottom_acc = BinaryAccuracy().to(device)
    
    train_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    train_inst_f1 = BinaryF1Score().to(device)
    # train_inst_ap = BinaryAveragePrecision().to(device)
    train_bottom_iou = BinaryJaccardIndex().to(device)

    val_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    val_inst_acc = BinaryAccuracy().to(device)
    val_bottom_acc = BinaryAccuracy().to(device)

    val_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    val_inst_f1 = BinaryF1Score().to(device)
    # val_inst_ap = BinaryAveragePrecision().to(device)
    val_bottom_iou = BinaryJaccardIndex().to(device)

    iters = len(train_loader)
    ema_decay = wandb.config['ema_decay_per_epoch']**(1/iters)
    print(f'EMA decay: {ema_decay}')
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
    
    best_acc = 0.
    stopper = EarlyStopping(patience=10, min_delta=5e-5, mode='max', verbose=True)
    save_path = 'output'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = init_logger(os.path.join(save_path, 'log.txt'))
    for epoch in range(wandb.config['epochs']):
        logger.info(f'------------- Now start epoch {epoch}------------- ')
        model.train()
        # train_per_inst_acc = []
        train_losses = []
        train_bar = tqdm(train_loader)
        for data in train_bar:
            graphs = data["graph"].to(device, non_blocking=True)
            inst_label = data["inst_labels"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]
            bottom_label = graphs.ndata["bottom_y"]
            
            # Zero the gradients
            opt.zero_grad(set_to_none=True)
            
            # Forward pass
            seg_pred, inst_pred, bottom_pred = model(graphs)

            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = instance_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)
            loss = wandb.config['seg_a'] * loss_seg + \
                wandb.config['inst_a'] * loss_inst + \
                wandb.config['bottom_a'] * loss_bottom
            train_losses.append(loss.item())

            lr = opt.param_groups[0]["lr"]
            info = "Epoch:%d LR:%f Seg:%f Inst:%f Bottom:%f Total:%f" % (
                    epoch, lr, loss_seg, loss_inst, loss_bottom, loss)
            train_bar.set_description(info)

            # # Backward pass
            loss.backward()
            opt.step()

            # Update the moving average with the new parameters from the last optimizer step
            ema.update()
            
            train_seg_acc.update(seg_pred, seg_label)
            train_seg_iou.update(seg_pred, seg_label)
            train_inst_acc.update(inst_pred, inst_label)
            train_inst_f1.update(inst_pred, inst_label)
            train_bottom_acc.update(bottom_pred, bottom_label)
            train_bottom_iou.update(bottom_pred, bottom_label)
        
        scheduler.step()
        # batch end
        mean_train_loss = np.mean(train_losses).item()
        mean_train_seg_acc = train_seg_acc.compute().item()
        mean_train_seg_iou = train_seg_iou.compute().item()
        mean_train_inst_acc = train_inst_acc.compute().item()
        mean_train_inst_f1 = train_inst_f1.compute().item()
        mean_train_bottom_acc = train_bottom_acc.compute().item()
        mean_train_bottom_iou = train_bottom_iou.compute().item()
        
        logger.info(f'train_loss : {mean_train_loss}, \
                      train_seg_acc: {mean_train_seg_acc}, \
                      train_seg_iou: {mean_train_seg_iou}, \
                      train_inst_acc: {mean_train_inst_acc}, \
                      train_inst_f1: {mean_train_inst_f1}, \
                      train_bottom_acc: {mean_train_bottom_acc}, \
                      train_bottom_iou: {mean_train_bottom_iou}')
        wandb.log({'epoch': epoch, 
                   'train_loss': mean_train_loss, 
                   'train_seg_acc': mean_train_seg_acc, 
                   'train_seg_iou': mean_train_seg_iou, 
                   'train_inst_acc': mean_train_inst_acc, 
                   'train_inst_f1': mean_train_inst_f1, 
                   'train_bottom_acc': mean_train_bottom_acc, 
                   'train_bottom_iou': mean_train_bottom_iou
                    })
        
        train_seg_acc.reset()
        train_inst_acc.reset()
        train_bottom_acc.reset()
        train_seg_iou.reset()
        train_inst_f1.reset()
        # train_inst_ap.reset()
        train_bottom_iou.reset()
        
        # eval
        with torch.no_grad():
            with ema.average_parameters():
                model.eval()
                # val_per_inst_acc = []
                val_losses = []
                for data in tqdm(val_loader):
                    graphs = data["graph"].to(device)
                    inst_label = data["inst_labels"].to(device)
                    seg_label = graphs.ndata["seg_y"]
                    bottom_label = graphs.ndata["bottom_y"]
                    
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        seg_pred, inst_pred, bottom_pred = model(graphs)
                                                                
                        loss_seg = seg_loss(seg_pred, seg_label)
                        loss_inst = instance_loss(inst_pred, inst_label)
                        loss_bottom = bottom_loss(bottom_pred, bottom_label)
                        loss = wandb.config['seg_a'] * loss_seg + \
                                wandb.config['inst_a'] * loss_inst + \
                                wandb.config['bottom_a'] * loss_bottom
                    val_losses.append(loss.item())

                    val_seg_acc.update(seg_pred, seg_label)
                    val_seg_iou.update(seg_pred, seg_label)
                    val_inst_acc.update(inst_pred, inst_label)
                    val_inst_f1.update(inst_pred, inst_label)
                    val_bottom_acc.update(bottom_pred, bottom_label)
                    val_bottom_iou.update(bottom_pred, bottom_label)
                # val end
                mean_val_loss = np.mean(val_losses).item()
                mean_val_seg_acc = val_seg_acc.compute().item()
                mean_val_seg_iou = val_seg_iou.compute().item()
                mean_val_inst_acc = val_inst_acc.compute().item()
                mean_val_inst_f1 = val_inst_f1.compute().item()
                mean_val_bottom_acc = val_bottom_acc.compute().item()
                mean_val_bottom_iou = val_bottom_iou.compute().item()
                                                          
                logger.info(f'val_loss : {mean_val_loss}, \
                              val_seg_acc: {mean_val_seg_acc}, \
                              val_seg_iou: {mean_val_seg_iou}, \
                              val_inst_acc: {mean_val_inst_acc}, \
                              val_inst_f1: {mean_val_inst_f1}, \
                              val_bottom_acc: {mean_val_bottom_acc}, \
                              val_bottom_iou: {mean_val_bottom_iou}')
                wandb.log({'epoch': epoch, 
                           'val_loss': mean_val_loss, 
                           'val_seg_acc': mean_val_seg_acc, 
                           'val_seg_iou': mean_val_seg_iou, 
                           'val_inst_acc': mean_val_inst_acc, 
                           'val_inst_f1': mean_val_inst_f1, 
                           'val_bottom_acc': mean_val_bottom_acc, 
                           'val_bottom_iou': mean_val_bottom_iou
                            })
                
                val_seg_acc.reset()
                val_seg_iou.reset()
                val_inst_acc.reset()
                val_inst_f1.reset()
                # val_inst_ap.reset()
                val_bottom_acc.reset()
                val_bottom_iou.reset()

                cur_acc = mean_val_seg_iou + mean_val_inst_f1 + mean_val_bottom_iou
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    logger.info(f'best metric: {cur_acc}, model saved')
                    torch.save(model.state_dict(), os.path.join(save_path, "weight_%d-epoch.pth"%(epoch)))
                stopper(cur_acc, model, os.path.join(save_path, "best_model.pth"))
                if stopper.early_stop:
                    logger.info(f'Early stopping triggered at epoch {epoch}')
                    break
          # epoch end
        
    # training end test
    graphs = train_dataset.graphs() # no need to load graphs again !
    test_dataset = MFInstSegDataset(root_dir=dataset, graphs=graphs, split='test', 
                                     center_and_scale=False, normalize=True, random_rotate=False,
                                     dataset_type=dataset_type, num_threads=8)
    test_loader = test_dataset.get_dataloader(batch_size=wandb.config['batch_size'], pin_memory=True)

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_inst_acc = BinaryAccuracy().to(device)
    test_bottom_acc = BinaryAccuracy().to(device)
    
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    test_inst_f1 = BinaryF1Score().to(device)
    # test_inst_ap = BinaryAveragePrecision().to(device)
    test_bottom_iou = BinaryJaccardIndex().to(device)

    with torch.no_grad():
        logger.info(f'------------- Now start testing ------------- ')
        model.eval()
        # test_per_inst_acc = []
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            inst_label = data["inst_labels"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]
            bottom_label = graphs.ndata["bottom_y"]
            
            # Forward pass
            seg_pred, inst_pred, bottom_pred = model(graphs)

            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = instance_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)
            loss = wandb.config['seg_a'] * loss_seg + \
                   wandb.config['inst_a'] * loss_inst + \
                   wandb.config['bottom_a'] * loss_bottom
            test_losses.append(loss.item())
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            test_inst_acc.update(inst_pred, inst_label)
            test_inst_f1.update(inst_pred, inst_label)
            test_bottom_acc.update(bottom_pred, bottom_label)
            test_bottom_iou.update(bottom_pred, bottom_label)
        
        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        mean_test_inst_acc = test_inst_acc.compute().item()
        mean_test_inst_f1 = test_inst_f1.compute().item()
        mean_test_bottom_acc = test_bottom_acc.compute().item()
        mean_test_bottom_iou = test_bottom_iou.compute().item()
        
        logger.info(f'test_loss : {mean_test_loss}, \
                      test_seg_acc: {mean_test_seg_acc}, \
                      test_seg_iou: {mean_test_seg_iou}, \
                      test_inst_acc: {mean_test_inst_acc}, \
                      test_inst_f1: {mean_test_inst_f1}, \
                      test_bottom_acc: {mean_test_bottom_acc}, \
                      test_bottom_iou: {mean_test_bottom_iou}')
        wandb.log({'test_loss': mean_test_loss, 
                   'test_seg_acc': mean_test_seg_acc, 
                   'test_seg_iou': mean_test_seg_iou, 
                   'test_inst_acc': mean_test_inst_acc, 
                   'test_inst_f1': mean_test_inst_f1, 
                   'test_bottom_acc': mean_test_bottom_acc, 
                   'test_bottom_iou': mean_test_bottom_iou
                    })
