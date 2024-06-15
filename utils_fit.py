import os
import random
import numpy as np 
import torch
from loss import focal_loss, reg_l1_loss
from tqdm import tqdm
from loss import get_lr
from calc_coco_val import calculate_eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def fit_one_epoch(model_train, model,optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, cocoGt,classes,folder,best_mean_AP,local_rank=0):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            hm, wh, offset  = model_train(batch_images)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            
            loss            = c_loss + wh_loss + off_loss

            total_loss      += loss.item()
            total_c_loss    += c_loss.item()
            total_r_loss    += wh_loss.item() + off_loss.item()

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
                loss            = c_loss + wh_loss + off_loss

                total_loss      += loss.item()
                total_c_loss    += c_loss.item()
                total_r_loss    += wh_loss.item() + off_loss.item()
                

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
            
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            hm, wh, offset  = model_train(batch_images)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            loss            = c_loss + wh_loss + off_loss

            val_loss        += loss.item()


            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(".", 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        
        calculate_eval(model,cocoGt,classes,folder)
        try:
            cocoDt = cocoGt.loadRes("detection_results.json")
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            mean_ap = cocoEval.stats[0]  # This is the mAP at IoU thresholds from .50 to .95
            mean_ap_05 = cocoEval.stats[1]
            mean_ap_075 = cocoEval.stats[2] 
        except:
            mean_ap,mean_ap_05,mean_ap_075 = 0.0,0.0,0.0        

        print(f"Mean Average Precision (mAP) across IoU thresholds [0.50, 0.95]: {mean_ap:.3f}")
        if mean_ap > best_mean_AP:
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(".", "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(".", "last_epoch_weights.pth"))
    return mean_ap


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False