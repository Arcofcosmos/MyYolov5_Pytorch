'''
Author: TuZhou
Version: 1.0
Date: 2021-08-19 18:48:06
LastEditTime: 2021-08-19 20:07:35
LastEditors: TuZhou
Description: 
FilePath: \my_yolov5\train.py
'''
#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
import yaml

# import os,sys
# sys.path.append("./") 

from nets.yolov5 import YoloBody
from nets.yolo_training import LossHistory, YOLOLoss, weights_init
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.k_means_anchors import calculate_anchors


#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

        
def fit_one_epoch(net, yolo_loss,epoch,train_iteration,val_iteration,train_dataLoder,genval,Epoch,cuda):
    if Tensorboard:
        global train_tensorboard_step, val_tensorboard_step
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=train_iteration,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataLoder):
            if iteration >= train_iteration:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            total_loss += loss.item()

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            if Tensorboard:
                # 将loss写入tensorboard，每一步都写
                writer.add_scalar('Train_loss', loss, train_tensorboard_step)
                train_tensorboard_step += 1

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    # 将loss写入tensorboard，下面注释的是每个世代保存一次
    # if Tensorboard:
    #     writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)
    net.eval()
    print('Start Validation')
    with tqdm(total=val_iteration, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= val_iteration:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            # 将loss写入tensorboard, 下面注释的是每一步都写
            # if Tensorboard:
            #     writer.add_scalar('Val_loss', loss, val_tensorboard_step)
            #     val_tensorboard_step += 1
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    # 将loss写入tensorboard，每个世代保存一次
    if Tensorboard:
        writer.add_scalar('Val_loss',val_loss / (val_iteration+1), epoch)
    loss_history.append_loss(total_loss/(train_iteration+1), val_loss/(val_iteration+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(train_iteration+1),val_loss/(val_iteration+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(train_iteration+1),val_loss/(val_iteration+1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    yaml_path = './cfg/cfg.yaml'
    yaml_file = Path(yaml_path)
    with open(yaml_file, encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)


    Tensorboard = yaml_dict['Tensorboard']
    anchors_flag = yaml_dict['anchors_flag']        #是否自定义锚框

    Cuda = yaml_dict['Cuda']
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = yaml_dict['normalize']

    input_shape = yaml_dict['input_shape']
    # print(input_shape[1])
    # exit()
    #----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    anchors_path = yaml_dict['anchors_path']
    classes_path = yaml_dict['classes_path']   
    dataset_path = yaml_dict['dataset_path']
    #------------------------------------------------------#
    #   mosaic 马赛克数据增强 True or False 
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic = yaml_dict['mosaic']
    Cosine_lr = yaml_dict['Cosine_lr']
    smoooth_label = yaml_dict['smoooth_label']
    model_cfg_path = yaml_dict['model_cfg_paht']

    #如果需要则重新计算数据集锚框
    if anchors_flag:
        calculate_anchors(dataset_anno_path=dataset_path, anchorsPath=anchors_path, SIZE = input_shape[0])

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    #class_names = get_classes(classes_path)
    #anchors = get_anchors(anchors_path)
    anchors = yaml_dict['anchors']
    print(anchors)
    exit()

    num_classes = yaml_dict['num_classes']


    #------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改classes_path和对应的txt文件
    #------------------------------------------------------#
    model = YoloBody(yaml_path = model_cfg_path, number_anchors = len(anchors[0]))
    weights_init(model)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = "trained_model/yolo4_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize)
    loss_history = LossHistory("logs/")

    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = './datasets/WZRY/train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    if Tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir='logs',flush_secs=60)
        if Cuda:
            graph_inputs = torch.randn(1,3,input_shape[0],input_shape[1]).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.randn(1,3,input_shape[0],input_shape[1]).type(torch.FloatTensor)
        writer.add_graph(model, graph_inputs)
        train_tensorboard_step  = 1
        val_tensorboard_step    = 1

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 1e-3
        Batch_size      = 4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        
        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        optimizer       = optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        train_dataLoder             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        val_dataLoader         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)

        train_iteration      = num_train // Batch_size
        val_iteration  = num_val // Batch_size
        
        if train_iteration == 0 or val_iteration == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch, train_iteration, val_iteration,train_dataLoder, val_dataLoader,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        lr              = 1e-4
        Batch_size      = 2
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100

        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        optimizer       = optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        train_dataLoder             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        val_dataLoader         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)

        train_iteration      = num_train // Batch_size
        val_iteration  = num_val // Batch_size
        
        if train_iteration == 0 or val_iteration == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,train_iteration,val_iteration,train_dataLoder,val_dataLoader,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
