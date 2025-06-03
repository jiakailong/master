"""
Training script for enhanced lung point cloud classification using MSG model with graph convolution
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.LungDataLoader import LungDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training (reduced for enhanced model)')
    parser.add_argument('--model', default='pointnet2_cls_msg_enhanced', help='model name [pointnet2_cls_msg_enhanced]')
    parser.add_argument('--num_category', default=4, type=int, help='number of categories (4 for lung data)')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    
    # 特征相关参数
    parser.add_argument('--use_intensity', action='store_true', default=True, help='use intensity information')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normal information')
    
    # 图卷积相关参数
    parser.add_argument('--use_edge_conv', action='store_true', default=True, help='use EdgeConv in graph layers')
    parser.add_argument('--use_graph_conv', action='store_true', default=True, help='use GraphConv in graph layers')
    parser.add_argument('--use_self_attn', action='store_true', default=True, help='use Self-Attention in graph layers')
    
    # 数据相关参数
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--data_path', type=str, default='/media/jiang/jkl/project/dataset/lung_point_cloud', help='path to dataset')
    
    # 训练策略参数
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0
        
    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def test(model, loader, num_class=4, args=None):
    """测试函数，适应增强模型"""
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc_mean = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc_mean


def apply_data_augmentation(points, training=True):
    """改进的数据增强，适合肺部点云"""
    if training:
        # 随机丢弃点 (较小的丢弃率，保持肺部结构)
        points = provider.random_point_dropout(points, max_dropout_ratio=0.1)
        
        # 随机缩放 (较小的缩放范围)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3], 
                                                              scale_low=0.9, 
                                                              scale_high=1.1)
        
        # 随机平移 (较小的平移范围)
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], 
                                                       shift_range=0.05)
        
        # 添加噪声 (较小的噪声)
        points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3], 
                                                        sigma=0.005, 
                                                        clip=0.02)
        
        # 随机旋转 (仅绕z轴小角度旋转，保持肺部方向)
        points[:, :, 0:3] = provider.rotate_point_cloud_z(points[:, :, 0:3])
    
    return points


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('lung_classification_enhanced')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_path

    train_dataset = LungDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = LungDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    log_string('Train dataset size: %d' % len(train_dataset))
    log_string('Test dataset size: %d' % len(test_dataset))

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    
    # 复制增强的模型文件
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils_enhanced.py', str(exp_dir))
    shutil.copy('./train_classification_enhanced.py', str(exp_dir))

    # 创建增强模型实例
    classifier = model.get_model(
        num_class, 
        normal_channel=args.use_normals, 
        use_intensity=args.use_intensity,
        use_edge_conv=args.use_edge_conv,
        use_graph_conv=args.use_graph_conv,
        use_self_attn=args.use_self_attn
    )
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # 打印模型信息
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log_string(f'Total parameters: {total_params:,}')
    log_string(f'Trainable parameters: {trainable_params:,}')

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # 优化器设置
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # 学习率调度器 - 使用余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epoch, 
        eta_min=args.learning_rate * 0.01
    )
    
    # 预热调度器
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=args.warmup_epochs
    )

    # 早停机制
    early_stopping = EarlyStopping(patience=args.patience)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    
    # 训练历史记录
    train_losses = []
    train_accs = []
    val_accs = []

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        log_string('Learning rate: %f' % optimizer.param_groups[0]['lr'])
        
        mean_correct = []
        epoch_losses = []
        classifier = classifier.train()

        # 学习率调度
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # 应用数据增强
            points = apply_data_augmentation(points, training=True)
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            epoch_losses.append(loss.item())
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        train_accs.append(train_instance_acc)
        
        log_string('Train Instance Accuracy: %f, Train Loss: %f' % (train_instance_acc, train_loss))

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class, args=args)
            val_accs.append(instance_acc)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            # 保存最佳模型
            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                }
                torch.save(state, savepath)
            
            # 早停检查
            if early_stopping(instance_acc):
                log_string(f'Early stopping triggered after {epoch + 1} epochs')
                break
                
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)