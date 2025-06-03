import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False, use_intensity=True):
        super(get_model, self).__init__()
        # 计算输入通道数
        in_channel = 0
        if use_intensity:
            in_channel += 1  # intensity channel
        if normal_channel:
            in_channel += 3  # normal channels
        
        self.normal_channel = normal_channel
        self.use_intensity = use_intensity
        
        # MSG架构的第一层：多尺度分组
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512, 
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[16, 32, 128], 
            in_channel=in_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        
        # MSG架构的第二层：多尺度分组
        # 输入通道数 = 第一层输出的特征维度之和 (64+128+128=320)
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128, 
            radius_list=[0.2, 0.4, 0.8], 
            nsample_list=[32, 64, 128], 
            in_channel=320,
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        
        # 第三层：全局特征提取
        # 输入通道数 = 第二层输出的特征维度之和 + 3 (128+256+256+3=643)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=640 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True
        )
        
        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        
        # 分离坐标和特征
        if xyz.size()[1] > 3:
            if self.use_intensity and self.normal_channel:
                # 如果同时使用强度和法向量：[x,y,z,intensity,nx,ny,nz]
                norm = xyz[:, 3:, :]  # 强度 + 法向量
                xyz = xyz[:, :3, :]   # 坐标
            elif self.use_intensity:
                # 只使用强度：[x,y,z,intensity]
                norm = xyz[:, 3:4, :]  # 只有强度
                xyz = xyz[:, :3, :]    # 坐标
            elif self.normal_channel:
                # 只使用法向量：[x,y,z,nx,ny,nz]
                norm = xyz[:, 3:6, :]  # 法向量
                xyz = xyz[:, :3, :]    # 坐标
            else:
                norm = None
        else:
            norm = None
        
        # 通过MSG层进行特征提取
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # 分类头
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss