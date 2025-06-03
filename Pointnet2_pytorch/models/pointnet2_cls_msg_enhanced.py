import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils_enhanced import PointNetSetAbstractionMsgWithOriginalGraph, PointNetSetAbstractionWithOriginalGraph


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False, use_intensity=True,
                 use_edge_conv=True, use_graph_conv=True, use_self_attn=True):
        super(get_model, self).__init__()
        # 计算输入通道数
        in_channel = 0
        if use_intensity:
            in_channel += 1  # intensity channel
        if normal_channel:
            in_channel += 3  # normal channels
        
        self.normal_channel = normal_channel
        self.use_intensity = use_intensity
        self.use_edge_conv = use_edge_conv
        self.use_graph_conv = use_graph_conv
        self.use_self_attn = use_self_attn
        
        # 第一层：降采样到512点，同时在原始点云上进行图卷积
        self.sa1 = PointNetSetAbstractionMsgWithOriginalGraph(
            npoint=512, 
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[16, 32, 128], 
            in_channel=in_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            k_neighbors=16,
            original_k_neighbors=32,  # 原始点云使用更多邻居
            use_edge_conv=use_edge_conv,
            use_graph_conv=use_graph_conv,
            use_self_attn=use_self_attn
        )
        
        # 第二层：降采样到128点，继续在原始点云上进行图卷积
        self.sa2 = PointNetSetAbstractionMsgWithOriginalGraph(
            npoint=128, 
            radius_list=[0.2, 0.4, 0.8], 
            nsample_list=[32, 64, 128], 
            in_channel=320,  # 64+128+128=320
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            k_neighbors=12,
            original_k_neighbors=24,
            use_edge_conv=use_edge_conv,
            use_graph_conv=use_graph_conv,
            use_self_attn=use_self_attn
        )
        
        # 第三层：全局特征聚合，同样在原始点云上进行最后的图卷积
        self.sa3 = PointNetSetAbstractionWithOriginalGraph(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=640 + 3,  # 128+256+256=640, 加上3维坐标
            mlp=[256, 512, 1024], 
            group_all=True,
            k_neighbors=8,
            original_k_neighbors=16,
            use_edge_conv=use_edge_conv,
            use_graph_conv=use_graph_conv,
            use_self_attn=use_self_attn
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
                xyz_coord = xyz[:, :3, :]   # 坐标
            elif self.use_intensity:
                # 只使用强度：[x,y,z,intensity]
                norm = xyz[:, 3:4, :]  # 只有强度
                xyz_coord = xyz[:, :3, :]    # 坐标
            elif self.normal_channel:
                # 只使用法向量：[x,y,z,nx,ny,nz]
                norm = xyz[:, 3:6, :]  # 法向量
                xyz_coord = xyz[:, :3, :]    # 坐标
            else:
                norm = None
                xyz_coord = xyz[:, :3, :]
        else:
            norm = None
            xyz_coord = xyz
        
        # 保存原始点云信息，用于图构建
        original_xyz = xyz_coord.clone()
        original_points = norm.clone() if norm is not None else None
        
        # 三层特征提取 - 传递原始点云信息
        l1_xyz, l1_points = self.sa1(
            xyz_coord, norm, 
            original_xyz=original_xyz, 
            original_points=original_points
        )
        
        l2_xyz, l2_points = self.sa2(
            l1_xyz, l1_points,
            original_xyz=original_xyz, 
            original_points=original_points
        )
        
        l3_xyz, l3_points = self.sa3(
            l2_xyz, l2_points,
            original_xyz=original_xyz, 
            original_points=original_points
        )
        
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