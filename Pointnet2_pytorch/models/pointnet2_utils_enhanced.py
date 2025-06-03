import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    计算两组点之间的平方距离
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    根据索引获取点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球查询
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn(xyz, k):
    """
    构建k近邻图
    Input:
        xyz: points, [B, N, 3]
        k: number of neighbors
    Return:
        idx: k-nearest neighbors, [B, N, k]
    """
    batch_size, num_points, _ = xyz.shape
    
    # 计算距离矩阵
    dist = square_distance(xyz, xyz)  # [B, N, N]
    
    # 获取k+1个最近邻（包括自己）
    _, idx = torch.topk(dist, k+1, dim=-1, largest=False, sorted=True)
    
    # 移除自己，只保留k个邻居
    idx = idx[:, :, 1:]  # [B, N, k]
    
    return idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class EdgeConv(nn.Module):
    """
    EdgeConv层实现
    """
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, edges=None):
        """
        Input:
            x: [B, C, N]
            edges: [B, N, k] or None
        Output:
            edge_feature: [B, C_out, N]
        """
        batch_size, num_dims, num_points = x.size()
        
        if edges is None:
            # 重新计算k近邻
            xyz = x[:, :3, :].permute(0, 2, 1)  # [B, N, 3]
            edges = knn(xyz, self.k)  # [B, N, k]
        
        current_k_val = edges.size(-1)
        
        # 获取邻居特征
        x_neighbors = index_points(x.permute(0, 2, 1), edges)  # [B, N, k, C]
        x_neighbors = x_neighbors.permute(0, 3, 2, 1)  # [B, C, k, N]
        
        # 中心点特征
        x_central = x.unsqueeze(2).repeat(1, 1, current_k_val, 1)  # [B, C, k, N]
        
        # 边特征 = [中心点特征, 邻居特征 - 中心点特征]
        edge_feature = torch.cat([x_central, x_neighbors - x_central], dim=1)  # [B, 2*C, k, N]
        
        # 应用卷积
        edge_feature = self.conv(edge_feature)  # [B, out_channels, k, N]
        
        # 聚合邻居特征
        edge_feature = edge_feature.max(dim=2, keepdim=False)[0]  # [B, out_channels, N]
        
        return edge_feature

class GraphConv(nn.Module):
    """
    基础图卷积层
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, edges):
        """
        Input:
            x: [B, C, N]
            edges: [B, N, k]
        Output:
            out: [B, C_out, N]
        """
        batch_size, num_dims, num_points = x.size()
        k = edges.size(-1)
        
        # 获取邻居特征并聚合
        x_neighbors = index_points(x.permute(0, 2, 1), edges)  # [B, N, k, C]
        x_neighbors = x_neighbors.mean(dim=2)  # [B, N, C] - 平均聚合
        x_neighbors = x_neighbors.permute(0, 2, 1)  # [B, C, N]
        
        # 应用卷积
        out = self.conv(x_neighbors)
        
        return out

class SelfAttentionLayer(nn.Module):
    """
    自注意力层
    """
    def __init__(self, in_channels, out_channels, k=20, heads=8):
        super(SelfAttentionLayer, self).__init__()
        self.k = k
        self.heads = heads
        
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.head_dim = out_channels // heads
        
        self.q_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.out_conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, edges=None):
        """
        Input:
            x: [B, C, N]
            edges: [B, N, k] or None
        Output:
            out: [B, C_out, N]
        """
        B, C, N = x.size()
        
        if edges is None:
            # 重新计算k近邻
            xyz = x[:, :3, :].permute(0, 2, 1)  # [B, N, 3]
            edges = knn(xyz, self.k)  # [B, N, k]
        
        current_k_val = edges.size(-1)
        
        # 计算Q, K, V
        Q = self.q_conv(x)  # [B, out_channels, N]
        K = self.k_conv(x)  # [B, out_channels, N]
        V = self.v_conv(x)  # [B, out_channels, N]
        
        # 重塑为多头格式
        Q = Q.view(B, self.heads, self.head_dim, N)  # [B, heads, head_dim, N]
        K = K.view(B, self.heads, self.head_dim, N)
        V = V.view(B, self.heads, self.head_dim, N)
        
        # 为每个点获取其邻居的K和V
        K_neighbors = index_points(K.permute(0, 1, 3, 2).contiguous().view(B*self.heads, N, self.head_dim), 
                                   edges.unsqueeze(1).repeat(1, self.heads, 1, 1).view(B*self.heads, N, current_k_val))
        K_neighbors = K_neighbors.view(B, self.heads, N, current_k_val, self.head_dim).permute(0, 1, 4, 2, 3)
        
        V_neighbors = index_points(V.permute(0, 1, 3, 2).contiguous().view(B*self.heads, N, self.head_dim), 
                                   edges.unsqueeze(1).repeat(1, self.heads, 1, 1).view(B*self.heads, N, current_k_val))
        V_neighbors = V_neighbors.view(B, self.heads, N, current_k_val, self.head_dim).permute(0, 1, 4, 2, 3)
        
        # 计算注意力分数
        Q_expanded = Q.unsqueeze(-1)  # [B, heads, head_dim, N, 1]
        attention_scores = torch.sum(Q_expanded * K_neighbors, dim=2)  # [B, heads, N, k]
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, heads, N, k]
        
        # 应用注意力权重
        attention_weights = attention_weights.unsqueeze(2)  # [B, heads, 1, N, k]
        attended_features = torch.sum(attention_weights * V_neighbors, dim=-1)  # [B, heads, head_dim, N]
        
        # 合并多头
        attended_features = attended_features.view(B, -1, N)  # [B, out_channels, N]
        
        # 输出投影
        out = self.out_conv(attended_features)
        
        return out

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsgWithOriginalGraph(nn.Module):
    """
    增强的PointNet++多尺度消息传递层，整合图结构
    降采样仅用于选择中心点，图卷积在原始点云上进行
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, 
                 k_neighbors=16, original_k_neighbors=24, use_edge_conv=True, 
                 use_graph_conv=False, use_self_attn=False):
        super(PointNetSetAbstractionMsgWithOriginalGraph, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.k_neighbors = k_neighbors
        self.original_k_neighbors = original_k_neighbors
        self.use_edge_conv = use_edge_conv
        self.use_graph_conv = use_graph_conv
        self.use_self_attn = use_self_attn
        
        # PointNet++原始多尺度处理部分
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        
        # 基于原始点云的图卷积层
        self.original_graph_layers = nn.ModuleList()
        
        # 计算图卷积的输入和输出通道数
        if in_channel <= 0:
            # 如果没有特征通道，先通过MLP将坐标转换为特征
            self.coord_mlp = nn.Sequential(
                nn.Conv1d(3, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(64, 32, 1, bias=False),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(negative_slope=0.2)
            )
            graph_in_channel = 32
        else:
            self.coord_mlp = None
            graph_in_channel = in_channel
        
        # 确保输出通道数能被注意力头数整除
        graph_out_channel = max(32, ((graph_in_channel * 2) // 8) * 8)
        
        if use_edge_conv:
            self.original_graph_layers.append(EdgeConv(graph_in_channel, graph_out_channel, k=original_k_neighbors))
        if use_graph_conv:
            self.original_graph_layers.append(GraphConv(graph_in_channel, graph_out_channel))
        if use_self_attn:
            self.original_graph_layers.append(SelfAttentionLayer(graph_in_channel, graph_out_channel, k=original_k_neighbors))
        
        # 原始图卷积特征到采样点的映射
        out_channels = sum([m[-1] for m in mlp_list])
        if len(self.original_graph_layers) > 0:
            original_graph_out_channels = graph_out_channel * len(self.original_graph_layers)
            self.graph_feature_mapping = nn.Sequential(
                nn.Conv1d(original_graph_out_channels, out_channels//2, 1, bias=False),
                nn.BatchNorm1d(out_channels//2),
                nn.LeakyReLU(negative_slope=0.2)
            )
            
            # 特征融合层 - 结合PointNet++特征和图卷积特征
            self.fusion = nn.Sequential(
                nn.Conv1d(out_channels + out_channels//2, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.graph_feature_mapping = None
            self.fusion = None

    def forward(self, xyz, points, original_xyz=None, original_points=None):
        """
        Input:
            xyz: 输入点坐标, [B, C, N] 
            points: 输入点特征, [B, D, N]
            original_xyz: 原始点云坐标, [B, C, M], 如果为None则使用xyz
            original_points: 原始点云特征, [B, D, M], 如果为None则使用points
        Return:
            new_xyz: 采样点坐标, [B, C, S]
            new_points: 新特征, [B, D', S]
        """
        # 如果没有提供原始点云，则使用输入点云
        if original_xyz is None:
            original_xyz = xyz
        if original_points is None:
            original_points = points
            
        # 处理输入格式
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
            
        original_xyz_trans = original_xyz.permute(0, 2, 1)  # [B, M, C]
        if original_points is not None:
            original_points_trans = original_points.permute(0, 2, 1)  # [B, M, D]
        else:
            original_points_trans = None

        B, N, C = xyz.shape
        B_orig, M, C_orig = original_xyz_trans.shape
        S = self.npoint
        
        # 1. 在原始点云上进行图卷积
        original_graph_features = []
        if len(self.original_graph_layers) > 0:
            # 在原始点云上构建k近邻图
            original_edges = knn(original_xyz_trans, self.original_k_neighbors)
            
            # 准备图卷积的输入特征
            if original_points_trans is not None:
                # 将原始点云特征转换为正确格式 [B, D, M]
                original_features = original_points_trans.permute(0, 2, 1)
            else:
                # 如果没有特征，使用坐标作为特征并通过MLP处理
                if self.coord_mlp is not None:
                    original_features = self.coord_mlp(original_xyz_trans.permute(0, 2, 1))  # [B, 32, M]
                else:
                    # 这种情况不应该发生，但为了安全起见
                    original_features = original_xyz_trans.permute(0, 2, 1)  # [B, 3, M]
            
            # 在原始点云上应用图卷积
            for layer in self.original_graph_layers:
                graph_feat = layer(original_features, original_edges)  # [B, D, M]
                original_graph_features.append(graph_feat)
            
            # 合并原始图卷积特征
            if len(original_graph_features) > 0:
                original_graph_concat = torch.cat(original_graph_features, dim=1)  # [B, D*n_layers, M]
                original_graph_processed = self.graph_feature_mapping(original_graph_concat)  # [B, out_channels//2, M]
        
        # 2. 标准的PointNet++降采样和特征提取
        # FPS采样 - 仅用于选择中心点
        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)  # [B, S, C]
        
        # 多尺度分组和特征提取（标准PointNet++流程）
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D+3, K, S]
            
            # 应用多层感知机
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
                
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)  # 转换回 [B, C, S]
        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, sum(D'), S]
        
        # 3. 将原始图卷积特征映射到采样点
        if len(self.original_graph_layers) > 0 and len(original_graph_features) > 0:
            # 将原始点云的图卷积特征映射到采样点
            sampled_graph_features = self.interpolate_features(
                original_xyz_trans, new_xyz.permute(0, 2, 1), original_graph_processed
            )  # [B, out_channels//2, S]
            
            # 融合PointNet++特征和图卷积特征
            combined_features = torch.cat([new_points_concat, sampled_graph_features], dim=1)
            new_points = self.fusion(combined_features)
        else:
            new_points = new_points_concat
        
        return new_xyz, new_points

    def interpolate_features(self, source_xyz, target_xyz, source_features):
        """
        将原始点云的特征插值到采样点
        Input:
            source_xyz: 原始点云坐标, [B, M, 3]
            target_xyz: 目标点坐标, [B, S, 3]
            source_features: 原始点云特征, [B, D, M]
        Return:
            interpolated_features: 插值后的特征, [B, D, S]
        """
        B, M, _ = source_xyz.shape
        B, S, _ = target_xyz.shape
        
        if S == 1:
            # 如果只有一个目标点，使用全局平均
            interpolated_features = source_features.mean(dim=2, keepdim=True)  # [B, D, 1]
        else:
            # 计算距离
            dists = square_distance(target_xyz, source_xyz)  # [B, S, M]
            
            # 选择最近的3个点进行插值
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, S, 3]
            
            # 距离倒数权重
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, S, 3]
            
            # 获取最近邻特征并加权平均
            source_features_trans = source_features.permute(0, 2, 1)  # [B, M, D]
            nearest_features = index_points(source_features_trans, idx)  # [B, S, 3, D]
            interpolated_features = torch.sum(nearest_features * weight.unsqueeze(-1), dim=2)  # [B, S, D]
            interpolated_features = interpolated_features.permute(0, 2, 1)  # [B, D, S]
        
        return interpolated_features

class PointNetSetAbstractionWithOriginalGraph(nn.Module):
    """
    单一尺度版本，用于最后的全局聚合层
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, 
                 k_neighbors=16, original_k_neighbors=24, use_edge_conv=True, 
                 use_graph_conv=False, use_self_attn=False):
        super(PointNetSetAbstractionWithOriginalGraph, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.k_neighbors = k_neighbors
        self.original_k_neighbors = original_k_neighbors
        self.use_edge_conv = use_edge_conv
        self.use_graph_conv = use_graph_conv
        self.use_self_attn = use_self_attn
        self.group_all = group_all
        
        # 原始MLP层
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        # 基于原始点云的图卷积层
        self.original_graph_layers = nn.ModuleList()
        
        # 计算输入特征维度（去掉坐标维度）
        graph_in_channel = in_channel - 3
        # 确保输出通道数能被注意力头数整除
        graph_out_channel = max(32, ((graph_in_channel * 2) // 8) * 8) if graph_in_channel > 0 else 32
        
        if use_edge_conv:
            self.original_graph_layers.append(EdgeConv(graph_in_channel, graph_out_channel, k=original_k_neighbors))
        if use_graph_conv:
            self.original_graph_layers.append(GraphConv(graph_in_channel, graph_out_channel))
        if use_self_attn:
            self.original_graph_layers.append(SelfAttentionLayer(graph_in_channel, graph_out_channel, k=original_k_neighbors))
        
        # 特征融合层
        if len(self.original_graph_layers) > 0:
            original_graph_out_channels = graph_out_channel * len(self.original_graph_layers)
            self.graph_feature_mapping = nn.Sequential(
                nn.Conv1d(original_graph_out_channels, mlp[-1]//2, 1, bias=False),
                nn.BatchNorm1d(mlp[-1]//2),
                nn.LeakyReLU(negative_slope=0.2)
            )
            
            self.fusion = nn.Sequential(
                nn.Conv1d(mlp[-1] + mlp[-1]//2, mlp[-1], 1, bias=False),
                nn.BatchNorm1d(mlp[-1]),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.graph_feature_mapping = None
            self.fusion = None

    def forward(self, xyz, points, original_xyz=None, original_points=None):
        """
        全局聚合版本的前向传播
        """
        if original_xyz is None:
            original_xyz = xyz
        if original_points is None:
            original_points = points
            
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]
            
        original_xyz_trans = original_xyz.permute(0, 2, 1)  # [B, M, C]
        if original_points is not None:
            original_points_trans = original_points.permute(0, 2, 1)  # [B, M, D]

        # 1. 在原始点云上进行图卷积
        original_graph_features = []
        if len(self.original_graph_layers) > 0 and original_points_trans is not None:
            original_edges = knn(original_xyz_trans, self.original_k_neighbors)
            original_features = original_points_trans.permute(0, 2, 1)  # [B, D, M]
            
            for layer in self.original_graph_layers:
                graph_feat = layer(original_features, original_edges)
                original_graph_features.append(graph_feat)
            
            if len(original_graph_features) > 0:
                original_graph_concat = torch.cat(original_graph_features, dim=1)
                original_graph_processed = self.graph_feature_mapping(original_graph_concat)

        # 2. 标准的全局聚合
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, xyz.shape[2]).to(xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], xyz.shape[2])
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.view(points.shape[0], 1, points.shape[1], points.shape[2])], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # 非全局版本的处理...
            pass

        # 特征提取网络
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        point_feat = torch.max(new_points, 2)[0]  # [B, D', S]
        
        # 3. 融合图卷积特征
        if len(self.original_graph_layers) > 0 and len(original_graph_features) > 0:
            # 对于全局聚合，直接使用全局平均的图特征
            global_graph_features = original_graph_processed.mean(dim=2, keepdim=True)  # [B, D, 1]
            combined_features = torch.cat([point_feat, global_graph_features], dim=1)
            new_points = self.fusion(combined_features)
        else:
            new_points = point_feat
        
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points