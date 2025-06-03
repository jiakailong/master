import os
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def load_ply_file(file_path):
    """
    Load PLY file and extract x, y, z, intensity
    """
    points = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Find the end of header
        header_end = 0
        vertex_count = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line.strip() == 'end_header':
                header_end = i + 1
                break
        
        # Read vertex data
        for i in range(header_end, header_end + vertex_count):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    x, y, z, intensity = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z, intensity])
    
    return np.array(points, dtype=np.float32)

def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Farthest point sampling
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class LungDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_intensity = getattr(args, 'use_intensity', True)  # 默认使用强度信息
        self.num_category = 4  # 固定为4个类别 (0, 1, 2, 3)
        
        # 创建类别映射
        self.classes = {'0': 0, '1': 1, '2': 2, '3': 3}
        
        # 构建数据路径
        self.datapath = []
        for class_name in ['0', '1', '2', '3']:
            class_dir = os.path.join(self.root, class_name, split)
            if os.path.exists(class_dir):
                ply_files = [f for f in os.listdir(class_dir) if f.endswith('.ply')]
                for ply_file in ply_files:
                    full_path = os.path.join(class_dir, ply_file)
                    self.datapath.append((class_name, full_path))
        
        print(f'The size of {split} data is {len(self.datapath)}')
        
        if self.uniform:
            self.save_path = os.path.join(root, f'lung_data_{split}_{self.npoints}pts_fps.dat')
        else:
            self.save_path = os.path.join(root, f'lung_data_{split}_{self.npoints}pts.dat')
        
        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f'Processing data {self.save_path} (only running in the first time)...')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    class_name, file_path = self.datapath[index]
                    cls = self.classes[class_name]
                    cls = np.array([cls]).astype(np.int32)
                    
                    # 加载PLY文件
                    point_set = load_ply_file(file_path)
                    
                    # 确保有足够的点
                    if len(point_set) < self.npoints:
                        # 如果点数不够，进行重复采样
                        indices = np.random.choice(len(point_set), self.npoints, replace=True)
                        point_set = point_set[indices]
                    elif self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]
                    
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print(f'Load processed data from {self.save_path}...')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
    
    def __len__(self):
        return len(self.datapath)
    
    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            class_name, file_path = self.datapath[index]
            cls = self.classes[class_name]
            label = np.array([cls]).astype(np.int32)
            
            # 加载PLY文件
            point_set = load_ply_file(file_path)
            
            # 确保有足够的点
            if len(point_set) < self.npoints:
                indices = np.random.choice(len(point_set), self.npoints, replace=True)
                point_set = point_set[indices]
            elif self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
        
        # 标准化坐标
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        # 提取强度值到
        if self.use_intensity and point_set.shape[1] >= 4:
            intensity = point_set[:, 3:4]
            point_set[:, 3:4] = intensity
        else:
            # 如果不使用强度信息，只返回坐标
            point_set = point_set[:, 0:3]
        
        return point_set, label[0]
    
    def __getitem__(self, index):
        return self._get_item(index)