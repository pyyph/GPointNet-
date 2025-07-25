import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm, Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.autograd import Variable

# 核心修改：将 torch_geometric 的导入移至文件顶部
try:
    from torch_geometric.nn import PointConv, fps, radius, global_max_pool
except ImportError:
    # 如果 torch_geometric 未安装，则设置一个标志或打印警告
    PointConv, fps, radius, global_max_pool = None, None, None, None
    print("Warning: torch_geometric not found. PointNet++ related models will be unavailable.")


# 新增：Transformer 模块
class TransformerBlock(nn.Module):
    """
    一个基础的 Transformer 模块，包含多头自注意力和一个前馈网络。
    """
    def __init__(self, in_channels, num_heads=8, ff_dim=2048):
        """
        初始化 Transformer 模块。
        :param in_channels: int, 输入特征的维度
        :param num_heads: int, 多头注意力中的头数
        :param ff_dim: int, 前馈网络中间层的维度
        """
        super(TransformerBlock, self).__init__()
        # 核心修改：移除 batch_first=True 以兼容 PyTorch 1.8.0
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(in_channels)
        self.layernorm2 = nn.LayerNorm(in_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, in_channels)
        ) #TransformerBlock 类实现了 Transformer 架构中的基本组件，包括多头自注意力机制和前馈网络，用于捕获输入数据中的复杂特征关系

    def forward(self, x):
        """
        Transformer 模块的前向传播。
        :param x: Tensor, 输入张量，形状为 (B, N, C)，其中 B 是批量大小, N 是点的数量, C 是特征维度。
        :return: Tensor, 输出张量，形状与输入相同。
        """
        # 核心修改：调整维度以匹配 PyTorch 1.8.0 的 MultiheadAttention API (N, B, C)
        # 在 TransformerBlock 的 forward 方法中，调整了输入张量的维度以匹配 PyTorch 1.8.0 的 MultiheadAttention API
        x_transposed = x.permute(1, 0, 2)
        # 自注意力部分
        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
        # 将维度转置回来 (B, N, C)
        attn_output = attn_output.permute(1, 0, 2)
        # 残差连接和层归一化
        x = self.layernorm1(x + attn_output)
        # 前馈网络部分
        ff_output = self.feed_forward(x)
        # 第二个残差连接和层归一化
        x = self.layernorm2(x + ff_output)
        return x

def knn(x, k):
    # x: [B, C, N]
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return idx

class EdgeConvTorch(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.size()
        idx = knn(x, k=self.k)  # [B, N, k]

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()  # [B, N, C]
        feature = x.view(B * N, -1)[idx, :].view(B, N, self.k, C)  # [B, N, k, C]
        x = x.view(B, N, 1, C).repeat(1, 1, self.k, 1)  # [B, N, k, C]
        edge_feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)  # [B, 2C, N, k]

        out = self.mlp(edge_feature)  # [B, out_channels, N, k]
        out = torch.max(out, dim=-1)[0]  # [B, out_channels, N]
        return out


class energy_point_default(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        net_local, net_global = [], []
        prev = config.point_dim
        self.pnt_axis = 2 if config.swap_axis else 1
        for h in config.hidden_size[0]:
            layer = residual_Conv1d(h) if h==prev else torch.nn.Conv1d(prev, h, 1)
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02)
            # torch.nn.init.zeros_(layer.bias)
            net_local.append(layer)
            if config.batch_norm == "bn":
                net_local.append(torch.nn.BatchNorm1d(h)) # Question exists
            elif config.batch_norm == "ln":
                net_local.append(torch.nn.LayerNorm(config.num_point))
            elif config.batch_norm == "lnm":
                net_local.append(torch.nn.LayerNorm([h, config.num_point]))
            elif config.batch_norm == "in":
                net_local.append(torch.nn.InstanceNorm1d(h)) # Question exists
            if config.activation != "":
                net_local.append(getattr(torch.nn, config.activation)())
            prev = h
        for h in config.hidden_size[1]:
            layer = residual_Linear(h) if h==prev else torch.nn.Linear(prev, h)
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02)
            # torch.nn.init.zeros_(layer.bias)
            net_global.append(layer)
            if config.activation != "":
                net_global.append(getattr(torch.nn, config.activation)())
            prev = h
        net_global.append(torch.nn.Linear(prev, 1))
        self.local = torch.nn.Sequential(*net_local)
        self.edge_conv = EdgeConvTorch(
            in_channels=config.hidden_size[0][-1],
            out_channels=config.hidden_size[0][-1],  # 保持维度一致
            k=20  # 你可以调这个K值
        )

        self.globals = torch.nn.Sequential(*net_global)

    def forward(self, point_cloud, out_local=False, out_every_layer=False):

        local = self.local(point_cloud)
        if out_local:
            return local
        out = self.globals(torch.mean(local, self.pnt_axis))
        return out

    def _output_all(self, pcs):

        res = []
        for layer in self.local:
            pcs = layer(pcs)
            if type(layer) is torch.nn.LayerNorm:
                res.append(pcs)
        return res

class residual_Conv1d(torch.nn.Module):

    def __init__(self, h):
        super().__init__()
        self.layer = torch.nn.Conv1d(h, h, 1)
    def forward(self, x):
        return self.layer(x) + x

class residual_Linear(torch.nn.Module):

    def __init__(self, h):
        super().__init__()
        self.layer = torch.nn.Linear(h, h)
    def forward(self, x):
        return self.layer(x) + x

class energy_point_residual(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        net_local, net_global = [], []
        prev = config.point_dim
        self.pnt_axis = 2 if config.swap_axis else 1
        for h in config.hidden_size[0]:
            layer = torch.nn.Conv1d(prev, h, 1)
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02)
            # torch.nn.init.zeros_(layer.bias)
            net_local.append(layer)
            if config.batch_norm == "bn":
                net_local.append(torch.nn.BatchNorm1d(h)) # Question exists
            elif config.batch_norm == "ln":
                net_local.append(torch.nn.LayerNorm(config.num_point))
            elif config.batch_norm == "lnm":
                net_local.append(torch.nn.LayerNorm([h, config.num_point]))
            elif config.batch_norm == "in":
                net_local.append(torch.nn.InstanceNorm1d(h)) # Question exists
            if config.activation != "":
                net_local.append(getattr(torch.nn, config.activation)())
            prev = h
        for h in config.hidden_size[1]:
            layer = torch.nn.Linear(prev, h)
            # # Inherit from tf
            # torch.nn.init.normal_(layer.weight, 0, 0.02)
            # torch.nn.init.zeros_(layer.bias)
            net_global.append(layer)
            if config.activation != "":
                net_global.append(getattr(torch.nn, config.activation)())
            prev = h
        net_global.append(torch.nn.Linear(prev, 1))
        self.local = torch.nn.Sequential(*net_local)
        self.globals = torch.nn.Sequential(*net_global)

    def forward(self, point_cloud, out_local=False):

        local = self.local(point_cloud)
        local = self.edge_conv(local)
        if out_local:
            return torch.mean(local, self.pnt_axis)
        out = self.globals(torch.mean(local, self.pnt_axis))
        return out

class energy_point_pointnet(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(energy_point_pointnet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# Inherit from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        if PointConv is None:
            raise ImportError("torch_geometric is not installed. Please install it to use SAModule.")
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        if fps is None or radius is None:
            raise ImportError("torch_geometric is not installed. Please install it to use SAModule.")
        
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        # SAModule的输出x是特征，pos是坐标，batch是批次索引
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        if global_max_pool is None:
            raise ImportError("torch_geometric is not installed. Please install it to use GlobalSAModule.")
        
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class energy_point_pointnet2(torch.nn.Module):
    def __init__(self, config=None):
        super(energy_point_pointnet2, self).__init__()

        if PointConv is None:
            print("Warning: torch_geometric not found. PointNet++ model will not be fully functional.")
            self.use_pointnet_modules = False
            return
        
        self.use_pointnet_modules = True
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.transformer1 = TransformerBlock(in_channels=128, num_heads=4) # 在第一个SA模块后添加Transformer
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.transformer2 = TransformerBlock(in_channels=256, num_heads=8) # 在第二个SA模块后添加Transformer
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)
        if config:
            self.batch_i = torch.arange(config.batch_size, device=config.device).repeat(config.num_point,1).t().reshape(-1)
        else:
            # 提供一个默认值，以防config为None
            self.batch_i = None


    def forward(self, data):
        if not self.use_pointnet_modules:
            raise RuntimeError("torch_geometric is required for PointNet++ forward pass but is not installed.")

        # 确保输入数据是 (B, N, C) 的形式
        if data.dim() == 3 and data.shape[1] == 3:
             # 如果是 (B, C, N)，转换为 (B, N, C)
             data = data.permute(0, 2, 1)
        
        # 准备初始数据
        pos = data.reshape(-1, 3) # 将所有点云平铺 (B*N, 3)
        batch_size = data.shape[0]
        num_points = data.shape[1]
        
        # 在 energy_point_pointnet2 类中，动态生成 batch_i，以适应不同的批量大小。
        # 根据输入数据的批量大小动态生成批次索引，提高了模型的灵活性。
        if self.batch_i is None or len(self.batch_i) != batch_size * num_points:
             # 动态创建 batch 索引
             batch = torch.arange(batch_size, device=data.device).repeat_interleave(num_points)
        else:
             batch = self.batch_i[:batch_size * num_points]

        # 在 energy_point_pointnet2 类中，添加了两个 TransformerBlock 实例，分别在第一个和第二个 SAModule 之后。
        # 两个 TransformerBlock 实例分别在第一个和第二个 SAModule 之后，用于在局部特征提取后应用 Transformer 编码器，以捕获更复杂的特征关系。
        # 第一个SA模块
        x, pos, batch = self.sa1_module(None, pos, batch)
        
        # 第一个Transformer模块
        # SAModule的输出x是 (N', C)，需要变回 (B, N'/B, C)
        # 注意：由于fps采样，每个点云剩下的点数可能不同，不能直接reshape
        # 我们需要根据batch索引来处理
        # 这是一个简化的假设，即每个批次项的点数在采样后仍然相同
        num_points_sa1 = pos.shape[0] // batch_size
        x = x.view(batch_size, num_points_sa1, -1)
        x = self.transformer1(x)
        x = x.reshape(-1, x.shape[2]) # 变回 (N', C)

        # 第二个SA模块
        x, pos, batch = self.sa2_module(x, pos, batch)

        # 第二个Transformer模块
        num_points_sa2 = pos.shape[0] // batch_size
        x = x.view(batch_size, num_points_sa2, -1)
        x = self.transformer2(x)
        x = x.reshape(-1, x.shape[2]) # 变回 (N', C)

        # 全局SA模块
        x, pos, batch = self.sa3_module(x, pos, batch)

        # 分类头
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x
