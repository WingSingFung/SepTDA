import torch
import torch.nn as nn
from packaging.version import parse as V

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class TimeSegmentation(nn.Module):
    """时间分割和合并模块
    
    用于将时间序列数据分割成重叠的片段，然后重新合并。
    这对于处理长序列数据特别有用，可以减少内存使用并提高计算效率。
    """
    
    def __init__(self, segment_size: int = 96):
        """初始化时间分割模块
        
        Args:
            segment_size (int): 分割片段的大小，默认为96
        """
        super().__init__()
        self.segment_size = segment_size
    
    def split_feature(self, x):
        """将特征分割成重叠的片段
        
        Args:
            x (torch.Tensor): 输入特征 [B, D, T]
            
        Returns:
            torch.Tensor: 分割后的特征 [B, D, segment_size, n_chunks]
        """
        B, D, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, D, self.segment_size, -1)

    def merge_feature(self, x, length=None):
        """将分割的片段合并回原始序列
        
        Args:
            x (torch.Tensor): 分割的特征 [B, D, L, n_chunks]
            length (int, optional): 目标长度，如果为None则自动计算
            
        Returns:
            torch.Tensor: 合并后的特征 [B, D, length]
        """
        B, D, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, D * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, D, length)