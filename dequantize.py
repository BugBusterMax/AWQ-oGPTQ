#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer


def unpack_weights(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """解包量化权重"""
    qweight = qweight.numpy().astype(np.uint32)
    unpacked_shape = (qweight.shape[0] * 32 // bits, qweight.shape[1])
    unpacked = np.zeros(unpacked_shape, dtype=np.uint32)
    mask = (1 << bits) - 1
    for i in range(qweight.shape[0]):
        for j in range(32 // bits):
            start_idx = i * (32 // bits) + j
            if start_idx < unpacked.shape[0]:
                unpacked[start_idx] = (qweight[i] >> (bits * j)) & mask
    return torch.from_numpy(unpacked.astype(np.int32))


def unpack_zeros(qzeros: torch.Tensor, bits: int) -> torch.Tensor:
    """解包零点"""
    qzeros = qzeros.numpy().astype(np.uint32)
    unpacked_shape = (qzeros.shape[0], qzeros.shape[1] * 32 // bits)
    unpacked = np.zeros(unpacked_shape, dtype=np.uint32)
    mask = (1 << bits) - 1
    for i in range(qzeros.shape[1]):
        for j in range(32 // bits):
            start_idx = i * (32 // bits) + j
            if start_idx < unpacked.shape[1]:
                unpacked[:, start_idx] = (qzeros[:, i] >> (bits * j)) & mask
    return torch.from_numpy(unpacked.astype(np.int32)) + 1


def dequantize_weight(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
                      g_idx: torch.Tensor, bits: int, infeatures: int, outfeatures: int) -> torch.Tensor:
    """修复的反量化函数，保证准确性同时提升性能"""
    # 解包权重和零点
    unpacked_qweight = unpack_weights(qweight, bits)  # [infeatures, outfeatures]
    unpacked_qzeros = unpack_zeros(qzeros, bits)      # [outfeatures, groups]
    
    # 转置scales以匹配维度 [outfeatures, groups]
    scales = scales.t().contiguous()
    unpacked_qzeros = unpacked_qzeros.t().contiguous()  # [groups, outfeatures] -> [outfeatures, groups]
    
    # 创建输出权重矩阵
    weight = torch.zeros((outfeatures, infeatures), dtype=torch.float16)
    
    # 按组处理以提升性能，但保持逻辑正确性
    unique_groups = torch.unique(g_idx)
    
    for group_id in unique_groups:
        # 找到属于当前组的输入特征索引
        group_mask = (g_idx == group_id)
        input_indices = torch.where(group_mask)[0]
        
        if len(input_indices) == 0:
            continue
            
        # 批量处理当前组的所有输入特征
        group_qweight = unpacked_qweight[input_indices]  # [group_size, outfeatures]
        group_scales = scales[:, group_id].unsqueeze(0)  # [1, outfeatures]
        group_zeros = unpacked_qzeros[:, group_id].unsqueeze(0)  # [1, outfeatures]
        
        # 向量化计算：(量化值 - 零点) * 缩放因子
        group_weight = (group_qweight.float() - group_zeros.float()) * group_scales.float()
        
        # 将结果写回到正确的位置 [outfeatures, group_size]
        weight[:, input_indices] = group_weight.t().half()
    
    return weight


def process_model(input_dir: str, output_dir: str, force: bool = False):
    """处理模型的主函数"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 输入验证
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    safetensors_path = input_path / "model.safetensors"
    if not safetensors_path.exists():
        raise ValueError(f"未找到模型权重文件: {safetensors_path}")
    
    if output_path.exists():
        if not force:
            raise ValueError(f"输出目录已存在: {output_dir}，使用 --force 强制覆盖")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"开始处理模型: {input_dir} -> {output_dir}")
    
    # 加载配置和tokenizer
    try:
        config = AutoConfig.from_pretrained(input_dir)
        print(f"已加载模型配置: {config.model_type}")
    except Exception as e:
        print(f"警告: 无法加载模型配置: {e}")
        config = None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(input_dir)
        print("已加载tokenizer")
    except Exception as e:
        print(f"警告: 无法加载tokenizer: {e}")
        tokenizer = None
    
    # 处理模型权重
    new_state_dict = {}
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        
        # 查找所有量化层
        quantized_layers = {key[:-8] for key in all_keys if key.endswith('.qweight')}
        print(f"发现 {len(quantized_layers)} 个量化层")
        
        # 处理每个参数
        processed_layers = set()
        
        for key in all_keys:
            tensor = f.get_tensor(key)
            
            # 检查是否为量化层的参数
            layer_name = None
            for ql in quantized_layers:
                if key.startswith(ql + '.'):
                    layer_name = ql
                    break
            
            if layer_name and key.endswith('.qweight') and layer_name not in processed_layers:
                print(f"反量化层: {layer_name}")
                
                # 获取量化参数
                qweight = tensor
                scales = f.get_tensor(f"{layer_name}.scales")
                qzeros = f.get_tensor(f"{layer_name}.qzeros")
                g_idx = f.get_tensor(f"{layer_name}.g_idx")
                
                # 推断量化参数
                infeatures = g_idx.shape[0]
                outfeatures = scales.shape[1]
                packed_rows = qweight.shape[0]
                
                # 推断量化位数
                if packed_rows * 32 == infeatures * 2:
                    bits = 2
                elif packed_rows * 32 == infeatures * 4:
                    bits = 4
                elif packed_rows * 32 == infeatures * 8:
                    bits = 8
                else:
                    raise ValueError(f"无法推断量化位数，层: {layer_name}")
                
                print(f"  - 检测到 {bits} 位量化")
                print(f"  - 输入特征: {infeatures}, 输出特征: {outfeatures}")
                
                try:
                    # 反量化权重
                    dequant_weight = dequantize_weight(
                        qweight, scales, qzeros, g_idx, bits, infeatures, outfeatures
                    )
                    new_state_dict[f"{layer_name}.weight"] = dequant_weight
                    print(f"  - 成功反量化到形状: {dequant_weight.shape}")
                    
                    # 处理偏置项
                    bias_key = f"{layer_name}.bias"
                    if bias_key in all_keys:
                        bias = f.get_tensor(bias_key)
                        new_state_dict[bias_key] = bias.half()
                    
                    processed_layers.add(layer_name)
                    
                except Exception as e:
                    print(f"  - 错误: 反量化失败: {e}")
                    raise
            
            # 处理非量化参数
            elif not any(key.startswith(ql + '.') for ql in quantized_layers):
                # 转换为FP16
                if tensor.dtype == torch.float32:
                    new_state_dict[key] = tensor.half()
                else:
                    new_state_dict[key] = tensor
                print(f"复制参数: {key} (形状: {tensor.shape}, 类型: {tensor.dtype})")
    
    print(f"总共处理 {len(new_state_dict)} 个参数张量")
    
    # 保存权重
    output_safetensors = output_path / "model.safetensors"
    print(f"保存权重到: {output_safetensors}")
    save_file(new_state_dict, output_safetensors)
    
    # 复制配置文件
    config_files = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "vocab.txt", "merges.txt", "special_tokens_map.json"
    ]
    
    for config_file in config_files:
        src_file = input_path / config_file
        if src_file.exists():
            dst_file = output_path / config_file
            shutil.copy2(src_file, dst_file)
            print(f"复制配置文件: {config_file}")
    
    # 保存配置和tokenizer
    if config is not None:
        config.save_pretrained(output_path)
        print("已保存模型配置")
    
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)
        print("已保存tokenizer")
    
    # 显示文件大小信息
    input_size = safetensors_path.stat().st_size / (1024**3)
    output_size = output_safetensors.stat().st_size / (1024**3)
    print(f"文件大小: {input_size:.2f}GB -> {output_size:.2f}GB ({output_size/input_size:.2f}x)")
    print("模型转换完成！")


def main():
    parser = argparse.ArgumentParser(
        description="将GPTQ量化的模型权重反量化为FP16格式（修复版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s ./gptq-model ./fp16-model
  %(prog)s ./gptq-model ./fp16-model --force
        """
    )
    parser.add_argument("input_dir", help="输入的GPTQ量化模型目录（包含model.safetensors）")
    parser.add_argument("output_dir", help="输出的FP16模型目录")
    parser.add_argument("--force", action="store_true", help="强制覆盖输出目录（如果存在）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    try:
        process_model(args.input_dir, args.output_dir, args.force)
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())