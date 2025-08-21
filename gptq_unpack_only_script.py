#!/usr/bin/env python3
"""
GPTQ到llama.cpp模型转换工具
将GPTQ量化的模型权重从int32格式反打包为8位整型，并清理量化参数
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


class GPTQToLlamaCppConverter:
    """GPTQ到llama.cpp的模型转换器"""
    
    def __init__(self, input_path: str, output_path: str, bits: int = 4, transpose_weights: bool = True):
        """
        初始化转换器
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径  
            bits: 量化位数 (2, 4, 8)
            transpose_weights: 是否转置权重矩阵 (默认True，适配llama.cpp格式)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.transpose_weights = transpose_weights
        
        if bits not in [2, 4, 8]:
            raise ValueError("只支持2, 4, 8位量化")
            
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def unpack_weights(self, qweight: torch.Tensor, original_shape: tuple, transpose: bool = False) -> torch.Tensor:
        """
        将int32打包的权重反打包为原始形状的8位整型
        
        Args:
            qweight: 打包的int32权重 
            original_shape: 原始权重形状 (infeatures, outfeatures)
            transpose: 是否对权重进行转置 (默认True，适配llama.cpp格式)
            
        Returns:
            反打包的8位权重张量
        """
        infeatures, outfeatures = original_shape
        
        # 转换为numpy进行位操作
        qweight_np = qweight.numpy().astype(np.uint32)
        
        # 创建输出数组
        unpacked = np.zeros((infeatures, outfeatures), dtype=np.uint8)
        
        elements_per_int32 = 32 // self.bits
        mask = (1 << self.bits) - 1
        
        for col in range(outfeatures):
            for packed_row in range(qweight_np.shape[0]):
                # 计算实际的行索引范围
                start_row = packed_row * elements_per_int32
                end_row = min(start_row + elements_per_int32, infeatures)
                
                packed_val = qweight_np[packed_row, col]
                
                # 解包每个元素
                for i in range(end_row - start_row):
                    actual_row = start_row + i
                    shift = i * self.bits
                    unpacked_val = (packed_val >> shift) & mask
                    unpacked[actual_row, col] = unpacked_val
        
        unpacked_tensor = torch.from_numpy(unpacked)
        
        # 根据需要进行转置
        if transpose:
            unpacked_tensor = unpacked_tensor.t().contiguous()
            print(f"权重已转置: {original_shape} -> {unpacked_tensor.shape}")
            
        return unpacked_tensor
    
    def get_original_shape(self, qweight_shape: tuple, outfeatures: int) -> tuple:
        """
        根据打包后的形状推算原始权重形状
        
        Args:
            qweight_shape: 打包后权重形状
            outfeatures: 输出特征数
            
        Returns:
            原始权重形状 (infeatures, outfeatures)
        """
        packed_infeatures = qweight_shape[0]
        infeatures = (packed_infeatures * 32) // self.bits
        return (infeatures, outfeatures)
    
    def convert_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        转换模型权重
        
        Returns:
            转换后的权重字典
        """
        model_file = self.input_path / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        converted_weights = {}
        
        print("正在加载和转换模型权重...")
        
        with safe_open(model_file, framework="pt", device="cpu") as f:
            # 获取所有张量名称
            tensor_names = f.keys()
            
            for name in tensor_names:
                tensor = f.get_tensor(name)
                
                # 处理量化的权重
                if name.endswith('.qweight'):
                    base_name = name[:-8]  # 移除 '.qweight' 后缀
                    
                    print(f"转换权重: {name}")
                    
                    # 查找对应的scales来获取outfeatures信息
                    scales_name = f"{base_name}.scales"
                    if scales_name in tensor_names:
                        scales = f.get_tensor(scales_name)
                        outfeatures = scales.shape[1]
                        
                        # 推算原始形状并反打包
                        original_shape = self.get_original_shape(tensor.shape, outfeatures)
                        unpacked_weight = self.unpack_weights(tensor, original_shape, self.transpose_weights)
                        
                        # 保存为原始权重名称
                        weight_name = f"{base_name}.weight"
                        converted_weights[weight_name] = unpacked_weight
                        
                # 跳过量化参数
                elif any(name.endswith(suffix) for suffix in ['.qzeros', '.scales', '.g_idx']):
                    print(f"跳过量化参数: {name}")
                    continue
                    
                # 保留其他参数（如bias, embeddings等）
                else:
                    print(f"保留参数: {name}")
                    converted_weights[name] = tensor.clone()
                    
        print(f"成功转换 {len(converted_weights)} 个张量")
        return converted_weights
    
    def copy_config_files(self):
        """复制配置文件到输出目录"""
        config_files = [
            "config.json",
            "generation_config.json", 
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "tokenizer.model"
        ]
        
        print("复制配置文件...")
        for filename in config_files:
            src_file = self.input_path / filename
            dst_file = self.output_path / filename
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"已复制: {filename}")
            else:
                print(f"配置文件不存在，跳过: {filename}")
    
    def update_config(self):
        """更新模型配置，移除GPTQ相关配置"""
        config_file = self.output_path / "config.json"
        
        if config_file.exists():
            print("更新模型配置...")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 移除GPTQ相关配置
            gptq_keys = [
                "quantization_config", 
                "gptq_bits",
                "gptq_groupsize", 
                "gptq_act_order",
                "gptq_desc_act"
            ]
            
            for key in gptq_keys:
                if key in config:
                    del config[key]
                    print(f"移除配置项: {key}")
            
            # 标记为转换后的模型
            config["_converted_from_gptq"] = True
            config["_conversion_bits"] = self.bits
            config["_weights_transposed"] = self.transpose_weights
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            print("配置文件已更新")
    
    def convert(self):
        """执行完整的转换流程"""
        print(f"开始转换: {self.input_path} -> {self.output_path}")
        print(f"量化位数: {self.bits}")
        print(f"权重转置: {'是' if self.transpose_weights else '否'}")
        
        # 转换权重
        converted_weights = self.convert_model_weights()
        
        # 保存转换后的权重
        output_model_file = self.output_path / "model.safetensors"
        print(f"保存转换后的模型到: {output_model_file}")
        save_file(converted_weights, output_model_file)
        
        # 复制配置文件
        self.copy_config_files()
        
        # 更新配置
        self.update_config()
        
        print("转换完成！")
        print(f"输出模型路径: {self.output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将GPTQ量化模型转换为llama.cpp兼容格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s ./gptq_model ./llamacpp_model --bits 4
  %(prog)s /path/to/gptq_model /path/to/output --bits 8 --verbose
  %(prog)s ./gptq_model ./llamacpp_model --no-transpose  # 不转置权重
        """
    )
    
    parser.add_argument(
        "input_path",
        help="输入GPTQ模型路径"
    )
    
    parser.add_argument(
        "output_path", 
        help="输出llama.cpp兼容模型路径"
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="量化位数 (默认: 4)"
    )
    
    parser.add_argument(
        "--transpose", "-t",
        action="store_true",
        default=True,
        help="转置权重矩阵以适配llama.cpp格式 (默认: True)"
    )
    
    parser.add_argument(
        "--no-transpose",
        action="store_false", 
        dest="transpose",
        help="不转置权重矩阵"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not Path(args.input_path).exists():
        print(f"错误: 输入路径不存在: {args.input_path}")
        return 1
    
    try:
        # 创建转换器并执行转换
        converter = GPTQToLlamaCppConverter(
            input_path=args.input_path,
            output_path=args.output_path,
            bits=args.bits,
            transpose_weights=args.transpose
        )
        
        converter.convert()
        
        print("\n✅ 转换成功完成!")
        print(f"转换后的模型已保存到: {args.output_path}")
        print("现在可以使用llama.cpp进行推理了。")
        
        return 0
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())