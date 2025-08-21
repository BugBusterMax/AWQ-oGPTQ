#!/usr/bin/env python3
"""
删除lm_head层并启用权重共享脚本
将模型的lm_head层删除，设置tie_word_embeddings=true，让embedding层权重共享给输出层
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Set

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class LMHeadRemover:
    """lm_head层删除和权重共享配置器"""
    
    def __init__(self, input_path: str, output_path: str):
        """
        初始化处理器
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def get_lmhead_layer_names(self, tensor_names: Set[str]) -> Set[str]:
        """
        识别所有lm_head相关的层名称
        
        Args:
            tensor_names: 所有张量名称集合
            
        Returns:
            lm_head相关层名称集合
        """
        lmhead_patterns = [
            'lm_head.weight',
            'lm_head.bias', 
            'output.weight',
            'output.bias',
            'head.weight',
            'head.bias'
        ]
        
        lmhead_layers = set()
        
        for name in tensor_names:
            for pattern in lmhead_patterns:
                if name.endswith(pattern) or pattern in name:
                    lmhead_layers.add(name)
                    break
                    
        return lmhead_layers
    
    def process_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        处理模型权重，删除lm_head层
        
        Returns:
            处理后的权重字典
        """
        model_file = self.input_path / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        processed_weights = {}
        
        print("正在加载和处理模型权重...")
        
        with safe_open(model_file, framework="pt", device="cpu") as f:
            # 获取所有张量名称
            tensor_names = set(f.keys())
            
            # 识别lm_head层
            lmhead_layers = self.get_lmhead_layer_names(tensor_names)
            
            if lmhead_layers:
                print(f"发现以下lm_head层将被删除:")
                for layer in sorted(lmhead_layers):
                    print(f"  - {layer}")
            else:
                print("警告: 未找到lm_head层，可能已经被删除或使用不同的命名")
            
            # 处理每个张量
            for name in tensor_names:
                if name in lmhead_layers:
                    print(f"跳过lm_head层: {name}")
                    continue
                else:
                    tensor = f.get_tensor(name)
                    processed_weights[name] = tensor.clone()
                    
        print(f"保留了 {len(processed_weights)} 个张量")
        print(f"删除了 {len(lmhead_layers)} 个lm_head相关张量")
        
        return processed_weights
    
    def copy_config_files(self):
        """复制配置文件到输出目录"""
        config_files = [
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
        """更新模型配置，启用权重共享"""
        src_config_file = self.input_path / "config.json"
        dst_config_file = self.output_path / "config.json"
        
        if not src_config_file.exists():
            print("警告: config.json不存在，跳过配置更新")
            return
            
        print("更新模型配置...")
        
        with open(src_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 备份原始设置
        original_tie_embeddings = config.get("tie_word_embeddings", False)
        
        # 设置权重共享
        config["tie_word_embeddings"] = True
        
        # 添加处理标记
        config["_lmhead_removed"] = True
        config["_original_tie_word_embeddings"] = original_tie_embeddings
        
        # 如果有vocab_size，确保它与embedding层匹配
        if "vocab_size" in config:
            print(f"词汇表大小: {config['vocab_size']}")
        
        with open(dst_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"配置已更新:")
        print(f"  tie_word_embeddings: {original_tie_embeddings} -> True")
        print(f"  已标记lm_head层已删除")
    
    def verify_embedding_layer(self, weights: Dict[str, torch.Tensor]):
        """
        验证embedding层是否存在
        
        Args:
            weights: 权重字典
        """
        embedding_patterns = [
            'embed_tokens.weight',
            'embeddings.weight', 
            'word_embeddings.weight',
            'wte.weight'
        ]
        
        found_embeddings = []
        for name in weights.keys():
            for pattern in embedding_patterns:
                if pattern in name:
                    found_embeddings.append(name)
                    break
        
        if found_embeddings:
            print(f"找到embedding层:")
            for emb in found_embeddings:
                shape = weights[emb].shape
                print(f"  {emb}: {shape}")
                print(f"    词汇表大小: {shape[0]}, 隐藏维度: {shape[1]}")
        else:
            print("警告: 未找到embedding层，请检查模型结构")
    
    def process(self):
        """执行完整的处理流程"""
        print(f"开始处理: {self.input_path} -> {self.output_path}")
        
        # 处理权重
        processed_weights = self.process_model_weights()
        
        # 验证embedding层
        self.verify_embedding_layer(processed_weights)
        
        # 保存处理后的权重
        output_model_file = self.output_path / "model.safetensors"
        print(f"保存处理后的模型到: {output_model_file}")
        save_file(processed_weights, output_model_file)
        
        # 复制配置文件
        self.copy_config_files()
        
        # 更新配置
        self.update_config()
        
        print("处理完成！")
        print(f"输出模型路径: {self.output_path}")
        print("现在模型将使用embedding层权重作为输出层权重")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="删除lm_head层并启用权重共享",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s ./model_with_lmhead ./model_tied_embeddings
  %(prog)s /path/to/input_model /path/to/output_model --verbose

说明:
  此脚本将删除模型的lm_head层并设置tie_word_embeddings=true，
  使模型使用embedding层的权重作为输出层权重，这可以减少模型大小。
        """
    )
    
    parser.add_argument(
        "input_path",
        help="输入模型路径"
    )
    
    parser.add_argument(
        "output_path", 
        help="输出模型路径"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要删除的层，不实际执行"
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not Path(args.input_path).exists():
        print(f"错误: 输入路径不存在: {args.input_path}")
        return 1
    
    # 检查输出路径是否与输入路径相同
    if Path(args.input_path).resolve() == Path(args.output_path).resolve():
        print("错误: 输出路径不能与输入路径相同，请指定不同的输出目录")
        return 1
    
    try:
        if args.dry_run:
            # 仅显示要删除的层
            model_file = Path(args.input_path) / "model.safetensors"
            if not model_file.exists():
                print(f"错误: 模型文件不存在: {model_file}")
                return 1
                
            with safe_open(model_file, framework="pt", device="cpu") as f:
                tensor_names = set(f.keys())
                
            remover = LMHeadRemover(args.input_path, args.output_path)
            lmhead_layers = remover.get_lmhead_layer_names(tensor_names)
            
            print("预览模式 - 将要删除的层:")
            if lmhead_layers:
                for layer in sorted(lmhead_layers):
                    print(f"  - {layer}")
                print(f"\n总共将删除 {len(lmhead_layers)} 个层")
            else:
                print("  无lm_head层需要删除")
                
            print(f"\n配置更改:")
            print(f"  tie_word_embeddings 将设置为 true")
            
        else:
            # 执行实际处理
            remover = LMHeadRemover(
                input_path=args.input_path,
                output_path=args.output_path
            )
            
            remover.process()
            
            print("\n✅ 处理成功完成!")
            print(f"处理后的模型已保存到: {args.output_path}")
            print("模型现在使用共享的embedding权重作为输出层")
        
        return 0
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())