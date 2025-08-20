import argparse
import torch
from transformers import AutoTokenizer, GenerationConfig

# AutoGPTQ 的类（来自 auto-gptq 包）
try:
    from auto_gptq import AutoGPTQForCausalLM
except Exception as e:
    raise ImportError("需要安装 auto-gptq：pip install auto-gptq") from e

def load_quantized_model(model_dir: str,
                         model_basename: str = None,
                         device: str = "cuda:0",
                         use_triton: bool = False,
                         use_safetensors: bool = True,
                         strict: bool = False):

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # from_quantized 参数随版本会略有不同；这里使用常见签名
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        model_basename=model_basename,   # 可为 None
        use_safetensors=use_safetensors,
        device=device,
        use_triton=use_triton,
        strict=strict
    )
    return model, tokenizer

def generate(model, tokenizer, prompt: str, max_new_tokens=256, temperature=0.8, top_p=0.95, do_sample=True):

    # tokenization
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # generation config（你也可以改用 transformers.GenerationConfig）
    gen_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    # 有些 auto-gptq 的模型实现封装了 generate 方法兼容 transformers
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens
        )

    # 解码（去掉 prompt 部分）
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # 可选：只返回生成补充（去掉 prompt 前缀）
    if output.startswith(prompt):
        return output[len(prompt):].strip()
    else:
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="量化模型所在目录（含 config/tokenizer）")
    parser.add_argument("--model_basename", type=str, default=None, help="量化权重 basename（可选）")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备：cuda:0 或 cpu")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="输入提示语")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print("加载模型中...（这可能需要一些时间）")
    model, tokenizer = load_quantized_model(args.model_dir, model_basename=args.model_basename, device=args.device)
    print("模型加载完成，开始生成...")

    out = generate(model, tokenizer, prompt=args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print("=== 生成结果 ===")
    print(out)

if __name__ == "__main__":
    main()
