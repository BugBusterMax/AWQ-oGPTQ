from safetensors.torch import safe_open

# model_path = "/home/mzr/llama3.2_1b/model.safetensors"
model_path = "/home/mzr/convert/adjust/model.safetensors"
# model_path = "/home/mzr/quant_result/combine_quant_result_4bit/model.safetensors"

with safe_open(model_path, framework="pt", device="cpu") as f:
    print("✅ 所有参数名与 shape：\n")
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key:60} shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")