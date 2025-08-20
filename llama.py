import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *
from quant_linear import *
from transformers import AutoTokenizer, AutoConfig

import warnings
import copy

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)


    # my change
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    # my change
     


    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()


    # my change
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # my change


    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                original_weight = subset[name].weight.data.clone()
                scale, zero, g_idx, error = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
                gptq[name].free()


        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)


    # my change
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    # my change


    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()


    # my change
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # my change

    
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    testenc = testenc.cpu()
    model.config.use_cache = use_cache
def find_embedding_layer(model):
    print("🔍 Search for the embedding layer in all modules...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and 'embed' in name.lower():
            print(f"✅ Find the embedding layer: {name}")
            return module, name
    return None, None
def add_separate_lm_head_advanced(model, config=None):

    if config is None:
        if hasattr(model, 'config'):
            config = model.config
        else:
            warnings.warn("Configuration information cannot be obtained. Processing will be attempted to continue.")
            config = None
    tie_embeddings = False
    if config is not None:
        tie_embeddings = getattr(config, 'tie_word_embeddings', False)
        print(f"📋 tie_word_embeddings: {tie_embeddings}")
    
    if not tie_embeddings:
        print("✅ The configuration shows that weight sharing is not enabled. No modification is required.")
        return model, False

    embed_layer, embed_name = find_embedding_layer(model)
    if embed_layer is None:
        raise ValueError("❌ The embedding layer cannot be found. Please check the model architecture.")

    has_lm_head = hasattr(model, 'lm_head')
    if has_lm_head:
        current_lm_head = model.lm_head
        if isinstance(current_lm_head, nn.Linear):

            weights_shared = torch.equal(current_lm_head.weight, embed_layer.weight)
            same_memory = current_lm_head.weight.data_ptr() == embed_layer.weight.data_ptr()
            
            if not weights_shared and not same_memory:
                print("✅ The lm_head is already independent and does not require any modification.")
                return model, False
            
            print(f"🔧 Current state of lm_head: Weight sharing={same_memory}, Equal values={weights_shared}")
        else:
            print(f"⚠️ lm_head is not a Linear layer, type: {type(current_lm_head)}")
    

    vocab_size, hidden_size = embed_layer.weight.shape
    device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype
    new_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    

    with torch.no_grad():
        if has_lm_head and hasattr(model.lm_head, 'weight'):
            new_lm_head.weight.data.copy_(model.lm_head.weight.data)
        else:
            new_lm_head.weight.data.copy_(embed_layer.weight.data)
    
    new_lm_head = new_lm_head.to(device=device, dtype=dtype)
    
    model.lm_head = new_lm_head
    
    if config is not None:
        config.tie_word_embeddings = False
        model.config = config
        print("📝 Updated configuration: tie_word_embeddings = False")
    
    weights_equal = torch.equal(model.lm_head.weight, embed_layer.weight)
    same_memory = model.lm_head.weight.data_ptr() == embed_layer.weight.data_ptr()
    
    
    success = not weights_shared and not same_memory
    return model, success

def llama_pack3(model, quantizers, wbits, groupsize, config):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    model, success = add_separate_lm_head_advanced(model, config)

    if success:
        print("✅ Successfully added the independent lm_head layer")
    print('Done.')
    return model

def llama_pack4(model, quantizers, wbits, groupsize, config):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant4(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [Quant4Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    model, success = add_separate_lm_head_advanced(model, config)

    if success:
        print("✅ Successfully added the independent lm_head layer")
    print('Done.')
    return model
def llama_pack(model, quantizers, wbits, groupsize, config):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    model, success = add_separate_lm_head_advanced(model, config)

    if success:
        print("✅ Successfully added the independent lm_head layer")
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    model = get_llama(args.model)
    config = AutoConfig.from_pretrained(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    # weights_before = copy.deepcopy(model.state_dict())
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)
    # weights_changed = False
    # for name, param in model.named_parameters():
    #     if not torch.equal(param.data, weights_before[name]):
    #         print(f"参数 {name} 已改变")
    #         weights_changed = True
    # datasets = ['wikitext2', 'ptb', 'c4'] 
    # if args.new_eval:
    #     datasets = ['wikitext2', 'ptb-new', 'c4-new']
    # for dataset in datasets:
    #     dataloader, testloader = get_loaders(
    #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    #     )
    #     print(dataset)
    #     model_copy = copy.deepcopy(model)
    #     llama_eval(model_copy, testloader, DEV)

    if args.save:
        # if args.wbits == 3:
        #     llama_pack3(model, quantizers, args.wbits, args.groupsize, config)
        # elif args.wbits == 4:
        #     llama_pack4(model, quantizers, args.wbits, args.groupsize, config)
        llama_pack(model, quantizers, args.wbits, args.groupsize, config)
        # 方法1: 保存完整的Transformers兼容模型
        print("Saving quantized model...")

        # 保存模型权重和配置
        model.save_pretrained(args.save)
        # 保存tokenizer（如果需要）
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)

        # 保存量化配置信息（可选，用于记录量化参数）
        import json
        import os
        quant_config = {
            "bits": args.wbits,
            "group_size": args.groupsize,
            "damp_percent": args.percdamp,
            "desc_act": args.act_order,
            "static_groups": args.static_groups,
            "sym": args.sym,
            "true_sequential": args.true_sequential,
            "model_name_or_path": args.save,
            "model_file_base_name": "model",
            "quant_method": "gptq",
            "checkpoint_format": "gptq"
        }
        with open(f"{args.save}/quantize_config.json", "w") as f:
            json.dump(quant_config, f, indent=2)

        print(f"Model saved to {args.save}")
        # torch.save(model.state_dict(), args.save)