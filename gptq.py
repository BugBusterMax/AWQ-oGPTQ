import math
import time
import copy
import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())




    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, outlier_aware_order=True, lookahead_steps=4
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        scale = []
        zero = []
        now_idx = 1
        

        if static_groups:
            
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)


        if actorder:
            hessian_diag = torch.diag(H)
            if outlier_aware_order:

                weight_importance = hessian_diag
                weight_magnitude = torch.norm(W, dim=0)

                weight_std = torch.std(weight_magnitude)
                weight_mean = torch.mean(weight_magnitude)
                outlier_cols = weight_magnitude > (weight_mean + 4 * weight_std)

                sort_score = weight_importance.clone()
                sort_score[outlier_cols] *= 0.8
                perm = torch.argsort(sort_score, descending=True)
                
            else:
                perm = torch.argsort(hessian_diag, descending=True)
            # perm = torch.argsort(hessian_diag, descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H


        def estimate_lookahead_errors(W_block, Hinv_block, count):

            lookahead_errors = []
            for look_i in range(min(lookahead_steps, count)):

                w_look = W_block[:, look_i]
                d_look = Hinv_block[look_i, look_i]
                temp_quantizer = copy.deepcopy(self.quantizer)

                q_look = quantize(
                    w_look.unsqueeze(1), temp_quantizer.scale, 
                    temp_quantizer.zero, temp_quantizer.maxq
                ).flatten()
                err_look = (w_look - q_look) / d_look
                lookahead_errors.append(err_look)
            return lookahead_errors



        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]



            for i in range(count):

                w = W1[:, i]
                d = Hinv1[i, i]
                global_col_idx = i1 + i
                if groupsize != -1:
                    current_groupsize = groupsize
                    if not static_groups:
                        if global_col_idx % current_groupsize == 0:
                            end_idx = min(global_col_idx + current_groupsize, self.columns)
                            self.quantizer.find_params(W[:, global_col_idx:end_idx], weight=True)

                        if (global_col_idx // current_groupsize) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = global_col_idx
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                endi = min(i + lookahead_steps + 1, count)
                W2 = W1[:,i:endi]
                Hinv2 = Hinv1[i:endi, i:endi]
                if lookahead_steps > 0:
                    lookahead_errors = estimate_lookahead_errors(
                        W2, Hinv2, count - i - 1
                    )
                else:
                    lookahead_errors = []
                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d

                if lookahead_errors:
                    adj = []
                    for n in range(len(lookahead_errors)):

                        lookahead_err = lookahead_errors[n]
                        err_correlation = torch.dot(err1.flatten(), lookahead_err.flatten()) / (torch.norm(err1) * torch.norm(lookahead_err) + 1e-8)
                        correlation_factor = err_correlation.clamp(-0.5, 0.5)
                        broadcast_adjustment = 0.25 * correlation_factor
                        distance_decay = 0.9 ** n
                        adj.append(broadcast_adjustment * distance_decay)
                    final_adjustment = 1.0 + torch.mean(torch.tensor(adj))

                    err1_adjusted = err1 * final_adjustment
                else:
                    err1_adjusted = err1
                W1[:, i:] -= err1_adjusted.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1_adjusted
                # W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                # Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())
        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, error

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()