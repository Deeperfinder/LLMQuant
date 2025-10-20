
import inspect
import time
import torch
import functools
from torch import nn
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict

from .scale import apply_scale, apply_clip
from quant.nn_models.modules.linear import get_concrete_linear_module
from quant.utils.awq_calib_utils import get_calib_dataset
from quant.quantization.base.quantizer import BaseQuantizer
from quant.utils.common_utils import (
    append_str_prefix,
    get_op_name,
    set_op_name,
    get_best_device,
    get_named_linears,
    exculde_layers_to_not_quantize,
    clear_memory
)

class AWQQuantizer(BaseQuantizer):
    def __init__(
        self,
        modelforCausalLM, # 各模型的模型类比如Qwen2ModelForCausalLM，它的from_pretained返回的model
        model,   # AutoModelForCausalLM.from_pretrained 返回的 model, huggingface返回的model对象
        model_type,
        tokenizer,
        quant_config,
        quant_method,
        w_bit,
        group_size,
        zero_point,
        calib_data, # str
        duo_scaling, # true of false
        modules_to_not_convert=None,
        fake_quant=False, # true时为fake quant，false为real quant
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024, # used in seearch best scale and compute loss, 避免在校准 token 数量很大时一次性分配两个巨大的 FP32/FP16 张量导致 OOM。通过 max_chunk_memory（默认通常设几百 MB）动态决定 chunk size，既能跑在 24 GB 显卡上，也能跑在 80 GB 显卡上，实现“一次代码，到处可跑”
    ) -> None:
        super(BaseQuantizer, self).__init__()
        self.awq_model = modelforCausalLM 
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.quant_method = quant_method
        self.w_bit = 4
        self.group_size = group_size
        self.zero_point = zero_point
        self.calib_data = calib_data
        self.duo_scaling = duo_scaling
        self.fake_quant = fake_quant
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        if self.model_type == "qwen3_moe":
            self.max_calib_samples = max_calib_samples + 128 # increase calib nums for qwen3 moe in case line 763 assert error
        else:
            self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        # 返回值这里不能命名为self.modules，因为BaseQuantizer是一个torch.nn.Module，它也有self.modules成员
        # self.inps表示捕获到的第一个layer的输入，用于layer1-layern的calib
        # self.target_modules表示模型的所有layers
        self.target_modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def quantize(self):
        # 遍历每个decoderlayer
        for i in tqdm(range(len(self.target_modules)), desc="AWQ"):
            start = time.perf_counter()
            # 决策出当前decoder layer的best device， 然后把layer weights传到best device
            common_device = next(self.target_modules[i].parameters()).device # next为获取该module的第一个参数
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda" + str( i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()
                
                self.target_modules[i] = self.target_modules[i].to(best_device)
                common_device = next(self.target_modules[i].parameters()).device
            
            # 把第0个decoder layer的输入传到GPU、best device
            self.inps = self.inps.to(common_device)
            # 把emb table 传到GPU 、 best device
            self.awq_model.move_embed(self.model, common_device)
            
            # 返回第i个decoder layer上所有linear的name和torch.nn.Linear的映射字典
            # {'self_attn.q_proj': Linear(in_features=5120, out_features=5120, bias=True), 
            # 'self_attn.k_proj': Linear(in_features=5120, out_features=1024, bias=True), 
            # 'self_attn.v_proj': Linear(in_features=5120, out_features=1024, bias=True), 
            # 'self_attn.o_proj': Linear(in_features=5120, out_features=5120, bias=False), 
            # 'mlp.gate_proj': Linear(in_features=5120, out_features=13824, bias=False), 
            # 'mlp.up_proj': Linear(in_features=5120, out_features=13824, bias=False), 
            # 'mlp.down_proj': Linear(in_features=13824, out_features=5120, bias=False)}
            named_linear = get_named_linears(self.target_modules[i])

            named_linear = exculde_layers_to_not_quantize(
                named_linear, self.modules_to_not_convert
            )
            # calib, 返回每个decoderlayer中每个linear的input activations,送去apply scale后再送去决定clip
            # 即采集输入 {linear name: input act}
            input_feat = self._get_input_feat(self.target_modules[i], named_linear)

            end0 = time.perf_counter()
            clear_memory()
            print("[info] the get_input_feat time per layer is ", end0 - start)

            # attn qkvo 为1个， gate和up为1个，down为1个
            module_config: List[dict] = self.awq_model.get_layers_for_scaling(
                self.target_modules[i], input_feat, self.module_kwargs
            )

            # 搜寻每个linear的best scale， 依赖于input_feat， 即input activation
            # scales[0]
            # ('input_layernorm' prev op name, ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj') layer name,
            #   best scales: tensor([1.3828, 1.2344, 1.2500,  ..., 1.2812, 1.1250, 1.1406], dtype=torch.bfloat16))
            scales_list = [
                self._search_best_scale(self.target_modules[i], **layer) 
                for layer in module_config
            ]
            # * 把上面搜索出来的best scale乘到当前layer的weight, activation对应的scale融合到prev op，因为输入activation一直都在变,
            # * 如果等到输入来的时候再去乘scale，会很低效
            # （W * S） 是要做量化的， 量化完之后和activation * S^-1 相乘
            # W = (W*S), X = (S^-1 * X)
            apply_scale(self.target_modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.target_modules[i]) + "."
            )
            end1 = time.perf_counter()
            print("[info] the apply_scale time per layer is ", end1 - end0)
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.target_modules[i], named_linear, input_feat, common_device
                )
                # 将上面搜索出来的best clip 乘到当前layer的weight
                apply_clip(self.target_modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.target_modules[i]) + "."
                )
            end2 = time.perf_counter()
            print("[info] the apply_clip time per layer is ", end2 - end1)
            # 开始执行量化
            # fp16 weights -> int4 weights
            # self.target_modules[i] 目前是一个fp16 decoder layer， apply_quant 后变为int4 decoder layer
            if not self.fake_quant:
                self._apply_quant(self.target_modules[i], named_linear, common_device)
            clear_memory()
            end3 = time.perf_counter()
            print("[info] the apply_quant time per layer is ", end3 - end2)

    def init_quant(self, n_samples=128, max_seq_len=512):
        # n x decoder layers
        modules = self.awq_model.get_model_layers(self.model) # from pretrained 返回的mode(Qwen2ForCausalLM)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split="validation"
        )
        samples = torch.cat(samples, dim=0)

        inps = [] 
        layer_kwargs = {}
        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device) # 这里要做一次推理，获取该layer的输入
        # 如何捕获到layer0的input
            # 方法：hack一个hook
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.ori_module = module

                # 继承原始模块的所有属性，以防calib的时候报AttributeError: 'Catcher' object has no attribute 'attention_type'
                for name, value in module.__dict__.items():
                    setattr(self, name, value)
            
            def forward(self, *args, **kwargs):
                # assume first input
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError # 提前退出，避免继续进行推理
            
        original_module = modules[0]
        modules[0] = Catcher(original_module)
        try:
            # 将sample上的数据，放到和model第一个参数相同的device上
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:
            pass
        modules[0] = original_module
        # modules[0] = modules[0].ori_module  # restore
        # prepare_inputs_for_generation的解释：输入samples，根据当前状态动态调整模型的输入
        # ep.prefill阶段，inputs如下：
        # inputs = {
        #     "input_ids": initial_input,  # 初始输入 [batch_size, seq_len]
        #     "attention_mask": mask,      # 初始掩码
        #     "use_cache": True
        # }
        # model_inputs = model.prepare_inputs_for_generation(**inputs)
        # decode阶段，inputs如下：有了kv cache，且input_ids的shape不一样了
        # inputs = {
        #     "input_ids": new_token,      # 新生成的 token [batch_size, 1]
        #     "past_key_values": past,     # 上一轮的 KV Cache
        #     "attention_mask": updated_mask,  # 扩展掩码
        # }
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # layer_kwargs: dict_keys(['cache_position', 'input_ids', 'inputs_embeds', 'position_ids', 'past_key_value', 'output_attentions', 'use_cache', 'position_embeddings'])
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]
        # 节省显存
        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()
        return modules, layer_kwargs, inps
    
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape # [5120, 5120]
        if self.group_size > 0: # for deepseek and group size > 0
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size) # [5120x40, 128]
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        # 1. 非对称量化的scale和zp的计算公式
        # scale = (absmax - absmin) / 255, zp = clip(-round(absmin/scale), 0, 255)
        # qx = clip(round(x / scale -zp))
        # 2. 对称量化的scale计算公式
        # scale = absmax / (2^N -1) -1
        # qx = clip(round(x / scale), -128, 127)
        
        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True) # [5120x40, 1] 即列上每128个元素的最大值
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:   
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit -1) -1
            min_int = - (2 ** self.w_bit -1)
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w/scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0
        scales = scales.view(org_w_shape[0], -1) # [5120x40, 1] ==> [5120, 40]
        w = w.reshape(org_w_shape)

        return w, scales, zeros #[5120, 40]

    def _apply_quant(self, module, named_linears : Dict[str, nn.Linear], common_device):
        for name, linear_layer in named_linears.items():
            print("[info] apply quant linear name: ", name)
            linear_layer = linear_layer.to(common_device).half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )
            # 1.拿到linear的scales和zeros
            scales = scales.t().contiguous()
            if zeros is not None:
                zeros = zeros.t().contiguous()# (out channel, in channel)=>(in channel, out channel)
            # 2.基于scales和zeros创建quantized linear
            q_linear_module = get_concrete_linear_module(self.quant_method) 
            # quantized linear的初始化
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros, # [in channels/128,out channels ]
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            # 3.用quantized linear替换老的torch.nn.Linear, 
            # 即用q_linear 的属性来替换原有的torch.nn.Linear中的属性
            set_op_name(module, name, q_linear)
            clear_memory()


    def _get_input_feat(self, layer, named_linears):
        # 在每个linear module上注册这个钩子，等到forward的时候，运行到该linear，将自动触发
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x) # {linear name: input act}
        
        input_feat = defaultdict(list)
        handles = []
        
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # 返回的self.inps为当前入参layer的input
        self.inps = self.inps.to(next(layer.parameters()).device) # 同步GPU的位置，防止multi-gpu
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)
        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        
        for h in handles:
            h.remove()

        def cat_and_assert(k,v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"{k} has a zero dimension. This can happen if no data was passed through (e.g. an expert in MoE not being activated). "
                "Try increasing max_calib_samples (warning: this can significantly increase quantization time and memory usage.)"
            )
            return x
        
        input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}
        return input_feat
    
    @torch.no_grad()
    def _module_forward(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        module_kwargs: Dict
    ) -> torch.Tensor:
        
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]
                
                module_output.append(partial_output.cpu())
            module_output = torch.cat(module_output, dim=0)
        
        return module_output
            
    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
            过滤掉目标模块（module）的 forward 方法不支持的参数，确保传入的关键字参数（inputs_kwargs）
            不会因为transformers版本差异或参数不匹配导致模块的前向传播（forward）失败

            Args:
                inputs_kwargs (`dict`):
                    The input dictionary to pass to the model layer
                module (`torch.nn.Module`):
                    Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    # prev_op=module.input_layernorm,
    # layers=[
    #     module.self_attn.q_proj,
    #     module.self_attn.k_proj,
    #     module.self_attn.v_proj,
    # ],
    # inp=input_feat["self_attn.q_proj"],
    # module2inspect=module.self_attn,
    # kwargs=module_kwargs,
    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)
        # [STEP 1] 计算[out channels, in channels]下，每列的weight 按照group size大小先归一化， 然后再求得均值
        weight = torch.cat([_m.weight for _m in layers], dim=0)  # concat([out1,in], [out2,in] ==> [out1+out2, in])
        org_shape = weight.shape
        weight = weight.view(-1, self.group_size) # [(out1 + out2) * in / 128, 128]
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6) # 每一行对应一个group 128， 求得每一行的最大值, 即128个group的max值
        w_scale = w_scale.view(org_shape) # [out1+out2, in]
        w_mean = w_scale.mean(0) #[in] tensor.mean(dim=0) 会 压缩（squeeze）掉 dim=0，结果的 shape 是原 shape 去掉第 0 维。对每个in channel的 output channel个weight求平均值
        clear_memory(weight) 

        # [STEP 2] 计算[bs*tokens, in channels]下，每列的activation均值
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0) # 行=bs*numtokens
        num_channels = inp_flat.size(1) # 列=hidden_size
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32, 每个元素的大小

        # Calculate chunk size dynamically based on max_chunk_memory 分chunk_size大小来进行求和，节省显存
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels)) 
        chunk_size = min(chunk_size, num_elements)

        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)  
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0) # 对0维度求和并且累加，然后就只剩下1维度的res
            x_sum += chunk_sum.to(inp.device)

        # [in_channel], [token, in_channel]下每个channel的mean, 这里的sum是每个channel中所有token的sum，然后再/num_elements
        # 即求得了每一列的均值
        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3] Compute output of module(即W*X), 对module2inspect(对于qkv是attn，对于gate up down是mlp)做fwd，求它的out
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            # enbale deepseek v3 
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4] Compute loss，基于x和w的均值，linear对应的module2inspect的fwd结果
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers ]),
            best_scales,
        )
    
    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat, common_device):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            if any([_ in name for _ in avoid_clipping]):
                continue
            named_linears[name].to(common_device)
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()
        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # 每隔stepsize选择input feat的一个数，直到取够n sample token
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size] # [1, n_sample_token, ngroup, groupsize]
        
        oc_batch_size = 256 if org_w_shape[0] % 256 ==0 else 64
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size] # 256
            # 保存clip 前后每个group的最大值
            org_max_val = w.abs().amax(dim=-1, keepdim=True)
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val)
            input_feat = input_feat.to(w.device)
            # groudtruth, WX
            org_out = (input_feat * w).sum(dim=-1)

            # gridsearch clip位置，得到数据的最大最小阈值，然后截断，求得Q(W) X
            for i_s in range([int(max_shrink * n_grid)]):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # out channel, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                 # group id
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                # 某个group的best max val
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        
        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1) # [out channel, n_group, 1]
    
    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module, # nn.attn module / nn.mlp module
        linears2scale: List[nn.Linear], # qkv linear, gateupdown linear
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")
        org_weights = {k: v.cpu() for k,v in module2inspect.state_dict().items()}

        device = x.device
        # [in] => [in]
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)
        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid
            # AWQ 原文确实只拿 activation 的 magnitude 作为「重要性」的 proxy
            # 但 open-source 的 auto-awq 后来为了稳定收敛给了一个可选的 duo_scaling 分支，把 weight 的均值也拉进来做一个几何加权
            # 好处有：
            # 1.防止极端通道的activation 很小，按论文方法几乎把 scale 压到 0，导致量化后权重直接“消失”
            # 2.搜索空间平滑：用 (x^ratio) / (w^(1-ratio)) 的形式，把原来只有一条 activation 曲线变成 ratio∈[0,1] 的连续曲线，网格搜索更稳定，不容易出现 loss 突然爆炸
            # 3.在 Llama 405B / DeepSeek 671B 这类大模型上，作者发现 activation 极度稀疏，只靠 activation 会低估某些“权重很大但很少被激活”的通道的重要性，于是给 weight 也留了一个“话语权”。
            if self.duo_scaling:
                scales = (x_mean.pow(ratio)) / (w_mean.pow(1-ratio) + 1e-4).clamp(min=1e-4)
            else:
                scales = (x_mean.pow(ratio)).clamp(min=1e-4).view(-1)
            # 但是为什么最终scale是下式？作用是
            # 1.平衡scale的范围，x_mean.pow(ratio) 得到的值可能会有较大的动态范围，即最大值最小值之间的差距可能很大，通过除以 (scales.max() * scales.min()).sqrt()，可以将缩放因子的范围调整到一个更合理的区间，避免某些通道的缩放因子过大或过小，从而导致量化后的权重分布不均匀
            # 2.保持数值稳定性：如果scale的范围过大，可能会导致数值不稳定，例如在后续的乘法和除法操作中出现溢出或下溢      
            scales = scales / (scales.max()*scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)
            
            # 对于极端数据处理：不做AWQ量化
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s) * s^-1: 对应论文提到的fuse to previous op, 和smoothquant的处理一样
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # Q(W * s) * s^ -1 * X, 对module2inspect（attn mlp）里面的linear 做forward
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)
            # fp16_output = W * X
            # compute mean squared error(L2 norm) between fp16_output and int_w_output
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            # 还原weight，以便做下一个iter的grid search
            module2inspect.load_state_dict(org_weights)
        if best_ratio == -1:
            raise Exception(f"Failed to find best scale for {module2inspect}")
        
        assert torch.isnan(best_scales).sum == 0, best_scales
        return best_scales.detach().cpu()
    
    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # 根据max_chunk_memory动态计算chunk_size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # compute loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss
        
        # Normalize the loss by the total number of elements
        loss /= num_elements
       
        return loss
