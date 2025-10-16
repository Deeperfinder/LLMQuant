from quant.core.base import BaseModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import(
    Qwen2DecoderLayer as OldQwen2DecoderLayer,
    Qwen2ForCausalLM as OldQwen2ForCausalLM,
)
class Qwen2ModelForCausalLM(BaseModelForCausalLM):
    layer_type = "Qwen2DecoderLayer"

    @staticmethod
    def get_model_layers(model: OldQwen2ForCausalLM):
        # OldQwen2ForCausalLM.Qwen2Model.layers
        return model.model.layers
    
    # embedding词表move到对应的device
    @staticmethod 
    def move_embed(mode: OldQwen2ForCausalLM, device: str):
        pass
    
    # 将layer的各个成员例举出来
    @staticmethod
    def get_layers_for_scaling(module: OldQwen2DecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op = module.input_layernorm,
                layers = [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attention 
        # 这里如果是GQA / MQA类型（k/v linear weights shape < q linear weights shape)
        # ，v_proj的out feats不全是head， 但是o_proj的out feats是全head
        # 所以这种不兼容使得o_proj找出来的scale对于x来说很难fuse到v_proj里面去
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # linear 1
        layers.append(
            dict(
                prev_op = module.post_attention_layernorm,
                layers = [module.mlp.gate_proj, module.mlp.up_proj],
                inp = input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear2 
        layers.append(
            dict(
                prev_op = module.mlp.up_proj,
                layers = [module.mlp.down_proj],
                inp = input_feat["mlp.down_proj"]
            )
        )
        return layers
