import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from safetensors.torch import load_file
from collections import defaultdict

def load_tokenizer_only(model_size, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(f'/dlabdata1/llama2_hf/Llama-2-{model_size}-hf', use_auth_token=hf_token)
    return tokenizer

def load_unemb_only(model_size):
    if model_size == '70B' or model_size == '70b':
        lm_head_state_dict = load_file("/dlabdata1/llama2_hf/Llama-2-70b-hf/model-00015-of-00015.safetensors")
        norm_state_dict = load_file("/dlabdata1/llama2_hf/Llama-2-70b-hf/model-00014-of-00015.safetensors")
    if model_size == '7B' or model_size == '7b':
        lm_head_state_dict = load_file('/dlabdata1/llama2_hf/Llama-2-7b-hf/model-00002-of-00002.safetensors')
        norm_state_dict = lm_head_state_dict 
    if model_size == '13B' or model_size == '13b':
        lm_head_state_dict = load_file("/dlabdata1/llama2_hf/Llama-2-13b-hf/model-00003-of-00003.safetensors")
        norm_state_dict = lm_head_state_dict
        
    norm_params = norm_state_dict['model.norm.weight'].detach().clone()
    lm_head_params = lm_head_state_dict['lm_head.weight'].detach().clone()
    
    
    lm_head = nn.Linear(*lm_head_params.shape[::-1], bias=False)
    lm_head.weight.requires_grad_(False)
    lm_head.weight.copy_(lm_head_params)
    norm = transformers.models.llama.modeling_llama.LlamaRMSNorm(len(norm_params))
    norm.weight.requires_grad_(False)
    norm.weight.copy_(norm_params)
    unemb = nn.Sequential(norm, lm_head)
    if torch.cuda.device_count() >= 2:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap your model with DataParallel
        unemb = nn.DataParallel(unemb, device_ids=[0, 1])
        # Move your model to the first device
        unemb = unemb.cuda()
    else:
        unemb.cuda()
    return unemb

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False
    #https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L422
    def forward(self, *args, **kwargs):
        if self.act_as_identity:
            #print(kwargs)
            kwargs['attention_mask'] += kwargs['attention_mask'][0, 0, 0, 1]*torch.tril(torch.ones(kwargs['attention_mask'].shape,
                                                                                                   dtype=kwargs['attention_mask'].dtype,
                                                                                                   device=kwargs['attention_mask'].device),
                                                                                        diagonal=-1)
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False

class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.up_proj = mlp.up_proj
        self.gate_proj = mlp.gate_proj
        self.act_fn = mlp.act_fn
        self.down_proj = mlp.down_proj
        self.neuron_interventions = {}
        self.post_activation = None
    
    def forward(self, x):
        post_activation = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        self.post_activation = post_activation.detach().cpu()
        output = self.down_proj(post_activation)
        if len(self.neuron_interventions) > 0:
            print('performing intervention: mlp_neuron_interventions')
            for neuron_idx, mean in self.neuron_interventions.items():
                output[:, :, neuron_idx] = mean
        return output
    
    def reset(self):
        self.neuron_interventions = {}
    
    def freeze_neuron(self, neuron_idx, mean):
        self.neuron_interventions[neuron_idx] = mean

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_post_activation = None # these are the hidden neurons of the MLP
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        self.add_to_last_tensor = None
        self.output = None
        self.output_normalized = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if self.add_to_last_tensor is not None:
            print('performing intervention: add_to_last_tensor')
            output[0][:, -1, :] += self.add_to_last_tensor
        self.output = output[0]
        self.output_normalized = self.norm(output[0].to(self.norm.weight.device))
        self.block_output_unembedded = self.unembed_matrix(self.output_normalized.to(self.unembed_matrix.weight.device))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output.to(self.unembed_matrix.weight.device)))
        attn_output += args[0].to(attn_output.device)
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output.to(self.post_attention_layernorm.weight.device)))
        self.mlp_post_activation = self.block.mlp.post_activation
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output.to(self.unembed_matrix.weight.device)))
        return output

    def mlp_freeze_neuron(self, neuron_idx, mean):
        self.block.mlp.freeze_neuron(neuron_idx, mean) 

    def block_add_to_last_tensor(self, tensor):
        self.add_to_last_tensor = tensor

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.block.mlp.reset()
        self.add_to_last_tensor = None

    def get_attn_activations(self):
        return self.block.self_attn.activations

class LlamaHelper:
    def __init__(self, dir='/dlabdata1/llama2_hf/Llama-2-7b-hf', hf_token=None, device=None, load_in_8bit=True, use_embed_head=False, device_map='auto'):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(dir, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(dir, use_auth_token=hf_token,
                                                          device_map=device_map,
                                                          load_in_8bit=load_in_8bit)
        self.use_embed_head = True
        W = list(self.model.model.embed_tokens.parameters())[0].detach()
        self.head_embed = torch.nn.Linear(W.size(1), W.size(0), bias=False)
        self.head_embed.to(W.dtype)
        self.norm = self.model.model.norm
        with torch.no_grad():
            self.head_embed.weight.copy_(W) 
        self.head_embed.to(self.model.device)
        self.head_unembed = self.model.lm_head
        #self.model = self.model.to(self.device)
        self.device = next(self.model.parameters()).device
        if use_embed_head:
            head = self.head_embed
        else:
            head = self.head_unembed
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, head, self.model.model.norm)


    def set_embed_head(self):
        self.use_embed_head = True
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i].unembed_matrix = self.head_embed

    def set_unembed_head(self):
        self.use_embed_head = False
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i].unembed_matrix = self.head_unembed
    
    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    def generate_intermediate_text(self, layer_idx, prompt, max_length=100, temperature=1.0):
        layer = self.model.model.layers[layer_idx]
        for _ in range(max_length):
            self.get_logits(prompt)
            next_id = self.sample_next_token(layer.block_output_unembedded[:,-1], temperature=temperature)
            #next_token = self.tokenizer.decode(next_id)
            prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[1:]+[next_id])
            if next_id == self.tokenizer.eos_token_id:
                break
        return prompt

    def sample_next_token(self, logits, temperature=1.0):
        assert temperature >= 0, "temp must be geq 0"
        if temperature == 0:
            return self._sample_greedy(logits)
        return self._sample_basic(logits/temperature)
        
    def _sample_greedy(self, logits):
        return logits.argmax().item()

    def _sample_basic(self, logits):
        return torch.distributions.categorical.Categorical(logits=logits).sample().item()
    
    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids).logits
          return logits

    def set_neuron_intervention(self, layer_idx, neuron_idx, mean):
        self.model.model.layers[layer_idx].mlp_freeze_neuron(neuron_idx, mean)

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def set_add_to_last_tensor(self, layer, tensor):
      print('setting up intervention: add tensor to last soft token')
      self.model.model.layers[layer].block_add_to_last_tensor(tensor)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(indices.detach().cpu().numpy().tolist(), tokens, probs_percent)))

    def logits_all_layers(self, text, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_block=True):
        res = defaultdict(list)
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            if return_block:
                res['block_2'] += [layer.block_output_unembedded.detach().cpu()]
            if return_attn_mech:
                res['attn'] += [layer.attn_mech_output_unembedded.detach().cpu()]
            if return_intermediate_res:
                res['block_1'] += [layer.intermediate_res_unembedded.detach().cpu()]
            if return_mlp:
                res['mlp'] += [layer.mlp_output_unembedded.detach().cpu()]
        for k,v in res.items():
            res[k] = torch.cat(v, dim=0)
        if len(res) == 1:
            return list(res.values())[0]
        return res

    def latents_all_layers(self, text, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_mlp_post_activation=False, return_block=True, normalized=True):
        if return_attn_mech or return_intermediate_res or return_mlp:
            raise NotImplemented("not implemented")
        self.get_logits(text)
        tensors = []
        if return_block:
            for i, layer in enumerate(self.model.model.layers):
                if normalized:
                    tensors += [layer.output.detach().cpu()]
                else:
                    tensors += [layer.output_normalized.detach().cpu()]
        elif return_mlp_post_activation:
            for i, layer in enumerate(self.model.model.layers):
                tensors += [layer.mlp_post_activation.detach().cpu()]
        return torch.cat(tensors, dim=0)
        
    def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        print('Prompt:', text)
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism')
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream')
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 'Block output')