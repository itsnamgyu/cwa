from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig, Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding


class ClosingLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ClosingLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class ClosingLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ClosingLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.ne(1).any():
            # we only consider the perplexity testing use case
            raise NotImplementedError("Custom attention mask is not supported yet")
        if past_key_values is not None:
            raise NotImplementedError("KV cache is not supported yet")

        bs = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        seq_len = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        # create placeholder 4d attention mask to suppress mask computation in super().forward
        # these will be overwritten in ClosingLlamaDecoderLayer.forward
        attention_mask = torch.ones(bs, 1, seq_len, seq_len, dtype=torch.long, device=input_ids.device)

        return super().forward(input_ids, attention_mask, position_ids, past_key_values, *args, **kwargs)


class ClosingLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        # if layer_idx < 12 or layer_idx >=28:
        #     self.window_size = 2048 + 1024
        # else:
        #     self.window_size = 2048 - 1024
        self.window_size = 2560 - layer_idx * 32  # 1024 to 3072 on 32 layers (llama 3 8b)
        # self.window_size = 1024 + layer_idx * 64  # 1024 to 3072 on 32 layers (llama 3 8b)
        self.sink_size = 4
        print(f"Layer {layer_idx} window size: {self.window_size}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]
        dtype, device = hidden_states.dtype, hidden_states.device
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device)

        local_window_size = self.window_size - self.sink_size
        top_mask = torch.triu(causal_mask, diagonal=1)
        bottom_mask = None
        if sequence_length > local_window_size:
            bottom_mask = torch.tril(causal_mask, diagonal=-local_window_size)
        causal_mask[:, :self.sink_size].zero_()  # + sink
        if bottom_mask is not None:
            causal_mask = torch.maximum(bottom_mask, causal_mask)  # - bottom + sink
        causal_mask = torch.minimum(causal_mask, top_mask)  # - bottom + sink - top

        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        return super().forward(hidden_states, causal_mask, *args, **kwargs)
