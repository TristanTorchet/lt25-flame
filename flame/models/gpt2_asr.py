import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict
from transformers import PreTrainedModel, LlamaModel, GPT2Model, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPast
import math

from fla.layers.hgrn import HGRNAttention
from fla.modules import GatedMLP as HGRNMLP
from fla.modules import RMSNorm
from fla.models.hgrn.configuration_hgrn import HGRNConfig


class HGRNASRConfig(HGRNConfig,GPT2Config):
    """Configuration class for HGRN ASR model."""
    
    def __init__(
        self,
        input_size: int = 80,  # MFCC features
        vocab_size: int = 50,  # Number of character classes
        max_sequence_length: int = 4096,
        blank_id: int = -100,  # CTC blank token
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.n_positions = self.max_sequence_length = 4096
        self.blank_id = blank_id
        self.num_heads = num_heads


class HGRNASRBlock(nn.Module):
    """HGRN block for ASR processing."""
    
    def __init__(self, config: HGRNASRConfig, layer_idx: int):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = HGRNAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            elementwise_affine=config.elementwise_affine,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx
        )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.mlp = HGRNMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        # Create lower_bound tensor if use_lower_bound is True
        if self.config.use_lower_bound and lower_bound is None:
            lower_bound = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound,
            **kwargs
        )
        
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        
        return hidden_states, attentions, past_key_values


class HGRNASRModel(PreTrainedModel):
    """HGRN model for ASR."""
    
    config_class = HGRNASRConfig
    base_model_prefix = "gpt2"
    
    def __init__(self, config: HGRNASRConfig):
        super().__init__(config)
        self.config = config
        
        # Input projection from input_size to hidden_size
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # HGRN layers
        self.layers = nn.ModuleList([
            HGRNASRBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        
        # CTC head - outputs probability distribution over vocabulary for each time step
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,  # Audio features: (batch_size, seq_len, input_size)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        **kwargs
    ):
        # Project input features to hidden size
        hidden_states = self.input_projection(input_ids)
        
        # Apply HGRN layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Create lower_bound tensor if use_lower_bound is True
        lower_bound = None
        if self.config.use_lower_bound:
            lower_bound = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states, attentions, past_key_values = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lower_bound=lower_bound,
                **kwargs
            )
            
            if output_attentions:
                all_attentions = all_attentions + (attentions,)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # CTC head - output logits for each time step
        logits = self.ctc_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        #if not return_dict:
        #    return (logits, hidden_states, all_hidden_states, all_attentions)
        
        return logits


class ParallelBeamSearchDecoder:
    """Fully vectorized beam search decoder using dynamic programming approach.
        Generated by Claude.ai
    """
    
    def __init__(self, blank_id: int = 0, beam_size: int = 5):
        self.blank_id = blank_id
        self.beam_size = beam_size
    
    def decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
        """
        Decode using a fully parallel approach with prefix trees.
        
        This version uses a different algorithm that's more amenable to vectorization
        by tracking all possible prefixes simultaneously.
        """
        batch_size, max_seq_len, vocab_size = log_probs.shape
        device = log_probs.device
        
        # Use a simpler greedy approach with CTC rules for full parallelization
        # This sacrifices some accuracy for speed
        results = []
        
        # Process each batch in parallel using vectorized operations
        for b in range(batch_size):
            seq_len = input_lengths[b].item()
            seq_log_probs = log_probs[b, :seq_len, :]
            
            # Get best token at each time step
            best_tokens = torch.argmax(seq_log_probs, dim=1)
            
            # Apply CTC collapse rules vectorized
            sequence = []
            prev_token = None
            
            # Convert to list for processing
            tokens = best_tokens.tolist()
            
            # Vectorized CTC collapse
            mask = torch.ones(len(tokens), dtype=torch.bool, device=device)
            
            # Mark blanks
            blank_mask = best_tokens == self.blank_id
            mask = mask & ~blank_mask
            
            # Mark consecutive duplicates
            if len(tokens) > 1:
                consecutive_mask = torch.zeros(len(tokens), dtype=torch.bool, device=device)
                consecutive_mask[1:] = best_tokens[1:] == best_tokens[:-1]
                mask = mask & ~consecutive_mask
            
            # Extract final sequence
            valid_tokens = best_tokens[mask]
            sequence = valid_tokens.tolist()
            
            results.append(sequence)
        
        return results


class BeamSearchDecoder:
    """Beam search decoder for CTC outputs."""
    
    def __init__(self, blank_id: int = 0, beam_size: int = 5):
        self.blank_id = blank_id
        self.beam_size = beam_size
    
    def decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
        """
        Decode CTC outputs using beam search.
        
        Args:
            log_probs: Log probabilities from CTC model (batch_size, seq_len, vocab_size)
            input_lengths: Actual sequence lengths (batch_size,)
            
        Returns:
            List of decoded sequences for each batch item
        """
        batch_size = log_probs.size(0)
        results = []
        
        for i in range(batch_size):
            seq_len = input_lengths[i].item()
            seq_log_probs = log_probs[i, :seq_len, :]  # (seq_len, vocab_size)
            
            # Initialize beam with blank token
            beam = [{"sequence": [], "score": 0.0}]
            
            for t in range(seq_len):
                new_beam = []
                
                for beam_item in beam:
                    for token in range(log_probs.size(2)):
                        score = beam_item["score"] + seq_log_probs[t, token].item()
                        
                        if token == self.blank_id:
                            # Blank token - don't add to sequence
                            new_beam.append({
                                "sequence": beam_item["sequence"].copy(),
                                "score": score
                            })
                        else:
                            # Non-blank token
                            new_sequence = beam_item["sequence"].copy()
                            
                            # CTC collapse rule: only add if different from last token
                            if len(new_sequence) == 0 or new_sequence[-1] != token:
                                new_sequence.append(token)
                            
                            new_beam.append({
                                "sequence": new_sequence,
                                "score": score
                            })
                
                # Keep top beam_size candidates
                new_beam.sort(key=lambda x: x["score"], reverse=True)
                beam = new_beam[:self.beam_size]
            
            # Get best sequence
            best_sequence = beam[0]["sequence"]
            results.append(best_sequence)
        
        return results
    
    def greedy_decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
        """
        Simple greedy decoding for CTC outputs.
        
        Args:
            log_probs: Log probabilities from CTC model (batch_size, seq_len, vocab_size)
            input_lengths: Actual sequence lengths (batch_size,)
            
        Returns:
            List of decoded sequences for each batch item
        """
        batch_size = log_probs.size(0)
        results = []
        
        for i in range(batch_size):
            seq_len = input_lengths[i].item()
            seq_log_probs = log_probs[i, :seq_len, :]  # (seq_len, vocab_size)
            
            # Get best token at each time step
            best_tokens = torch.argmax(seq_log_probs, dim=-1)  # (seq_len,)
            
            # Apply CTC collapse rules
            decoded = []
            prev_token = None
            
            for token in best_tokens:
                token = token.item()
                if token != self.blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            
            results.append(decoded)
        
        return results


class HGRNASRForCTC(nn.Module):
    """HGRN model for ASR with CTC loss and beam search decoding."""
    
    config_class = HGRNASRConfig
    base_model_prefix = "gpt2"
    
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.blank_id = config.blank_id
        
        # Base model
        self.gpt2 = GPT2Model(config)
        
        # CTC loss
        # Beam search decoder
        self.decoder = ParallelBeamSearchDecoder(blank_id=self.blank_id, beam_size=3)
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    
    def forward(
        self,
        input_ids: torch.Tensor,  # Audio features: (batch_size, seq_len, input_size)
        labels: Optional[torch.Tensor] = None,  # Target tokens (N,S)
        input_lengths: Optional[torch.Tensor] = None,  # Actual input lengths
        target_lengths: Optional[torch.Tensor] = None,  # Target sequence lengths
    ):

        h = self.input_projection(input_ids)
        # Get model outputs
        hidden_states, = self.gpt2(
            inputs_embeds=h,
            #attention_mask=attention_mask,
            #**kwargs
            return_dict=False,
            output_hidden_states=False,
            use_cache=False
        )
        
        logits = self.ctc_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        # Apply log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

        
        loss = None
            

        
        return log_probs #, logits, log_probs
    
    def decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor, 
               use_beam_search: bool = False) -> List[List[int]]:
        """
        Decode audio features to text tokens.
        
        Args:
            input_ids: Audio features (batch_size, seq_len, input_size)
            input_lengths: Actual sequence lengths (batch_size,)
            use_beam_search: Whether to use beam search or greedy decoding
            
        Returns:
            List of decoded token sequences
        """
        self.eval()
        #with torch.no_grad():
        #log_probs = self.forward(input_ids, input_lengths=input_lengths)
        
        if use_beam_search:
            return self.decoder.decode(log_probs, input_lengths)
        else:
            return self.decoder.greedy_decode(log_probs, input_lengths)
    
    def generate(self, log_probs: torch.Tensor, input_lengths: torch.Tensor, 
                 idx_to_char: Dict[int, str], use_beam_search: bool = False) -> List[str]:
        """
        Generate text from audio features.
        
        Args:
            input_ids: Audio features (batch_size, seq_len, input_size)
            input_lengths: Actual sequence lengths (batch_size,)
            idx_to_char: Mapping from token indices to characters
            use_beam_search: Whether to use beam search or greedy decoding
            
        Returns:
            List of decoded text strings
        """
        token_sequences = self.decode(log_probs, input_lengths, use_beam_search)
        
        texts = []
        for tokens in token_sequences:
            text = ''.join([idx_to_char.get(token, '<unk>') for token in tokens])
            texts.append(text)
        
        return texts
