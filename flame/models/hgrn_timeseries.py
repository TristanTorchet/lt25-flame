import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from fla.layers.hgrn import HGRNAttention
from fla.modules import GatedMLP as HGRNMLP
from fla.modules import RMSNorm
from fla.models.hgrn.configuration_hgrn import HGRNConfig


class HGRNTimeSeriesConfig(HGRNConfig):
    """Configuration class for HGRN time series classification model."""
    
    def __init__(
        self,
        input_size: int = 1,
        num_classes: int = 10,
        max_sequence_length: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        # Remove vocab_size since we're not doing language modeling
        if hasattr(self, 'vocab_size'):
            delattr(self, 'vocab_size')


class HGRNTimeSeriesBlock(nn.Module):
    """HGRN block for time series processing."""
    
    def __init__(self, config: HGRNTimeSeriesConfig, layer_idx: int):
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


class HGRNTimeSeriesModel(PreTrainedModel):
    """HGRN model for time series classification."""
    
    config_class = HGRNTimeSeriesConfig
    base_model_prefix = "hgrn_timeseries"
    
    def __init__(self, config: HGRNTimeSeriesConfig):
        super().__init__(config)
        self.config = config
        
        # Input projection from input_size to hidden_size
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        
        # HGRN layers
        self.layers = nn.ModuleList([
            HGRNTimeSeriesBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,  # For compatibility with existing training loop
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs
    ):
        # input_ids is actually our time series data with shape (batch_size, seq_len, input_size)
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
        
        # Global average pooling over sequence dimension for classification
        pooled_output = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        if not return_dict:
            return (logits, hidden_states, all_hidden_states, all_attentions)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        ), logits


class HGRNTimeSeriesForSequenceClassification(PreTrainedModel):
    """HGRN model for sequence classification with proper loss computation."""
    
    config_class = HGRNTimeSeriesConfig
    base_model_prefix = "hgrn_timeseries"
    
    def __init__(self, config: HGRNTimeSeriesConfig):
        super().__init__(config)
        self.config = config
        self.num_classes = config.num_classes
        
        # Base model
        self.hgrn_timeseries = HGRNTimeSeriesModel(config)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Get model outputs
        outputs, logits = self.hgrn_timeseries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        # Return in format compatible with existing training loop
        class TimeSeriesOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return TimeSeriesOutput(loss, logits)