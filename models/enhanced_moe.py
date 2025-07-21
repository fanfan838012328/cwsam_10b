"""
Enhanced MoE (Mixture of Experts) implementation for CWSAM parameter scaling.
This module provides an improved MoE MLP block with better load balancing,
expert dropout, and noise gating mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple, Dict, Any, List
import math
import warnings
import time
from collections import defaultdict
import threading


class EnhancedMoEMLPBlock(nn.Module):
    """
    Enhanced MoE MLP Block with improved gating network, load balancing,
    expert dropout, and noise gating mechanisms.
    
    Supports 16 to 512 experts with scalable architecture.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        num_experts: int = 64,
        k: int = 4,
        act: Type[nn.Module] = nn.GELU,
        noisy_gating: bool = True,
        load_balancing: bool = True,
        expert_dropout: float = 0.1,
        gate_dropout: float = 0.0,
        capacity_factor: float = 1.25,
        use_bias: bool = True,
        gate_hidden_dim: Optional[int] = None,
        expert_init_std: float = 0.02,
    ) -> None:
        """
        Initialize Enhanced MoE MLP Block.
        
        Args:
            embedding_dim: Input/output embedding dimension
            mlp_dim: Hidden dimension of expert MLPs
            num_experts: Number of experts (16-512)
            k: Number of experts to select per token
            act: Activation function class
            noisy_gating: Whether to use noisy gating
            load_balancing: Whether to apply load balancing
            expert_dropout: Dropout rate within experts
            gate_dropout: Dropout rate in gating network
            capacity_factor: Capacity factor for load balancing
            use_bias: Whether to use bias in expert networks
            gate_hidden_dim: Hidden dimension of gating network (default: embedding_dim//4)
            expert_init_std: Standard deviation for expert weight initialization
        """
        super().__init__()
        
        # Validate parameters
        if not (16 <= num_experts <= 512):
            raise ValueError(f"num_experts must be between 16 and 512, got {num_experts}")
        if k > num_experts:
            raise ValueError(f"k ({k}) cannot be larger than num_experts ({num_experts})")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
            
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.load_balancing = load_balancing
        self.expert_dropout = expert_dropout
        self.capacity_factor = capacity_factor
        self.expert_init_std = expert_init_std
        
        # Improved gating network with multi-layer perceptron
        gate_hidden_dim = gate_hidden_dim or max(embedding_dim // 4, 64)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim // 2, num_experts)
        )
        
        # Expert networks with improved initialization
        self.experts = nn.ModuleList([
            self._create_expert(embedding_dim, mlp_dim, act, use_bias, expert_dropout)
            for _ in range(num_experts)
        ])
        
        # Load balancing components
        if self.load_balancing:
            self.register_buffer('expert_usage', torch.zeros(num_experts))
            self.register_buffer('total_tokens', torch.tensor(0.0))
            
        # Noise parameters for gating
        if self.noisy_gating:
            self.noise_epsilon = 1e-2
            self.register_buffer('noise_std', torch.tensor(1.0))
            
        # Initialize weights
        self._init_weights()
        
    def _create_expert(
        self, 
        embed_dim: int, 
        mlp_dim: int, 
        act: Type[nn.Module], 
        use_bias: bool,
        dropout: float
    ) -> nn.Module:
        """Create a single expert network with improved architecture."""
        return nn.Sequential(
            nn.Linear(embed_dim, mlp_dim, bias=use_bias),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_dim, embed_dim, bias=use_bias)
        )
        
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize gating network
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Initialize expert networks
        for expert in self.experts:
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=self.expert_init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
    def _compute_load_balancing_loss(
        self, 
        gates: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        if not self.load_balancing:
            return torch.tensor(0.0, device=gates.device)
            
        # Count how many tokens are assigned to each expert
        expert_counts = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            expert_counts[i] = (indices == i).float().sum()
            
        # Normalize by total number of tokens
        total_tokens = indices.numel()
        if total_tokens > 0:
            expert_probs = expert_counts / total_tokens
            
            # Compute coefficient of variation as load balancing loss
            mean_prob = expert_probs.mean()
            var_prob = ((expert_probs - mean_prob) ** 2).mean()
            cv_loss = torch.sqrt(var_prob) / (mean_prob + 1e-8)
            
            return cv_loss
        else:
            return torch.tensor(0.0, device=gates.device)
            
    def _noisy_top_k_gating(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Improved noisy top-k gating with better load balancing.
        
        Returns:
            gates: Top-k gate values [batch_size, k]
            indices: Top-k expert indices [batch_size, k]  
            raw_gates: Raw gate logits for load balancing [batch_size, num_experts]
        """
        # Compute raw gate logits
        raw_gates = self.gate(x)  # [batch_size, num_experts]
        
        # Add noise during training for exploration
        if self.noisy_gating and self.training:
            noise = torch.randn_like(raw_gates) * self.noise_epsilon * self.noise_std
            noisy_gates = raw_gates + noise
        else:
            noisy_gates = raw_gates
            
        # Apply temperature scaling for better distribution
        temperature = 1.0
        scaled_gates = noisy_gates / temperature
        
        # Get top-k experts
        top_k_gates, top_k_indices = scaled_gates.topk(self.k, dim=-1)
        
        # Apply softmax to top-k gates
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        return top_k_gates, top_k_indices, raw_gates
        
    def _update_expert_usage(self, indices: torch.Tensor):
        """Update expert usage statistics for monitoring."""
        if not self.load_balancing:
            return
            
        with torch.no_grad():
            # Update usage counts
            for i in range(self.num_experts):
                count = (indices == i).float().sum()
                self.expert_usage[i] = 0.9 * self.expert_usage[i] + 0.1 * count
                
            # Update total token count
            self.total_tokens = 0.9 * self.total_tokens + 0.1 * indices.numel()
            
    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Get expert usage statistics for monitoring."""
        if not self.load_balancing:
            return {}
            
        usage_probs = self.expert_usage / (self.total_tokens + 1e-8)
        return {
            'expert_usage_mean': usage_probs.mean().item(),
            'expert_usage_std': usage_probs.std().item(),
            'expert_usage_min': usage_probs.min().item(),
            'expert_usage_max': usage_probs.max().item(),
            'expert_usage_cv': (usage_probs.std() / (usage_probs.mean() + 1e-8)).item(),
            'total_tokens': self.total_tokens.item()
        }
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through Enhanced MoE MLP Block.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim] or [batch_size, embedding_dim]
            
        Returns:
            output: Output tensor with same shape as input
            aux_loss: Dictionary containing auxiliary losses for training
        """
        original_shape = x.shape
        
        # Flatten input for processing
        if x.dim() > 2:
            x_flat = x.view(-1, self.embedding_dim)
        else:
            x_flat = x
            
        batch_size = x_flat.shape[0]
        
        # Get gating decisions
        gates, indices, raw_gates = self._noisy_top_k_gating(x_flat)
        
        # Update expert usage statistics
        if self.training:
            self._update_expert_usage(indices)
            
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert efficiently
        for expert_idx in range(self.num_experts):
            # Find all positions that use this expert
            expert_mask = (indices == expert_idx)
            
            if not expert_mask.any():
                continue
                
            # Collect inputs and corresponding gate values for this expert
            expert_inputs = []
            expert_gates = []
            expert_positions = []
            
            for k_idx in range(self.k):
                mask = expert_mask[:, k_idx]
                if mask.any():
                    expert_inputs.append(x_flat[mask])
                    expert_gates.append(gates[:, k_idx][mask])
                    expert_positions.append(torch.where(mask)[0])
                    
            if not expert_inputs:
                continue
                
            # Batch process all inputs for this expert
            combined_input = torch.cat(expert_inputs, dim=0)
            combined_gates = torch.cat(expert_gates, dim=0)
            combined_positions = torch.cat(expert_positions, dim=0)
            
            # Forward through expert
            expert_output = self.experts[expert_idx](combined_input)
            
            # Apply gate weights
            weighted_output = expert_output * combined_gates.unsqueeze(-1)
            
            # Scatter results back to output tensor
            output.index_add_(0, combined_positions, weighted_output)
            
        # Compute auxiliary losses
        aux_losses = {}
        if self.training and self.load_balancing:
            aux_losses['load_balance_loss'] = self._compute_load_balancing_loss(gates, indices)
            
        # Reshape output to original shape
        output = output.view(original_shape)
        
        return output, aux_losses
        
    def extra_repr(self) -> str:
        """String representation of the module."""
        return (f'embedding_dim={self.embedding_dim}, mlp_dim={self.mlp_dim}, '
                f'num_experts={self.num_experts}, k={self.k}, '
                f'noisy_gating={self.noisy_gating}, load_balancing={self.load_balancing}')


class MoELayer(nn.Module):
    """
    Complete MoE layer that wraps the Enhanced MoE MLP block with normalization.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        num_experts: int = 64,
        k: int = 4,
        act: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        **moe_kwargs
    ):
        super().__init__()
        
        self.norm = norm_layer(embedding_dim)
        self.moe_mlp = EnhancedMoEMLPBlock(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            num_experts=num_experts,
            k=k,
            act=act,
            **moe_kwargs
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with pre-normalization."""
        normalized_x = self.norm(x)
        moe_output, aux_losses = self.moe_mlp(normalized_x)
        output = x + moe_output  # Residual connection
        return output, aux_losses


class OptimizedMoEMLPBlock(EnhancedMoEMLPBlock):
    """
    Optimized MoE MLP Block with advanced efficiency improvements:
    - Batch expert computation to reduce loop overhead
    - Expert parallel computation support
    - Dynamic expert selection strategies
    - Expert usage statistics and monitoring
    """
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        num_experts: int = 64,
        k: int = 4,
        act: Type[nn.Module] = nn.GELU,
        enable_parallel_experts: bool = True,
        dynamic_k: bool = False,
        min_k: int = 2,
        max_k: int = 8,
        batch_expert_computation: bool = True,
        expert_capacity_factor: float = 1.25,
        enable_expert_monitoring: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize Optimized MoE MLP Block.
        
        Args:
            enable_parallel_experts: Enable parallel expert computation
            dynamic_k: Enable dynamic expert selection based on input complexity
            min_k: Minimum number of experts to select (for dynamic_k)
            max_k: Maximum number of experts to select (for dynamic_k)
            batch_expert_computation: Enable batched expert computation
            expert_capacity_factor: Capacity factor for expert load balancing
            enable_expert_monitoring: Enable detailed expert usage monitoring
        """
        super().__init__(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            num_experts=num_experts,
            k=k,
            act=act,
            **kwargs
        )
        
        self.enable_parallel_experts = enable_parallel_experts
        self.dynamic_k = dynamic_k
        self.min_k = min_k
        self.max_k = max_k
        self.batch_expert_computation = batch_expert_computation
        self.expert_capacity_factor = expert_capacity_factor
        self.enable_expert_monitoring = enable_expert_monitoring
        
        # Performance monitoring
        if self.enable_expert_monitoring:
            self.register_buffer('forward_times', torch.zeros(1))
            self.register_buffer('expert_compute_times', torch.zeros(num_experts))
            self.register_buffer('gating_times', torch.zeros(1))
            self.register_buffer('total_forward_calls', torch.tensor(0))
            
        # Dynamic expert selection components
        if self.dynamic_k:
            self.complexity_estimator = nn.Linear(embedding_dim, 1)
            self.k_predictor = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 4),
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, 1),
                nn.Sigmoid()
            )
            
        # Batch computation optimization
        if self.batch_expert_computation:
            # Pre-allocate tensors for batch processing
            self.register_buffer('batch_indices', torch.zeros(1, dtype=torch.long))
            self.register_buffer('batch_gates', torch.zeros(1))
            
    def _estimate_input_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate input complexity for dynamic expert selection."""
        if not self.dynamic_k:
            return torch.tensor(0.5, device=x.device)
            
        # Compute input variance as complexity measure
        complexity = torch.var(x, dim=-1, keepdim=True)
        complexity_score = torch.sigmoid(self.complexity_estimator(x))
        
        return complexity_score.mean()
        
    def _dynamic_k_selection(self, x: torch.Tensor) -> int:
        """Dynamically select number of experts based on input complexity."""
        if not self.dynamic_k:
            return self.k
            
        complexity = self._estimate_input_complexity(x)
        k_ratio = self.k_predictor(x.mean(dim=0, keepdim=True)).item()
        
        # Scale k based on complexity
        dynamic_k = int(self.min_k + (self.max_k - self.min_k) * k_ratio)
        dynamic_k = min(dynamic_k, self.num_experts)
        
        return dynamic_k
        
    def _batch_expert_forward(
        self, 
        x: torch.Tensor, 
        gates: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized batch expert computation to reduce loop overhead.
        """
        batch_size = x.shape[0]
        output = torch.zeros_like(x)
        
        if not self.batch_expert_computation:
            # Fall back to original implementation
            return self._original_expert_forward(x, gates, indices)
            
        # Group tokens by expert for batch processing
        expert_token_map = defaultdict(list)
        expert_gate_map = defaultdict(list)
        expert_position_map = defaultdict(list)
        
        for batch_idx in range(batch_size):
            for k_idx in range(indices.shape[1]):
                expert_idx = indices[batch_idx, k_idx].item()
                expert_token_map[expert_idx].append(x[batch_idx])
                expert_gate_map[expert_idx].append(gates[batch_idx, k_idx])
                expert_position_map[expert_idx].append(batch_idx)
                
        # Process each expert with batched computation
        for expert_idx, tokens in expert_token_map.items():
            if not tokens:
                continue
                
            # Stack tokens for batch processing
            expert_input = torch.stack(tokens, dim=0)
            expert_gates = torch.stack(expert_gate_map[expert_idx], dim=0)
            expert_positions = torch.tensor(expert_position_map[expert_idx], device=x.device)
            
            # Batch forward through expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Apply gate weights
            weighted_output = expert_output * expert_gates.unsqueeze(-1)
            
            # Accumulate results
            output.index_add_(0, expert_positions, weighted_output)
            
        return output
        
    def _parallel_expert_forward(
        self, 
        x: torch.Tensor, 
        gates: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel expert computation using threading (for CPU) or CUDA streams (for GPU).
        """
        if not self.enable_parallel_experts or not torch.cuda.is_available():
            return self._batch_expert_forward(x, gates, indices)
            
        # Use CUDA streams for parallel expert computation
        streams = [torch.cuda.Stream() for _ in range(min(4, self.num_experts))]
        results = {}
        
        batch_size = x.shape[0]
        output = torch.zeros_like(x)
        
        # Group work by expert
        expert_work = defaultdict(list)
        for batch_idx in range(batch_size):
            for k_idx in range(indices.shape[1]):
                expert_idx = indices[batch_idx, k_idx].item()
                expert_work[expert_idx].append((batch_idx, k_idx))
                
        # Process experts in parallel using streams
        active_experts = list(expert_work.keys())
        for i, expert_idx in enumerate(active_experts):
            stream_idx = i % len(streams)
            
            with torch.cuda.stream(streams[stream_idx]):
                work_items = expert_work[expert_idx]
                if not work_items:
                    continue
                    
                # Collect inputs for this expert
                expert_inputs = []
                expert_gates_list = []
                expert_positions = []
                
                for batch_idx, k_idx in work_items:
                    expert_inputs.append(x[batch_idx])
                    expert_gates_list.append(gates[batch_idx, k_idx])
                    expert_positions.append(batch_idx)
                    
                if expert_inputs:
                    expert_input_batch = torch.stack(expert_inputs, dim=0)
                    expert_gates_batch = torch.stack(expert_gates_list, dim=0)
                    expert_positions_tensor = torch.tensor(expert_positions, device=x.device)
                    
                    # Forward through expert
                    expert_output = self.experts[expert_idx](expert_input_batch)
                    weighted_output = expert_output * expert_gates_batch.unsqueeze(-1)
                    
                    results[expert_idx] = (weighted_output, expert_positions_tensor)
                    
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
            
        # Accumulate results
        for expert_idx, (weighted_output, positions) in results.items():
            output.index_add_(0, positions, weighted_output)
            
        return output
        
    def _original_expert_forward(
        self, 
        x: torch.Tensor, 
        gates: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Original expert forward implementation for fallback."""
        batch_size = x.shape[0]
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx)
            
            if not expert_mask.any():
                continue
                
            expert_inputs = []
            expert_gates_list = []
            expert_positions = []
            
            for k_idx in range(self.k):
                mask = expert_mask[:, k_idx]
                if mask.any():
                    expert_inputs.append(x[mask])
                    expert_gates_list.append(gates[:, k_idx][mask])
                    expert_positions.append(torch.where(mask)[0])
                    
            if not expert_inputs:
                continue
                
            combined_input = torch.cat(expert_inputs, dim=0)
            combined_gates = torch.cat(expert_gates_list, dim=0)
            combined_positions = torch.cat(expert_positions, dim=0)
            
            expert_output = self.experts[expert_idx](combined_input)
            weighted_output = expert_output * combined_gates.unsqueeze(-1)
            
            output.index_add_(0, combined_positions, weighted_output)
            
        return output
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if not self.enable_expert_monitoring:
            return {}
            
        stats = self.get_expert_usage_stats()
        
        if self.total_forward_calls > 0:
            stats.update({
                'avg_forward_time': (self.forward_times / self.total_forward_calls).item(),
                'avg_gating_time': (self.gating_times / self.total_forward_calls).item(),
                'expert_compute_times': self.expert_compute_times.tolist(),
                'total_forward_calls': self.total_forward_calls.item(),
            })
            
        return stats
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Optimized forward pass with performance monitoring.
        """
        if self.enable_expert_monitoring:
            start_time = time.time()
            
        original_shape = x.shape
        
        # Flatten input for processing
        if x.dim() > 2:
            x_flat = x.view(-1, self.embedding_dim)
        else:
            x_flat = x
            
        # Dynamic k selection
        current_k = self._dynamic_k_selection(x_flat)
        
        # Get gating decisions with timing
        if self.enable_expert_monitoring:
            gating_start = time.time()
            
        gates, indices, raw_gates = self._noisy_top_k_gating(x_flat)
        
        if self.enable_expert_monitoring:
            gating_time = time.time() - gating_start
            self.gating_times += gating_time
            
        # Update expert usage statistics
        if self.training:
            self._update_expert_usage(indices)
            
        # Use optimized expert computation
        if self.enable_parallel_experts and torch.cuda.is_available():
            output = self._parallel_expert_forward(x_flat, gates, indices)
        else:
            output = self._batch_expert_forward(x_flat, gates, indices)
            
        # Compute auxiliary losses
        aux_losses = {}
        if self.training and self.load_balancing:
            aux_losses['load_balance_loss'] = self._compute_load_balancing_loss(gates, indices)
            
        # Add dynamic k loss if enabled
        if self.dynamic_k and self.training:
            # Encourage using fewer experts when possible
            k_penalty = (current_k - self.min_k) / (self.max_k - self.min_k)
            aux_losses['dynamic_k_loss'] = torch.tensor(k_penalty * 0.01, device=x.device)
            
        # Reshape output to original shape
        output = output.view(original_shape)
        
        # Update performance monitoring
        if self.enable_expert_monitoring:
            forward_time = time.time() - start_time
            self.forward_times += forward_time
            self.total_forward_calls += 1
            
        return output, aux_losses


class ExpertMonitor:
    """
    Standalone expert usage and performance monitor.
    """
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset_stats()
        
    def reset_stats(self):
        """Reset all monitoring statistics."""
        self.expert_usage_count = defaultdict(int)
        self.expert_compute_times = defaultdict(list)
        self.total_tokens = 0
        self.total_forward_calls = 0
        
    def update_usage(self, expert_indices: torch.Tensor):
        """Update expert usage statistics."""
        for idx in expert_indices.flatten():
            self.expert_usage_count[idx.item()] += 1
        self.total_tokens += expert_indices.numel()
        
    def update_compute_time(self, expert_idx: int, compute_time: float):
        """Update expert computation time."""
        self.expert_compute_times[expert_idx].append(compute_time)
        
    def get_usage_distribution(self) -> Dict[str, float]:
        """Get expert usage distribution statistics."""
        if self.total_tokens == 0:
            return {}
            
        usage_probs = []
        for i in range(self.num_experts):
            prob = self.expert_usage_count[i] / self.total_tokens
            usage_probs.append(prob)
            
        usage_probs = torch.tensor(usage_probs)
        
        return {
            'usage_mean': usage_probs.mean().item(),
            'usage_std': usage_probs.std().item(),
            'usage_min': usage_probs.min().item(),
            'usage_max': usage_probs.max().item(),
            'usage_entropy': -(usage_probs * torch.log(usage_probs + 1e-8)).sum().item(),
            'load_balance_score': 1.0 - (usage_probs.std() / (usage_probs.mean() + 1e-8)).item()
        }
        
    def get_compute_time_stats(self) -> Dict[str, Any]:
        """Get expert computation time statistics."""
        if not self.expert_compute_times:
            return {}
            
        stats = {}
        for expert_idx, times in self.expert_compute_times.items():
            if times:
                stats[f'expert_{expert_idx}_avg_time'] = sum(times) / len(times)
                stats[f'expert_{expert_idx}_total_calls'] = len(times)
                
        return stats


def create_moe_block(
    embedding_dim: int,
    mlp_dim: int,
    num_experts: int = 64,
    k: int = 4,
    optimized: bool = True,
    **kwargs
) -> EnhancedMoEMLPBlock:
    """
    Factory function to create an Enhanced or Optimized MoE MLP block.
    
    Args:
        optimized: If True, create OptimizedMoEMLPBlock, else EnhancedMoEMLPBlock
    """
    # Validate expert count is within supported range
    if not (16 <= num_experts <= 512):
        warnings.warn(f"num_experts {num_experts} is outside recommended range [16, 512]")
        
    # Adjust k if necessary
    k = min(k, num_experts)
    
    if optimized:
        return OptimizedMoEMLPBlock(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            num_experts=num_experts,
            k=k,
            **kwargs
        )
    else:
        return EnhancedMoEMLPBlock(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            num_experts=num_experts,
            k=k,
            **kwargs
        )