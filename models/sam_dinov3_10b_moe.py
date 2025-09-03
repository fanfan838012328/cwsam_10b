@register('sam_dinov3_10b_moe')
class SAM_DINOV3_10B_MoE(nn.Module):
    """SAM with DINOv3-10B MoE backbone"""
    
    def __init__(
        self,
        inp_size=None,
        encoder_mode=None,
        loss=None,
        num_classes=None,
        loss_weight=None,
        ignore_index=-100,
        resume=None,
        # MoE specific parameters
        num_experts=32,
        k=4,
        moe_layers=None,  # None表示使用默认层
        expert_capacity_factor=1.5,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DINOv3权重路径
        dinov3_weights_path = '/mnt/fanfq/data/fan/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth'
        
        # 创建DINOv3-10B MoE模型
        if resume is None:
            # 新训练时加载预训练权重
            self.image_encoder = create_dinov3_10b_moe(
                dinov3_checkpoint_path=dinov3_weights_path,
                num_experts=num_experts,
                k=k,
                moe_layers=moe_layers,
                expert_capacity_factor=expert_capacity_factor,
            )
        else:
            # 恢复训练时只构建模型结构，权重后续从checkpoint加载
            dinov3_model = torch.hub.load('dinov3-main', 'dinov3_vit7b16', source='local', pretrained=False)
            from .dinov3_moe import DINOV3MoEWrapper
            self.image_encoder = DINOV3MoEWrapper(
                dinov3_model=dinov3_model,
                moe_layers=moe_layers,
                num_experts=num_experts,
                k=k,
                expert_capacity_factor=expert_capacity_factor,
                noisy_gating=True,
            )
        
        # DINOv3输出维度
        dinov3_hidden_dim = 4096  # DINOv3-7B的隐藏维度
        
        # 投影层：DINOv3特征 -> SAM prompt特征
        mid_dim = 1024
        self.projection = nn.Sequential(
            nn.Conv2d(dinov3_hidden_dim, mid_dim, kernel_size=1, bias=False),
            LayerNorm2d(mid_dim),
            nn.Conv2d(mid_dim, encoder_mode['prompt_embed_dim'], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(encoder_mode['prompt_embed_dim']),
        )
        
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        
        # SAM解码器
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
        )
        
        # 损失函数设置
        self.loss_mode = loss
        self.ignore_index = ignore_index
        
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss(reduction='none')
        elif self.loss_mode == 'iou':
            if loss_weight is not None:
                pos_weight = torch.tensor(loss_weight, dtype=torch.float)
                self.criterionBCE = torch.nn.CrossEntropyLoss(
                    weight=pos_weight, ignore_index=self.ignore_index
                )
            else:
                self.criterionBCE = torch.nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index
                )
            self.criterionIOU = IOU()
        
        # 位置编码
        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])
        
        # MoE辅助损失权重
        self.moe_aux_loss_weight = 0.01
        
    def set_input(self, input, gt_mask, task_ids=None):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
    
    def get_dense_pe(self) -> torch.Tensor:
        """获取位置编码"""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def forward(self):
        bs = self.input.shape[0]
        
        # DINOv3-MoE特征提取
        dino_output = self.image_encoder.forward(self.input)
        
        # 获取MoE辅助损失
        self.moe_aux_loss = self.image_encoder.get_aux_loss()
        
        # 处理DINOv3输出格式
        if isinstance(dino_output, dict):
            # 如果输出是字典格式（包含patch tokens）
            if 'x_norm_patchtokens' in dino_output:
                dino_features = dino_output['x_norm_patchtokens']
            elif 'x_prenorm' in dino_output:
                # 去掉CLS token，只保留patch tokens
                dino_features = dino_output['x_prenorm'][:, 1:]  # 跳过CLS token
            else:
                # 取第一个输出
                dino_features = list(dino_output.values())[0]
                if dino_features.dim() == 3:  # (B, N+1, D) 格式
                    dino_features = dino_features[:, 1:]  # 去掉CLS token
        else:
            # 直接tensor输出
            if dino_output.dim() == 3:  # (B, N+1, D)
                dino_features = dino_output[:, 1:]  # 去掉CLS token
            else:
                dino_features = dino_output
        
        # 推导网格尺寸
        token_count = dino_features.shape[1]
        h = int(math.sqrt(token_count))
        if h * h != token_count:
            h = int(round(token_count ** 0.5))
        if h * h != token_count:
            raise RuntimeError(f"意外的token数量={token_count}，无法形成方形网格")
        w = h
        
        # 重塑为空间维度：(B, N, D) -> (B, D, H, W)
        dino_features = dino_features.permute(0, 2, 1).reshape(bs, -1, h, w)
        
        # 投影到SAM特征空间
        self.features = self.projection(dino_features)
        
        # SAM mask解码
        sparse_embeddings = torch.empty(
            (bs, 0, self.prompt_embed_dim), device=self.input.device
        )
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.pe_layer(h).unsqueeze(0),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, h, w
            ),
            multimask_output=False,
        )
        
        # 选择第一个mask token的输出
        low_res_masks = low_res_masks[:, 0]  # (bs, num_classes, h, w)
        
        # 上采样到原始分辨率
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks
    
    def infer(self, input, task_id=None):
        """推理接口"""
        bs = input.shape[0]
        
        # DINOv3-MoE特征提取
        dino_output = self.image_encoder.forward(input)
        
        # 处理输出格式（同forward）
        if isinstance(dino_output, dict):
            if 'x_norm_patchtokens' in dino_output:
                dino_features = dino_output['x_norm_patchtokens']
            elif 'x_prenorm' in dino_output:
                dino_features = dino_output['x_prenorm'][:, 1:]
            else:
                dino_features = list(dino_output.values())[0]
                if dino_features.dim() == 3:
                    dino_features = dino_features[:, 1:]
        else:
            if dino_output.dim() == 3:
                dino_features = dino_output[:, 1:]
            else:
                dino_features = dino_output
        
        # 网格重塑
        token_count = dino_features.shape[1]
        h = int(math.sqrt(token_count))
        if h * h != token_count:
            h = int(round(token_count ** 0.5))
        w = h
        
        dino_features = dino_features.permute(0, 2, 1).reshape(bs, -1, h, w)
        features = self.projection(dino_features)
        
        # SAM解码
        sparse_embeddings = torch.empty(
            (bs, 0, self.prompt_embed_dim), device=input.device
        )
        
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.pe_layer(h).unsqueeze(0),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, h, w
            ),
            multimask_output=False,
        )
        
        low_res_masks = low_res_masks[:, 0]
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """后处理masks"""
        masks = masks.squeeze(dim=1)
        masks = F.interpolate(
            masks,
            (self.inp_size, self.inp_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size, :input_size]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks
    
    def backward_G(self):
        """计算损失"""
        # 主任务损失
        main_loss = self.criterionBCE(
            self.pred_mask, torch.argmax(self.gt_mask, dim=1, keepdim=True).squeeze(1)
        )
        
        # MoE辅助损失
        aux_loss = self.moe_aux_loss if hasattr(self, 'moe_aux_loss') else 0.0
        
        # 总损失
        self.loss_G = main_loss + self.moe_aux_loss_weight * aux_loss
    
    def optimize_parameters(self):
        """优化参数"""
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()
    
    def set_requires_grad(self, nets, requires_grad=False):
        """设置梯度需求"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def get_parameter_stats(self):
        """获取参数统计信息"""
        return self.image_encoder.get_parameter_count()
    
    def get_moe_parameters(self):
        """获取MoE参数用于优化器设置"""
        moe_params = self.image_encoder.get_moe_parameters()
        projection_params = list(self.projection.parameters())
        decoder_params = list(self.mask_decoder.parameters())
        other_params = list(self.pe_layer.parameters()) + list(self.no_mask_embed.parameters())
        
        return {
            'moe_params': moe_params,
            'projection_params': projection_params,
            'decoder_params': decoder_params,
            'other_params': other_params,
            'trainable_params': moe_params + projection_params + decoder_params + other_params
        }

