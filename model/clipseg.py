import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
import sys
from einops.layers.torch import Rearrange, Reduce

class ConvexUpsampler(nn.Module):
    """
    Convex Upsampler to go from (bs, in_channels, H, W) to (bs, out_channels, H*K, W*K).
    Uses a 3x3 neighborhood convex combination.
    """
    def __init__(self, in_channels=64, out_channels=4, up_factor=16):
        super(ConvexUpsampler, self).__init__()
        self.up_factor = up_factor
        # 1) Predict weighting masks
        #    We want 9*(up_factor^2) weights per spatial position.
        self.weights_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 9 * (up_factor**2), kernel_size=1, stride=1, padding=0),
        )
        # 2) Predict coarse grasp maps (4 channels: pos, cos, sin, width)
        self.coarse_predictor = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: Coarse features of shape (bs, in_channels, H, W) 
               e.g. (bs, 64, 14, 14)

        Returns:
            upsampled: (bs, out_channels, H*K, W*K) 
                       e.g. (bs, 4, 224, 224)
        """
        bs, _, H, W = x.shape
        K = self.up_factor

        # 1) Predict weighting masks (bs, 9*K*K, H, W)
        weights = self.weights_predictor(x)
        # reshape to (bs, 9, K*K, H, W)
        weights = weights.view(bs, 9, K*K, H, W)
        # normalize across the 9 dimension so each 3x3 is a convex combination
        weights = F.softmax(weights, dim=1)  # (bs, 9, K*K, H, W)
        
        # 2) Predict coarse 4-channel grasp maps: (bs, 4, H, W)
        coarse_maps = self.coarse_predictor(x)
        
        # 3) Gather a 3x3 neighborhood of coarse_maps around each pixel
        #    - We'll use F.unfold to extract 3x3 patches over the entire feature map.
        #    - Then multiply by the weights and sum.
        # pad by 1 to safely gather 3x3 neighborhoods on edges
        coarse_maps_padded = F.pad(coarse_maps, (1,1,1,1), mode='replicate')
        
        # unfold => shape: (bs, out_channels*3*3, H*W)
        #   i.e. (bs, 4*9, 14*14) if out_channels=4
        unfolded = F.unfold(coarse_maps_padded, kernel_size=3, stride=1, padding=0) # (bs, 4*9, H*W)
        
        # reshape to (bs, out_channels, 9, H, W)
        unfolded = unfolded.view(bs, coarse_maps.shape[1], 9, H, W) # (bs, 4, 9, 14, 14) 

        # 4) Multiply by weights: 
        #    weights has shape (bs, 9, K*K, H, W)
        #    unfolded has shape (bs, out_channels, 9, H, W)
        # => broadcast across the '9' dimension and sum
        # after summing over 9, we get shape (bs, out_channels, K*K, H, W)
        upsampled = (unfolded.unsqueeze(3) * weights.unsqueeze(1)).sum(dim=2) # * (bs, 4, K*K, H, W)
        
        bs, c, kk, H_, W_ = upsampled.shape
        K = int(kk**0.5)  # e.g. if kk=256 -> K=16

        upsampled = upsampled.view(bs, c, K, K, H_, W_) # (bs, 4, 16, 16, 14, 14)
        # reorder dimensions -> (bs, out_channels, H*K, W*K)
        upsampled = upsampled.permute(0, 1, 4, 2, 5, 3).contiguous() # (bs, 4, 14, 16, 14, 16)
        upsampled = upsampled.view(bs, c, H_ * K, W_ * K) # (bs, 4, 224, 224)

        return upsampled

class FeatureMixerLayer(nn.Module):
    def __init__(self, num_token, token_dim, mlp_ratio):
        super().__init__()
        self.expanded_dim_t = int(num_token * mlp_ratio)
        self.mix_t = nn.Sequential(
            nn.LayerNorm(token_dim),
            Rearrange('b c n -> b n c'),
            nn.Linear(num_token, self.expanded_dim_t),
            nn.GELU(),
            nn.Linear(self.expanded_dim_t, num_token),
            Rearrange('b n c -> b c n'),
        )

        self.expanded_dim_c = int(token_dim * mlp_ratio)
        self.mix_c = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.expanded_dim_c),
            nn.GELU(),
            nn.Linear(self.expanded_dim_c , token_dim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # from (197, bs, 64) to (bs, 197, 64)

        # 2) Pass through the MLP-Mixer blocks (which each do their own rearrange internally)
        x = x + self.mix_t(x)
        x = x + self.mix_c(x)

        # 3) Permute back to (N, B, C) => (197, bs, 64)
        x = x.permute(1, 0, 2)
        return x

def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """ 
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses). 
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module. 
    """

    x_ = b.ln_1(x)
    q, k, v = F.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #  n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:


        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)
        
        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None,...]

        if attn_mask_type == 'all':
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]
        
    
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x


class CLIPSEG(nn.Module):
    def __init__(self,clip_version,reduce_dim=64,mlpmixer=False,stage1=False,stage2=False):
        super(CLIPSEG,self).__init__()

        self.clip_model,_ = clip.load(clip_version,device='cpu',jit=False)
        self.model = self.clip_model.visual
        self.stage1 = stage1
        self.stage2 = stage2
        
        self.version = clip_version

        self.extract_layers = (3,6,9) #(1,3,5,7,9) 
        self.cond_layer = 0
        self.rev_activations = False
        self.n_heads = 4

        self.depth = len(self.extract_layers)
        self.add_activations1 = True

        self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14), 'ViT-L/14': (16,16)}[self.version]
        self.path_shape = {'ViT-B/32': 32, 'ViT-B/16': 16, 'ViT-L/14': 14}[self.version]
        self.trans_conv_ks = {'ViT-B/32': (32, 32), 'ViT-B/16': (16, 16), 'ViT-L/14': (14,14)}[self.version]

        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        if(self.version=='ViT-B/16'):
            film_dim = 512
            self.reduce = nn.Linear(768,reduce_dim)
            self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(self.depth)])
        elif(self.version=='ViT-L/14'):
            film_dim = 768
            self.reduce = nn.Linear(1024,reduce_dim)
            self.reduces = nn.ModuleList([nn.Linear(1024, reduce_dim) for _ in range(self.depth)])
        
        self.film_mul = nn.Linear(film_dim,reduce_dim)
        self.film_add = nn.Linear(film_dim,reduce_dim)
        if(mlpmixer):
            self.blocks = nn.ModuleList([FeatureMixerLayer(num_token=677, token_dim=reduce_dim, mlp_ratio=4) for _ in range(len(self.extract_layers))])
        else:
            self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=self.n_heads) for _ in range(len(self.extract_layers))])
        self.conv1 = nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(reduce_dim)


        # Projections for same size output
        # self.mask_output = nn.ConvTranspose2d(reduce_dim, 1, self.trans_conv_ks, stride=self.trans_conv_ks)
        # self.pos_output = nn.ConvTranspose2d(reduce_dim, 1, self.trans_conv_ks, stride=self.trans_conv_ks)
        # self.cos_output = nn.ConvTranspose2d(reduce_dim, 1, self.trans_conv_ks, stride=self.trans_conv_ks)
        # self.sin_output = nn.ConvTranspose2d(reduce_dim, 1, self.trans_conv_ks, stride=self.trans_conv_ks)
        # self.width_output = nn.ConvTranspose2d(reduce_dim,1,self.trans_conv_ks,stride=self.trans_conv_ks)

        # Projections for (104,104) output
        self.mask_output = nn.ConvTranspose2d(reduce_dim, 1, kernel_size=8, stride=4, padding=0, output_padding=0)
        if(self.stage2 and not self.stage1):
            self.pos_output = nn.ConvTranspose2d(reduce_dim, 1, kernel_size=8, stride=4, padding=0, output_padding=0)
            self.cos_output = nn.ConvTranspose2d(reduce_dim, 1, kernel_size=8, stride=4, padding=0, output_padding=0)
            self.sin_output = nn.ConvTranspose2d(reduce_dim, 1, kernel_size=8, stride=4, padding=0, output_padding=0)
            self.width_output = nn.ConvTranspose2d(reduce_dim, 1, kernel_size=8, stride=4, padding=0, output_padding=0)
        #self.convex_upsampler = ConvexUpsampler(in_channels=reduce_dim, out_channels=4, up_factor=self.trans_conv_ks[0])

    def compute_conditional(self, conditional):

        dev = next(self.parameters()).device

        if type(conditional) in {list, tuple}:

            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)
                
        else:
            if conditional in self.precomputed_prompts:
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else:
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]
        
        return cond

    def rescaled_pos_emb(self, new_size):
        assert len(new_size) == 2
        
        if(self.version=='ViT-B/16'):
            pos_emb_size = 768
        else:
            pos_emb_size = 1024
        a = self.model.positional_embedding[1:].T.view(1, pos_emb_size, *self.token_shape)
        b = F.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(pos_emb_size, new_size[0]*new_size[1]).T
        return torch.cat([self.model.positional_embedding[:1], b])

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None):
        
        #print("Extract layers = ",extract_layers)
        #print("Length of clip model = ",len(self.model.transformer.resblocks))
        with torch.no_grad():

            inp_size = x_inp.shape[2:]

            x = self.model.conv1(x_inp)  # shape = [bs, width, grid, grid] width=768, grid=22

            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [bs, grid^2, width]
            
            x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            
            standard_n_tokens = 50 if self.model.conv1.kernel_size[0] == 32 else 197

            if x.shape[1] != standard_n_tokens: 
                new_shape = int(math.sqrt(x.shape[1]-1)) # = grid
                x = x + self.rescaled_pos_emb((new_shape, new_shape)).to(x.dtype)[None,:,:]
            else:
                x = x + self.model.positional_embedding.to(x.dtype)

            
            x = self.model.ln_pre(x) # Layer normalization
            
            
            x = x.permute(1, 0, 2)  # NLD -> LND

            activations, affinities = [], []
            
            for i, res_block in enumerate(self.model.transformer.resblocks):
                    # The input is processed through the clip's pipeline and activations and weights are extracted after every residual block.
                if mask is not None:
                    mask_layer, mask_type, mask_tensor = mask
                    if mask_layer == i or mask_layer == 'all':
                        size = int(math.sqrt(x.shape[0] - 1))
                        
                        attn_mask = (mask_type, F.interpolate(mask_tensor.unsqueeze(1).float(), (size, size)).view(mask_tensor.shape[0], size * size))
                        
                    else:
                        attn_mask = None
                else:
                    attn_mask = None
                
                # res_block is a residual attention block. 
                x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True, attn_mask=attn_mask)
                
                # x has a size of [grid^2+1,bs,768]
                # aff_per_head is the attention weights of dimension [192,grid^2+1,grid^2+1]
                #print("x shape = ",x.shape)
                if i in extract_layers:
                    
                    # if i is one of the layers whose weights are to be extracted
                    affinities += [aff_per_head]

                    activations += [x]
                    

                if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                    print('early skip')
                    break
                
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_post(x[:, 0, :])

            if self.model.proj is not None:
                x = x @ self.model.proj

            return x, activations, affinities
    
    def get_cond_vec(self, conditional, batch_size):
        # compute conditional from a single string
        #conditional is a tuple of strings, with size corresponding to the batch_size
        if conditional is not None and type(conditional) == str:
            cond = self.compute_conditional(conditional) # features extracted from CLIP
            cond = cond.repeat(batch_size, 1) # cond is now a tensor of size (bs,512) after extracting textual features from CLIP

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2:
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        
        return cond

    def forward(self, img, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # img.shape -> (bs,3,416,416)
        # word.shape -> (bs,512)
        bs,dev = img.shape[0],img.device
        cond = self.get_cond_vec(word,bs).to(dtype=img.dtype,device=dev) # (bs,512)

        visual_q, activations, _ = self.visual_forward(img, extract_layers=[0] + list(self.extract_layers)) # len(activations) = 1 + len(self.extract_layers)

        activation1 = activations[0]
        activations = activations[1:]

        _activations = activations[::-1] if not self.rev_activations else activations

        a = None

        
        for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)):

            # activation -> (197,bs,768)
            #print("Activation shape = ",activation.shape)
            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            if i == self.cond_layer:
                a = self.film_mul(cond) * a + self.film_add(cond) # heirarchical conditioning
            #print("a shape = ",a.shape)
            a = block(a)

        a = a[1:].permute(1,2,0) # (bs,reduce_dim,196)

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs,a.shape[1],size,size)

        a = F.relu(self.bn1(self.conv1(a))) # (bs,64,26,26)


        # if stage1, then just predict the mask output and set other maps to None
        # if stage 2 then predict mask output, use mask output as mask pooling for other maps

        if(self.stage1 and not self.stage2):
            pred = self.mask_output(a)
            grasp_qua_pred = None
            grasp_sin_pred = None
            grasp_cos_pred = None
            grasp_wid_pred = None
        elif(self.stage2 and not self.stage1):
            pred = self.mask_output(a)
            pred_copy = torch.sigmoid(pred.detach())
            pred_copy = (pred_copy>0.35).float()
            pred_copy = F.interpolate(pred_copy, size=(26, 26), mode='bilinear', align_corners=False)
            a_mask_pooled = a * pred_copy
            grasp_qua_pred = self.pos_output(a_mask_pooled)
            grasp_sin_pred = self.sin_output(a_mask_pooled)
            grasp_cos_pred = self.cos_output(a_mask_pooled)
            grasp_wid_pred = self.width_output(a_mask_pooled)

        if(self.training):
            # if stage 1 then use mask output to compute loss, set other losses to zero
            if(pred.shape[-2:] != mask.shape[-2:]):
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                if(self.stage2 and not self.stage1):
                    grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
                    grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
                    grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
                    grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()
            total_area = mask.shape[2] * mask.shape[3]
            coef = 1 - (mask.sum(dim=(2,3)) / total_area)

            # Generate weight
            weight = mask * 0.5 + 1

            loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
            if(self.stage2 and not self.stage1):
        
                grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
                grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
                grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
                grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)
                total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss

                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = grasp_qua_loss.item()
                loss_dict["m_sin"] = grasp_sin_loss.item()
                loss_dict["m_cos"] = grasp_cos_loss.item()
                loss_dict["m_wid"] = grasp_wid_loss.item()
            elif(self.stage1 and not self.stage2):
                total_loss = loss
                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = 0
                loss_dict["m_sin"] = 0
                loss_dict["m_cos"] = 0
                loss_dict["m_wid"] = 0
            if(self.stage1 and not self.stage2):
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), total_loss, loss_dict
            elif(self.stage2 and not self.stage1):
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
        else:
            if(self.stage1 and not self.stage2):
                return pred.detach(), mask
            elif(self.stage2 and not self.stage1):
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)



        # gt_mask = mask.float()  
        # # Interpolate
        # gt_mask_26 = F.interpolate(gt_mask, size=(26, 26), mode='bilinear', align_corners=False)
        

        # a_mask_pooled = a * gt_mask_26
        
        # pred  = self.mask_output(a) # (bs,1,104,104)
        # grasp_qua_pred = self.pos_output(a_mask_pooled) # (bs,1,104,104)
        # grasp_cos_pred = self.cos_output(a_mask_pooled) # (bs,1,104,104)
        # grasp_sin_pred = self.sin_output(a_mask_pooled) # (bs,1,104,104)
        # grasp_wid_pred = self.width_output(a_mask_pooled) # (bs,1,104,104)

        # #pred = self.mask_output(a)
        # # upsampled_pred = self.convex_upsampler(a)
        # # grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = torch.split(upsampled_pred, 1, dim=1)

        # if(self.training):
        #     if pred.shape[-2:] != mask.shape[-2:]: # mask is of shape (bs,1,416,416)
        #         mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
        #         grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
        #         grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
        #         grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
        #         grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()
            
        #     # Ratio Augmentation
        #     total_area = mask.shape[2] * mask.shape[3]
        #     coef = 1 - (mask.sum(dim=(2,3)) / total_area)

        #     # Generate weight
        #     weight = mask * 0.5 + 1

        #     loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
        #     grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
        #     grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
        #     grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
        #     grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

        #     # @TODO adjust coef of different loss items
        #     total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss

        #     loss_dict = {}
        #     loss_dict["m_ins"] = loss.item()
        #     loss_dict["m_qua"] = grasp_qua_loss.item()
        #     loss_dict["m_sin"] = grasp_sin_loss.item()
        #     loss_dict["m_cos"] = grasp_cos_loss.item()
        #     loss_dict["m_wid"] = grasp_wid_loss.item()

        #     # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
        #     # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

        #     return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
        # else:
        #     return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

