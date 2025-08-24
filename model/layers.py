import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x
    
class MultiTaskProjectorPP(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3,stage1=False,stage2=False,use_gt_obj_masks=False):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.stage1 = stage1
        self.stage2 = stage2
        self.use_gt_obj_masks = use_gt_obj_masks

        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            )

        self.vis_mask = nn.Conv2d(in_dim, in_dim, 1)
        if(self.stage2 and not self.stage1):
            self.vis_grasp = nn.Conv2d(in_dim, in_dim*4, 1)
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word,mask):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        gt_mask = mask.detach().clone()
        # print("GT mask shape", gt_mask.shape) (bs,1,416,416)
        # print("GT mask contents", torch.unique(gt_mask,return_counts=True)) # (More 0s than 1s)
        x = self.vis(x)
        # print("x shape", x.shape) # (bs,512,104,104)
        # print("x contents", torch.unique(x,return_counts=True)) # (
        # import sys
        # sys.exit(0)
        

        x_mask = self.vis_mask(x)
        
        B, C, H, W = x_mask.size()
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)

        # print("weight shape", weight.shape) # (bs,512,3,3)
        # print("weight contents", torch.unique(weight,return_counts=True)) # (More 0s than 1s)
        # print("bias shape", bias.shape) # (bs,512)
        # print("bias contents", torch.unique(bias,return_counts=True)) # (More
        

        x_mask = x_mask.reshape(1, B * C, H, W)
        mask_out = F.conv2d(x_mask,
                    weight,
                    padding=self.kernel_size // 2,
                    groups=weight.size(0),
                    bias=bias)
        # print("mask_out shape", mask_out.shape) # (bs,512,104,104)
        # print("mask_out contents", torch.unique(mask_out,return_counts=True)) # (More
        
        mask_out = mask_out.transpose(0, 1) # (bs,1,104,104)

        if(self.stage2 and not self.stage1):
            #print("Is this called????")
            x_grasp = self.vis_grasp(x)
            x_grasp = torch.tensor_split(x_grasp, 4, dim=1)
            grasp_qua_x = x_grasp[0]
            grasp_sin_x = x_grasp[1]
            grasp_cos_x = x_grasp[2]
            grasp_wid_x = x_grasp[3]

            mask_out_copy = torch.sigmoid(mask_out.detach())
            mask_out_copy = (mask_out_copy>0.35).float()

            if(self.use_gt_obj_masks):
                # If using GT object masks, use the gt_mask as mask_out_copy
                mask_out_copy = gt_mask.bool().float()

            # import sys
            # sys.exit(0)
            # else:
                # print("Using GT object masks")
            # print("mask_out_copy contents", torch.unique(mask_out_copy,return_counts=True)) More 0s than 1s
            # print("mask_out_copy shape", mask_out_copy.shape) # (bs,1,104,104)
            mask_out_104 = F.interpolate(mask_out_copy, size=(104, 104), mode='bilinear', align_corners=False)
            #print("mask_out_copy shape", mask_out_104.shape) # (bs,1,104,104)
            #print("mask_out_copy contents", torch.unique(mask_out_104,return_counts=True)) # (More 0s than 1s)
            #mask_out = mask_out_104
            grasp_qua_x = grasp_qua_x * mask_out_104
            grasp_sin_x = grasp_sin_x * mask_out_104
            grasp_cos_x = grasp_cos_x * mask_out_104
            grasp_wid_x = grasp_wid_x * mask_out_104

            grasp_qua_x = grasp_qua_x.reshape(1, B * C, H, W)
            grasp_sin_x = grasp_sin_x.reshape(1, B * C, H, W)
            grasp_cos_x = grasp_cos_x.reshape(1, B * C, H, W)
            grasp_wid_x = grasp_wid_x.reshape(1, B * C, H, W)

            grasp_qua_out = F.conv2d(grasp_qua_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
            grasp_sin_out = F.conv2d(grasp_sin_x,
                                weight,
                                padding=self.kernel_size // 2,
                                groups=weight.size(0),
                                bias=bias)

            grasp_cos_out = F.conv2d(grasp_cos_x,
                                weight,
                                padding=self.kernel_size // 2,
                                groups=weight.size(0),
                                bias=bias)
            
            grasp_wid_out = F.conv2d(grasp_wid_x,
                                weight,
                                padding=self.kernel_size // 2,
                                groups=weight.size(0),
                                bias=bias)
                
            
            grasp_qua_out = grasp_qua_out.transpose(0, 1)
            grasp_sin_out = grasp_sin_out.transpose(0, 1)
            grasp_cos_out = grasp_cos_out.transpose(0, 1)
            grasp_wid_out = grasp_wid_out.transpose(0, 1)
        else:
            grasp_qua_out,grasp_sin_out, grasp_cos_out, grasp_wid_out = None,None,None,None
            

            # b, 1, 104, 104

        return mask_out, grasp_qua_out, grasp_sin_out, grasp_cos_out, grasp_wid_out


class MultiTaskProjector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3,use_max_pool=False):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.use_max_pool = use_max_pool
        # visual projector
        # if(self.use_max_pool is True):
        #     self.vis = nn.Sequential(  # os16 -> os4
        #         nn.Upsample(scale_factor=2, mode='bilinear'),
        #         conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
        #         nn.Upsample(scale_factor=2, mode='bilinear'),
        #         conv_layer(in_dim * 2, in_dim, 3, padding=1),
        #         nn.Conv2d(in_dim, in_dim*4, 1))
        # else:

        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim*5, 1))

        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word,infer_mask=None):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        #if(self.use_max_pool==False):
        x = torch.tensor_split(x, 5, dim=1) # no tensor_split api in torch 1.7, please use it in higher version
        mask_x = x[0]
        grasp_qua_x = x[1]
        grasp_sin_x = x[2]
        grasp_cos_x = x[3]
        grasp_wid_x = x[4]
        B, C, H, W = mask_x.size()
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)

        mask_x = mask_x.reshape(1, B * C, H, W)
        mask_out = F.conv2d(mask_x,
                    weight,
                    padding=self.kernel_size // 2,
                    groups=weight.size(0),
                    bias=bias)
        mask_out = mask_out.transpose(0, 1)
            #print("mask_out shape", mask_out.shape) # (bs,1,104,104)
            #print(torch.unique(mask_out,return_counts=True))
        #else:
            # infer_mask = infer_mask.float()
            # infer_mask = F.interpolate(infer_mask, size=(104, 104), mode='bilinear', align_corners=False)
            # x = torch.tensor_split(x,4,dim=1)
            # mask_out = infer_mask
            # ##print("mask_out shape", mask_out.shape) # (bs,1,104,104)
            # #print(torch.unique(mask_out,return_counts=True))
            # grasp_qua_x = x[0]
            # grasp_sin_x = x[1]
            # grasp_cos_x = x[2]
            # grasp_wid_x = x[3]
            # B, C, H, W = grasp_qua_x.size()
            # word = self.txt(word)
            # weight, bias = word[:, :-1], word[:, -1]
            # weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)

        
        
        #print("mask_out shape", mask_out.shape) # (bs,1,104,104)
        #gt_mask = mask_out
        #print("gt mask shape", gt_mask.shape)

        if(self.use_max_pool):
            #infer_mask = infer_mask.float()
            #infer_mask = F.interpolate(infer_mask, size=(104, 104), mode='bilinear', align_corners=False)
            # create a copy of mask_out that doesnt require grad
            #mask_out_copy = mask_out.clone()
             # creates a copy of mask_out that doesn't require grad. What happens to mask_out? Does it still require grad? 
            mask_out_copy = torch.sigmoid(mask_out.detach())
            mask_out_copy = (mask_out_copy>0.35).float()
            mask_out_104 = F.interpolate(mask_out_copy, size=(104, 104), mode='bilinear', align_corners=False)
            #print("mask_out_104 contents", torch.unique(mask_out_104,return_counts=True))
            grasp_qua_x = grasp_qua_x * mask_out_104
            grasp_sin_x = grasp_sin_x * mask_out_104
            grasp_cos_x = grasp_cos_x * mask_out_104
            grasp_wid_x = grasp_wid_x * mask_out_104
            #gt_mask = infer_mask.float()  
            # Interpolate
            #gt_mask_104 = F.interpolate(gt_mask, size=(104, 104), mode='bilinear', align_corners=False)
            # Multiply onto the grasp channels (optional: threshold or clamp if needed)
            # grasp_qua_x = grasp_qua_x * infer_mask
            # grasp_sin_x = grasp_sin_x * infer_mask
            # grasp_cos_x = grasp_cos_x * infer_mask
            # grasp_wid_x = grasp_wid_x * infer_mask

        


        # 1, b*256, 104, 104
        
        grasp_qua_x = grasp_qua_x.reshape(1, B * C, H, W)
        grasp_sin_x = grasp_sin_x.reshape(1, B * C, H, W)
        grasp_cos_x = grasp_cos_x.reshape(1, B * C, H, W)
        grasp_wid_x = grasp_wid_x.reshape(1, B * C, H, W)


        
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        
        
        grasp_qua_out = F.conv2d(grasp_qua_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
        grasp_sin_out = F.conv2d(grasp_sin_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)

        grasp_cos_out = F.conv2d(grasp_cos_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
        grasp_wid_out = F.conv2d(grasp_wid_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
            
        
        grasp_qua_out = grasp_qua_out.transpose(0, 1)
        grasp_sin_out = grasp_sin_out.transpose(0, 1)
        grasp_cos_out = grasp_cos_out.transpose(0, 1)
        grasp_wid_out = grasp_wid_out.transpose(0, 1)
        # b, 1, 104, 104

        return mask_out, grasp_qua_out, grasp_sin_out, grasp_cos_out, grasp_wid_out


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
            -1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in