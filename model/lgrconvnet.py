from .layers import ResidualBlock
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

class LGRCONVNET(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, channel_size=32, dropout=False, prob=0.0, clip_version='ViT-B/16',training=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.y_flatten = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 104),
            nn.GELU(),
        )

        self.mask_output = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=2)
        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.training = training
        self.dropout_mask = nn.Dropout(p=prob)
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # Setup language modality
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def _load_and_freeze_clip(self, clip_version, device=None):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def _encode_text(self, raw_text, device=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1) # [bs, default_context_length]
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.lang_model.encode_text(texts).float()
    
    def forward(self, img, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''

        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Encode Text
        device = x.device
        y_feats = self._encode_text(word, device=device)
        #y_feats = word.float().to(device)
        y_feats = self.y_flatten(y_feats) # (bs, 104)

        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 104).unsqueeze(1).expand(-1,128, -1, -1) # (bs, 128, 104, 104)

        # combine textual features with visual features
        x = torch.clone(x).detach() + y_feats

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x) # (bs, 32, 417, 417)
        
        if(self.dropout):
            pred = self.mask_output(self.dropout_mask(x))
            grasp_qua_pred = self.pos_output(self.dropout_pos(x))
            grasp_cos_pred = self.cos_output(self.dropout_cos(x))
            grasp_sin_pred = self.sin_output(self.dropout_sin(x))
            grasp_wid_pred = self.width_output(self.dropout_wid(x))
        else:
            pred = self.mask_output(x)
            grasp_qua_pred = self.pos_output(x)
            grasp_cos_pred = self.cos_output(x)
            grasp_sin_pred = self.sin_output(x)
            grasp_wid_pred = self.width_output(x)
        
        #print('pred', pred.shape) # (bs,1,104,104)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
                grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
                grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
                grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()

            # Ratio Augmentation
            total_area = mask.shape[2] * mask.shape[3]
            coef = 1 - (mask.sum(dim=(2,3)) / total_area)

            # Generate weight
            weight = mask * 0.5 + 1

            loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
            grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
            grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
            grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
            grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

            # @TODO adjust coef of different loss items
            total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss

            loss_dict = {}
            loss_dict["m_ins"] = loss.item()
            loss_dict["m_qua"] = grasp_qua_loss.item()
            loss_dict["m_sin"] = grasp_sin_loss.item()
            loss_dict["m_cos"] = grasp_cos_loss.item()
            loss_dict["m_wid"] = grasp_wid_loss.item()

            # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
            # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

            return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
        else:
            return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)



