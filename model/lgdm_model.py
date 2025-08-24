import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from ruamel.yaml import YAML
yaml = YAML(typ='safe')  # or 'rt' for round-trip parsing

# If your repository has a module `inference.models.grasp_model`, import from there:

import model.lgdm.albef.utils as utils
from model.lgdm.albef.models.tokenization_bert import BertTokenizer
from model.lgdm.albef.models.model_retrieval import ALBEF

###############################################################################
# Helper structures - adjust as needed
###############################################################################
filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

class LanguageGraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(LanguageGraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc, prompt, query):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, prompt, query)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

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


class LGDM(LanguageGraspModel):
    """
    LGDM model that integrates a diffusion pipeline
    + ALBEF-based text/image encoding
    + Basic conv/transposed-conv layers for generating grasp predictions.
    """
    def __init__(self, 
                 input_channels=4, 
                 output_channels=1, 
                 channel_size=32, 
                 dropout=False, 
                 prob=0.0, 
                 clip_version='ViT-B/32'):
        super().__init__()
        
        # ---------------------------------------------------------------------
        # Convolutional encoder/decoder backbone
        # ---------------------------------------------------------------------
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)

        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], 
                                         stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], 
                                         stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], 
                                         stride=strides[5], padding=5, output_padding=1)

        # ---------------------------------------------------------------------
        # Text features
        # ---------------------------------------------------------------------
        # CLIP as an additional (possibly unused) language model
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version)

        # ALBEF initialization
        self._init_albef()  # sets up self.albef, self.tokenizer

        # Flatten + MLP for text features
        self.y_flatten = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 2888),
            nn.GELU(),
        )

        # ---------------------------------------------------------------------
        # Output heads
        # ---------------------------------------------------------------------
        self.mask_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.pos_output   = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output   = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output   = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        # ---------------------------------------------------------------------
        # Weights initialization
        # ---------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def _init_albef(self):
        """
        Initialize the ALBEF model and BERT tokenizer for grounding 
        or attention-based feature extraction.
        """
        import argparse

        class WrapperArgument:
            def __init__(self):
                pass
            def add_attribute(self, name, value):
                setattr(self, name, value)

        args = WrapperArgument()
        args.config = '/data/vineet/workspace/CROG/model/lgdm/albef/configs/Grounding.yaml'
        args.gradcam_mode = 'itm'
        args.block_num = 8
        args.text_encoder = 'bert-base-uncased'
        args.device = 'cuda'
        args.world_size = 1
        args.dist_url = 'env://'
        args.distributed = True

        utils.init_distributed_mode(args)

        with open(args.config, 'r') as f:
            config = yaml.load(f)
        self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
        self.albef = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=self.tokenizer)

    def forward(self, x, img, t, query, alpha, idx, prompt=None):
        """
        Forward pass:
          x: (B,1,H,W)    # e.g. noise / partial shape from the diffusion
          img: (B,3,224,224)
          t: Timestep in diffusion
          query: List[str] textual queries
          alpha, idx: Additional controls for the diffusion steps

        Returns:
            pos_output: 
        """
        # 1) Encode text with ALBEF
        device = img.device
        text_input = self.tokenizer(query, padding='longest', max_length=30, return_tensors="pt").to(device)
        image_atts, y = self.albef(img, text_input, alpha, idx)

        # 2) Process attention
        self.full_image_atts = self._process_attention_mask(image_atts).to(device)

        # 3) Combine the image with attention
        # e.g. multiply each channel by the attention mask
        r_channel, g_channel, b_channel = img[:, 0], img[:, 1], img[:, 2]
        r_channel = r_channel * self.full_image_atts
        g_channel = g_channel * self.full_image_atts
        b_channel = b_channel * self.full_image_atts
        img = torch.cat([r_channel.unsqueeze(1), g_channel.unsqueeze(1), b_channel.unsqueeze(1)], dim=1)

        # 4) Flatten text embedding
        y = y[:, 0].unsqueeze(1)  # shape (B,1,768)? or (B,1,x)
        y = self.y_flatten(y)
        y = y.view(-1, 8, 19, 19)  # example shape from your code

        # 5) Forward CNN
        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))

        # (Optionally) add textual features if you want:
        # img = torch.clone(img).detach() + y

        img = F.relu(self.convt1(img))
        img = F.relu(self.convt2(img))
        img = F.relu(self.convt3(img))

        # 6) Prediction heads
        mask_output = self.mask_output(img)
        pos_output   = self.pos_output(img)
        cos_output   = self.cos_output(img)
        sin_output   = self.sin_output(img)
        width_output = self.width_output(img)

        # For diffusion guidance, store them as attributes
        self.guiding_point = pos_output
        self.mask_output_str = mask_output
        self.pos_output_str   = pos_output
        self.cos_output_str   = cos_output
        self.sin_output_str   = sin_output
        self.width_output_str = width_output

        # The final output from your forward pass can be the pos_output 
        # if the diffusion pipeline expects that as a "sample".
        # Often you'll return pos_output + x to incorporate the diffusion noise:
        pos_output = x + pos_output

        return pos_output

    def compute_loss(self, yc, mask_pred,pos_pred, cos_pred, sin_pred, width_pred):
        """
        Compute MSE losses between predicted fields and ground truth (yc).
          yc: [y_pos, y_cos, y_sin, y_width]
        """
        y_mask,y_pos, y_cos, y_sin, y_width = yc

        mask_loss = F.mse_loss(mask_pred,y_mask)
        p_loss     = F.mse_loss(pos_pred,   y_pos)
        cos_loss   = F.mse_loss(cos_pred,   y_cos)
        sin_loss   = F.mse_loss(sin_pred,   y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': mask_loss+p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'mask_loss': mask_loss,
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
            },
            'pred': {
                'mask': mask_pred,
                'pos':   pos_pred,
                'cos':   cos_pred,
                'sin':   sin_pred,
                'width': width_pred
            }
        }

    def _get_contrastive_loss(self, x, y, temperature=1.0):
        """
        Example contrastive loss (not fully used in code).
        """
        x = F.normalize(x, dim=0, p=2)
        y = F.normalize(y, dim=0, p=2)
        similarity = F.cosine_similarity(x, y, dim=0) / temperature
        contrastive_loss = -torch.log(torch.exp(similarity) / torch.sum(torch.exp(similarity), dim=0))
        return torch.mean(contrastive_loss)

    def _process_attention_mask(self, image_atts):
        """
        Convert the 14x14 patch-level attention into a 224x224 mask.
        """
        bs, _ = image_atts.data.shape
        W, H, w, h = 224, 224, 14, 14
        ps = W // w
        image_atts = image_atts[:, 1:].view(-1, w, h).float()  # remove CLS token if needed

        full_image_atts = torch.zeros(bs, W, H)
        for i in range(bs):
            for j in range(w):
                for k in range(h):
                    full_image_atts[i, j*16:(j+1)*16, k*16:(k+1)*16] = image_atts[i, j, k]
        return full_image_atts

    def _load_and_freeze_clip(self, clip_version, device=None):
        """
        Load CLIP in eval mode and freeze parameters.
        """
        clip_model, clip_preprocess = clip.load(clip_version, device=device, jit=False)
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def _encode_text(self, raw_text, device=None):
        """
        Basic function to tokenize and encode text with CLIP.
        (Potentially not used if you're using ALBEF instead.)
        """
        max_text_len = 20
        default_context_length = 77
        context_length = max_text_len + 2
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
        zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], 
                               dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)
        return self.lang_model.encode_text(texts).float()

