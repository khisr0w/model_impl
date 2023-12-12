""" Link (more or less) to the original implementations:
    - https://github.com/huggingface/pytorch-image-models
    - https://www.youtube.com/watch?v=ovB0ddFtzzA

    Original Paper:
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
      Link: https://arxiv.org/abs/2010.11929
"""

import numpy as np
import timm
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):

    """
    Parameters:
        - img_size : int
        - patch_size : int
        - in_chans : int
        - embed_dim : int

    Internal Attributes:
        - n_patches : int
        - proj : nn.Conv2d
            Conv for splitting into patches and embedding
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # NOTE(Abid): Conv2d is the hybrid architecture proposal of the paper,
        #             the original version has a MLP instead.
        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

    def forward(self, x):
        """
            Parameters:
                - x : torch.Tensor
                    Shape (batch, in_chans, img_size, img_size)

            Return: torch.Tensor
                    Shape (batch, n_patches, embed_dim)
        """
        x = self.proj(x) # Shape (batch, embed_dim, n_patches ** 0.5, n_patches ** 5)
        x = x.flatten(2) # Shape (batch, embed_dim, n_patches)
        x = x.transpose(1, 2) # Shape (batch, n_patches, embed_dim)

        return x

class MultiHeadAttention(nn.Module):
    """
        Parameters:
            - dim : int
                input and output dim of per token features
            - n_heads : int
            - qkv_bias : bool
            - attn_p : float
                Dropout probability for qkv
            - proj_p : float
                Dropout probability for output

        Attributes:
            - scale : float
                For scaled dot product
            - qkv : nn.Linear
            - proj : nn.Linear
            - attn_drop, proj_drop : nn.Dropout
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0, proj_p=0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
            Parameters:
                - x : torch.Tensor
                      Shape (batch, n_patches + 1, dim)

            Returns:
                - out : torch.Tensor
                        Shape (batch, n_patches + 1, dim)
        """
        batch, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # Shape : (batch, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(batch, n_tokens, 3, self.n_heads, self.head_dim) # Shape : (batch, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # Shape : (3, batch, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) # Shape : (batch, n_heads, head_dim, n_patches+1)
        dot = (q @ k_t) * self.scale # Shape : (batch, n_heads, n_patches+1, n_patches+1)
        attention = dot.softmax(dim=-1)
        attention = self.attn_drop(attention)

        weighted_avg = attention @ v # Shape : (batch, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # Shape : (batch, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # Shape : (batch, n_patches+1, dim)

        x = self.proj(weighted_avg) # Shape : (batch, n_patches+1, dim)
        x = self.proj_drop(x)

        return x

class FeedForward(nn.Module):
    """
        Parameters:
            - in_features: int
            - hidden_features : int
            - out_features : int
            - p : float
                Dropout

        Attributes:
            - fc : nn.Linear
            - act : nn.GELU
                Activation
            - fc2 : nn.Linear
            - drop : nn.Dropout
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
            Parameters:
                - x : torch.Tensor
                      Shape (batch, n_patches +1, in_features)

            Returns:
                - torch.Tensor
                  Shape (batch, n_patches +1, out_features)
        """

        x = self.fc1(x) # Shape (batch, n_patches +1, hidden_features)
        x = self.act(x) 
        x = self.drop(x)
        x = self.fc2(x) # Shape (batch, n_patches +1, out_features)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """
        Parameters:
            - dim : int
                Embedding
            - n_heads : int
            - mlp_ratio : float
                Determines the hidden dimension size of the MLP wrt dim
            - qkv_bias : bool
            - p, attn_p : float

        Attributes:
            - norm1, norm2 : LayerNorm
            - attention : Attention
            - mlp : MLP
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mha = MultiHeadAttention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.ff = FeedForward(in_features=dim,
                              hidden_features=hidden_features,
                              out_features=dim)

    def forward(self, x):
        """
            Parameters
                - x : torch.Tensor
                      Shape (batch, n_patches + 1, dim)
            Returns
                - torch.Tensor
                  Shape (batch, n_patches + 1, dim)
        """

        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    """
        Parameters:
            - img_size : int
                Square image assumed
            - patch_size: int
            - in_chans : int
            - n_classes : int
            - embed_dim : int
            - depth : int
                Number of blocks
            - mlp_ratio : float
            - qkv_bias : bool
            - p, attn_p : float

        Attributes:
            - patch_embed : PatchEmbed
            - cls_token : nn.Parameter
                Prepended parameter used for classification, the first one in order
            - pos_emb : nn.Parameter
            - pos_drop : nn.Dropout
            - blocks: nn.ModuleList
            - norm : nn.LayerNorm
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768,
                 depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.Sequential(*[Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # NOTE(Abid): According to the original paper, the MLP here should have a single hidden layer in training phase.
        self.pred_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
            Parameters:
                - x : torch.Tensor
                      Shape (batch, in_chans, img_size, img_size)

            Returns:
                - logits : torch.Tensor
                           Shape (batch, n_classes)
        """

        batch = x.shape[0]
        x = self.patch_embed(x) # (batch, n_patches, embed_dim)
        cls_token = self.cls_token.expand(batch, -1, -1) # Shape (batch, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # Shape (batch, n_patches + 1, embed_dim)
        x = x + self.pos_emb
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_token_final = x[:, 0] # Shape (batch, embed_dim) --> Only the first patch is used
        x = self.pred_head(cls_token_final) # (batch, n_classes)

        return x

# --------| Verifying the Model |----------
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()

print(type(model_official))

custom_config = {
    "img_size" : 384,
    "patch_size" : 16,
    "in_chans" : 3,
    "embed_dim" : 768,
    "depth" : 12,
    "n_heads" : 12,
    "qkv_bias" : True,
    "mlp_ratio" : 4
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()

for (n_o, p_o), (n_c, p_c) in zip (model_official.named_parameters(),
                                   model_custom.named_parameters()):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_c.data, p_o.data)

inputs = torch.rand(1, 3, 384, 384)
res_c = model_custom(inputs)
res_o = model_official(inputs)

assert get_n_params(model_custom) == get_n_params(model_official)
assert_tensors_equal(res_c, res_o)

torch.save(model_custom, "vit_model.pth")


# ---------| Inference |----------
from PIL import Image

k = 10

imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("vit_model.pth")
model.eval()

img = (np.array(Image.open("elephant.jpg")) / 128) - 1
inputs = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32) # Shape (1, in_chans, img_size, img_size)
logits = model(inputs) # Shape (1, n_classes)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")
