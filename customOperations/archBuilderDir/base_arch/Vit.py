import torch
import torch.nn as nn

# Patch Embedding Layer for ViT
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=64, img_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Learnable projection layer that converts the input image into patches
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional embedding (to keep positional awareness)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)  # Apply convolution to create patches
        x = x.flatten(2).transpose(1, 2)  # Flatten patches into shape: [B, num_patches, emb_dim]
        x = x + self.pos_embed  # Add learnable positional embedding
        return x

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Apply attention and MLP blocks with residual connections
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16, emb_dim=64, depth=6, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_dim=emb_dim, img_size=img_size)
        
        # Transformer encoder blocks, stacked `depth` times
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.patch_embed(x)  # Convert image to patch embeddings
        x = self.transformer(x)  # Apply transformer blocks
        return x

# ViT Decoder (Upsampling to match original resolution)
class ViTDecoder(nn.Module):
    def __init__(self, emb_dim=64, out_channels=3):
        super().__init__()
        
        # Decoder uses transposed convolutions (deconvolution) to upsample the features back
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decode(x)

# Complete ViT Model (Encoder-Decoder Architecture)
class VisionTransformer(nn.Module):
    def __init__(self, img_size=512, patch_size=16, emb_dim=64, depth=6, heads=4, mlp_dim=128, out_channels=3):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, emb_dim, depth, heads, mlp_dim)
        self.decoder = ViTDecoder(emb_dim, out_channels)

    def forward(self, x):
        # Encoder outputs patch embeddings
        enc = self.encoder(x)  # [B, num_patches, emb_dim]
        
        # Decoder processes the embeddings to reconstruct the image
        out = self.decoder(enc)  # [B, 3, H, W]
        
        return out
