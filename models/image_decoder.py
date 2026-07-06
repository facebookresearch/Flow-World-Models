import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, *, dropout=0.0, num_groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, num_groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x)


class DINOv3ImageDecoder(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        decoder_cfg = cfg_model.decoder
        in_channels = int(decoder_cfg.input_dim)
        hidden_dim = int(decoder_cfg.hidden_dim)
        num_res_blocks = int(decoder_cfg.num_res_blocks)
        dropout = float(decoder_cfg.dropout)
        num_groups = int(decoder_cfg.num_groups)
        upsample_channels = list(decoder_cfg.upsample_channels)

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, dropout=dropout, num_groups=num_groups)
                for _ in range(num_res_blocks)
            ]
        )

        upsample_blocks = []
        current_channels = hidden_dim
        for next_channels in upsample_channels:
            upsample_blocks.append(
                UpsampleBlock(
                    current_channels, int(next_channels), num_groups=num_groups
                )
            )
            upsample_blocks.append(
                ResidualBlock(
                    int(next_channels), dropout=dropout, num_groups=num_groups
                )
            )
            current_channels = int(next_channels)
        self.upsampler = nn.Sequential(*upsample_blocks)

        self.output_head = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(current_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image_latents):
        x = self.input_proj(image_latents)
        x = self.res_blocks(x)
        x = self.upsampler(x)
        reconstructed_images = self.output_head(x)
        return {"reconstructed_images": reconstructed_images}
