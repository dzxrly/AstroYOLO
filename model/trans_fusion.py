import torch
import torch.nn as nn


class TransMLP(nn.Module):
    def __init__(self, in_feature, out_feature, expand_dim_ratio=4):
        super(TransMLP, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(
                in_features=in_feature,
                out_features=int(in_feature * expand_dim_ratio)
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(in_feature * expand_dim_ratio),
                out_features=out_feature
            )
        )

    def forward(self, x):
        return self.feedforward(x)


class TransFusion(nn.Module):
    def __init__(self, input_channel, hidden_feature, head_num=4, dropout=0.2, trans_block=4):
        super(TransFusion, self).__init__()
        assert trans_block >= 1, '[Error]: Transformer block number must >= 1 !'
        self.linear_projection = nn.Conv2d(input_channel, hidden_feature, 1)
        self.fusion_mhsa = nn.MultiheadAttention(
            embed_dim=hidden_feature,
            num_heads=head_num,
            dropout=dropout,
            vdim=hidden_feature * 2,
            batch_first=True
        )
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_feature,
            num_heads=head_num,
            dropout=dropout,
            batch_first=True
        )
        self.feature_transform = nn.Linear(hidden_feature * 2, hidden_feature)
        self.input_norm = nn.LayerNorm(hidden_feature)
        self.fusion_mhsa_norm = nn.LayerNorm(hidden_feature * 2)
        self.mhsa_norm = nn.LayerNorm(hidden_feature)
        self.ffn_norm = nn.LayerNorm(hidden_feature)
        self.ffn = TransMLP(hidden_feature, hidden_feature)
        self.mhas_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.trans_block = trans_block

    def forward(self, inputs_1, inputs_2):
        b, c, h, w = inputs_1.shape
        inputs_1 = self.linear_projection(inputs_1)
        inputs_2 = self.linear_projection(inputs_2)
        inputs_1 = inputs_1.flatten(2).permute(0, 2, 1)  # b, hidden_c, h, w -> b, h*w, hidden_c
        inputs_2 = inputs_2.flatten(2).permute(0, 2, 1)  # b, hidden_c, h, w -> b, h*w, hidden_c
        inputs = torch.cat([inputs_1, inputs_2], dim=2)
        res = self.feature_transform(inputs)  # hidden_c * 2 -> hidden_c
        inputs = self.fusion_mhsa_norm(inputs)
        inputs, _ = self.fusion_mhsa(self.input_norm(inputs_1), self.input_norm(inputs_2), inputs)  # hidden_c
        inputs = self.ffn_norm(inputs + res)
        res = inputs
        inputs = self.ffn(inputs) + res
        for _ in range(self.trans_block - 1):
            res = inputs
            inputs = self.mhsa_norm(inputs)
            inputs, _ = self.mhsa(inputs, inputs, inputs)
            inputs = self.ffn_norm(inputs + res)
            res = inputs
            inputs = self.ffn(inputs) + res
        inputs = inputs.permute(0, 2, 1).reshape(b, c * 2, h, w)  # b, h*w, hidden_c -> b, hidden_c, h, w
        return inputs
