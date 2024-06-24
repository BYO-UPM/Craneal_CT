import monai
import torch
from monai.networks.nets import UNETR
from monai.losses import DiceLoss



model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(512, 512, 64),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed='perceptron',
    norm_name='instance',
    res_block=True,
    dropout_rate=0.0,
)

loss = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

