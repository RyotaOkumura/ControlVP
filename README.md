# vanishing_point
### memo
- python 3.11.8
- pip 24.0
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- git clone https://github.com/huggingface/diffusers
- https://huggingface.co/docs/diffusers/training/controlnet を参考にセットアップ



12L4oy5Y8Dk5ESuMix-KdDQgUGaEO4oUE

12L4oy5Y8Dk5ESuMix-KdDQgUGaEO4oUE
SD2_Finetune.ckpt

import gdown

url = 'https://drive.google.com/uc?id=12L4oy5Y8Dk5ESuMix-KdDQgUGaEO4oUE'
out_path = 'SD2_Finetune.ckpt'
gdown.download(url, out_path, quiet=False)
