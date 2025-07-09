'''
nohup python preprocess/extract-t5.py 0 2 >> t5.log 2>&1 &
nohup python preprocess/extract-t5.py 1 2 >> t5.log 2>&1 &

0/1: What part does this process do
2:   It consists of two parts in total.
'''
txt_path="../data/train_data_prompts.txt"
opt_root="/mnt/data/mjc/temp"
checkpoint_dir="/mnt/data/mjc/Index-anisora"
import os
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
num_processes = int(os.environ.get("GPUS_PER_NODE", "8"))
from tqdm import tqdm
import os

meta_root="/mnt/data/mjc/processed/metadata"
all_video=[(os.path.join(meta_root,ii,"metadata.npz"),ii) for ii in os.listdir(meta_root)]
all_video=sorted(all_video,key=lambda x:x[1])
if local_rank==0:
    print(all_video[:3])
all_video=all_video[35000:35500]
num_each_rank = len(all_video) // num_processes+1
metadatas = all_video[round(local_rank * num_each_rank) : round((local_rank + 1) * num_each_rank)]

import pdb,torch
from wan.modules.t5 import T5EncoderModel
print(local_rank)
text_encoder = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=torch.device('cpu'),
    checkpoint_path=os.path.join(checkpoint_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
    tokenizer_path=os.path.join(checkpoint_dir, 'google/umt5-xxl'),
    shard_fn=None,
)
text_encoder.model=text_encoder.model.to(local_rank)
context = text_encoder(["A cartoon character is talking. High quality 2D cartoon animation."], local_rank)[0].cpu()#torch.Size([138, 4096])#"In this scene, a man with a beard is seen tending to a woman who lies in bed, her face illuminated by the soft glow of a nearby light source. The man, dressed in a blue robe adorned with intricate designs, holds a bowl, possibly containing a healing potion or a magical elixir. The woman, clad in a pink garment, appears to be resting or possibly unwell, as she lies on her side with her eyes closed. The setting suggests a historical or medieval context, with the dimly lit room and the man's attire evoking a sense of timelessness and mystery. "

torch.save(context,"/mnt/data/mjc/Index-anisora/anisoraV2_gpu/High quality 2D cartoon animation.pt")

os.makedirs(opt_root,exist_ok=True)
import numpy as np
for meta_path,name in tqdm(metadatas):
    try:
        #text=np.load(meta_path,allow_pickle=True)['arr_0'].item()["prompt"]
        name=str(name)+".pt"
        if os.path.exists("%s/%s"%(opt_root,name)):continue
        context = text_encoder([text], local_rank)[0].cpu()#torch.Size([138, 4096])#"In this scene, a man with a beard is seen tending to a woman who lies in bed, her face illuminated by the soft glow of a nearby light source. The man, dressed in a blue robe adorned with intricate designs, holds a bowl, possibly containing a healing potion or a magical elixir. The woman, clad in a pink garment, appears to be resting or possibly unwell, as she lies on her side with her eyes closed. The setting suggests a historical or medieval context, with the dimly lit room and the man's attire evoking a sense of timelessness and mystery. "
        save_path="%s/%s"%(opt_root,name)
        torch.save(context,save_path)
    except Exception as e:
        print(e)

