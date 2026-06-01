# DUSt3r: Synthetic Fine-Tuning Project

This repository is a modified version of the original [DUSt3R repository by Naver Labs](https://github.com/naver/dust3r). 

### Our Modifications:
* Added parameter-efficient fine-tuning capabilities (freezing specific layers in `training_freeze.py`).
* Created preprocessing scripts for the BlendedMVS dataset (`preprocess_blendedMVS_E.py`).
* Implemented evaluation and assessment notebooks for synthetic fine-tuning benchmarks.


---

### To run assessment

Change the path to the models in the python script

```bash
python fine-tune_assessment.py

```

---

### For processing data

```bash
python datasets_preprocess/preprocess_blendedMVS_E.py \
    --blendedmvs_dir data/RanScenes2 \
    --precomputed_pairs data/RanScenes2/blendedmvs_pairs.npy \
    --output_dir data/RanScenes2_processed

```

---

### Demo for 512 linear model

```bash
python demo.py --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth --server_name 0.0.0.0

```

---

### Training and validating on SyScenes3

```bash
torchrun --nproc_per_node=1 train_freeze.py \
    --train_dataset "4000 @ BlendedMVS(split='train', ROOT='data/SyScenes3_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) " \
    --test_dataset "2000 @ BlendedMVS(split='val', ROOT='data/SyScenes3_processed', aug_crop=16, resolution=(512,384), transform=ColorJitter) " \
    --trainable "downstream_head1,downstream_head2,dec_blocks2,decoder_embed,dec_norm" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 30 --batch_size 1 --accum_iter 16 \
    --save_freq 1 --keep_freq 100 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_linear_512_SyScenes3_2T1V30EP"

```

#### demo for on SyScenes3:

```bash
python demo.py --weights checkpoints/dust3r_linear_512_SyScenes3_2T1V30EP/checkpoint-best.pth --server_name 0.0.0.0

```

---

### Training and validating on RanScenes2

```bash
torchrun --nproc_per_node=1 train_freeze.py \
    --train_dataset "3600 @ BlendedMVS(split='train', ROOT='data/RanScenes2_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) " \
    --test_dataset "1800 @ BlendedMVS(split='val', ROOT='data/RanScenes2_processed', aug_crop=16, resolution=(512,384), transform=ColorJitter) " \
    --trainable "downstream_head1,downstream_head2,dec_blocks2,decoder_embed,dec_norm" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 30 --batch_size 1 --accum_iter 16 \
    --save_freq 1 --keep_freq 100 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_linear_512_RanScenes2_1T1V30EP"

```

#### demo for on RanScenes2:

```bash
python demo.py --weights checkpoints/dust3r_linear_512_RanScenes2_1T1V30EP/checkpoint-best.pth --server_name 0.0.0.0

```
