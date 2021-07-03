This is the code of VidFace.
TRAIN:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4324 basicsr/train.py -opt options/train/EDVR/unt2t_vit_h48_final.yml --launcher pytorch
TEST:
    TUFS-145K:
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4323 basicsr/test.py -opt options/test/EDVR/test_vox.yml --launcher pytorch
    IJBC:
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4323 basicsr/test.py -opt options/test/EDVR/test_ijbc.yml --launcher pytorch