# MMPedestron

[ECCV2024] This is the official implementation of the paper "When Pedestrian Detection Meets Multi-Modal Learning: Generalist Model and Benchmark Dataset".

Authors: [Yi Zhang](https://scholar.google.com/citations?hl=en&user=hzR7V5AAAAAJ), [Wang ZENG](https://scholar.google.com/citations?user=u_RNsOUAAAAJ&hl=en), [Sheng Jin](https://scholar.google.com/citations?user=wrNd--oAAAAJ&hl=en), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=en), [Ping Luo](https://scholar.google.com/citations?user=aXdjxb4AAAAJ&hl=en), [Wentao Liu](https://scholar.google.com/citations?user=KZn9NWEAAAAJ&hl=en)

## Configs and Models

### Region proposal performance
1. Prtrained Stage

| Method&Config | Backbone|                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
|  [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_pretrained.py) | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1XCiQHElKkhCVAWCGsXXVgn0Tw-2LWJMy/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1nnkvS0n0EWuXD72rz_c3lw) |

2. CrowdHuman

|                                Method&Config                                | Backbone |                                                                    Download                                                                    |
|:---------------------------------------------------------------------------:| :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/crowdhuman/mmpedestron_crowdhuman_2x.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1cAuHOpLgl-p5BpVJLso3MxStolAMzspd/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1VPnzk5--iiQ_WABDx2zHvQ) |

3.COCO-Person

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron finetune](configs/mmpedestron/coco_exp/mmpedestron_coco.py) | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1hao7Y0NZOy9gTp4Q-kQD_TlCFnJ04iQB/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1rBYbCFNK8KrnvBGQe1irCQ) |

4.FLIR

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/flir_exp/mmpedestron_flir_2x.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1st8Iwal_43wGeFiBi3VBatiV4zCYAnuS/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1EtqNyZwHCGL4tVLOiX1o9w) |

5.PEDRo

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/pedro_exp/mmpedestron_pedro.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/15eYXeXo0iINntD6DaFE8ODy6jMLn_vPc/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1FtVPqUmtFlCWGTkOzHxfqg) |
| [MMPedestron(10% train data)](configs/mmpedestron/pedro_exp/mmpedestron_pedro_10p.py) | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1ndHSDJ8DTYXl0yQqUcwAFNaGqYz4Wt2f/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1GykXNV5eb8huGwy5-Y5YBA) |
| [Co-Dino](configs/baseline_exp/pedro_evs/co_dino/co_dino_5scale_r50_1x_pedro_evs.py) | Res50 | - |
| [YOLOX](configs/baseline_exp/yolox_base/yolox_x_8x8_300e_base.py) | CSPDarknet | - |
| [Meta Transformer](configs/baseline_exp/meta_transformer_base/cascade_mask_rcnn_meta_transformer_adapter_base_fpn.py) | ViTAdapter | - |
| [Faster R-CNN](configs/baseline_exp/pedro_evs/faster-rcnn/faster_rcnn_r50_fpn_1x_pedro_evs.py) | Res50 | - |

6.LLVIP Datasets

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1ddAtS5Oz3cvPtR7b7kOiYerNQstD9PHM/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1lRF3MQraKXVn5v5uM3YTxw) |
| [Co-Dino RGB](configs/baseline_exp/llvip/co_dino/co_dino_5scale_r50_1x_llvip_rgb.py), [Co-Dino IR](configs/baseline_exp/llvip/co_dino/co_dino_5scale_r50_1x_llvip_ir.py)  | Res50 | - |
| [YOLOX RGB](configs/baseline_exp/llvip/yolox/yolox_x_1x_llvip_rgb.py), [YOLOX IR](configs/baseline_exp/llvip/yolox/yolox_x_1x_llvip_ir.py) | CSPDarknet | - |
| [Meta Transformer RGB](configs/baseline_exp/llvip/meta_transformer/meta_transformer_b_llvip_rgb_1x.py), [Meta Transformer IR](configs/baseline_exp/llvip/meta_transformer/meta_transformer_b_llvip_ir_1x.py) | ViTAdapter | - |
| [Faster R-CNN RGB](configs/baseline_exp/llvip/faster-rcnn/faster_rcnn_r50_fpn_1x_llvip_rgb.py), [Faster R-CNN IR](configs/baseline_exp/llvip/faster-rcnn/faster_rcnn_r50_fpn_1x_llvip_ir.py) | Res50 | - |

7.InoutDoor Datasets

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1ddAtS5Oz3cvPtR7b7kOiYerNQstD9PHM/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1lRF3MQraKXVn5v5uM3YTxw) |
| [Co-Dino RGB](configs/baseline_exp/inoutdoor/co_dino/co_dino_5scale_r50_1x_inoutdoor_rgb.py), [Co-Dino Depth](configs/baseline_exp/inoutdoor/co_dino/co_dino_5scale_r50_1x_inoutdoor_depth.py)  | Res50 | - |
| [YOLOX RGB](configs/baseline_exp/inoutdoor/yolox/yolox_x_1x_inoutdoor_rgb.py), [YOLOX Depth](configs/baseline_exp/inoutdoor/yolox/yolox_x_1x_inoutdoor_depth.py) | CSPDarknet | - |
| [Meta Transformer RGB](configs/baseline_exp/inoutdoor/meta_transformer/meta_transformer_b_inoutdoor_rgb_1x.py), [Meta Transformer Depth](configs/baseline_exp/inoutdoor/meta_transformer/meta_transformer_b_inoutdoor_depth_1x.py) | ViTAdapter | - |
| [Faster R-CNN RGB](configs/baseline_exp/inoutdoor/faster-rcnn/faster_rcnn_r50_fpn_1x_inoutdoor_rgb.py), [Faster R-CNN Depth](configs/baseline_exp/inoutdoor/faster-rcnn/faster_rcnn_r50_fpn_1x_inoutdoor_depth.py) | Res50 | - |

8.STCrowd Datasets

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1ddAtS5Oz3cvPtR7b7kOiYerNQstD9PHM/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1lRF3MQraKXVn5v5uM3YTxw) |
| [Co-Dino RGB](configs/baseline_exp/stcrowd/co_dino/co_dino_5scale_r50_1x_stcrowd_rgb.py), [Co-Dino Lidar](configs/baseline_exp/stcrowd/co_dino/co_dino_5scale_r50_1x_stcrowd_lidar.py)  | Res50 | - |
| [YOLOX RGB](configs/baseline_exp/stcrowd/yolox/yolox_x_1x_stcrowd_rgb.py), [YOLOX Lidar](configs/baseline_exp/stcrowd/yolox/yolox_x_1x_stcrowd_lidar.py) | CSPDarknet | - |
| [Meta Transformer RGB](configs/baseline_exp/stcrowd/meta_transformer/meta_transformer_b_stcrowd_rgb_1x.py), [Meta Transformer Lidar](configs/baseline_exp/stcrowd/meta_transformer/meta_transformer_b_stcrowd_lidar_1x.py) | ViTAdapter | - |
| [Faster R-CNN RGB](configs/baseline_exp/stcrowd/faster-rcnn/faster_rcnn_r50_fpn_1x_stcrowd_rgb.py), [Faster R-CNN Lidar](configs/baseline_exp/stcrowd/faster-rcnn/faster_rcnn_r50_fpn_1x_stcrowd_lidar.py) | Res50 | - |

9.EventPed Datasets

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/1ddAtS5Oz3cvPtR7b7kOiYerNQstD9PHM/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1lRF3MQraKXVn5v5uM3YTxw) |
| [Co-Dino RGB](configs/baseline_exp/evs_human/co_dino/co_dino_5scale_r50_1x_evs_human_rgb.py), [Co-Dino Lidar](configs/baseline_exp/evs_human/co_dino/co_dino_5scale_r50_1x_evs_human_evs.py)  | Res50 | - |
| [YOLOX RGB](configs/baseline_exp/evs_human/yolox/yolox_x_1x_evs_human_rgb.py), [YOLOX Lidar](configs/baseline_exp/evs_human/yolox/yolox_x_1x_evs_human_evs.py) | CSPDarknet | - |
| [Meta Transformer RGB](configs/baseline_exp/evs_human/meta_transformer/meta_transformer_b_evs_human_rgb_1x.py), [Meta Transformer Lidar](configs/baseline_exp/evs_human/meta_transformer/meta_transformer_b_evs_human_evs_1x.py) | ViTAdapter | - |
| [Faster R-CNN RGB](configs/baseline_exp/evs_human/faster_rcnn/faster_rcnn_r50_fpn_1x_evs_human_rgb.py), [Faster R-CNN Lidar](configs/baseline_exp/evs_human/faster_rcnn/faster_rcnn_r50_fpn_1x_evs_human_evs.py) | Res50 | - |

9.Fusion Exp

9-1 LLVIP

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/15eYXeXo0iINntD6DaFE8ODy6jMLn_vPc/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1FtVPqUmtFlCWGTkOzHxfqg) |
| [Early-Fusion](configs/baseline_exp/llvip/faster-rcnn-ef/faster_rcnn_r50_fpn_ef_1x_llvip.py) | Res50 | - |
| [FPN-Fusion](configs/baseline_exp/llvip/faster-rcnn-mf/faster_rcnn_r50_fpn_mf_1x_llvip.py) | Res50 | - |
| [ProbEN RGB](configs/baseline_exp/llvip/faster_rcnn_pf/faster_rcnn_r50_fpn_1x_llvip_rgb_infer_only.py), [ProbEN IR](configs/baseline_exp/llvip/faster_rcnn_pf/faster_rcnn_r50_fpn_1x_llvip_ir_infer_only.py) | Res50 | - |
| [CMX](configs/baseline_exp/llvip/cmx/faster_rcnn_cmx_fpn_1x_llvip.py) | SwinTransformer | - |

9-2 InOutDoor

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/15eYXeXo0iINntD6DaFE8ODy6jMLn_vPc/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1FtVPqUmtFlCWGTkOzHxfqg) |
| [Early-Fusion](configs/baseline_exp/inoutdoor/faster-rcnn-ef/faster_rcnn_r50_fpn_ef_1x_inoutdoor.py) | UNIXViT | - |
| [FPN-Fusion](configs/baseline_exp/inoutdoor/faster-rcnn-mf/faster_rcnn_r50_fpn_mf_1x_inoutdoor.py) | Res50 | - |
| [ProbEN RGB](configs/baseline_exp/inoutdoor/faster-rcnn_pf/faster_rcnn_r50_fpn_1x_inoutdoor_rgb_infer_only.py), [ProbEN Depth](configs/baseline_exp/inoutdoor/faster-rcnn_pf/faster_rcnn_r50_fpn_1x_inoutdoor_depth_infer_only.py) | Res50 | - |
| [CMX](configs/baseline_exp/inoutdoor/cmx/faster_rcnn_cmx_fpn_1x_inoutdoor.py) | SwinTransformer | - |

9-1 STCrowd

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/15eYXeXo0iINntD6DaFE8ODy6jMLn_vPc/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1FtVPqUmtFlCWGTkOzHxfqg) |
| [Early-Fusion](configs/baseline_exp/stcrowd/faster-rcnn-ef/faster_rcnn_r50_fpn_ef_1x_stcrowd.py) | Res50 | - |
| [FPN-Fusion](configs/baseline_exp/stcrowd/faster-rcnn-mf/faster_rcnn_r50_fpn_mf_1x_stcrowd.py) | Res50 | - |
| [ProbEN RGB](configs/baseline_exp/stcrowd/faster-rcnn-pf/faster_rcnn_r50_fpn_1x_stcrowd_rgb_infer_only.py), [ProbEN Lidar](configs/baseline_exp/stcrowd/faster-rcnn-pf/faster_rcnn_r50_fpn_1x_stcrowd_lidar_infer_only.py) | Res50 | - |
| [CMX](configs/baseline_exp/stcrowd/cmx/faster_rcnn_cmx_fpn_1x_stcrowd.py) | SwinTransformer | - |


9-1 EventPed

| Method&Config | Backbone |                                                                    Download                                                                    |
| :----: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| [MMPedestron](configs/mmpedestron/mix_datasets/mmpedestron_mix5datasets_best.py)  | UNIXViT | [Gooogle Drive](https://drive.google.com/file/d/15eYXeXo0iINntD6DaFE8ODy6jMLn_vPc/view?usp=sharing), [Baidu Yun (Code: mmpd)](https://pan.baidu.com/s/1FtVPqUmtFlCWGTkOzHxfqg) |
| [Early-Fusion](configs/baseline_exp/evs_human/faster_rcnn_ef/faster_rcnn_r50_fpn_ef_1x_evs_human.py) | Res50 | - |
| [FPN-Fusion](configs/baseline_exp/evs_human/faster_rcnn_mf/faster_rcnn_r50_fpn_mf_1x_evs_human.py) | Res50 | - |
| [ProbEN RGB](configs/baseline_exp/evs_human/faster_rcnn_pf/faster_rcnn_r50_fpn_1x_evs_human_rgb_infer_only.py), [ProbEN Event](configs/baseline_exp/evs_human/faster_rcnn_pf/faster_rcnn_r50_fpn_1x_evs_human_evs_infer_only.py) | Res50 | - |
| [CMX](configs/baseline_exp/evs_human/cmx/faster_rcnn_cmx_fpn_1x_evs_human.py) | SwinTransformer | - |



## Installation

### Prepare environment
1. Create a conda virtual environment and activate it.

```shell
conda create -n mmpedestron python=3.6
conda activate mmpedestron
```

2. Install requirements, we recommend you to install requirements by env_deploy.sh

```shell
conda install cudatoolkit=10.1

sh env_deploy.sh
```
## Data Preparation
Please obtain the datasets repo from the following: [MMPD-Dataset](https://github.com/jin-s13/MMPD-Dataset)

## Training

Manage training jobs with Slurm

```shell
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPUS}
```

## Testing

Manage testing jobs with Slurm

```shell
sh tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT} ${GPUS}
```

## License
Codes and data are freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please contact Mr. Sheng Jin (jinsheng13[at]foxmail[dot]com). We will send the detail agreement to you.

## Citation
if you find our paper and code useful in your research, please consider giving a star and citation :)

```bibtex
@inproceedings{zhang2024when,
  title={When Pedestrian Detection Meets Multi-Modal Learning: Generalist Model and Benchmark Dataset},
  author={Zhang, Yi and Zeng, Wang and Jin, Sheng and Qian, Chen and Luo, Ping and Liu, Wentao},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  month={September}
}
```
