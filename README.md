# MapleGrasp: Mask-guided Feature Pooling for Language-driven Efficient Robotic Grasping (ArXiv 2025)

Created by Vineet, Naman Patel, Prashanth Krishnamurthy, Ramesh Karri and Farshad Khorrami

This is an official PyTorch implementation of the baseline end-to-end model [MapleGrasp](https://www.arxiv.org/abs/2506.06535) of our work. The implementation of our MapleGrasp model is based on the [CROG](https://github.com/HilbertXu/CROG) model, thanks for their amazing work!

Robotic manipulation of unseen objects via natural language commands remains challenging. Language driven robotic grasping (LDRG) predicts stable grasp poses from natural language queries and RGB-D images. We propose MapleGrasp, a novel framework that leverages mask-guided feature pooling for efficient vision-language driven grasping. Our two-stage training first predicts segmentation masks from CLIP-based vision-language features. The second stage pools features within these masks to generate pixel-level grasp predictions, improving efficiency, and reducing computation. Incorporating mask pooling results in a 7% improvement over prior approaches on the OCID-VLG benchmark. Furthermore, we introduce RefGraspNet, an open-source dataset eight times larger than existing alternatives, significantly enhancing model generalization for open-vocabulary grasping. MapleGrasp scores a strong grasping accuracy of 89\% when compared with competing methods in the RefGraspNet benchmark. Our method achieves comparable performance to larger Vision-Language-Action models on the LIBERO benchmark, and shows significantly better generalization to unseen tasks. Real-world experiments on a Franka arm demonstrate 73% success rate with unseen objects, surpassing competitive baselines by 11%.

## Preparation

1. Environment
   - use the environment.yml file to create the conda env.
2. Datasets
   - Please download the [OCID-VLG](https://github.com/gtziafas/OCID-VLG) dataset by following the instructions and directory formatting.

## Training

```
python train_maplegrasp.py --config config/OCID-VLG/maplegrasp_multiple_r50.yaml
```

Please appropriately modify the stage 1 and stage 2 parameters in the yaml file before running.

## Evaluation

```
python test.py --config config/OCID-VLG/maplegrasp_multiple_r50.yaml
```


**Please remember to modify the path to the dataset in config files.**


## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation
If you find our work useful in your research, please consider citing:
```
@article{bhat2025maplegrasp,
  title={MapleGrasp: Mask-guided Feature Pooling for Language-driven Efficient Robotic Grasping},
  author={Bhat, Vineet and Patel, Naman and Krishnamurthy, Prashanth and Karri, Ramesh and Khorrami, Farshad},
  journal={arXiv preprint arXiv:2506.06535},
  year={2025}
}
```
