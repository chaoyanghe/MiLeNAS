# MiLeNAS: Efficient Neural Architecture Search via Mixed-level Reformulation
This is the source code for the following paper:
```
@inproceedings{he2020milenas,
  title={Milenas: Efficient neural architecture search via mixed-level reformulation},
  author={He, Chaoyang and Ye, Haishan and Shen, Li and Zhang, Tong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11993--12002},
  year={2020}
}
```


## Installation
For Linux OS:

```
conda create --name milenas python=3
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

For Macbook:

```
conda create --name milenas python=3
conda install pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```


NOTE: PyTorch > 1.0 is supported.

## Datasets
For CIFAR10, it will be automatically downloaded.
For ImageNet, please download it manually and set the path accordingly.

## Architecture Search
To carry out architecture search using 2nd-order approximation, run
```
cd cnn && python train_search.py --unrolled     # for conv cells on CIFAR-10
cd rnn && python train_search.py --unrolled     # for recurrent cells on PTB
```


## Architecture Evaluation (Training the searched model from scratch)
To evaluate our best cells by training from scratch, run
```
# CIFAR-10
CUDA_VISIBLE_DEVICES=0 sh evaluation/run_eval_cifar10.sh "0" GDAS_MIXED_LEVEL2 6000 0.030 saved_models

# ImageNet
CUDA_VISIBLE_DEVICES=0 sh evaluation/run_eval_imagenet.sh "0" GDAS_MIXED_LEVEL2 6000 0.030 /home/chaoyanghe/sourcecode/dataset/cv/ImageNet
```
Please change the `--arch` in `evaluation/run_eval_imagenet.sh` and `evaluation/run_eval_cifar10.sh`, 'arch' is defined at `genotypes.py`


## Visualization
```
python visualization/visualize.py GDAS_MIXED_LEVEL2
```
where `GDAS_MIXED_LEVEL2` is the architecture defined in `genotypes.py`.

## Citation
If you use any part of this code in your research, please cite our paper:
```
@inproceedings{he2020milenas,
  title={Milenas: Efficient neural architecture search via mixed-level reformulation},
  author={He, Chaoyang and Ye, Haishan and Shen, Li and Zhang, Tong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11993--12002},
  year={2020}
}
```
We also publish FedNAS, a neural architecture search method for federated deep learning, please also cite if it is related to your research:
```
@inproceedings{FedNAS,
  title={FedNAS: Federated Deep Learning via Neural Architecture Search},
  author={He, Chaoyang and Annavaram, Murali and Avestimehr, Salman},
  booktitle={CVPR 2020 Workshop on Neural Architecture Search and Beyond for Representation Learning},
  year={2020},
}
```
