import argparse
from typing import List
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.covariance import LedoitWolf
from sklearn.manifold import TSNE
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.models.efficientnet import EfficientNet
import timm

import datasets.mvtec as mvtec


def parse_args():
    parser = argparse.ArgumentParser('MahalanobisAD')
    parser.add_argument("-m", "--model_name", type=str, default='tf_efficientnet_b4')
    parser.add_argument("-s", "--save_path", type=str, default="./result")
    parser.add_argument("-p", "--pool_method", type=str, default="avg")
    return parser.parse_args()


def main():

    args = parse_args()

    dim_reduction_model= TSNE(n_components=2)

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = timm.create_model(args.model_name, pretrained=True)
    model.to(device)
    model.eval()

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

    total_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(
            class_name=class_name, is_train=True)
        train_dataloader = DataLoader(
            train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(
            class_name=class_name, is_train=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size=32, pin_memory=True)

        train_outputs = [[] for _ in range(9)]
        test_outputs = [[] for _ in range(9)]
        youden_index_thresholds = []

        # extract train set features
        train_feat_filepath = os.path.join(
            args.save_path, 'temp', 'train_%s_%s.pkl' % (class_name, args.model_name))
        if not os.path.exists(train_feat_filepath):
            for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    feats = extract_features(x.to(device), model, args.pool_method)

                for f_idx, feat in enumerate(feats):
                    train_outputs[f_idx].append(feat)

            # fitting a multivariate gaussian to features extracted from every level of ImageNet pre-trained model
            for t_idx, train_output in enumerate(train_outputs):
                mean = torch.mean(
                    torch.cat(train_output, 0).squeeze(), dim=0).cpu().detach().numpy()
                # covariance estimation by using the Ledoit. Wolf et al. method
                cov = LedoitWolf().fit(
                    torch.cat(train_output, 0).squeeze().cpu().detach().numpy()).covariance_
                train_outputs[t_idx] = [mean, cov]

            # save extracted feature
            with open(train_feat_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature distribution from: %s' %
                  train_feat_filepath)
            with open(train_feat_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            gt_list.extend(y.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                feats = extract_features(x.to(device), model, args.pool_method)

            for feat_idx, feat in enumerate(feats):
                test_outputs[feat_idx].append(feat)

        for test_idx, test_output in enumerate(test_outputs):
            test_outputs[test_idx] = torch.cat(
                test_output, 0).squeeze().cpu().detach().numpy()

        # calculate Mahalanobis distance per each level of EfficientNet
        dist_list = []
        each_level_dist = {}
        for test_idx, test_output in enumerate(test_outputs):
            mean = train_outputs[test_idx][0]
            cov_inv = np.linalg.inv(train_outputs[test_idx][1])
            print(f"level, {test_idx}, mean shape, {mean.shape}")
            dist = [mahalanobis(sample, mean, cov_inv)
                    for sample in test_output]
            each_level_dist[test_idx + 1] = np.array(dist)
            dist_list.append(np.array(dist))

        # Anomaly score is followed by unweighted summation of the Mahalanobis distances
        # scores = np.sum(np.array(dist_list), axis=0)
        scores = each_level_dist[7]

        # calculate image-level ROC AUC score
        fpr, tpr, thresholds = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        youden_index_thresholds.append(thresholds[np.argmax(tpr-fpr)])
        print(f"{class_name}, youden index, {thresholds[np.argmax(tpr-fpr)]:.1f}")
        tn, fp, fn, tp = confusion_matrix(
            gt_list, scores >= thresholds[np.argmax(tpr-fpr)]).flatten()
        print(
            f"conf matrix, tn, fp, fn, tp, {tn, fp, fn, tp}")
        # print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        # plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))
        print('%s ROCAUC: %.3f, mean th: %.1f' % (class_name, roc_auc, np.mean(youden_index_thresholds)))
        plt.plot(fpr, tpr, label='%s ROCAUC: %.3f, mean th:%.1f' %
                 (class_name, roc_auc, np.mean(youden_index_thresholds)))

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    plt.title('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.save_path, 'roc_curve_%s.png' %
                args.model_name), dpi=200)



def extract_features(inputs: torch.Tensor,model:EfficientNet, pool_method:str)-> List:
    """ Returns list of the feature at each level of the EfficientNet """
    if pool_method == "avg":
        pool = F.adaptive_avg_pool2d
    elif pool_method == "max":
        pool = F.adaptive_max_pool2d
    feat_list = []

    # Stem
    x = model.conv_stem(inputs)
    x = model.bn1(x)
    x = model.act1(x)
    feat_list.append(pool(x, 1))

    # Blocks: 2~8 layer
    for _, block_layer in enumerate(model.blocks, start=2):
        x = block_layer(x)
        feat_list.append(pool(x, 1))

    # Head
    x = model.conv_head(x)
    x = model.bn2(x)
    x = model.act2(x)
    feat_list.append(pool(x, 1))

    return feat_list

if __name__ == '__main__':
    main()
