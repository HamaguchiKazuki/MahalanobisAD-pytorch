import argparse
import datetime as dt
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from timm.models.efficientnet import EfficientNet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# import datasets.mvtec as mvtec

TRANSFORM_IMAGE = T.Compose([T.Resize(224, InterpolationMode.LANCZOS),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])


def parse_args():
    parser = argparse.ArgumentParser('MahalanobisAD')
    parser.add_argument("-m", "--model_name", type=str,
                        default='tf_efficientnet_b4')
    parser.add_argument("-s", "--save_path", type=str, default="./result")
    parser.add_argument("-p", "--pool_method", type=str, default="avg")
    parser.add_argument("-r", "--root_path", type=str, default="../data")
    parser.add_argument("-v", "--ver", type=int, default=1)
    return parser.parse_args()

# unsupervised learning using pre-trained efficient net, result is Mahalanobis distance of sum all level(1~8 layers)


def main():

    args = parse_args()
    # dataset path
    steel_ball_path = os.path.join(
        args.root_path, '50_defect_images_and_50_defect_free_images', 'processed', f'ver{args.ver}')
    timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = timm.create_model(args.model_name, pretrained=True)
    model.to(device)
    model.eval()

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'csv'), exist_ok=True)

    train_datasets = datasets.ImageFolder(os.path.join(
        steel_ball_path, "train"), transform=TRANSFORM_IMAGE)

    print(train_datasets.class_to_idx)

    # k hold cross validation
    k = 4
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    train_data, train_label = np.array(train_datasets.imgs)[
        :, 0], np.array(train_datasets.imgs)[:, 1]

    # totalzation param
    result_df = pd.DataFrame(index=range(k+1), columns=["ROCAUC", "threshold"])
    total_roc_auc = []
    total_youden_index_threshold = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_data, train_label)):

        train_dataset = Subset(train_datasets, tr_idx)
        valid_dataset = Subset(train_datasets, va_idx)

        train_dataloader = DataLoader(
            train_dataset, batch_size=32, pin_memory=True, shuffle=True)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=32, pin_memory=True, shuffle=False)

        train_outputs = [[] for _ in range(9)]
        valid_outputs = [[] for _ in range(9)]

        # extract train set all level (1~8) features
        train_feat_filepath = os.path.join(
            args.save_path, 'temp', f'train_steel_ball_{args.model_name}.pkl')
        if not os.path.exists(train_feat_filepath):
            for (x, y) in tqdm(train_dataloader, f'| feature extraction | train fold {fold} | steel_ball |'):
                # model prediction
                with torch.no_grad():
                    feats = extract_features(
                        x.to(device), model, args.pool_method)
                for feat_idx, feat in enumerate(feats):
                    train_outputs[feat_idx].append(feat)

            # fitting a multivariate gaussian to features extracted from every level of ImageNet pre-trained model
            for train_idx, train_output in enumerate(train_outputs):
                mean = torch.mean(
                    torch.cat(train_output, 0).squeeze(), dim=0).cpu().detach().numpy()
                # covariance estimation by using the Ledoit. Wolf et al. method
                cov = LedoitWolf().fit(
                    torch.cat(train_output, 0).squeeze().cpu().detach().numpy()).covariance_
                # append each level (1~8) mean, cov
                train_outputs[train_idx] = [mean, cov]

            # save extracted feature
            with open(train_feat_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature distribution from: %s' %
                  train_feat_filepath)
            with open(train_feat_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        # extract valid set features
        for (x, y) in tqdm(valid_dataloader, f'| feature extraction | valid fold {fold} | steel_ball '):
            gt_list.extend(y.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                feats = extract_features(x.to(device), model, args.pool_method)
            for feat_idx, feat in enumerate(feats):
                valid_outputs[feat_idx].append(feat)

        # anomaly now label 1,2,3,... to 1 only
        gt_list = np.array(gt_list)
        print(gt_list)
        gt_list[gt_list > 1] = 1
        print(gt_list)

        for valid_idx, valid_output in enumerate(valid_outputs):
            valid_outputs[valid_idx] = torch.cat(
                valid_output, 0).squeeze().cpu().detach().numpy()

        # calculate Mahalanobis distance per each level of EfficientNet
        dist_list = []
        each_level_dist = {}
        for valid_idx, valid_output in enumerate(valid_outputs):
            mean = train_outputs[valid_idx][0]
            cov_inv = np.linalg.inv(train_outputs[valid_idx][1])
            dist = [mahalanobis(sample, mean, cov_inv)
                    for sample in valid_output]
            each_level_dist[valid_idx + 1] = np.array(dist)
            dist_list.append(np.array(dist))

        # Anomaly score is followed by unweighted summation of the Mahalanobis distances
        scores = np.sum(np.array(dist_list), axis=0)

        # calculate image-level ROC AUC score
        fpr, tpr, thresholds = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        selected_threds = thresholds[np.argmax(tpr-fpr)]

        plt.plot(fpr, tpr, label='%d fold ROCAUC: %.3f, th:%.2f' %
                 (fold, roc_auc, selected_threds))

        # TODO totalization
        print(
            f"fold {fold}, youden index, {selected_threds:.1f}")
        tn, fp, fn, tp = confusion_matrix(
            gt_list, scores >= selected_threds).flatten()
        print(
            f"conf matrix, tn, fp, fn, tp, {tn, fp, fn, tp}")
        print('fold %d, ROCAUC: %.3f' % (fold, roc_auc))
        total_youden_index_threshold.append(selected_threds)

        result_df["ROCAUC"][fold] = roc_auc
        result_df["threshold"][fold] = selected_threds

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    print('Average threshold: %.3f' % np.mean(total_youden_index_threshold))

    plt.title('Average steel_ball ROCAUC: %.3f' % np.mean(total_roc_auc))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.save_path,
                f'roc_curve_{args.model_name}_timestamp_{timestamp}.png'), dpi=200)

    # TODO save parameter Average ROC-AUC, Average threshold
    result_df["ROCAUC"][k] = np.mean(total_roc_auc)
    result_df["threshold"][k] = np.mean(total_youden_index_threshold)
    result_df.to_csv(os.path.join(args.save_path, 'csv',
                     f"roc_threshold_{args.model_name}__timestamp_{timestamp}.csv"))

    # TODO test dataset validation
    print("test")
    test_threshold = result_df["threshold"][k]
    print(f"test threshold, {test_threshold}")

    test_dataset = datasets.ImageFolder(os.path.join(
        steel_ball_path, "test"), transform=TRANSFORM_IMAGE)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, pin_memory=True)

    test_outputs = [[] for _ in range(9)]

    gt_list = []
    # extract test set features
    for (x, y) in tqdm(test_dataloader, '| feature extraction | test | steel_ball |'):
        gt_list.extend(y.cpu().detach().numpy())
        # model prediction
        with torch.no_grad():
            feats = extract_features(x.to(device), model, args.pool_method)

        for feat_idx, feat in enumerate(feats):
            test_outputs[feat_idx].append(feat)

    for test_idx, test_output in enumerate(test_outputs):
        test_outputs[test_idx] = torch.cat(
            test_output, 0).squeeze().cpu().detach().numpy()

    # anomaly now label 1,2,3,... to 1 only
    gt_list = np.array(gt_list)
    print(gt_list)
    gt_list[gt_list > 1] = 1
    print(gt_list)

    # calculate Mahalanobis distance per each level of EfficientNet
    dist_list = []
    each_level_dist = {}
    for test_idx, test_output in enumerate(test_outputs):
        mean = train_outputs[test_idx][0]
        cov_inv = np.linalg.inv(train_outputs[test_idx][1])
        # print(f"level, {test_idx}, mean shape, {mean.shape}")
        dist = [mahalanobis(sample, mean, cov_inv)
                for sample in test_output]
        each_level_dist[test_idx + 1] = np.array(dist)
        dist_list.append(np.array(dist))

    scores = np.sum(np.array(dist_list), axis=0)

    tn, fp, fn, tp = confusion_matrix(
        gt_list, scores >= test_threshold).flatten()
    print(
        f"conf matrix, tn, fp, fn, tp, {tn, fp, fn, tp}")

    # TODO save test tp, fp, fn, tp


def extract_features(inputs: torch.Tensor, model: EfficientNet, pool_method: str) -> List:
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


# def transform_image(resize):
#     T.Compose([T.Resize(resize, Image.LANCZOS),
#                T.ToTensor(),
#                T.Normalize(mean=[0.485, 0.456, 0.406],
#                            std=[0.229, 0.224, 0.225])])

# class ImageTransform():
#     def __init__(self, resize=224):
#         self.data_transform = {
#             T.Compose([T.Resize(resize, Image.LANCZOS),
#                        T.ToTensor(),
#                        T.Normalize(mean=[0.485, 0.456, 0.406],
#                        std=[0.229, 0.224, 0.225])])
#         }

#     def __call__(self, img):
#         return self.data_transform(img)

if __name__ == '__main__':
    main()
