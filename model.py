import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

import nibabel
import SimpleITK as sitk
import numpy as np
import skan
from skimage.morphology import skeletonize_3d, binary_opening, binary_closing, ball
from vtkplotter import *


import os
import nrrd
import csv
import nrrd
import pandas as pd
from numpy.linalg import svd


from PIL import Image
import math

""" 
Extractor
"""
class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))

class DenseBlock(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):
            super(DenseBlock, self).__init__()
            self.num_layers = num_layers
            self.k0 = input_channels
            self.k = growth_rate
            self.layers = self.__make_layers()

    def __make_layers(self):
            layer_list = []
            for i in range(self.num_layers):
                layer_list.append(nn.Sequential(
                    BN_Conv2d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0),
                    BN_Conv2d(4 * self.k, self.k, 3, 1, 1)
                ))
            return layer_list

    def forward(self, x):
            feature = self.layers[0](x)
            out = torch.cat((x, feature), 1)
            for i in range(1, len(self.layers)):
                feature = self.layers[i](out)
                out = torch.cat((feature, out), 1)
            return out

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, patch_count=16, in_chans=1, embed_dim=512,):
        super().__init__()
        patch_stride = img_size // patch_count
        patch_padding = (patch_stride * (patch_count - 1) + patch_size - img_size) // 2
        num_patches = patch_count ** 2

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.feature = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_padding)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.feature(x).flatten(2).transpose(1, 2)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=16, bias=False, scale=None, feature_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(dim, dim * 2, bias=bias)
        self.feature = nn.Linear(dim, dim)
        self.feature_drop = nn.Dropout(feature_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.feature(x)
        x = self.feature_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class DDB(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., bias=False, scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads, bias=bias, scale=scale,featurte_drop=drop )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class ICM(nn.Module):
    def init(self):
        super(ICM, self).init()
        selfresnet = ResBlock(n_chans=256)
        self.feature = nn.Sequential(*list(self.resnet.children())[:-2])
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, current_img, prev_imgs, next_imgs):
        current_features = self.feature_extractor(current_img)
        prev_features = [self.feature_extractor(img) for img in prev_imgs]
        next_features = [self.feature_extractor(img) for img in next_imgs]

        current_features = self.conv1x1(current_features)
        current_features = self.relu(current_features)

        prev_corr = self.calculate_correlation(current_features, prev_features)
        next_corr = self.calculate_correlation(current_features, next_features)

        semantic_features = torch.cat((current_features, prev_corr, next_corr), dim=1)
        return semantic_features

    def calculate_correlation(self, current_features, other_features):
        batch_size, channels, height, width = current_features.size()
        correlation_map = torch.zeros(batch_size, 1, height, width).to(current_features.device)

        for features in other_features:
            correlation_map += torch.sum(current_features * features, dim=1, keepdim=True)

        return correlation_map

class PatchPoint(nn.Module):
    def init(self, input_size, patch_count, tanh=True,weights=(1., 1.)):
        super(PatchPoint, self).init()
        self.input_size = input_size
        self.patch_count = patch_count
        self.generate_NewPoint()
        self.tanh = tanh
        self.weights = weights

    def generate_NewPoint(self):
        NewPoint = []
        patch_stride = 1. / self.patch_count
        for i in range(self.patch_count):
            for j in range(self.patch_count):
                y = (0.5+i)*patch_stride
                x = (0.5+j)*patch_stride
                NewPoint.append([x, y])
        NewPoint = torch.as_tensor(NewPoint)
        self.register_buffer("NewPoint", NewPoint)

    def generate_offsets(self):
        return (self.NewPoints - self.NewPoint) * self.input_size

    def resimple(self, rel_codes):
        NewPoints=self.NewPoint
        pixel = 1. / self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel if self.tanh else rel_codes[:, :, 0] * pixel / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel if self.tanh else rel_codes[:, :, 1] * pixel / wy

        pred_NewPoint = torch.zeros_like(rel_codes)

        ref_x = NewPoints[:, 0].unsqueeze(0)
        ref_y = NewPoints[:, 1].unsqueeze(0)

        pred_NewPoint[:, :, 0] = dx + ref_x
        pred_NewPoint[:, :, 1] = dy + ref_y
        pred_NewPoint = pred_NewPoint.clamp_(min=0., max=1.)

        return pred_NewPoint

    def forward(self, patchs, model_offset=None):
        self.NewPoint = self.resimple(patchs)
        return self.NewPoint

class Extractor(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], patch_embeds=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4 = patch_embeds

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches + 1, embed_dims[3]))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        CS = 0
        self.block1 = nn.ModuleList([DDB(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        CS += depths[0]
        self.block2 = nn.ModuleList([DDB(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        CS += depths[1]
        self.block3 = nn.ModuleList([DDB(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        CS += depths[2]
        self.block4 = nn.ModuleList([DDB(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], bias=bias, scale=scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[CS + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[3])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # new
        self.patch_embed2.reset_offset()
        self.patch_embed3.reset_offset()
        self.patch_embed4.reset_offset()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        CS = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[CS + i]

        CS += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[CS + i]

        CS += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[CS + i]

        CS += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[CS + i]

    def forward_features(self, x):
        B = x.shape[0]
        aux_results = []

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        if isinstance(x, tuple):
            aux_results.append(x[1])
            x = x[0]
        x = x + self.pos_embed2
        x = self.pos_dros2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        if isinstance(x, tuple):
            aux_results.append(x[1])
            x = x[0]
        x = x + self.pos_embed3
        x = self.pos_dros3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        if isinstance(x, tuple):
            aux_results.append(x[1])
            x = x[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed4
        x = self.pos_dros4(x)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x, x


""" 
Decoder
"""
class GDBlock(nn.Module):
    def __init__(self, input_channels):
        super(GDBlock, self).__init__()
        self.input_channels = input_channels
        self.out_channels = int(input_channels / 2)

        self.s1_feature = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())        
        self.s1 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s1_new = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        
        self.s2_feature = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s2_new = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        
        self.s3_feature = nn.Sequential(
            nn.Conv2d(self.input_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 5, 1, 2),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.s3_new = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.out_channels), nn.ReLU())

        self.gd_results = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        s1_input = self.s1_feature(x)
        s1 = self.s1(s1_input)
        s1_new = self.s1_new(s1)

        s2_input = self.s2_feature(x) + s1_new
        s2 = self.s2(s2_input)
        s2_new = self.s2_new(s2)

        s3_input = self.s3_feature(x) + s2_new
        s3 = self.s3(s3_input)
        s3_new = self.s3_new(s3)

        out = self.gd_results(torch.cat((s1_new, s2_new, s3_new), 1))

        return out

class Decoder(nn.Module):
    def __init__(self, channel1, channel2):
        super(Decoder, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.input_high_level_feature = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 5, 1, 2),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.beta_1 = nn.Parameter(torch.ones(1))
        self.beta_2 = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()
        self.output_map = nn.Conv2d(self.channel1, 1, 5, 1, 2)

    def forward(self, x, y, higher_level_prediction):
        # /upsample
        up = self.up(y)

        input_high_level_feature = self.input_high_level_feature(higher_level_prediction)
        foreground_feature = x * input_high_level_feature
        background_feature = x * (1 - input_high_level_feature)

        f1 = GDBlock(foreground_feature)
        f2 = GDBlock(background_feature)

        refine_feature1 = up - (self.beta_2 * f1)
        refine_feature1 = self.bn1(refine_feature1)
        refine_feature1 = self.relu1(refine_feature1)

        refine_feature2 = refine_feature1 + (self.beta_1 * f2)
        refine_feature2 = self.bn2(refine_feature2)
        refine_feature2 = self.relu2(refine_feature2)

        output_map = self.output_map(refine_feature2)

        return refine_feature2, output_map

""" 
Guider
"""
# If you do not need CPR images, you can decouple the guider module

# get centerline
def get_centerline(image_path):

    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)

    vtkImage = Volume(image)
    s = vtkImage.isosurface(1).alpha(0.5)

    image = binary_opening(image)
    image=vtk.vtkImageData(data)
    image = vtk.vtkImageGaussianSmooth(image)

    skele = skeletonize_3d(image)
    skeletonNet = skan.Skeleton(skele)

    paths = [skeletonNet.path_coordinates(i) for i in range(skeletonNet.paths.shape[0])]
    #print(paths)

    s = [s]
    for path in paths:
        p = np.array(path)
        if p.shape[0] == 2:
            spl = Spline(p, smooth=1, degree=1)
        else:
            spl = Spline(p, smooth=1)
        s.append(spl)
    # show(s, axes=0)
    return s



# CPR 

'''
The function cpr(img_name, center_line_name), 
where img_name represents the path of the image and center_line_name represents the path of the centerline point. 
The former supports .nii.gz and .mha files. The latter only supports .npy files.
'''

def update_list1(size, start, P, list1, last_point):
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                new_x, new_y, new_z = start[0] + i, start[1] + j, start[2] + k
                if new_x >= 0 and new_x < size[0] and new_y >= 0 and new_y < size[1] and new_z >= 0 and new_z < size[2]:
                    if P[new_x, new_y, new_z] == 0:
                        if [new_x, new_y, new_z] not in list1:
                            list1.append([new_x, new_y, new_z])
                            key = str(new_x) + '+' + str(new_y) + '+' + str(new_z)
                            if key in last_point.keys():
                                print('error')
                            last_point[key] = start


def find_point_list(thin_label_name, start, end):
    thin_label = sitk.GetArrayFromImage(sitk.ReadImage(thin_label_name))
    data = thin_label.copy().astype(np.float)
    data[data < 0.005] = 0.005
    size = data.shape
    cost = 1 / data
    last_point = {}
    P = np.zeros(cost.shape)
    key = str(start[0]) + '+' + str(start[1]) + '+' + str(start[2])
    P[start[0], start[1], start[2]] = 1
    last_point[key] = [-1, -1, -1]
    list1 = []
    update_list1(size, start, P, list1, last_point)
    iter_num = 0
    while P[end[0], end[1], end[2]] == 0:
        iter_num = iter_num + 1
        if iter_num > 30000:
            print('failure')
            return
        if iter_num % 100 == 0:
            print(len(list1), len(last_point))
        cost_min = 301
        index = -1
        for i in range(0, len(list1)):
            if cost_min > cost[list1[i][0], list1[i][1], list1[i][2]]:
                cost_min = cost[list1[i][0], list1[i][1], list1[i][2]]
                index = i
        P[list1[index][0], list1[index][1], list1[index][2]] = 1
        update_list1(size, [list1[index][0], list1[index][1], list1[index][2]], P, list1, last_point)
        del list1[index]
    last = end.copy()
    path = []
    while last[0] != -1:
        path.append(last)
        last = last_point[str(last[0]) + '+' + str(last[1]) + '+' + str(last[2])]
    path_arr = path[0][np.newaxis, :]
    for i in range(1, len(path)):
        path_arr = np.concatenate([path_arr, np.array(path[i])[np.newaxis, :]])
    np.save('test_path.npy', path_arr)
    print('have saved the npy')
    return 0


def cpr_process(img, path):
    y_list, p_list = [], []
    '''
    planar according to the XZ plane, so the indexes are 0 and 2. 
    if want to planar in the XY plane, please change the index to 0 and 1, i.e. replace as path[*][0] and path[*][1]. The YZ plane is similar.
    '''
    y_list.append(0)
    p_list.append(img[path[0][0], :, path[0][2]])
    for i in range(1, len(path)):
        delta_y = math.sqrt(math.pow(path[i][0] - path[i - 1][0], 2) + math.pow(path[i][2] - path[i - 1][2], 2))
        y_list.append(y_list[-1] + delta_y)
        p_list.append(img[path[i][0], :, path[i][2]])

    new_img = p_list[0][np.newaxis, :]
    for i in range(1, math.ceil(y_list[-1])):
        index = []
        for j in range(0, len(y_list)):
            if i + 1 >= y_list[j] >= i - 1:
                index.append(j)
        new_row = np.zeros((p_list[0].shape[0],))
        for j in index:
            new_row = new_row + p_list[j]
        new_row = new_row / len(index)
        new_img = np.concatenate([new_img, new_row[np.newaxis, :]])
    print(new_img.shape)
    return new_img


def cpr(img_name, center_line_name):
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
    path = np.load(center_line_name)
    path_b = path[0]
    path_e = path[-1]
    label = np.zeros(img.shape)
    for i in range(0, path.shape[0]):
        label[int(path[i][0]), int(path[i][1]), int(path[i][2])] = 1
    label = label.astype(np.float)
    for i in range(1, 6):
        path = np.concatenate([np.array([path_b[0] - i, path_b[1], path_b[2]])[np.newaxis, :], path], axis=0)
    for i in range(1, 6):
        path = np.concatenate([path, np.array([path_e[0] + i, path_e[1], path_e[2]])[np.newaxis, :]], axis=0)
    print(path[:, 0].max() - path[:, 0].min(), path[:, 1].max() - path[:, 1].min(), path[:, 2].max() - path[:, 2].min())
    print(img.shape)
    new_img = cpr_process(img, path)
    new_label = cpr_process(label, path)
    img_slicer = (((new_img - new_img.min()) / (new_img.max() - new_img.min())) * 255).astype(np.uint8)
    img_slicer = Image.fromarray(img_slicer)
    img_slicer = img_slicer.convert("RGB")
    img_slicer = np.array(img_slicer)
    index_label = np.where(new_label > 0.2)
    for i in range(0, len(index_label[0])):
        img_slicer[index_label[0][i], index_label[1][i]] = [255, 0, 0]
    img_slicer = Image.fromarray(img_slicer)
    img_slicer = img_slicer.transpose(Image.ROTATE_180)
    # img_slicer.save('show_img.png')


'''
Need to first convert the centerline image into a series of sequential points. 
A minimum path method for continuous centerline extraction, 
which is called find_point_list(thin_label_name, start, end), 
where thin_label_name is the path of 3D image, start is the coordinates of the starting point of the center line and end is the coordinates of the ending point of the center line.
This function will save the center line as an .npy file.

for example：
if __name__ == '__main__':
    root_path = './example/'
    img_file = root_path +'img.nii'
    thin_file = root_path + 'thin.nii'
    start, end = [71,107,170],[186,124,166]
    find_point_list = (thin_file, start,end)
    center_line = './cen.npy'
    cpr(img_file, center_line)
'''


# graph optimal transport
def cost_matrix(x, y):
    x = x.div(torch.norm(x, p=2, dim=0, keepdim=True) )
    y = y.div(torch.norm(y, p=2, dim=0, keepdim=True) )
    cost_distance = 1 - torch.mm(torch.transpose(y, 0, 1), x)
    return cost_distance

def OT(C, n, m, miu, nu, beta=0.5):
	# C is the distance matrix
	sigma = torch.ones(int(m), 1).float().cuda()/m
	T = torch.ones(n, m).cuda()
	C = torch.exp(-C/beta).float()
	for t in range(15):
		T = C * T
		for k in range(1):
			delta = miu / torch.squeeze(torch.matmul(T, sigma))
			sigma = torch.unsqueeze(nu,1) / torch.matmul(torch.transpose(T,0,1), torch.unsqueeze(delta,1))
		T = torch.unsqueeze(delta,1) * T * sigma.transpose(1,0)
	return T.detach()

def W_distance(C, n, m):
    T = OT(C, n, m)
    distance = torch.trace(torch.matmul(C, T.t()))
    return W_distance


def init_layer(layer, activation_type = 'tanh'):
    if isinstance(layer, nn.Conv2d):
        if activation_type == 'tanh':
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")


def init_bn(bn):
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)


class DCTN(nn.Module):

    def __init__(self):
        super(MyNET2, self).__init__()
        self.conv_block_down1 = Extractor(in_channels=1, out_channels=64)
        self.conv_block_down2 = Extractor(in_channels=64, out_channels=128)
        self.conv_block_down3 = Extractor(in_channels=128, out_channels=256)
        self.conv_block_down4 = Extractor(in_channels=256, out_channels=512)

        self.conv_block = ConvBlock_Down(in_channels=512, out_channels=512)

        self.conv_block_up4 = Decoder(in_channels=512, out_channels=256)
        self.conv_block_up3 = Decoder(in_channels=256, out_channels=128)
        self.conv_block_up2 = Decoder(in_channels=128, out_channels=64)
        self.conv_block_up1 = Decoder(in_channels=64, out_channels=32)


	self.conv_block_guide4 = Guider(in_channels=512, out_channels=256)
        self.conv_block_guide3 = Guider(in_channels=256, out_channels=128)
        self.conv_block_guide2 = Guider(in_channels=128, out_channels=64)
        self.conv_block_guide1 = Guider(in_channels=64, out_channels=32)

        self.final_conv = nn.Conv1d(32, 4, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1,x1_pooled = self.conv_block_down1(x)
        x2,x2_pooled = self.conv_block_down2(x1)
        x3,x3_pooled = self.conv_block_down3(x2)
        x4,x4_pooled = self.conv_block_down4(x3)

        out1,_ = self.conv_block(x4,is_pool=False)

        # Decoder
        out1-0 = self.conv_block_up4(out1, x4)
        out1-1 = self.conv_block_up3(out1-0, x3)
        out1-2 = self.conv_block_up2(out1-1, x2)
        out1-3 = self.conv_block_up1(out1-2, x1)
	    
        out1 = self.final_conv(out1-3)

        out2-0 = self.conv_block_guide4(out1-3, x4_pooled)
        out2-1 = self.conv_block_guide3(out1-2 x3_pooled)
        out2-2 = self.conv_block_guide2(out1-1, x2_pooled)
        out2-3 = self.conv_block_guide1(out1-0, x1_pooled)
	    
        out2 = self.final_conv(out2-3)

        # return output
        return out1, out2
