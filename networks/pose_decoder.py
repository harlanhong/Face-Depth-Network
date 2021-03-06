# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
import pdb
import torch.nn.functional as F
# from options import MonodepthOptions
# options = MonodepthOptions()
# opts = options.parse()
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,opts=None):
        super(PoseDecoder, self).__init__()
        self.opts = opts
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.convs[("intrinsics", 'focal')] = nn.Conv2d(256, 2, kernel_size = 3,stride = 1,padding = 1)
        self.convs[("intrinsics", 'offset')] = nn.Conv2d(256, 2, kernel_size = 3,stride = 1,padding = 1)
        # self.convs[("intrinsics", 'focal')] = nn.Linear(in_features = 256, out_features = 2)
        # self.convs[("intrinsics", 'offset')] = nn.Linear(in_features = 256, out_features = 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        feat = cat_features
        for i in range(2):
            feat = self.convs[("pose", i)](feat)
            feat = self.relu(feat)
        out = self.convs[("pose", 2)](feat)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        #add_intrinsics_head
        scales = torch.tensor([self.opts.width,self.opts.height]).cuda()
        focals = F.softplus(self.convs[("intrinsics", 'focal')](feat)).mean(3).mean(2)*scales
        offset = (F.softplus(self.convs[("intrinsics", 'offset')](feat)).mean(3).mean(2)+0.5)*scales
        # pdb.set_trace()
        eyes = torch.eye(2).cuda()
        b,xy = focals.shape
        focals = focals.unsqueeze(-1).expand(b,xy,xy)
        eyes = eyes.unsqueeze(0).expand(b,xy,xy)
        intrin = focals*eyes
        # intrin = torch.diag_embed(focals, offset=0, dim1=-2, dim2=-1)
        offset = offset.view(b,2,1).contiguous()
        intrin = torch.cat([intrin,offset],-1)
        pad = torch.tensor([0.0,0.0,1.0]).view(1,1,3).expand(b,1,3).cuda()
        intrinsics = torch.cat([intrin,pad],1)
        return axisangle, translation,intrinsics

    def forward1(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        feat = cat_features
        for i in range(2):
            feat = self.convs[("pose", i)](feat)
            feat = self.relu(feat)
        out = self.convs[("pose", 2)](feat)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        #add_intrinsics_head
        pool_feat = self.global_pool(feat).squeeze(-1).squeeze(-1)
        scales = torch.tensor([self.opts.width,self.opts.height]).cuda()
        focals = F.softplus(self.convs[("intrinsics", 'focal')](pool_feat))*scales
        offset = (F.softplus(self.convs[("intrinsics", 'offset')](pool_feat))+0.5)*scales
        #focals = F.softplus(self.convs[("intrinsics",'focal')](feat).mean(3).mean(2))
        #offset = F.softplus(self.convs[("intrinsics",'offset')](feat).mean(3).mean(2))
        # pdb.set_trace()
        # eyes = torch.eye(2).cuda()
        b,xy = focals.shape
        # focals = focals.unsqueeze(-1).expand(b,xy,xy)
        # eyes = eyes.unsqueeze(0).expand(b,xy,xy)
        # intrin = focals*eyes

        intrin = torch.diag_embed(focals, offset=0, dim1=-2, dim2=-1)
        offset = offset.view(b,2,1).contiguous()
        intrin = torch.cat([intrin,offset],-1)
        pad = torch.tensor([0.0,0.0,1.0]).view(1,1,3).expand(b,1,3).cuda()
        intrinsics = torch.cat([intrin,pad],1)
        return axisangle, translation,intrinsics

class PatchPoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,opts=None):
        super(PatchPoseDecoder, self).__init__()
        self.opts = opts
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.convs[("intrinsics", 'focal')] = nn.Conv2d(256, 2, kernel_size = 3,stride = 1,padding = 1)
        self.convs[("intrinsics", 'offset')] = nn.Conv2d(256, 2, kernel_size = 3,stride = 1,padding = 1)
        # self.convs[("intrinsics", 'focal')] = nn.Linear(in_features = 256, out_features = 2)
        # self.convs[("intrinsics", 'offset')] = nn.Linear(in_features = 256, out_features = 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        pdb.set_trace()
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        feat = cat_features
        for i in range(2):
            feat = self.convs[("pose", i)](feat)
            feat = self.relu(feat)
        out = self.convs[("pose", 2)](feat)

        # out = out.mean(3).mean(2)
        bs,c,w,h = out.shape
        out = 0.01 * out.view(bs, self.num_frames_to_predict_for, 6,w,h)

        axisangle = out[:,:,:3,:,:]
        translation = out[:,:,3:,:,:]
        #add_intrinsics_head
        scales = torch.tensor([self.opts.width,self.opts.height]).cuda()
        focals = F.softplus(self.convs[("intrinsics", 'focal')](feat)).mean(3).mean(2)*scales
        offset = (F.softplus(self.convs[("intrinsics", 'offset')](feat)).mean(3).mean(2)+0.5)*scales
        # pdb.set_trace()
        eyes = torch.eye(2).cuda()
        b,xy = focals.shape
        focals = focals.unsqueeze(-1).expand(b,xy,xy)
        eyes = eyes.unsqueeze(0).expand(b,xy,xy)
        intrin = focals*eyes
        # intrin = torch.diag_embed(focals, offset=0, dim1=-2, dim2=-1)
        offset = offset.view(b,2,1).contiguous()
        intrin = torch.cat([intrin,offset],-1)
        pad = torch.tensor([0.0,0.0,1.0]).view(1,1,3).expand(b,1,3).cuda()
        intrinsics = torch.cat([intrin,pad],1)
        return axisangle, translation,intrinsics

class PixelPoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,opts=None):
        super(PixelPoseDecoder, self).__init__()
        self.opts = opts
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(6, 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.convs[("intrinsics", 'focal')] = nn.Conv2d(256, 2, kernel_size = 3,stride = 1,padding = 1)
        self.convs[("intrinsics", 'offset')] = nn.Conv2d(256, 2, kernel_size = 3,stride = 1,padding = 1)
        # self.convs[("intrinsics", 'focal')] = nn.Linear(in_features = 256, out_features = 2)
        # self.convs[("intrinsics", 'offset')] = nn.Linear(in_features = 256, out_features = 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        # last_features = [f[-1] for f in input_features]
        # pdb.set_trace()
        # cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = self.relu(self.convs["squeeze"](input_features))
        # cat_features = torch.cat(cat_features, 1)

        feat = cat_features
        for i in range(2):
            feat = self.convs[("pose", i)](feat)
            feat = self.relu(feat)
        out = self.convs[("pose", 2)](feat)

        # out = out.mean(3).mean(2)
        bs,c,w,h = out.shape
        out = 0.01 * out.view(bs, self.num_frames_to_predict_for, 6,w,h)

        axisangle = out[:,:,:3,:,:]
        translation = out[:,:,3:,:,:]
        #add_intrinsics_head
        scales = torch.tensor([self.opts.width,self.opts.height]).cuda()
        focals = F.softplus(self.convs[("intrinsics", 'focal')](feat)).mean(3).mean(2)*scales
        offset = (F.softplus(self.convs[("intrinsics", 'offset')](feat)).mean(3).mean(2)+0.5)*scales
        # pdb.set_trace()
        eyes = torch.eye(2).cuda()
        b,xy = focals.shape
        focals = focals.unsqueeze(-1).expand(b,xy,xy)
        eyes = eyes.unsqueeze(0).expand(b,xy,xy)
        intrin = focals*eyes
        # intrin = torch.diag_embed(focals, offset=0, dim1=-2, dim2=-1)
        offset = offset.view(b,2,1).contiguous()
        intrin = torch.cat([intrin,offset],-1)
        pad = torch.tensor([0.0,0.0,1.0]).view(1,1,3).expand(b,1,3).cuda()
        intrinsics = torch.cat([intrin,pad],1)
        return axisangle, translation,intrinsics
