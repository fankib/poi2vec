#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config

class POI2VEC(nn.Module):
    def __init__(self, poi_cnt, user_cnt, id2route, id2lr, id2prob):
        super(POI2VEC, self).__init__()

        # attributes
        route_cnt = np.power(2, config.route_depth)-1
        self.id2route = id2route
        self.id2lr = np.array(id2lr)
        self.id2prob = np.array(id2prob)

        # models
        self.poi_weight = nn.Embedding(poi_cnt, config.feat_dim, padding_idx=0)
        self.poi_weight.weight.data.normal_(config.weight_m, config.weight_v)
        self.user_weight = nn.Embedding(user_cnt, config.feat_dim, padding_idx=0)
        self.user_weight.weight.data.normal_(config.weight_m, config.weight_v)
        self.route_weight = nn.Embedding(route_cnt, config.feat_dim, padding_idx=0)
        self.route_weight.weight.data.normal_(config.weight_m, config.weight_v)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, context, target, device):
        #target = map(int, target)
        target = [int(t) for t in target]
        route = Variable(torch.from_numpy(self.id2route[target]))\
                        .contiguous().view(-1, config.route_count*config.route_depth).long().to(device)
                        # batch x (route_coutn(4) x route_dept(22))
        lr = Variable(torch.from_numpy(self.id2lr[target]))\
                        .view(-1, config.route_count*(config.route_depth)).float().to(device)
                        # batch x (route_count(4) x route_depth(21))
        prob = Variable(torch.from_numpy(self.id2prob[target]))\
                        .view(-1, config.route_count).float().to(device) # batch x route_count(4)

        context = self.poi_weight(context) # batch x context_len(32) x feat_dim(200)
        route = self.route_weight(route) # batch x (route_count(4) x route_depth(22)) x feat_dim(200)
        user = self.user_weight(user) # batch x feat_dim(200)
        target = Variable(torch.from_numpy(np.asarray(target)).long()).to(device)
        target = self.poi_weight(target)

        phi_context = torch.sum(context, dim=1, keepdim=True).permute(0,2,1) # batch x feat_dim x 1
        psi_context = torch.bmm(route, phi_context) # batch x (route_count x route_depth) x 1
        psi_context = self.sigmoid(psi_context).view(-1, config.route_count*config.route_depth)

        psi_context = (torch.pow(torch.mul(psi_context, 2), lr) - psi_context)\
                        .view(-1, config.route_count, config.route_depth)

        pr_path = 1
        for i in range(config.route_depth):
            pr_path = torch.mul(psi_context[:,:,i], pr_path)
        pr_path = torch.sum(torch.mul(pr_path, prob), 1)
        
        pr_user = torch.mm(user, self.poi_weight.weight.t())
        pr_user = torch.sum(torch.exp(pr_user), 1)
        pr_user = torch.div(torch.exp(torch.sum(torch.mul(target, user), 1)), pr_user)
        #pr_ult = 1.0-torch.sum(torch.mul(pr_user, pr_path))
        
        pr_ult = pr_path

        return pr_ult
        
class Rec:
    # Rectangle for calculate overlaped area
    def __init__(self, rectangle):
        top, down, left, right = rectangle
        self.top = top
        self.down = down
        self.left = left
        self.right = right

    def overlap(self, a): 
        dx = min(self.top, a.top) - max(self.down, a.down)
        dy = min(self.right, a.right) - max(self.left, a.left)
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            # error
            return -1

class Node:
# Tree Node
    theta = 0.5
    count = 0 
    leaves = []

    def __init__(self, west, east, north, south, level):
        self.left = None
        self.right = None
        self.west = west
        self.east = east
        self.north = north
        self.south = south
        self.level = level
        Node.count += 1
        self.count = Node.count

    def build(self):
        # even : horizen, odd : vertical
        if self.level%2 == 0:
            if (self.east - (self.west+self.east)/2) > 2*Node.theta:
                self.left = Node(self.west, (self.west+self.east)/2, self.north, self.south, self.level+1)
                self.right = Node((self.west+self.east)/2, self.east, self.north, self.south, self.level+1)
                self.left.build()
                self.right.build()
            else:
                Node.leaves.append(self)
        else:
            if (self.north - (self.north+self.south)/2) > 2*Node.theta:
                self.left = Node(self.west, self.east, self.north, (self.north+self.south)/2, self.level+1)
                self.right = Node(self.west, self.east, (self.north+self.south)/2, self.south, self.level+1)
                self.left.build()
                self.right.build()
            else:
                Node.leaves.append(self)

    def find_route(self, coords):
        latitude, longitude = coords
        if self.left == None:
            prev_route = [self.count]
            prev_lr = []
            return prev_route, prev_lr

        # left : 0, right : 1
        if self.level%2 == 0:
            if self.left.east < latitude:
                prev_route, prev_lr = self.right.find_route((latitude, longitude))
                prev_lr.append(1)
            else:
                prev_route, prev_lr = self.left.find_route((latitude, longitude))
                prev_lr.append(0)
        else:
            if self.left.south < longitude:
                prev_route, prev_lr = self.left.find_route((latitude, longitude))
                prev_lr.append(0)
            else:
                prev_route, prev_lr = self.right.find_route((latitude, longitude))
                prev_lr.append(1)
        prev_route.append(self.count)
        return prev_route, prev_lr

    def find_idx(self, idx):
        # find in leaves
        for leaf in Node.leaves:
            if leaf.count == idx:
                return leaf.north, leaf.south, leaf.west, leaf.east
