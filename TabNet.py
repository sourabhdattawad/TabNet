#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np

from sparsemax import Sparsemax
sparsemax = Sparsemax(dim=1)

# GLU 
def glu(act, n_units):
    
    act[:, :n_units] = act[:, :n_units].clone() * torch.nn.Sigmoid()(act[:, n_units:].clone())     
    
    return act

class TabNetModel(nn.Module):
    
    def __init__(
        self,
        columns = 3,
        num_features = 3,
        feature_dims = 128,
        output_dim  =64,
        num_decision_steps =6,
        relaxation_factor = 0.5,
        batch_momentum = 0.001,
        virtual_batch_size = 2,
        num_classes = 2,
        epsilon = 0.00001
    ):
        
        super().__init__()
        
        self.columns = columns
        self.num_features  = num_features
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        self.feature_transform_linear1 = torch.nn.Linear(num_features, self.feature_dims * 2, bias=False)
        self.BN = torch.nn.BatchNorm1d(num_features, momentum = batch_momentum)
        self.BN1 = torch.nn.BatchNorm1d(self.feature_dims * 2, momentum = batch_momentum)
        
        self.feature_transform_linear2 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear3 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        self.feature_transform_linear4 = torch.nn.Linear(self.feature_dims * 2, self.feature_dims * 2, bias=False)
        
        self.mask_linear_layer = torch.nn.Linear(self.feature_dims * 2-output_dim, self.num_features, bias=False)
        self.BN2 = torch.nn.BatchNorm1d(self.num_features, momentum = batch_momentum)
        
        self.final_classifier_layer = torch.nn.Linear(self.output_dim, self.num_classes, bias=False)
    
    def encoder(self, data):
        
        batch_size = data.shape[0]
        features = self.BN(data)
        output_aggregated = torch.zeros([batch_size, self.output_dim])
        
        masked_features = features
        mask_values = torch.zeros([batch_size, self.num_features])
        
        aggregated_mask_values = torch.zeros([batch_size, self.num_features])
        complemantary_aggregated_mask_values =torch.ones([batch_size, self.num_features])
        
        total_entropy = 0

        for ni in range(self.num_decision_steps):
            
            if ni==0:
                
                transform_f1  = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2      = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)
            
            else:

                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2      = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

                # GLU 
                transform_f2 = (glu(norm_transform_f2, self.feature_dims) +transform_f1) * np.sqrt(0.5)

                transform_f3 = self.feature_transform_linear3(transform_f2)
                norm_transform_f3 = self.BN1(transform_f3)

                transform_f4 = self.feature_transform_linear4(norm_transform_f3)
                norm_transform_f4 = self.BN1(transform_f4)

                # GLU
                transform_f4 = (glu(norm_transform_f4, self.feature_dims) + transform_f3) * np.sqrt(0.5)
                
                decision_out = torch.nn.ReLU(inplace=True)(transform_f4[:, :self.output_dim])
                # Decision aggregation
                output_aggregated  = torch.add(decision_out, output_aggregated)
                scale_agg = torch.sum(decision_out, axis=1, keepdim=True) / (self.num_decision_steps - 1)
                aggregated_mask_values  = torch.add( aggregated_mask_values, mask_values * scale_agg)

                features_for_coef = (transform_f4[:, self.output_dim:])
                               
                if ni<(self.num_decision_steps-1):

                    mask_linear_layer = self.mask_linear_layer(features_for_coef)
                    mask_linear_norm = self.BN2(mask_linear_layer)
                    mask_linear_norm  = torch.mul(mask_linear_norm, complemantary_aggregated_mask_values)
                    mask_values = sparsemax(mask_linear_norm)
                    
                    complemantary_aggregated_mask_values = torch.mul(complemantary_aggregated_mask_values,self.relaxation_factor - mask_values)
                    total_entropy = torch.add(total_entropy,torch.mean(torch.sum(-mask_values * torch.log(mask_values + self.epsilon),axis=1)) / (self.num_decision_steps - 1))
                    masked_features = torch.mul(mask_values , features)
           
        return  output_aggregated, total_entropy
     
    def classify(self, output_logits):

        logits = self.final_classifier_layer(output_logits)
        predictions = torch.nn.Softmax(dim=1)(logits)

        return logits, predictions
       
