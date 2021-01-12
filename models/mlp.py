import torch
import os
import numpy as np
from torch import nn

class MLP(nn.Module):
	def __init__(self, args, in_size, mlp_size, out_size):
		super(MLP, self).__init__()
		self.args = args

		self.in_size = in_size
		self.mlp_size = mlp_size
		self.out_size = out_size

		self.hidden_0 = nn.Linear(in_features=self.in_size, out_features=self.mlp_size, bias=True)
		self.activation_0 = nn.Tanh()
		self.hidden_1 = nn.Linear(in_features=self.mlp_size, out_features=self.out_size, bias=True)

	def forward(self, x):
		"""

		:param x: [batch_size, in_size]
		:return:
		"""
		# [batch_size, mlp_size]
		output_0 = self.hidden_0(x)
		output_0 = self.activation_0(output_0)

		# [batch_size, out_size]
		output_1 = self.hidden_1(output_0)

		return output_1



