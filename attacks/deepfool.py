import torch
import numpy as np
import os
from copy import deepcopy
from torch import nn
from torch.autograd.gradcheck import zero_gradients

class DeepFool(nn.Module):
	def __init__(self, args, num_classes, max_iters, overshoot=0.02):
		super(DeepFool, self).__init__()
		self.args = args
		self.num_classes = num_classes
		self.loops_needed = None

		self.max_iters = max_iters
		self.overshoot = overshoot
		self.loops = 0
		
	def forward(self, vecs, net_, target=None):
		"""

		:param vecs: [batch_size, vec_size]
		:param net_: FFNN in our case
		:param target:
		:return:
		"""
		net = deepcopy(net_)
		sent_vecs = deepcopy(vecs.data)
		input_shape = sent_vecs.size()

		f_vecs = net.forward(sent_vecs).data

		I = torch.argsort(f_vecs, dim=1, descending=True)
		I = I[:, 0:self.num_classes]

		# this is actually the predicted label
		label = I[:, 0]

		if target is not None:
			I = target.unsqueeze(1)
			if self.args.dataset == 'imdb':
				num_classes = 2
			elif self.args.dataset == 'agnews':
				num_classes = 4
			else:
				print('Unrecognized dataset {}'.format(self.args.dataset))
		else:
			num_classes = I.size(1)

		pert_vecs = deepcopy(sent_vecs)
		r_tot = torch.zeros(input_shape)
		check_fool = deepcopy(sent_vecs)

		k_i = label
		loop_i = 0

		# pre-define an finish_mask, [batch_size], all samples are not finished at first
		finish_mask = torch.zeros((input_shape[0], 1), dtype=torch.float)
		finished = torch.ones_like(finish_mask)
		self.loops_needed = torch.zeros((input_shape[0],))

		if torch.cuda.is_available():
			r_tot = r_tot.cuda()
			finish_mask = finish_mask.cuda()
			finished = finished.cuda()
			self.loops_needed = self.loops_needed.cuda()

		# every sample needs to be finished, and total loops should be smaller than max_iters
		while torch.sum(finish_mask >= finished) != input_shape[0] and loop_i < self.max_iters:
			x = pert_vecs.requires_grad_(True)
			fs = net.forward(x)

			pert = torch.ones(input_shape[0])*np.inf
			w = torch.zeros(input_shape)

			if torch.cuda.is_available():
				pert = pert.cuda()
				w = w.cuda()

			# fs[sample_index, I[sample_index, sample_label]]
			logits_label_sum = torch.gather(fs, dim=1, index=label.unsqueeze(1)).sum()
			logits_label_sum.backward(retain_graph=True)
			grad_orig = deepcopy(x.grad.data)

			for k in range(1, num_classes):
				if target is not None:
					k = k - 1
					if k > 0:
						break

				zero_gradients(x)
				# fs[sample_index, I[sample_index, sample_class]]
				logits_class_sum = torch.gather(fs, dim=1, index=I[:, k].unsqueeze(1)).sum()
				logits_class_sum.backward(retain_graph=True)

				# [batch_size, n_channels, height, width]
				cur_grad = deepcopy(x.grad.data)
				w_k = cur_grad - grad_orig

				# fs[sample_index, I[sample_index, sample_class]] - fs[sample_index, I[sample_index, sample_label]]
				f_k = torch.gather(fs, dim=1, index=I[:, k].unsqueeze(1)) - torch.gather(fs, dim=1, index=label.unsqueeze(1))
				f_k = f_k.squeeze(-1)

				# element-wise division
				pert_k = torch.div(torch.abs(f_k), self.norm_dim(w_k))

				valid_pert_mask = pert_k < pert

				new_pert = pert_k + 0.
				new_w = w_k + 0.

				valid_pert_mask = valid_pert_mask.bool()
				pert = torch.where(valid_pert_mask, new_pert, pert)
				# index by valid_pert_mask

				valid_w_mask = torch.reshape(valid_pert_mask, shape=(input_shape[0], 1)).float()
				valid_w_mask = valid_w_mask.bool()

				w = torch.where(valid_w_mask, new_w, w)

			r_i = torch.mul(torch.clamp(pert, min=1e-4).reshape(-1, 1), w)
			r_i = torch.div(r_i, self.norm_dim(w).reshape((-1, 1)))

			r_tot_new = r_tot + r_i

			# if get 1 for cur_update_mask, then the sample has never changed its label, we need to update it
			cur_update_mask = (finish_mask < 1.0).byte()
			if torch.cuda.is_available():
				cur_update_mask = cur_update_mask.cuda()

			cur_update_mask = cur_update_mask.bool()

			r_tot = torch.where(cur_update_mask, r_tot_new, r_tot)

			# r_tot already filtered with cur_update_mask, no need to do again
			pert_vecs = sent_vecs + r_tot
			check_fool = sent_vecs + (1.0 + self.overshoot) * r_tot

			k_i = torch.argmax(net.forward(check_fool.requires_grad_(True)), dim=-1).data

			if target is None:
				# in untargeted version, we finish perturbing when the network changes its predictions to the advs
				finish_mask += ((k_i != label)*1.0).reshape((-1, 1)).float()
				# print(torch.sum(finish_mask >= finished))
			else:
				# in targeted version, we finish perturbing when the network classifies the advs as the target class
				finish_mask += ((k_i == target)*1.0).reshape((-1, 1)).float()

			loop_i += 1
			self.loops += 1
			self.loops_needed[cur_update_mask.squeeze()] = loop_i

			r_tot.detach_()
			check_fool.detach_()
			r_i.detach_()
			pert_vecs.detach_()

		# grad is not really need for deepfool, used here as an additional check
		x = pert_vecs.requires_grad_(True)
		fs = net.forward(x)

		torch.sum(torch.gather(fs, dim=1, index=k_i.unsqueeze(1)) - torch.gather(fs, dim=1, index=label.unsqueeze(1))).backward(retain_graph=True)

		grad = deepcopy(x.grad.data)
		grad = torch.div(grad, self.norm_dim(grad).unsqueeze(1))

		label = deepcopy(label.data)

		if target is not None:
			# in targeted version, we move an adv towards the true class, but we do not want to cross the boundary
			pert_vecs = deepcopy(pert_vecs.data)
			return grad, pert_vecs, label
		else:
			# check_fool should be on the other side of the decision boundary
			check_fool_vecs = deepcopy(check_fool.data)
			return grad, check_fool_vecs, label

	@staticmethod
	def norm_dim(w):
		norms = []
		for idx in range(w.size(0)):
			norms.append(w[idx].norm())
		norms = torch.stack(tuple(norms), dim=0)

		return norms
