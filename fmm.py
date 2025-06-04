"""
Module responsible for Fourier-domain Moment Matching (FMM)
for data augmentation and domain adaptation.
This module computes spectral statistics (mean, standard deviation, covariance)
and performs moment matching in the Fourier domain.
"""
import math
import torch as ch
from torchvision import transforms
import numpy as np
import random
import sys
import torch.fft # Must import fft lib before using fft functions
from os.path import exists
from torch.optim import SGD, lr_scheduler, Adam
import torch.nn as nn
from math import sqrt

from PIL import Image

# IMPORTANT: This path append is specific to the original environment.
# For your setup, ensure 'robustness' library and 'sqrtm.py' are correctly
# installed and accessible in your Python environment.
sys.path.append('/home/kiran/robustness/robustness')
from sqrtm import matsqrt, inverse, BLOCK_DIAG_THRESHOLD, get_diagonal_sub_matrices

class FMM(object):
	"""
	Fourier-domain Moment Matching (FMM) class.
	Matches spectral moments (mean, std, or covariance) of a source dataset
	to a target dataset.
	"""
	def __init__(self, target_dataset, batch_size, source_loader=None, \
					source_dataset=None, \
				    match_cov=True, target_loader=None, mean_only=False, \
					ledoit_wolf_correction=False, \
					block_diag=False, large_img_sample_size=9999, \
					use_2D_dft=False):
		"""
		Initializes the FMM object.

		Args:
			target_dataset (str): Name of the target dataset (e.g., 'cifar10', 'mnist').
			batch_size (int): Batch size for data loaders.
			source_loader (torch.utils.data.DataLoader): DataLoader for the source dataset.
			source_dataset (str, optional): Name of the source dataset.
			match_cov (bool, optional): If True, matches covariance; otherwise, matches mean/std. Defaults to True.
			target_loader (torch.utils.data.DataLoader, optional): DataLoader for the target dataset.
				Required if `target_dataset` is not one of the built-in ones handled by `fmm.py`.
			mean_only (bool, optional): If True, only matches the mean (requires `match_cov=False`). Defaults to False.
			ledoit_wolf_correction (bool, optional): Apply Ledoit-Wolf covariance correction. Defaults to False.
			block_diag (bool or int, optional): If True or an integer, uses a block-diagonal approximation
				for covariance matrices. If an integer, it specifies the block size. Defaults to False.
			large_img_sample_size (int, optional): Maximum number of large images to sample
				for statistics computation to reduce runtime. Defaults to 9999.
			use_2D_dft (bool, optional): If True, uses 2D DFT (on H, W dimensions) instead of 3D DFT (on C, H, W).
				Defaults to False.
		"""
		self.diag_lambda = 0.001 # For making covariance matrices full-rank (adding diagonal noise).
		self.mean_only = mean_only
		self.block_diag = block_diag
		self.target_dataset = target_dataset
		self.source_dataset = source_dataset
		self.use_2D_dft = use_2D_dft

		print(f'==> FMM large img sample size: {large_img_sample_size}')

		if use_2D_dft:
			print('==> Using 2D DFT')

		if block_diag:
			assert match_cov == True, '`block_diag` must be used with `match_cov=True`'
			print(f'==> Block Diag approx., block-size: {block_diag}')
			
		if self.mean_only:
			assert not match_cov, '`mean_only` cannot be used with `match_cov=True`'

		self.MAX_N_LARGE_IMAGES = large_img_sample_size # Sample size limit for large images

		self.batch_size = batch_size

		self.target_loaders = []

		# Load target dataset loaders based on dataset name
		if target_dataset in ['mnist', 'mnistm', 'svhn', 'usps', 'synth']:
			print(f'Loading target loader {target_dataset}')
			# robustness datasets import
			from .datasets import DATASETS
			data_path = f'/home/i2r/datasets/{target_dataset}'
			tdataset = DATASETS[target_dataset](data_path)
			# num_workers=0 to avoid BrokenPipeError with certain dataloaders
			train_loader, val_loader = tdataset.make_loaders(0, batch_size, data_aug=True)
			self.target_loaders = [val_loader] # Target is typically the test set of the target domain
		
		# NOTES on dataloaders:
		# - Each dataloader per distortion contains all severity levels (severity=None returns all severities).
		# - Avoid multiple workers for these dataloaders (see num_workers=0) to prevent BrokenPipeError.
		# - Use drop_last=True (implicitly handled by some loaders) to ensure input and target batch sizes are equal.
		# - Use shuffle to randomly sample from target dataset.
		# - Use fixed batch size to match source minibatches.

		elif target_dataset == 'cifar10':
			# For reverse FMM or other specific CIFAR10 use cases
			from . import datasets # robustness datasets
			cifar10_dataset = datasets.CIFAR('/tmp')
			train_loader, val_loader = cifar10_dataset.make_loaders(0, batch_size, data_aug=True)
			self.target_loaders = [train_loader]

		elif 'OfficeHome' in target_dataset or 'Cityscapes' in target_dataset:
			self.target_loaders = [target_loader] # Assume target_loader is provided externally

		elif 'Digits' in target_dataset:
			self.target_loaders = [target_loader] # Assume target_loader is provided externally

		elif 'ImageNet50_' in target_dataset or 'CIFAR10C_' in target_dataset or \
			'Camelyon17' in target_dataset or target_dataset in ['fmow', 'iwildcam', 'camelyon17']:
			self.target_loaders = [target_loader] # Assume target_loader is provided externally

		else:
			raise Exception(f'Unrecognized target dataset: {target_dataset}')

		print(f'*** Number of target data loaders: {len(self.target_loaders)} ***')

		import time
		start_time = time.time()

		self.match_cov = match_cov
		print('Computing statistics for amplitude spectrum')

		if self.match_cov:
			print('==> Matching Covariance')
			print(f'==> Ledoit-Wolf covariance correction: {ledoit_wolf_correction}')
		else:
			if self.mean_only:
				print('*** Matching mean only ***')
			else:
				print('*** Matching mean/std only ***')

		assert len(self.target_loaders) == 1, 'Only one target loader is expected for statistics computation.'

		# Ensure source_loader is the underlying torch DataLoader if it's wrapped by DataPrefetcher
		if source_loader and hasattr(source_loader, 'loader'):
			source_loader = source_loader.loader 

		if self.match_cov:
			print(f'Computing stats for target dataset train loader: {target_dataset}')
			if self.use_2D_dft:
				assert not ledoit_wolf_correction, 'Ledoit-Wolf correction unimplemented for 2D DFT'
				self.target_mean, self.target_cov = self.get_2D_spectral_cov(self.target_loaders[0])
			else:
				self.target_mean, self.target_cov = self.get_spectral_cov(self.target_loaders[0], ledoit_wolf_correction=ledoit_wolf_correction)

			# Get source stats
			print(f'Computing stats for source dataset loader: {source_dataset}')
			if self.use_2D_dft:
				assert not ledoit_wolf_correction, 'Ledoit-Wolf correction unimplemented for 2D DFT'
				self.source_mean, self.source_cov = self.get_2D_spectral_cov(source_loader)
			else:
				self.source_mean, self.source_cov = self.get_spectral_cov(source_loader, ledoit_wolf_correction=ledoit_wolf_correction)

			# Compute square root of target covariance matrix
			if self.use_2D_dft:
				if self.block_diag:
					self.target_cov_sqrt = [] # Channel-wise COV matrix sqrts (list of lists of blocks)
					for channel_covs in self.target_cov:
						self.target_cov_sqrt.append(matsqrt(channel_covs, block_diag=self.block_diag))
				else:
					# List of matrices, one per channel
					self.target_cov_sqrt = [matsqrt(channel_covs, block_diag=self.block_diag) for channel_covs in self.target_cov]
			else:
				# Single matrix for 3D DFT
				self.target_cov_sqrt = matsqrt(self.target_cov, block_diag=self.block_diag)

			del self.target_cov # Free up memory

			# Move target_cov_sqrt to GPU
			if self.block_diag:
				if self.use_2D_dft:
					for channel in range(len(self.target_cov_sqrt)):
						self.target_cov_sqrt[channel] = [mat.cuda() for mat in self.target_cov_sqrt[channel]]
				else:
					self.target_cov_sqrt = [mat.cuda() for mat in self.target_cov_sqrt]
			else:
				if self.use_2D_dft:
					for channel in range(len(self.target_cov_sqrt)):
						self.target_cov_sqrt[channel] = self.target_cov_sqrt[channel].cuda()
				else:
					self.target_cov_sqrt = self.target_cov_sqrt.cuda() # Move full matrix to GPU

			self.target_mean = self.target_mean.cuda()
			print('Target spectral mean shape', self.target_mean.shape)

			# Compute inverse square root of source covariance matrix
			if self.use_2D_dft:
				if self.block_diag:
					self.source_cov_inverse_sqrt = [] # Channel-wise COV matrix inverse sqrts
					for channel_covs in self.source_cov:
						self.source_cov_inverse_sqrt.append(matsqrt(inverse(channel_covs, block_diag=self.block_diag), block_diag=self.block_diag))
				else:
					# List of matrices, one per channel
					self.source_cov_inverse_sqrt = [matsqrt(inverse(channel_covs, block_diag=self.block_diag), block_diag=self.block_diag) for channel_covs in self.source_cov] 
			else:
				# Single matrix for 3D DFT
				self.source_cov_inverse_sqrt = matsqrt(inverse(self.source_cov, block_diag=self.block_diag), block_diag=self.block_diag)

			del self.source_cov # Free up memory

			# Move source_cov_inverse_sqrt to GPU
			if self.block_diag:
				if self.use_2D_dft:
					for channel in range(len(self.source_cov_inverse_sqrt)):
						self.source_cov_inverse_sqrt[channel] = [mat.cuda() for mat in self.source_cov_inverse_sqrt[channel]]
				else:
					self.source_cov_inverse_sqrt = [mat.cuda() for mat in self.source_cov_inverse_sqrt]
			else:
				if self.use_2D_dft:
					for channel in range(len(self.source_cov_inverse_sqrt)):
						self.source_cov_inverse_sqrt[channel] = self.source_cov_inverse_sqrt[channel].cuda()
				else:
					self.source_cov_inverse_sqrt = self.source_cov_inverse_sqrt.cuda() # Move entire matrix to GPU

			self.source_mean = self.source_mean.cuda()
			print('Source spectral mean shape', self.source_mean.shape)

			# Compute the coloring matrix: (Cov(S)**(-1/2)) * (Cov(T)**1/2)
			# Use float64 for coloring matrix computation for precision
			if self.block_diag:
				if self.use_2D_dft:
					self.coloring_matrix = []
					for channel in range(len(self.source_cov_inverse_sqrt)):
						self.coloring_matrix.append([torch.matmul(x.float(), y.float()) for (x,y) in zip(self.source_cov_inverse_sqrt[channel], self.target_cov_sqrt[channel])])
				else:
					self.coloring_matrix = [torch.matmul(x.float(), y.float()) for (x,y) in zip(self.source_cov_inverse_sqrt, self.target_cov_sqrt)]
			else:
				if self.use_2D_dft:
					self.coloring_matrix = [torch.matmul(x.float(), y.float()) for (x,y) in zip(self.source_cov_inverse_sqrt, self.target_cov_sqrt)]
				else:
					self.coloring_matrix = torch.matmul(self.source_cov_inverse_sqrt.float(), self.target_cov_sqrt.float())

			# Delete large intermediate matrices to free up GPU memory
			if hasattr(self, 'source_cov_inverse_sqrt'): del self.source_cov_inverse_sqrt
			if hasattr(self, 'target_cov_sqrt'): del self.target_cov_sqrt

		else:
			# Match only mean/std deviation
			print(f'Computing stats for target dataset train loader: {target_dataset}')

			spec = 'amp' # Amplitude spectrum
			target_stats_file = f'{target_dataset}_target_{spec}_mean_std.npy'
			source_stats_file = f'{source_dataset}_source_{spec}_mean_std.npy'

			# Attempt to load pre-computed target stats
			if exists(target_stats_file):
				target_stats = np.load(target_stats_file, allow_pickle=True).item()
				mean, std = target_stats['mean'], target_stats['std']
				print(f'*** Loading saved target stats from file {target_stats_file} ***')
				self.target_mean, self.target_std = torch.Tensor(mean).cuda(), torch.Tensor(std).cuda()
				print(self.target_mean.shape, self.target_std.shape)
			else:
				self.target_mean, self.target_std = self.get_spectral_mean_std(self.target_loaders[0])
				# Move to GPU
				self.target_mean, self.target_std = self.target_mean.cuda(), self.target_std.cuda()
				print('Target spectral mean and standard deviation shapes', self.target_mean.shape, self.target_std.shape)

			# Get source stats
			print(f'Computing stats for source dataset loader: {source_dataset}')
			# Attempt to load pre-computed source stats
			if exists(source_stats_file):
				source_stats = np.load(source_stats_file, allow_pickle=True).item()
				mean, std = source_stats['mean'], source_stats['std']
				print(f'*** Loading saved source stats from file {source_stats_file} ***')
				self.source_mean, self.source_std = torch.Tensor(mean).cuda(), torch.Tensor(std).cuda()
				print(self.source_mean.shape, self.source_std.shape)
			else:
				self.source_mean, self.source_std = self.get_spectral_mean_std(source_loader)
				# Move to GPU
				self.source_mean, self.source_std = self.source_mean.cuda(), self.source_std.cuda()
				print('Source spectral mean and standard deviation shapes', self.source_mean.shape, self.source_std.shape)

			# Compute the coloring matrix for mean/std matching (element-wise)
			self.coloring_matrix = (1 / self.source_std) * self.target_std

		print(f'Total {time.time()-start_time:.2f} seconds to compute statistics.')

	def get_mean_amp_spec(self, loader):
		"""
		Computes the mean amplitude spectrum across a dataset loader.
		(This method is not used in the current FMM __init__ or __call__ but might be for debugging/analysis).
		"""
		mean = None
		n = len(loader.dataset) # Size of dataset (number of samples)

		print(f'Computing spectral mean for {n} samples')

		for _, (x,y) in enumerate(loader):
			# Compute amplitude spectrum (3D DFT)
			x = ch.abs(ch.fft.fftn(x, dim=(1,2,3))) # (N,C,H,W) -> amplitude spectrum

			batch_sum = x.sum(dim=0) # Sum across batch dimension (N) -> (C,H,W)

			if mean is None:
				mean = batch_sum
			else:
				mean += batch_sum

		mean /= n
		return mean

	def get_2D_spectral_cov(self, loader, topk=None):
		"""
		Computes channel-wise mean and covariance of the 2D amplitude spectrum
		across a dataset loader.

		Args:
			loader (torch.utils.data.DataLoader): DataLoader for the dataset.
			topk (list, optional): Not fully implemented/used for top-k selection in covariance. Defaults to None.

		Returns:
			tuple: (mean, cov) - Mean and covariance of the 2D amplitude spectrum.
		"""
		mean, cov = None, None
		n = len(loader.dataset) # Size of dataset (number of samples)

		count_1, count_2 = 0, 0

		# --- Compute Mean ---
		for _, z in enumerate(loader):
			# Handle different data loader output formats (e.g., for CityscapesDataset)
			if type(z) is dict and self.target_dataset == 'CityscapesDataset':
				x = z['img']._data[0] # (N,3,512,512)
				assert x.shape[-3] == 3, "Expected 3 channels for CityscapesDataset image"
			else:
				x, y = z[0], z[1] # Assume (image, label) tuple

			# Calculate total dimension (C*H*W) for large image check
			dim = x.shape[-1] * x.shape[-2] * x.shape[-3]
			num_channels = x.shape[1]
			count_1 += x.shape[0] # Accumulate number of samples processed

			# Compute 2D DFT (on H, W dimensions) and get amplitude spectrum
			x = ch.abs(ch.fft.rfftn(x, dim=(2,3))) # (N,C,H,W) -> channel-wise amplitude spectrum

			if topk is not None:
				# This section for topk is not fully developed for covariance computation
				batch_sum = []
				for channel in range(num_channels):
					topk_x = x[:, channel, topk[channel]] # (N,k)
					batch_sum.append(topk_x.sum(dim=0).unsqueeze(0)) # (1,k)
				batch_sum = ch.cat(batch_sum, dim=0) # (C,k)
				assert batch_sum.shape == torch.Size([num_channels, topk[0].count_nonzero().item()])
			else:
				# Flatten H, W dimensions into a single dimension D for each channel
				batch_sum = x.flatten(start_dim=2, end_dim=3).sum(dim=0) # (N,C,H,W) -> (N,C,D) -> (C,D)

			if mean is None:
				mean = batch_sum
			else:
				mean += batch_sum

			# Stop accumulating for large images if sample size limit is reached
			if dim > BLOCK_DIAG_THRESHOLD:
				if count_1 > self.MAX_N_LARGE_IMAGES:
					break

		mean /= count_1 # Normalize mean by the number of samples

		print(f'Computing spectral mean and cov for {count_1} samples')

		# Initialize covariance storage (list of Nones for each channel)
		if self.block_diag:
			cov = [None, None, None] # Assuming 3 channels (RGB)
		else:
			cov = [None, None, None] # Assuming 3 channels (RGB)

		# --- Compute Covariance ---
		for _, z in enumerate(loader):
			if type(z) is dict and self.target_dataset == 'CityscapesDataset':
				x = z['img']._data[0]
				assert x.shape[-3] == 3
			else:
				x, y = z[0], z[1]

			x = ch.abs(ch.fft.rfftn(x, dim=(2,3))) # Channel-wise amplitude spectrum
	
			count_2 += x.shape[0]

			if topk is not None:
				batch_cov = []
				for channel in range(num_channels):
					topk_x = x[:, channel, topk[channel]].unsqueeze(1) # (N,1,k)
					batch_cov.append(topk_x)
				batch_cov = ch.cat(batch_cov, dim=1) # (N,C,k)
			else:
				batch_cov = x.flatten(start_dim=2, end_dim=3) # (N,C,H,W) -> (N, C, D)

			batch_cov -= mean # Center data (x - mean); (N, C, D)

			for channel in range(batch_cov.shape[1]): # Iterate over channels
				if self.block_diag:
					# Compute block-diagonal covariance for the current channel
					channel_batch_covs = self.get_block_covs(batch_cov[:,channel,:]) # Returns list of diagonal blocks
					if cov[channel] is None:
						cov[channel] = channel_batch_covs
					else:
						for block_idx in range(len(channel_batch_covs)):
							cov[channel][block_idx] += channel_batch_covs[block_idx]
				else:
					# Compute full covariance matrix for the current channel
					# sigma((x-u)*(x-u)^t) summed across samples
					temp_batch_cov = torch.matmul(batch_cov[:,channel,:].transpose(0,1), batch_cov[:,channel,:]) # (D,N) * (N,D) = (D,D)
					if cov[channel] is None:
						cov[channel] = temp_batch_cov
					else:
						cov[channel] += temp_batch_cov

			# Stop accumulating for large images if sample size limit is reached
			if dim > BLOCK_DIAG_THRESHOLD:
				if count_2 > self.MAX_N_LARGE_IMAGES:
					break

		assert count_1 == count_2, "Sample counts for mean and covariance computation do not match."

		print(f'==> Empirical covariance matrix. Adding diagonal noise to make invertible: {self.diag_lambda}')

		# Normalize covariance by (n-1) for Bessel's correction and add diagonal noise
		if type(cov) is list: # If channel-wise covariance (2D DFT)
			for channel_idx in range(len(cov)):
				if type(cov[channel_idx]) is list: # If block-diagonal
					for block_idx in range(len(cov[channel_idx])):
						cov[channel_idx][block_idx] /= (n-1)
						cov[channel_idx][block_idx] += (torch.eye(cov[channel_idx][block_idx].size(1)) * self.diag_lambda)
				else: # If full matrix per channel
					cov[channel_idx] /= (n-1)
					cov[channel_idx] += (torch.eye(cov[channel_idx].size(1)) * self.diag_lambda)
		else: # If single full covariance matrix (3D DFT)
			cov /= (n-1)
			cov += (torch.eye(cov.size(1)) * self.diag_lambda)

		return mean, cov

	def match_2D_mean_cov(self, x):
		"""
		Applies 2D Fourier-domain Moment Matching (mean and covariance) to input `x`.
		This method is used when `use_2D_dft` is True and `match_cov` is True.

		Args:
			x (torch.Tensor): Input image batch (N, C, H, W).

		Returns:
			torch.Tensor: Transformed image batch.
		"""
		fft = ch.fft.rfftn(x, dim=(2,3)) # Compute 2D DFT (on H, W dimensions)
		fmm_abs, angle = ch.abs(fft), ch.angle(fft) # Extract amplitude and phase

		original_shape = fmm_abs.shape
		# Flatten H, W dimensions to D for moment matching operations
		fmm_abs = fmm_abs.flatten(start_dim=2, end_dim=3) # (N,C,H,W) -> (N, C, D)

		# Center data by subtracting source mean
		fmm_abs -= self.source_mean

		# Whitening and coloring transformation
		for channel in range(x.shape[1]): # Iterate over channels
			if self.block_diag:
				# Apply block-diagonal multiplication for the current channel
				fmm_abs[:,channel,:] = self.block_multiply(fmm_abs[:,channel,:], self.coloring_matrix[channel])
			else:
				# Apply full matrix multiplication for the current channel
				# S = S * (Cov(S)**(-1/2)) * (Cov(T)**1/2)
				fmm_abs[:,channel,:] = torch.matmul(fmm_abs[:,channel,:], self.coloring_matrix[channel])
	
		# Add target mean
		fmm_abs += self.target_mean

		# Reshape back to original amplitude spectrum shape (N,C,H,W)
		fmm_abs = fmm_abs.reshape(original_shape)

		# Thresholding: Ensure amplitudes are non-negative
		fmm_abs[fmm_abs < 0] = 0

		# Reconstruct Fourier coefficients using transformed amplitude and original phase
		fft = fmm_abs * ch.exp((1j) * angle)
		# Inverse 2D DFT to get the transformed image
		fmm_x = ch.fft.irfftn(fft, dim=(2,3)).float()

		return fmm_x

	def get_spectral_cov(self, loader, ledoit_wolf_correction=False, topk=None):
		"""
		Computes the mean and covariance of the 3D amplitude spectrum across a dataset loader.

		Args:
			loader (torch.utils.data.DataLoader): DataLoader for the dataset.
			ledoit_wolf_correction (bool, optional): If True, applies Ledoit-Wolf shrinkage. Defaults to False.
			topk (list, optional): Not fully implemented/used for top-k selection. Defaults to None.

		Returns:
			tuple: (mean, cov) - Mean and covariance of the 3D amplitude spectrum.
		"""
		mean, cov = None, None
		count_1 = 0 # Number of samples used for mean computation

		# --- Compute Mean ---
		for _, z in enumerate(loader):
			x, y = z[0], z[1]

			# Handle specific dataset return formats (e.g., WILDS datasets like camelyon17)
			if type(x) is list and self.target_dataset in ['camelyon17', 'iwildcam']:
				x = x[0] # Use the weakly augmented input

			# Calculate total dimension (C*H*W) for large image check
			dim = x.shape[-1] * x.shape[-2] * x.shape[-3]
			count_1 += x.shape[0] # Accumulate number of samples processed

			# Compute 3D DFT (on C, H, W dimensions) and get amplitude spectrum
			x = ch.abs(ch.fft.rfftn(x, dim=(1,2,3))) # (N,C,H,W) -> amplitude spectrum

			if topk is not None:
				x = x[:, topk] # This topk usage is not fully specified for 3D DFT
				batch_sum = x.sum(dim=0)
			else:
				# Flatten C, H, W dimensions into a single dimension D
				batch_sum = x.flatten(start_dim=1, end_dim=3).sum(dim=0) # (N,C,H,W) -> (N,D) -> (D)
					
			if mean is None:
				mean = batch_sum
			else:
				mean += batch_sum

			# Stop accumulating for large images if sample size limit is reached
			if dim > BLOCK_DIAG_THRESHOLD:
				if count_1 > self.MAX_N_LARGE_IMAGES:
					break

		print(f'Computing spectral mean and cov for {count_1} samples')
		mean /= count_1 # Normalize mean by the number of samples

		count_2 = 0 # Number of samples used for covariance computation
		cov_2 = None # For Ledoit-Wolf correction

		# --- Compute Covariance ---
		for _, z in enumerate(loader):
			x, y = z[0], z[1]

			if type(x) is list and self.target_dataset in ['camelyon17', 'iwildcam']:
				x = x[0]

			count_2 += x.shape[0]

			x = ch.abs(ch.fft.rfftn(x, dim=(1,2,3))) # Amplitude spectrum

			if topk is not None:
				x = x[:, topk] # (N,k)
			else:
				x = x.flatten(start_dim=1, end_dim=3) # (N,C,H,W) -> (N, D)
			
			x -= mean # Center data (x - mean); (N, D)

			if self.block_diag:
				# Compute block-diagonal covariance
				batch_covs = self.get_block_covs(x) # Returns list of diagonal blocks
				if cov is None:
					cov = batch_covs
				else:
					for block_idx in range(len(batch_covs)):
						cov[block_idx] += batch_covs[block_idx]
			else:
				# Compute full covariance matrix: sigma((x-u)*(x-u)^t) summed across samples
				batch_cov = torch.matmul(x.transpose(0,1), x) # (D,N) * (N,D) = (D,D)

				if cov is None:
					cov = batch_cov
				else:
					cov += batch_cov
				
				if ledoit_wolf_correction:
					# For Ledoit-Wolf shrinkage, need sum of squared elements
					x2 = x**2
					batch_cov_2 = torch.matmul(x2.transpose(0,1), x2)
					if cov_2 is None:
						cov_2 = batch_cov_2
					else:
						cov_2 += batch_cov_2

			# Stop accumulating for large images if sample size limit is reached
			if dim > BLOCK_DIAG_THRESHOLD:
				if count_2 > self.MAX_N_LARGE_IMAGES:
					break

		assert count_1 == count_2, "Sample counts for mean and covariance computation do not match."

		if ledoit_wolf_correction:
			# Apply Ledoit-Wolf shrinkage (see sklearn for detailed implementation)
			n_features, n_samples = cov.shape[-1], count_1

			print('==> Computing Ledoit-Wolf correction')
			print(f'==> n_features: {n_features}')
			print(f'==> n_samples: {n_samples}')

			# Empirical covariance trace (diagonal elements sum)
			emp_cov_trace = cov.diag() / n_samples
			mu = ch.sum(emp_cov_trace) / n_features # Average of diagonal elements
		
			# Sum of squared coefficients of <X.T, X>
			delta_ = ch.sum(cov ** 2)
			delta_ /= n_samples**2

			# Sum of coefficients of <X2.T, X2>
			beta_ = ch.sum(cov_2)

			# Compute beta (part of shrinkage formula)
			beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
			
			# Compute delta (part of shrinkage formula)
			delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
			delta /= n_features
			
			# Get final shrinkage coefficient
			shrinkage = 0 if beta == 0 else min(beta, delta) / delta # min(beta, delta) prevents over-shrinking

			# Sklearn uses biased cov for shrunk_cov computation (no Bessel's correction here)
			cov /= n_samples

			print(f'==> Shrinkage: {float(shrinkage):.4f}')
			print(f'==> Mu: {float(mu):.4f}')

			# Shrunk covariance matrix
			cov = ((1.0 - shrinkage) * cov) + (shrinkage * mu * torch.eye(n_features, device=cov.device))

		else:
			print(f'==> Empirical covariance matrix. Adding diagonal noise to make invertible: {self.diag_lambda}')
			if type(cov) is list: # If block-diagonal
				for block_idx in range(len(cov)):
					cov[block_idx] /= (count_2-1) # Bessel's correction
					cov[block_idx] += (torch.eye(cov[block_idx].size(1), device=cov[block_idx].device) * self.diag_lambda)
			else: # If full matrix
				cov /= (count_2-1) # Bessel's correction for sample covariance
				cov += (torch.eye(cov.size(1), device=cov.device) * self.diag_lambda) # Add diagonal noise

		return mean, cov

	def get_block_covs(self, inp):
		"""
		Computes diagonal blocks of the covariance matrix for `inp.T @ inp`.
		Used for block-diagonal approximation.

		Args:
			inp (torch.Tensor): Centered data vector of shape (N, D).

		Returns:
			list: A list of torch.Tensor, where each tensor is a diagonal block.
		"""
		dim = inp.shape[1] # Dimension D
		inp_T = inp.transpose(0,1) # (D, N)

		block_covs = []
		# Calculate number of blocks based on `block_diag` size
		n_blocks = math.ceil(dim / self.block_diag)

		start, stop = 0, self.block_diag

		for i in range(n_blocks):
			if i == n_blocks - 1: # Last block might be smaller
				stop = dim

			# Compute covariance for the current block
			block_covs.append(torch.matmul(inp_T[start:stop,:], inp[:,start:stop])) # (B,N) @ (N,B) = (B,B)
				
			start = stop
			stop += self.block_diag

		del inp_T # Free up memory
		return block_covs

	def match_mean_cov(self, x):
		"""
		Applies 3D Fourier-domain Moment Matching (mean and covariance) to input `x`.
		This method is used when `use_2D_dft` is False and `match_cov` is True.

		Args:
			x (torch.Tensor): Input image batch (N, C, H, W).

		Returns:
			torch.Tensor: Transformed image batch.
		"""
		fft = ch.fft.rfftn(x, dim=(1,2,3)) # Compute 3D DFT (on C, H, W dimensions)
		original_abs, angle = ch.abs(fft), ch.angle(fft) # Extract amplitude and phase
		fmm_abs = original_abs.clone() # Clone to allow gradient flow to original_abs if needed

		original_shape = fmm_abs.shape
		# Flatten C, H, W dimensions to D for moment matching operations
		fmm_abs = fmm_abs.flatten(start_dim=1, end_dim=3) # (N,C,H,W) -> (N, D)

		# Center data by subtracting source mean
		fmm_abs -= self.source_mean

		# Whitening and coloring transformation
		if self.block_diag:
			# Apply block-diagonal multiplication
			fmm_abs = self.block_multiply(fmm_abs, self.coloring_matrix)
		else:
			# Apply full matrix multiplication: S = S * (Cov(S)**(-1/2)) * (Cov(T)**1/2)
			fmm_abs = torch.matmul(fmm_abs, self.coloring_matrix)
	
		# Add target mean
		fmm_abs += self.target_mean

		# Reshape back to original amplitude spectrum shape (N,C,H,W)
		fmm_abs = fmm_abs.reshape(original_shape)

		# Thresholding: Ensure amplitudes are non-negative
		fmm_abs[fmm_abs < 0] = 0

		# Reconstruct Fourier coefficients using transformed amplitude and original phase
		fft = fmm_abs * ch.exp((1j) * angle)
		# Inverse 3D DFT to get the transformed image
		fmm_x = ch.fft.irfftn(fft, dim=(1,2,3)).float()

		return fmm_x

	def block_multiply(self, inp, matrices):
		"""
		Performs block-wise matrix multiplication.
		Used when `block_diag` is enabled.

		Args:
			inp (torch.Tensor): Input tensor of shape (N, D).
			matrices (list): A list of matrices [ (M,M), (M,M), ... ] corresponding to blocks.

		Returns:
			torch.Tensor: Result of block-wise multiplication, shape (N, D).
		"""
		start, stop = 0, 0
		outputs = []

		for mat in matrices:
			block_size = mat.shape[0]
			stop += block_size
			# Multiply current input block with corresponding matrix block
			outputs.append(torch.matmul(inp[:,start:stop], mat)) # (N,M) * (M,M) = (N,M)
			start = stop

		# Concatenate all block outputs along the feature dimension
		outputs = ch.cat(outputs, dim=1) # (N,D)
		
		return outputs


	def get_spectral_mean_std(self, loader):
		"""
		Computes the mean and standard deviation per Fourier-mode (element-wise, not covariance).
		This method is used when `match_cov` is False.

		Args:
			loader (torch.utils.data.DataLoader): DataLoader for the dataset.

		Returns:
			tuple: (mean, std) - Mean and standard deviation of the amplitude spectrum.
		"""
		mean, var = None, None
		n = len(loader.dataset) # Size of dataset (number of samples)

		count_1, count_2 = 0, 0 # Number of samples used for mean and variance computation
		
		# --- Compute Mean ---
		for _, z in enumerate(loader):
			# Handle different data loader output formats
			if type(z) is dict and self.target_dataset == 'CityscapesDataset':
				x = z['img']._data[0]
				assert x.shape[-3] == 3
			else:
				x, y = z[0], z[1]

			if type(x) is list and self.target_dataset in ['camelyon17', 'iwildcam']:
				x = x[0] # Use the weakly augmented input

			# Compute DFT (2D or 3D based on use_2D_dft) and get amplitude spectrum
			if self.use_2D_dft:
				x = ch.abs(ch.fft.rfftn(x, dim=(2,3)))
			else:
				x = ch.abs(ch.fft.rfftn(x, dim=(1,2,3)))

			count_1 += x.shape[0] # Accumulate number of samples

			batch_sum = x.sum(dim=0) # Sum across batch dimension (N)

			if mean is None:
				mean = batch_sum
			else:
				mean += batch_sum

			# Stop accumulating for large images if sample size limit is reached
			# Check if total dimension (C*H*W) is above threshold
			if batch_sum.shape[-1] * batch_sum.shape[-2] * batch_sum.shape[-3] > BLOCK_DIAG_THRESHOLD:
				if count_1 > self.MAX_N_LARGE_IMAGES:
					break

		mean /= count_1 # Normalize mean

		print(f'Computing spectral mean and std across {count_1} samples')

		# --- Compute Variance ---
		for _, z in enumerate(loader):
			# Handle different data loader output formats
			if type(z) is dict and self.target_dataset == 'CityscapesDataset':
				x = z['img']._data[0]
				assert x.shape[-3] == 3
			else:
				x, y = z[0], z[1]

			if type(x) is list and self.target_dataset in ['camelyon17', 'iwildcam']:
				x = x[0]

			# Compute DFT (2D or 3D) and get amplitude spectrum
			if self.use_2D_dft:
				x = ch.abs(ch.fft.rfftn(x, dim=(2,3)))
			else:
				x = ch.abs(ch.fft.rfftn(x, dim=(1,2,3)))

			count_2 += x.shape[0] # Accumulate number of samples

			x -= mean # Center data (x - mean)
			x = x**2 # Square the centered data ((x-u)**2)
			x = x.sum(dim=0) # Sum across batch dimension (N) (sigma[(x-u)**2])

			if var is None:
				var = x
			else:
				var += x

			# Stop accumulating for large images if sample size limit is reached
			if batch_sum.shape[-1] * batch_sum.shape[-2] * batch_sum.shape[-3] > BLOCK_DIAG_THRESHOLD:
				if count_2 > self.MAX_N_LARGE_IMAGES:
					break

		assert count_1 == count_2, 'Sample counts for mean and variance computation are not equal.'
		
		var /= (count_2-1) # Bessel's correction for sample variance
		std = var ** 0.5 # Standard deviation is the square root of variance

		return mean, std

	def match_mean_std(self, x):
		"""
		Applies Fourier-domain Moment Matching (mean and standard deviation) to input `x`.
		This method is used when `match_cov` is False.

		Args:
			x (torch.Tensor): Input image batch (N, C, H, W).

		Returns:
			torch.Tensor: Transformed image batch.
		"""
		# Compute DFT (2D or 3D based on use_2D_dft)
		if self.use_2D_dft:
			x_fft = ch.fft.rfftn(x, dim=(2,3))
		else:
			x_fft = ch.fft.rfftn(x, dim=(1,2,3))

		original_abs, angle = ch.abs(x_fft), ch.angle(x_fft) # Extract amplitude and phase

		fmm_abs = original_abs.clone() # Clone to allow gradient flow

		# Center data by subtracting source mean
		fmm_abs -= self.source_mean

		if not self.mean_only:
			# Apply element-wise scaling using the coloring matrix (target_std / source_std)
			fmm_abs *= self.coloring_matrix

		# Add target mean
		fmm_abs += self.target_mean

		# Thresholding: Ensure amplitudes are non-negative
		fmm_abs[fmm_abs < 0] = 0

		# Reconstruct Fourier coefficients using transformed amplitude and original phase
		x_fft = fmm_abs * ch.exp((1j) * angle)

		# Inverse DFT (2D or 3D) to get the transformed image
		if self.use_2D_dft:
			transformed_x = ch.fft.irfftn(x_fft, dim=(2,3)).float()
		else:
			transformed_x = ch.fft.irfftn(x_fft, dim=(1,2,3)).float()

		return transformed_x

	def __call__(self, x, y=None):
		"""
		The main entry point for applying the FMM transformation.
		This method makes the FMM object callable.

		Args:
			x (torch.Tensor): Input image batch (N, C, H, W).
			y (torch.Tensor, optional): Labels corresponding to the batch (not used in transformation logic).

		Returns:
			torch.Tensor: Transformed image batch.
		"""
		if self.match_cov:
			if self.use_2D_dft:
				return self.match_2D_mean_cov(x)
			else:
				return self.match_mean_cov(x)
		else:
			return self.match_mean_std(x)

