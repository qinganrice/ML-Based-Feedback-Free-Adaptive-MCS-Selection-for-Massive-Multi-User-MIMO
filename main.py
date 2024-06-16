import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
import h5py

from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set


def resume_model(opt, model, optimizer):
	""" Resume model from checkpoint
	"""
	checkpoint = torch.load(opt.resume_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Model Restored from Epoch {}".format(checkpoint['epoch']))
	start_epoch = checkpoint['epoch'] + 1
	return start_epoch


def get_loaders(opt):
	""" Make dataloaders for train and validation sets
	"""
	# training loader
	### In sequential domain, we feed channel matix of 3 slots (2 previous + current slots)
	### In spatial domain, we feed UE x BS x 2 (real+img) channel matrix to extract inter-user correlation
	### The size of input should be [Batch_size(32), slots(3), BS(32), UE(4), 2]
	
	training_data = get_training_set()
	train_loader = torch.utils.data.DataLoader(
		training_data,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		pin_memory=True)


	# validation loader
	validation_data = get_validation_set()
	val_loader = torch.utils.data.DataLoader(
		validation_data,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=True)
	return train_loader, val_loader


def main_worker():
	opt = parse_opts()
	print(opt)

	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

	# tensorboard
	summary_writer = SummaryWriter(log_dir='tf_logs')

	# defining model
	model =  generate_model(opt, device)
	# get data loaders
	train_loader, val_loader = get_loaders(opt)

	# optimizer
	net_params = list(model.parameters())
	optimizer = torch.optim.Adam(net_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

	criterion = nn.CrossEntropyLoss()

	# resume model
	if opt.resume_path:
		start_epoch = resume_model(opt, model, optimizer)
	else:
		start_epoch = 1

	tr_acc = np.zeros((opt.n_epochs,))
	tt_acc = np.zeros((opt.n_epochs,))

	print(tr_acc.shape, tt_acc.shape)
	# start training
	for epoch in range(start_epoch, opt.n_epochs + 1):
		train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
		val_loss, val_acc = val_epoch(
			model, val_loader, criterion, device)

		tr_acc[epoch-1] = train_acc
		tt_acc[epoch-1] = val_acc

		# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
			summary_writer.add_scalar(
				'acc/train_acc', train_acc * 100, global_step=epoch)
			summary_writer.add_scalar(
				'acc/val_acc', val_acc * 100, global_step=epoch)

			state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
			print("Epoch {} model saved!\n".format(epoch))

	
	with h5py.File("./acc_results.hdf5", "w") as data_file:
		data_file.create_dataset("tr_acc", data=tr_acc)
		data_file.create_dataset("tt_acc", data=tt_acc)


if __name__ == "__main__":
	main_worker()