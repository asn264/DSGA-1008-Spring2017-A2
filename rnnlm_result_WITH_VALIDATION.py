'''
Aditi Nair (asn264) and Akash Shah (ass502)
Deep Learning Assignment Two

Loads our best model and reports perplexity on the test data. Please see write-up for details of model parameters.
'''

import argparse
import math
import data

import torch
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./final_rnnlm_PTB_asn264_ass502.m')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
args = parser.parse_args()

def batchify(data, bsz):

	'''Copied from starter code main.py file '''

	nbatch = data.size(0) // bsz
	data = data.narrow(0, 0, nbatch * bsz)
	data = data.view(bsz, -1).t().contiguous()
	return data


def evaluate(data_source):

	'''Copied from starter code main.py file '''

	total_loss = 0
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(eval_batch_size)
	for i in range(0, data_source.size(0) - 1, args.bptt):
		data, targets = get_batch(data_source, i, evaluation=True)
		output, hidden = model(data, hidden)
		output_flat = output.view(-1, ntokens)
		total_loss += len(data) * criterion(output_flat, targets).data
		hidden = repackage_hidden(hidden)
	return total_loss[0] / len(data_source)


def get_batch(source, i, evaluation=True):

	'''Copied from starter code main.py file '''

	seq_len = min(args.bptt, len(source) - 1 - i)
	data = Variable(source[i:i+seq_len], volatile=evaluation)
	target = Variable(source[i+1:i+1+seq_len].view(-1))
	return data, target


def repackage_hidden(h):
	'''Copied from starter code main.py file '''

	"""Wraps hidden states in new Variables, to detach them from their history."""
	if type(h) == Variable:
		return Variable(h.data)
	else:
		return tuple(repackage_hidden(v) for v in h)


with open(args.checkpoint, 'rb') as f:
	model = torch.load(f)

model.cpu()
model.eval()

criterion = nn.CrossEntropyLoss()

corpus = data.Corpus(args.data)
eval_batch_size = 10

valid_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

valid_loss = evaluate(valid_data)
test_loss = evaluate(test_data)

print 'Valid perplexity: ', math.exp(valid_loss)
print 'Test perplexity: ', math.exp(test_loss)

