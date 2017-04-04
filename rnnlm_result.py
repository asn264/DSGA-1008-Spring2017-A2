'''
Aditi Nair (asn264) and Akash Shah (ass502)
Deep Learning Assignment Two

Loads our best model and reports perplexity on the test data. Please see write-up for details of model parameters.
'''

import argparse
import torch
import math
import data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./dropout_point3_two_layer_200_hidden_200_embeddding.m')
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


with open(args.checkpoint, 'rb') as f:
	model = torch.load(f)

model.cpu()

corpus = data.Corpus(args.data)
eval_batch_size = 10
test_data = batchify(corpus.test, eval_batch_size)
test_loss = evaluate(test_data)

print 'Test perplexity: ', math.exp(test_loss)

