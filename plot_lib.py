from cProfile import label
from re import I
from tkinter import SW
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from tqdm import tqdm
import sys

def plot_style(x_label: str, y_label: str) -> None:
	"""General function to apply style format to all plots
	Args:
	    x_label (str): x-axis label
	    y_label (str): y-axis label
	"""
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	_, labels = plt.gca().get_legend_handles_labels()
	if len(labels)>0:
		plt.legend(frameon=False)

def plotAccLoss(trainInput:list, testInput:list, putVar:str, output_dir: str='models') -> None:
	""" Compare the loss and accuracy between training and testing.
	Args:
	    trainInput (list): list of loss or accuracy values of the training sample for each epoch
	    testInput (list): list of loss or accuracy values of the testing sample for each epoch
	    putVar (str): 'Loss' for plotting loss, 'Acc' for accuracy
	    output_dir (str): directory to save output files. Default: models/
	"""
	epochs = np.arange(1, len(trainInput) + 1)
	
	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(epochs, trainInput, 'o')
	plt.plot(epochs, testInput, 'o')
	if putVar == 'Loss':
		plt.legend(['training', 'testing'], loc='upper right')
		plot_style('epoch', 'loss')
		plt.yscale('log')
	elif putVar == 'Acc':
		plt.legend(['training', 'testing'], loc='upper right')
		plot_style('epoch', 'accuracy')
	else:
		print("ERROR: no such variable. Select from 'Loss' and 'Acc'.")
		sys.exit()

	#plt.ylim(0.5, 0.9)
	plt.savefig('{}/{}_compare.pdf'.format(output_dir, putVar))

def plotOutputScore(score: list, labels: list, weights: list, output_dir:str ='output') -> None:
	""" Evaluation plotting.
	Args:
	    score (list): output score for each jet at the evaluation
	    labels (list): labels for each corresponding jet
	    output_dire (str): output directory path. Default: output/
	"""

	outputScore = np.array(score)
	weights = np.array(weights)
	labels = np.array(labels)
	signal_score = outputScore[labels==1]
	bkg_score = outputScore[labels==0]
	cut = []
	purity = []
	purity_weighted = []
	sig_eff = []
	N_signal = []
	N_bkg = []
	significance = []
        
	## In most analysis, for MC signal and backgorund, they need to be reweigited. Apply the weight here if needed. Put 1 as temporary default value
	sWeight = weights[labels==1]
	bWeight = weights[labels==0]

	scanValue = np.linspace(0, 1, 100)
	for i in tqdm(range(len(scanValue)), desc="======= Calculating ROC curve score"):
		N_tagged = len(labels[outputScore>scanValue[i]])
		N_tagged_sweighted = np.sum(sWeight[signal_score>scanValue[i]])
		N_tagged_bweighted = np.sum(bWeight[bkg_score>scanValue[i]])
		if N_tagged == 0:
			break
		N_sig_tagged = len(signal_score[signal_score>scanValue[i]])

		## selected significance = s/sqrt(s+b)
		significance.append(N_sig_tagged * N_tagged_sweighted / np.sqrt(N_sig_tagged*N_tagged_sweighted + (N_tagged-N_sig_tagged)*N_tagged_bweighted))
		N_signal.append(N_sig_tagged*N_tagged_sweighted)
		N_bkg.append((N_tagged-N_sig_tagged)*N_tagged_bweighted)
		cut.append(scanValue[i])
		sig_eff.append(N_sig_tagged/len(signal_score))
		purity.append(N_sig_tagged/N_tagged)
		purity_weighted.append(N_sig_tagged*N_tagged_sweighted / (N_sig_tagged*N_tagged_sweighted + (N_tagged - N_sig_tagged)*N_tagged_bweighted))

	print("======= Plotting testing performance")
	pdf = matplotlib.backends.backend_pdf.PdfPages("{}/weight_results.pdf".format(output_dir))

	nbins=200
	## Output score comparison between signal and background
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(signal_score, bins=nbins, range=[0,1], density=True, label = 'signal', histtype='step')
	ax.hist(bkg_score, bins=nbins, range=[0,1], density=True, label = 'background', histtype='step')
	ax.set_yscale('log')
	plot_style('probability', 'density')
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	## plot significance as a function of output cut
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	color = 'tab:red'
	ax.plot(cut, significance, 'o', color=color)
	ax.set_ylabel('significance', color=color)
	ax.set_xlabel('output score')
	ax.tick_params(axis='y', labelcolor=color)

	ax2 = ax.twinx()
	color = 'tab:green'
	ax2.set_ylabel('N. of tagged events', color=color)
	ax2.plot(cut, N_signal, color='blue', label='signal')
	ax2.plot(cut, N_bkg, color='orange', label='background')
	ax2.tick_params(axis='y', labelcolor='black')
	ax2.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	## plot eff versus purity
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(cut, sig_eff, 'o', label='efficiency', color='blue')
	ax.plot(cut, purity, 'o', label='purity', color='orange')
	plot_style('output score', 'efficiency/purity')
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(sig_eff, purity, 'o')
	plot_style('signal efficiency', 'purity')
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	pdf.close()
