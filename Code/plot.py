import numpy as np
import pylab as pl 

def process_txt( txt_path ) : 
	f = open(txt_path, 'r')
	data = f.read()
	data = data.split('\n')
	data = data[1:-1]
	 
	 

	epoch = []

	train_loss = []

	valid_loss = []


	for i in range(len(data)):
		 line = data[i].split('\t')[1:]
		 epoch.append( int(line[0]))
		 train_loss.append(float(line[1]))
		 valid_loss.append(float(line[2]))

	return train_loss, valid_loss


     
train_loss, valid_loss = process_txt('log_training.txt')
pl.plot(train_loss, '--r', lw = 3, label = 'Train loss, ||X - X\'||')
pl.plot(valid_loss, '--b', lw = 3, label = 'Validation loss, ||X - X\'||')
pl.xlabel('Nb Epochs')
pl.ylabel('||X - X\'||')
pl.legend(loc = 'best')
pl.grid('on')
pl.title('Hidden units = 100, Learning rate = 0.001, 1-CD, batchsize = 100, validation ratio = 0.2, Validation Error = 67.86683')
pl.show()

