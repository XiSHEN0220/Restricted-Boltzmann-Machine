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

txt_files = ['log_training_1.txt',
			 'log_training_2.txt',
			'log_training_3.txt',
			'log_training_4.txt',
			'log_training_5.txt']

c = ['r', 'g', 'b', 'c', 'k']


for i in range(len(txt_files)) : 
	train_loss, valid_loss = process_txt( txt_files[i] )
	l = txt_files[i].replace('log_training_','train : ').replace('.txt', ' steps CD')
	pl.plot(train_loss, color = c[i], lw = 3, label = l)
	pl.plot(valid_loss, '--', color = c[i], lw = 3, label = l.replace('train', 'valid'))

pl.xlabel('Nb Epochs')
pl.ylabel('Reconstruction loss')
pl.legend(loc = 'best')
pl.grid('on')
pl.title('CD steps influence on trainning, Batch size = 100')
pl.show()


'''pl.plot(epoch, train_loss, '--r', lw = 3, label = 'Train loss, ||X - X\'||')
pl.plot(epoch, valid_loss, '--b', lw = 3, label = 'Validation loss, ||X - X\'||')
pl.xlabel('Nb Epochs')
pl.ylabel('||X - X\'||')
pl.legend(loc = 'best')
pl.grid('on')
pl.title('Hidden units = 100, Learning rate = 0.001, 1-CD, batchsize = 100, validation ratio = 0.2, Validation Error = 43.99160')
pl.show()'''

