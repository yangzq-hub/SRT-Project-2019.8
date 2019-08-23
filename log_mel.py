import os
import wave
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle 

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa

from general_tools import *
import python_speech_features as features
	# https://github.com/jameslyons/python_speech_features


##### 音频和音素文件用对齐程序输出的复制进来即可，不要改变文件名 #####
##### 可以处理多个文件，最后将所有结果输出到一个output.pkl文件 #####

##### THRESHOLD PARAMETER FOR VALID PHONEME JUDGEMENT #####
##### 为了数据更好，抛弃了太短的音素，下面的参数可以设置音素长度的下限值，单位为帧 #####
frame_threshold = 2

data_type = 'float32'

source_path_train = '/home/yangzq/train'
source_path_val = '/home/yangzq/val'
source_path_test = '/home/yangzq/test'

phonemes = ['sil', 'dh', 'ae', 'r', 'sp', 'w', 'ah', 'z', 't', 'ay', 'm', 'hh', 'eh', 'n', 'l', 'ow', 'uw', 'g', 'd', 'p', 'ey', 's', 'k', 'ao', 'iy', 'f', 'ih', 'v', 'uh', 'er', 'aa', 'y', 'b', 'oy', 'zh', 'ng', 'aw', 'th','sh','ch','jh']

	# With the use of CMU Dictionary there are 41 different phonemes


def find_phoneme (phoneme_idx):
	for i in range(len(phonemes)):
		if phoneme_idx == phonemes[i]:
			return i
	print("PHONEME NOT FOUND, NaN CREATED!")
	print("\t" + phoneme_idx + " wasn't found!")
	return -1


def create_logmel(filename):
	y, sr = librosa.load(filename, sr=None)
	# extract mel spectrogram feature
	melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=256, n_mels=128)
	# convert to log scale
	logmelspec = librosa.power_to_db(melspec)
	out=np.transpose(logmelspec)
	return out, out.shape[0]

def calc_norm_param(X):
	mean_val = np.mean(X,axis=0)
	std_val = np.std(X,axis=0)
	print(mean_val)
	print(std_val)
	return mean_val, std_val

def normalize(X, mean_val, std_val):
	for i in range(len(X)):
		X[i] = (X[i] - mean_val)/std_val
	return X

def set_type(X, type):
	for i in range(len(X)):
		X[i] = X[i].astype(type)
	return X


def preprocess_dataset(phn_fname, wav_fname):
	i = 0
	X = []
	Y = []
	fr = open(phn_fname)
	X_logmel, total_frames = create_logmel(wav_fname)
	total_frames = int(total_frames)
	for line in fr:
		line_split = line.rstrip('\n').split()
		if len(line_split) == 3:
			[start_time, end_time, phoneme] = line.rstrip('\n').split()
			start_frame = int(float(start_time)*1000/11.61) + 1
			end_frame = int(float(end_time)*1000/11.61) - 4
			if (end_frame - start_frame) >= frame_threshold and (end_frame - start_frame) <= 100:
				phoneme_num = find_phoneme(phoneme)
				if phoneme_num != -1:			
					for i in range(start_frame,end_frame):
						X.append(X_logmel[i])
						Y.append(phoneme_num)
	fr.close()
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

##### PREPROCESSING #####
X_train = np.empty(shape = [0,128])
y_train = np.empty(shape = [0])
for dirName, subdirList, fileList in os.walk(source_path_train):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn'): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		X = set_type(X, data_type)
		X_train = np.append(X_train,X,axis=0)
		y_train = np.append(y_train,y,axis=0)


print(X_train.shape)
print(y_train.shape)

X_val = np.empty(shape = [0,128])
y_val = np.empty(shape = [0])

for dirName, subdirList, fileList in os.walk(source_path_val):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn'): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		X = set_type(X, data_type)
		X_val = np.append(X_val,X,axis=0)
		y_val = np.append(y_val,y,axis=0)

print(X_val.shape)
print(y_val.shape)

X_test = np.empty(shape = [0,128])
y_test = np.empty(shape = [0])

for dirName, subdirList, fileList in os.walk(source_path_test):
	i = 0
	for fname in fileList:
		if not fname.endswith('.phn'): 
			continue
		phn_fname = dirName + '/' + fname
		wav_fname = dirName + '/' + fname[0:-11] + '.wav'
		i += 1
		print('Preprocessing file' + str(i))
		X, y =preprocess_dataset(phn_fname,wav_fname)									
		print('File ' + str(i) + ' preprocessing complete')
		print(X.shape)
		X = set_type(X, data_type)
		X_test = np.append(X_test,X,axis=0)
		y_test = np.append(y_test,y,axis=0)

print(X_test.shape)
print(y_test.shape)


mean_val,std_val = calc_norm_param(X_train)
X_train = normalize(X_train,mean_val,std_val)
X_test = normalize(X_test,mean_val,std_val)
X_val = normalize(X_val,mean_val,std_val)

print(np.mean(X_train,axis = 0))
print(np.std(X_train,axis = 0))
print(np.mean(X_val,axis = 0))
print(np.std(X_val,axis = 0))
print(np.mean(X_test,axis = 0))
print(np.std(X_test,axis = 0))

with open('train128.pkl', 'wb') as cPickle_file:
	cPickle.dump(
	[X_train, y_train], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)

with open('val128.pkl', 'wb') as cPickle_file:
	cPickle.dump(
	[X_val, y_val], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)

with open('test128.pkl', 'wb') as cPickle_file:
	cPickle.dump(
	[X_test, y_test], 
	cPickle_file, 
	protocol=cPickle.HIGHEST_PROTOCOL)

print('Preprocessing complete!')
print()


print('Total time: {:.3f}'.format(timeit.default_timer() - program_start_time))




