import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import gc
import pickle
import random
from warnings import warn
import string
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from scipy import stats
#from torchvision.ops import sigmoid_focal_loss
from sklearn.model_selection import ParameterGrid
import time
from statistics import mean
from torch.optim.lr_scheduler import StepLR, LambdaLR
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE



def save_obj(obj, name, path):
  with open(path+ name + '.pkl', 'wb') as f:
      pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
  with open(path+ name, 'rb') as f:
      return pickle.load(f)
  
def z_score_normalize(data):
    return (data - data.mean()) / data.std()

def generate_database_with_hold_out_memeff(Y,data_all,data_valid,data_test,data_train,balanced=70):  
  # index_hold_out: list of ids to hold_out
  # data_valid: list of validation subject ids
  # data_train: list of training subject ids
  # data_test: list of testing subject ids
  # data_all: list of all subject ids, in the same order as in Y
  # Y: (list of list of window labels, in the same order as in data_all. (Y[0] contains window labels of suject data_all[0])
  # balanced: If want to undersample non-spike windows to generated a balanced database

  total_nb_window=0
  nb_sub = len(Y)
  # Go through all subjects and add their number of windows to "total_nb_window"
  for s in range(nb_sub):
      total_nb_window = total_nb_window + len(Y[s])

  # filling "ids" array of shape(total_nb_windows,3)
  # For each window, ids[window]=[window_id, subject_id, label]
  # This big array is then used to go read the correct infos in windows_bi file during model training process
  ids = np.zeros((total_nb_window ,3),dtype=int)
  # "start" and "stop" index used to fill the correct part of "ids" when looping through the subjects
  start = 0
  stop = 0
  for i in range(nb_sub):
      nb_windows = len(Y[i])
      stop = stop + nb_windows
      # For each subject, generate window ids (going from 0 to nb_windows of subject i)
      win_id = np.expand_dims(np.linspace(0,nb_windows-1,num=nb_windows,dtype=int),axis=-1)
      # Generate a numpy array with repeated subject i id 
      sub_id = np.expand_dims(np.ones((nb_windows),dtype=int)*int(data_all[i]),axis=-1)
      # Fill "ids" array with [window_id, subject_id, label] for subject i
      ids[start:stop,:]=np.concatenate((win_id,sub_id,np.expand_dims(Y[i][:nb_windows],axis=-1)),axis=-1)
      start = start + nb_windows

  # From the "ids" array, extract valid, test and train subjects based on data_valid,data_test,data_train
  mask=np.isin(ids,data_test)
  X_test_ids = ids[mask[:,1]==True]
  mask=np.isin(ids,data_valid)
  X_valid_ids = ids[mask[:,1]==True]
  mask=np.isin(ids,data_train)
  X_train_ids = ids[mask[:,1]==True]

  #print("X_train_shape :",X_train_ids.shape)
  np.random.shuffle(X_train_ids)
  np.random.shuffle(X_valid_ids)

  if balanced:
      # undersample X_train_ids to get as many non-spike windows as spike windows
      np.random.shuffle(X_train_ids)
      X_train_ids = X_train_ids[X_train_ids[:, 2].argsort()[::-1]]
      nb_pos = X_train_ids[X_train_ids[:,2]==1].shape[0]
      nb_neg = X_train_ids[X_train_ids[:,2]==0].shape[0]
      X_train_ids = X_train_ids[:nb_pos+round(nb_neg/balanced),:]
      np.random.shuffle(X_train_ids)

      np.random.shuffle(X_test_ids)
      X_test_ids = X_test_ids[X_test_ids[:, 2].argsort()[::-1]]
      nb_pos = X_test_ids[X_test_ids[:,2]==1].shape[0]
      nb_neg = X_test_ids[X_test_ids[:,2]==0].shape[0]
      X_test_ids = X_test_ids[:nb_pos+round(nb_neg/balanced),:]
      np.random.shuffle(X_test_ids)

      np.random.shuffle(X_valid_ids)
      X_valid_ids = X_valid_ids[X_valid_ids[:, 2].argsort()[::-1]]
      nb_pos = X_valid_ids[X_valid_ids[:,2]==1].shape[0]
      nb_neg = X_valid_ids[X_valid_ids[:,2]==0].shape[0]
      X_valid_ids = X_valid_ids[:nb_pos+round(nb_neg/balanced),:]
      np.random.shuffle(X_valid_ids)
  
  return X_train_ids, X_test_ids, X_valid_ids #[item for sublist in Y_test for item in sublist]

class TimeSeriesDataSet(Dataset):
  """
  This is a custom dataset class

  """
  def __init__(self, X_ids, dim, path, path_add=None):
    self.X = X_ids
    self.path = path
    self.path_add = path_add
    self.dim = dim

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    win = np.array(self.X[index])[0]
    sub = np.array(self.X[index])[1]

    prefixe = 'data_raw_'
    suffixe = '_b3_windows_bi'
    path_data = self.path 

    # Opens windows_bi binary file
    f = open(path_data+prefixe+str(sub).zfill(3)+suffixe)
    # Set cursor position to 30 (nb time points)*274 (nb channels)*windows_id*4 because data is stored as float64 and dtype.itemsize = 8
    f.seek(self.dim[0]*self.dim[1]*win*4)
    # From cursor location, get data from 1 window
    sample = np.fromfile(f, dtype='float32', count=self.dim[0]*self.dim[1])
    # Reshape to create a 2D array (data from the binary file is just a vector)
    sample = sample.reshape(self.dim[1],self.dim[0])
    # Swap axis to have time point in first dim, nb channels in 2nd dim
    sample = np.swapaxes(sample,0,1)
    sample = z_score_normalize(sample)

    


    #print(np.mean(sample))
    #sample = normalize_by_window(sample)
    #sample = stats.zscore(sample,axis=None)
    # Add third dimension to be able to feed to CNN (CNN need 3 dim, image x, image y, RGB channels), here last dimension=1
    
    if len(self.dim) == 3:
      sample = np.expand_dims(sample,axis=0)
      #sample = np.expand_dims(sample,axis=-1)

    _x = sample
    _y = np.array(self.X[index])[2]
    #print("Valeurs X",_x)
    return _x, _y

def custom_distance(window1, window2):
    # Calculer la distance euclidienne pour chaque capteur
    distances = np.sqrt(np.sum((window1 - window2)**2, axis=1))
    # Somme des distances pour obtenir une distance globale
    return np.sum(distances)

def compute_distance_matrix(data):
    num_windows = data.shape[0]
    distance_matrix = np.zeros((num_windows, num_windows))
    
    for i in range(num_windows):
        for j in range(num_windows):
            distance_matrix[i, j] = custom_distance(data[i], data[j])
            
    return distance_matrix

##############################Some parameters##############################
sfreq = 150  # sampling frequency of the data in Hz
window_size = 0.2
dim = (int(sfreq*window_size), 274,1) # sample shape
batch_size=32
##############################CV##############################


path_extracted_data = '/sps/crnl/pmouches/data/MEG_PHRC_2006_preprocessedNEWAfterFeedback/'
path_writing_data = '/sps/crnl/tdejean/'
subjects= ['004','008','011','015','018','020','022','025','026','027','028','030','032','034','038','041','042','045','047','051','052','053','054','056','058','061','064','066','067','068','069','072','074','076','077','078','079','083','084','085','086','087','092','094','095','096','097','101','104','105','111','112','117','123']

sfreq = 150  # sampling frequency of the data in Hz
window_size = 0.2
dim = (int(sfreq*window_size), 274,1) # sample shape
batch_size=32

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Stockage des résultats sur tous les Folds
F1_list = []
M_sum=np.zeros((2,2))

#processes = []

rs = 0
kf = KFold(n_splits = 3,shuffle = True,random_state=rs)
#folds = list(kf.split(subjects))
fold=0



for train, test in kf.split(subjects):
    fold = fold+1
    print("Fold :",  fold)


    rs+=1
    Y=list()
    data_all=list()

    Y_add = list()
    data_add_spikes = list()


    train_val_subjects = [subjects[j] for j in train]
    test_subject = [subjects[j] for j in test]
    random.shuffle(train_val_subjects)

    # first 5 subjects for training
    data_train = [i for i in train_val_subjects[:-2]]
    data_train = list(map(int, data_train))

    
    # next subject for validation
    data_valid = [i for i in train_val_subjects[-2:]]
    data_valid = list(map(int, data_valid))
    

    # last subject for testing
    data_test = [i for i in test_subject]
    data_test = list(map(int, data_test))

    print("Data_train, data_valid, Data test",data_train,data_valid, data_test)

    #p = multiprocessing.Process(target=train_on_gpu, args=(fold, data_train, data_valid, data_test))
    #p.start()
    #processes.append(p)

    for ind, sub in enumerate(sorted(subjects)):
        data = load_obj('data_raw_'+sub.zfill(3)+'_b3_labels.pkl', path_extracted_data)
        Y.append(data)
        data_all.append(sub)

    data_all = list(map(int, data_all))


    X_train_ids, X_test_ids, X_valid_ids = generate_database_with_hold_out_memeff(Y,data_all, data_valid, data_test, data_train)
    
    train_window_dataset = TimeSeriesDataSet(X_train_ids.tolist(),dim,path_extracted_data)
    train_dataloader = DataLoader(train_window_dataset, batch_size=batch_size, shuffle=True)

    test_window_dataset = TimeSeriesDataSet(X_test_ids.tolist(),dim,path_extracted_data)
    test_dataloader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=True)

    valid_window_dataset = TimeSeriesDataSet(X_valid_ids.tolist(),dim,path_extracted_data)
    valid_dataloader = DataLoader(valid_window_dataset, batch_size=batch_size, shuffle=True)

    dataloaded = train_dataloader
    #print("Data :", dataloaded)
    input_tensor_list = []
    labels_tensor_list = []

    import numpy as np




    for i, data in enumerate(dataloaded, 0):
      inputs_batch, labels_batch = data
      #print("Labels_batch",labels_batch )      
      #print("Inputs_batch",inputs_batch )


      if inputs_batch.size(0) != 32:
        continue
      
      input_tensor_list.append(inputs_batch)
      labels_tensor_list.append(labels_batch)
      #print(type(inputs_batch), type(labels_batch) )

    inputs_array = np.concatenate([t.numpy() for t in input_tensor_list])
    labels_array = np.concatenate([t.numpy() for t in labels_tensor_list])
    #print("Inputs", inputs_array, inputs_array.shape)
    #print("Labels", labels_array, labels_array.shape)
    
    # Aplatir les données d'entrée
    std_values_pre = np.std(inputs_array, axis=2)
    std_values = np.squeeze(std_values_pre)
    #inputs_array_2d = inputs_array.reshape(inputs_array.shape[0], -1)
    print(std_values.shape)

    # Mise à l'échelle des données d'entrée
    scaler_inputs = StandardScaler()
    inputs_scaled = scaler_inputs.fit_transform(std_values)

    n_clusters = 2  # Supposons que nous voulons identifier 3 types de fenêtres
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(inputs_scaled)

    # Obtenir les labels des clusters pour chaque point de données
    labels2 = kmeans.labels_

    """
    distance_matrix = compute_distance_matrix(inputs_array)
    dbscan = DBSCAN(metric='precomputed')
    labels2 = dbscan.fit_predict(distance_matrix)
    # Visualiser les résultats

    # Aplatir les données pour la réduction de dimensionnalité
    data_flattened = inputs_array.reshape(inputs_array.shape[0], -1)
    data_2d = TSNE(n_components=2).fit_transform(data_flattened)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels2)  # colorer par labels si disponibles
    plt.title("Visualisation t-SNE")
    plt.show()
    """
    for i in range(n_clusters):
        plt.scatter(inputs_scaled[labels2 == i][:, 0], inputs_scaled[labels2 == i][:, 1], label=f'Cluster {i+1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    plt.savefig('/sps/crnl/tdejean/clustering_{}_std.png'.format(fold))
