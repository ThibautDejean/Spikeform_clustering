import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

df = pd.read_csv('/sps/crnl/tdejean/img/cluster3/graph_hybrid_balanced_test_cvpredictions_modified.csv')
clusters_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Pour stocker le nombre total de données pour chaque cluster
cluster_meaning = {0: 'True Negatives', 1: 'False Positive', 2: 'Fasle Negative', 3: 'True Positive'}
window_counting = 0
spike_counting = 0
no_spike_amp = 0
spike_amp = 0
delta_valid = 0
delta_unvalid = 0
validation = 0
unvalidation = 0
green_count_valid = 0
green_count_unvalid = 0


def normalize_window(window):
    min_val = np.min(window)
    max_val = np.max(window)
    return (window - min_val) / (max_val - min_val)

def amplitude(L, precut=None, postcut=None):
    a = 0
    if precut and postcut : 
        L2 = L[precut : postcut]
        a = max(L2) - min(L2)
    else : 
        a = max(L)-min(L)
    return(a)

timers_list = []
subject_names = {4: 'Benza', 23: 'Chekh', 26: 'Cinam', 40:'Ducfl_AllDataset1200Hz', 44:'Durag_AllDataset1200Hz', 75:'Liogier_AllDataset1200Hz', 80:'Magsy_AllDataset1200Hz', 81:'Malau_AllDataset1200Hz',104:'Plece_AllDataset1200Hz',105:'Rabsa_AllDataset1200Hz'}

for name in subject_names.values() :
    with open(f'/sps/crnl/tdejean/centering/Timers2/{name}.txt', 'r') as f:
        timers = sorted([float(line.strip())*150 for line in f.readlines()])
        timers_list += ((name, timers))

print(timers_list)
results = []

df2 = pd.read_csv("/sps/crnl/tdejean/img/cluster3/graph_hybrid_balanced_test_cvpredictions_modified.csv")


solution = []
for index, row in df2.iterrows():
    #closest_row=[]
    window_id = int(row['counter'])
    subject = int(row['subject'])
    timer = float(row['timing'])
    name = subject_names[subject]
    for i in range(0,len(timers_list),2):
        nom_patient = timers_list[i]
        timers_patient = timers_list[i+1]
        if name == nom_patient :
            
            for j in timers_patient :
                liste = [] 
                liste2 = 0
                a=0
                if abs(j - timer) <= 100 :
                    #print(j, timer, window_id)
                    liste.append([subject, window_id, j, timer])
                    a+=1
                    #print(liste)
                #print(a)
                if a>1 :
                    liste2 = min(liste, key=lambda x: x[2])
                    solution.append(liste2[0:3])
                if a==1 :
                    solution.append([subject, window_id, j, timer, j-timer])
                
print(solution)
print(len(solution))

results=[]
clusters_data = {0: [], 1: [], 2: [], 3:[]}
# Pour chaque patient unique

for subject in df['subject'].unique():
    # 2. Lire le fichier binaire correspondant

    with open(f"/sps/crnl/pmouches/data/MEG_PHRC_2006_preprocessedNEWAfterFeedback/data_raw_{str(subject).zfill(3)}_b3_windows_bi", "rb") as f:

        # Chaque fenêtre contient 30 flottants
        window_size_sensors = 30*274
        data = np.fromfile(f, dtype=np.float32)
        num_windows = len(data) // window_size_sensors
        windows_data_all = data.reshape(num_windows, window_size_sensors)

        all_windows_data = np.empty((num_windows, 274, 30))
        for i, win in enumerate(windows_data_all) :
            windows_data = win.reshape(274,30)
            normalized_windows_data = normalize_window(windows_data)

            all_windows_data[i]= normalized_windows_data

        #print(all_windows_data)
        mean_window_per_window = np.mean(all_windows_data, axis=1)

        #print(mean_window_per_window.shape)  # (num_windows, 30)
        #print(mean_window_per_window.max(axis=1), mean_window_per_window.max(axis=1).shape)


        #print(all_windows_data, np.shape(all_windows_data))
        
        # Extraire les données pour chaque sujet
        subject_data = df[df['subject'] == subject]
        
        # Pour stocker les données par cluster

        
        for index, row in subject_data.iterrows():
            window_id = int(row['counter'])
            test = float(row['test'])
            pred = float(row['pred'])
            cluster = int(row['cluster'])
            subject = int(row['subject'])

            if test == 1 :
                spike_counting += 1

                window_data = mean_window_per_window[window_id]
                clusters_data[cluster].append(window_data)
                clusters_counts[cluster] += 1
                window = np.array(window_data)
                amplitude_globale = amplitude(window)

                window = window*(1/amplitude_globale)
                derivative = np.diff(window)
                #print(derivative)
                minima = np.where((derivative[:-1] < 0) & (derivative[1:] > 0))[0] + 1

                peaks, _ = find_peaks(window, height=0.2, width = (3, 10))  # Vous pouvez ajuster la hauteur en fonction de vos données
                valid_peaks_diff = [peak for peak in peaks if any(abs(peak - min_pt) <= 5 for min_pt in minima)]

                valid_peaks_timing = [peak for peak in valid_peaks_diff if peak>=5 and peak<=25]

                valid_peaks_ampl = [peak for peak in valid_peaks_timing if amplitude(window, peak-5, peak+5)>=amplitude_globale*0.3]

                valid_peaks = valid_peaks_diff

                if len(valid_peaks) > 0 :
                    window_counting += 1
                    spike_amp += amplitude_globale
                    true_timer = 0

                    for true_spike in solution : 
                        if true_spike[0]==subject and true_spike[1]==window_id : 
                            true_timer = 15 + true_spike[4]

                    # Visualisation
                   
                    
                    if 0<true_timer<30  :
                        delta = []
                        plt.plot(window)
                        for peak in valid_peaks :
                            delta.append(abs(peak-true_timer))
                        indice_min = delta.index(min(delta))
                        peak = valid_peaks[indice_min]
                        if (abs(true_timer-peak))<=5:
                            validation +=1
                            delta_valid+=abs(peak-true_timer)
                            green_count_valid +=1
                        else :
                            unvalidation +=1
                            delta_unvalid+= abs(peak-true_timer)
                            green_count_unvalid +=1
                            
                        plt.axvline(x = true_timer, color = 'green', linestyle = '--')
                        plt.axvline(x=peak, color='r', linestyle='--')
                        plt.title("Identification des spikes")
                        plt.xlabel("Time points")
                        plt.ylabel("Amplitude")
                        plt.savefig(f'/sps/crnl/tdejean/img/cluster9/spike_centrering_sub{subject}_{window_id}.png')
                        plt.close()
                    
                        
                else : 
                    no_spike_amp += amplitude_globale


print('Nombre de spikes : ', spike_counting)
print('Nombre de fenetres  : ', window_counting)
print('Taux de détection : ',window_counting/spike_counting )

print('Nombre de pics bien positionnés :', validation)
print('Nombre de pics mal positionnés :', unvalidation)


print('Ecart moyen si détection valide :',delta_valid/green_count_valid )
print('Ecart moyen si détection non valide :',delta_unvalid/green_count_unvalid )
print('Ecart moyen global:',(delta_unvalid+delta_valid)/(green_count_unvalid+green_count_valid) )


print('amplitude moyenne no_spike : ',no_spike_amp/(spike_counting-window_counting))
print('amplitude moyenne spike : ',spike_amp/(window_counting))

