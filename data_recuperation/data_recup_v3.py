
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct
import random

df = pd.read_csv('/sps/crnl/tdejean/img/cluster3/graph_hybrid_balanced_test_cvpredictions_modified.csv')
clusters_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Pour stocker le nombre total de données pour chaque cluster
cluster_meaning = {0: 'True Negatives', 1: 'False Positive', 2: 'Fasle Negative', 3: 'True Positive'}
counting = 0

clusters_data = {0: [], 1: [], 2: [], 3:[]}
# Pour chaque patient unique
for subject in df['subject'].unique():
    # 2. Lire le fichier binaire correspondant

    with open(f"/sps/crnl/pmouches/data/MEG_PHRC_2006_preprocessedNEWAfterFeedback/data_raw_{str(subject).zfill(3)}_b3_windows_bi", "rb") as f:

        # Chaque fenêtre contient 30 flottants
        window_size = 30
        data = np.fromfile(f, dtype=np.float32)
        num_windows = len(data) // window_size
        windows_data = data.reshape(num_windows, window_size)
        
        # 3. Extraire les données pour chaque fenêtre
        subject_data = df[df['subject'] == subject]
        
        # Pour stocker les données par cluster

        
        for index, row in subject_data.iterrows():
            window_id = int(row['counter'])
            test = float(row['test'])
            pred = float(row['pred'])
            cluster = int(row['cluster'])
            subject = int(row['subject'])
            #if cluster == 1 : 
                #print(window_id)
            if subject == 4 :

                window_data = windows_data[window_id]
                clusters_data[cluster].append(window_data)
                clusters_counts[cluster] += 1
                counting += 1
            #print(clusters_data)



# 4. Calculer la courbe moyenne pour chaque cluster
# 5. Calculer les quartiles pour chaque cluster
for cluster, data in clusters_data.items():
    if data:  # Vérifier si des données existent pour ce cluster
        data_array = np.array(data)
        mean_curve = np.mean(data_array, axis=0)
        median_curve = np.median(data_array,axis=0)

        q1 = np.percentile(data_array, 25, axis=0)
        q3 = np.percentile(data_array, 75, axis=0)

        sample_indices = random.sample(range(data_array.shape[0]), min(10, data_array.shape[0]))
        sample_data = data_array[sample_indices]

        for curve in sample_data:
            plt.plot(curve, alpha=0.3, color='gray')
        
        # 6. Afficher la courbe moyenne
        plt.plot(mean_curve, label=f"Cluster {cluster}")
        #plt.plot(median_curve, label=f"Cluster {cluster}")

        #plt.fill_between(range(len(median_curve)), q1, q3, color='gray', alpha=0.5)
        #print(cluster, q1,q3)

    plt.title(cluster_meaning[cluster]+' mean curve')
    plt.legend()
    plt.savefig(f"/sps/crnl/tdejean/img/cluster3/GH_balanced_confusion__mean_curves_p4_{cluster}.png")
    plt.close()

# 7. Afficher le pourcentage de données utilisées pour chaque cluster
for cluster, data in clusters_data.items():
    percentage = (clusters_counts[cluster] / counting) * 100
    print(f"Pourcentage de données utilisées pour le cluster {cluster}: {percentage:.2f}%")

