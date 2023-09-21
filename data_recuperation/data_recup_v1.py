

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct

df = pd.read_csv('/sps/crnl/tdejean/img/cluster2/graph_hybrid_cvpredictions_with_clusters_modified.csv')

with open('/sps/crnl/tdejean/img/cluster2/cluster1.txt', 'r') as f:
    indices_cluster_1 = [int(line.strip()) for line in f]

with open('/sps/crnl/tdejean/img/cluster2/cluster2.txt', 'r') as g:
    indices_cluster_2 = [int(line.strip()) for line in g]

    # Initialiser la colonne cluster à 0
df['cluster'] = 0

# Mettre à jour les valeurs pour les clusters 1 et 2 en utilisant les indices
df.loc[indices_cluster_1, 'cluster'] = 1
df.loc[indices_cluster_2, 'cluster'] = 2

# Sauvegarder le DataFrame modifié
df.to_csv('/sps/crnl/tdejean/img/cluster2/graph_hybrid_cvpredictions_with_clusters_modified.csv', index=False)



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
        clusters_data = {0: [], 1: [], 2: []}
        
        for index, row in subject_data.iterrows():
            window_id = int(row['counter'])
            cluster = int(row['cluster'])
            window_data = windows_data[window_id]
            clusters_data[cluster].append(window_data)
            
        # 4. Calculer la courbe moyenne pour chaque cluster
        # 5. Calculer les quartiles pour chaque cluster
        for cluster, data in clusters_data.items():
            if data:  # Vérifier si des données existent pour ce cluster
                data_array = np.array(data)
                mean_curve = np.mean(data_array, axis=0)
                q1 = np.percentile(data_array, 25, axis=0)
                q3 = np.percentile(data_array, 75, axis=0)
                
                # 6. Afficher la courbe moyenne avec la zone grise pour les quartiles
                plt.plot(mean_curve, label=f"Cluster {cluster}")
                plt.fill_between(range(len(mean_curve)), q1, q3, color='gray', alpha=0.5)
    
    plt.legend()
    plt.savefig(f"/sps/crnl/tdejean/img/cluster3/Window_cluster_moy_reclustered_{subject}.png")
    plt.close()
