
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

# 1. Lire le fichier CSV

# Pour stocker les données par cluster
clusters_data = {0: [], 1: [], 2: []}
clusters_counts = {0: 0, 1: 0, 2: 0}  # Pour stocker le nombre total de données pour chaque cluster
colors = {0: 'red', 1: 'blue', 2: 'green'}

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
        
        for index, row in subject_data.iterrows():
            window_id = int(row['counter'])
            cluster = int(row['cluster'])
            test_value = int(row['test'])
            
            # Filtrer les données en fonction de la colonne test
            if (cluster == 0 and test_value == 0.0) or (cluster in [1, 2] and test_value == 1.0):
                window_data = windows_data[window_id]
                clusters_data[cluster].append(window_data)
            
            # Compter le nombre total de données pour chaque cluster
            clusters_counts[cluster] += 1

# 4. Calculer la courbe moyenne pour chaque cluster
# 5. Calculer les quartiles pour chaque cluster
for cluster, data in clusters_data.items():
    if data:  # Vérifier si des données existent pour ce cluster
        data_array = np.array(data)
        mean_curve = np.mean(data_array, axis=0)
        q1 = np.percentile(data_array, 25, axis=0)
        q3 = np.percentile(data_array, 75, axis=0)
        
        # 6. Afficher la courbe moyenne
        plt.plot(mean_curve, label=f"Cluster {cluster}")
        
        # Afficher les quartiles en pointillé
        plt.plot(q1, linestyle='--', color=plt.gca().lines[-1].get_color())
        plt.plot(q3, linestyle='--', color=plt.gca().lines[-1].get_color())

    plt.legend()
    plt.savefig(f"/sps/crnl/tdejean/img/cluster3/Window_cluster_moy_reclustered_all_{cluster}.png")
    plt.close()

# 7. Afficher le pourcentage de données utilisées pour chaque cluster
for cluster, data in clusters_data.items():
    percentage = (len(data) / clusters_counts[cluster]) * 100
    print(f"Pourcentage de données utilisées pour le cluster {cluster}: {percentage:.2f}%")

