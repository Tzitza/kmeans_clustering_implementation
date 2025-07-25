import numpy as np
import matplotlib.pyplot as plt
import time

def mykmeans(data, k, epsilon=1e-3, max_iter=100):
    """
    Υλοποιεί τον αλγόριθμο K-means με Ευκλείδεια απόσταση.
    Επιστρέφει τα κέντρα, τις ετικέτες των σημείων και το SSE για κάθε βήμα.
    """
    np.random.seed(int(time.time()))  # Τυχαίο seed βασισμένο στον χρόνο για διαφορετικά αποτελέσματα κάθε φορά
    
    # Τυχαία επιλογή αρχικών κέντρων από τα δεδομένα
    centers = data[np.random.choice(len(data), k, replace=False)]
    labels = np.zeros(len(data), dtype=int)  #Αρχικοποίηση ετικετών για κάθε σημείο
    sse_list = []  # Λίστα για το SSE ανά επανάληψη

    for iteration in range(max_iter):
        # Υπολογισμός αποστάσεων σημείων από τα κέντρα και ανάθεση στο πλησιέστερο cluster
        for i, point in enumerate(data):
            distances = np.linalg.norm(point - centers, axis=1) # Ευκλείδεια απόσταση
            labels[i] = np.argmin(distances) # Εύρεση του πλησιέστερου κέντρου

        # Υπολογισμός του SSE (Sum of Squared Errors) για το τρέχον βήμα
        sse = sum(np.linalg.norm(data[labels == i] - centers[i]) ** 2 for i in range(k))
        sse_list.append(sse)# Αποθήκευση του SSE    

        # Ενημέρωση κέντρων ως μέσος όρος των σημείων κάθε cluster
        new_centers = np.array([data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 else centers[i] for i in range(k)])

        # Οπτικοποίηση της τρέχουσας κατάστασης
        plot_kmeans_step(data, labels, centers, iteration)

        # Έλεγχος τερματισμού: Αν η αλλαγή στα κέντρα είναι μικρότερη από το epsilon
        if np.linalg.norm(new_centers - centers) < epsilon:
            break

        centers = new_centers # Ενημέρωση των κέντρων για την επόμενη επανάληψη

    return centers, labels, sse_list

def plot_kmeans_step(data, labels, centers, step):
    """
    Δημιουργεί γράφημα της κατάστασης του K-means σε κάθε βήμα.
    """
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=color, label=f'Cluster {i + 1}')
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='+', label='Centers')
    plt.title(f'K-means - Step {step + 1}')
    plt.legend()
    plt.show()

def generate_data():
    """
    Δημιουργεί δεδομένα από τρεις κανονικές κατανομές.
    Επιστρέφει έναν πίνακα με 150 σημεία (50 από κάθε κατανομή).
    """
    np.random.seed(42)  # Σταθερό seed για αναπαραγωγιμότητα
    n_samples = 150     # Συνολικός αριθμός σημείων
    # Συνδιακυμάνσεις και μέσες τιμές για τις τρεις κατανομές
    covs = [np.array([[0.29, 0.4], [0.4, 4]]),
            np.array([[0.29, 0.4], [0.4, 0.9]]),
            np.array([[0.64, 0], [0, 0.64]])]
    means = [[4, 0], [5, 7], [7, 4]]

     # Δημιουργία των σημείων
    data = np.vstack([np.random.multivariate_normal(mean, cov, n_samples // 3) for mean, cov in zip(means, covs)])
    return data

def plot_final_clustering(data, labels, centers):
    #Δημιουργία γράφήματος της τελικής ομαδοποίησης (Final Clustering)
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {i + 1}')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='+', s=100, label='Centers')
    plt.title('Final Clustering')
    plt.legend()
    plt.show()

def plot_sse(sse_list):

    #Δημιουργία γραφήματος για το SSE ανά επανάληψη.
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(sse_list) + 1), sse_list, marker='o')
    plt.title('SSE per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('SSE')
    plt.show()

# Εκτέλεση
data = generate_data()  # Δημιουργία δεδομένων
centers, labels, sse_list = mykmeans(data, k=3)  # Εκτέλεση του K-means
plot_final_clustering(data, labels, centers)  # Τελικό διάγραμμα clustering
plot_sse(sse_list)  # Γράφημα SSE
