##### K-Nearest Neighbors Graphs ########
from sklearn.metrics.pairwise import cosine_similarity

def get_knn_edge_index(x, k=4):
    sim_matrix = cosine_similarity(x)
    edge_list = []
    for i in range(x.shape[0]):
        topk = np.argsort(sim_matrix[i])[-(k+1):]  # include self
        for j in topk:
            if i != j:
                edge_list.append((i, j))
    return torch.tensor(edge_list, dtype=torch.long).t()
