import numpy as np
from scipy.sparse import csr_matrix
from time import time
import warnings
from sklearn.neighbors import NearestNeighbors
import matlab

eng=matlab.engine.start_matlab()

def S(k, t,Y_train,Y_test):
    # Assuming data is loaded from somewhere, e.g., from a file
    # Y_train and Y_test should be loaded or passed as arguments to the function
    # Y_train = np.load('path_to_Y_train.npy')
    # Y_test = np.load('path_to_Y_test.npy')

    options = {
        'Metric': 'Euclidean',#Cosine
        'WeightMode': 'HeatKernel',#Binary/Cosine
        'NeighborMode': 'KNN',
        'k': k,
        't': t
    }

    ntrn = Y_train.shape[0]
    ntest = Y_test.shape[0]
    
    temp2 = np.zeros((ntrn, ntest))
    for i in range(ntest):
        fmri = np.vstack([Y_train, Y_test[i, :]])
        print(fmri)
        print(options)
        temp,elapsed_time = constructW(fmri, options)
        print(temp)
        temp2[:, i] = temp[:-1, -1]

    S = np.zeros((ntrn, ntest))
    for i in range(ntest):
        sorted_indices = np.argsort(-temp2[:, i])
        selectidx = sorted_indices[1:k+1]
        S[selectidx, i] = -np.sort(-temp2[:, i])[1:k+1]

    return S

def EuDist2(X, Y=None, squared=True):
    """Compute squared Euclidean distances between data points."""
    if Y is None:
        Y = X
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = np.sum(diff**2, axis=-1)
    if not squared:
        distances = np.sqrt(distances)
    return distances

# Placeholder for LLE_Matrix function
def LLE_Matrix(data, k, regLLE):
    # Implement or convert this function from MATLAB if needed
    W = None  # Placeholder
    M = None  # Placeholder
    return W, M

def constructW(fea, options=None):
    # Melengkapi opsi di options
    if options is None:
        options = {}

    if "LLE" in options and options["LLE"]:
        start_time = time()
        W, M = LLE_Matrix(fea.T, options["k"], options["regLLE"])
        elapsed_time = time() - start_time
        return W, elapsed_time, M
    
    options.setdefault("Metric", "Cosine")
    
    if options["Metric"].lower() not in ["euclidean", "cosine"]:
        raise ValueError("Metric does not exist!")
    
    if options["Metric"].lower() == "cosine" and "bNormalized" not in options:
        options["bNormalized"] = 0
    
    options.setdefault("NeighborMode", "KNN")
    
    if options["NeighborMode"].lower() not in ["knn", "supervised"]:
        raise ValueError("NeighborMode does not exist!")
    
    if options["NeighborMode"].lower() == "supervised":
        if "bLDA" not in options:
            options["bLDA"] = 0
        if options["bLDA"]:
            options["bSelfConnected"] = 1
        if "k" not in options:
            options["k"] = 0
        if "gnd" not in options:
            raise ValueError("Label(gnd) should be provided under 'Supervised' NeighborMode!")
        if fea.shape[0] != len(options["gnd"]):
            raise ValueError("gnd doesn't match with fea!")
    
    options.setdefault("WeightMode", "Binary")
    
    if options["WeightMode"].lower() not in ["binary", "heatkernel", "cosine"]:
        raise ValueError("WeightMode does not exist!")
    
    if options["WeightMode"].lower() == "heatkernel" and options["Metric"].lower() != "euclidean":
        warnings.warn("'HeatKernel' WeightMode should be used under 'Euclidean' Metric!")
        options["Metric"] = "euclidean"
        if "t" not in options:
            options["t"] = 1
    
    if options["WeightMode"].lower() == "cosine" and options["Metric"].lower() != "cosine":
        warnings.warn("'Cosine' WeightMode should be used under 'Cosine' Metric!")
        options["Metric"] = "cosine"
        if "bNormalized" not in options:
            options["bNormalized"] = 0
    
    options.setdefault("bSelfConnected", 1)
    
    start_time = time()
    
    if "gnd" in options:
        nSmp = len(options["gnd"])
    else:
        nSmp = fea.shape[0]
    
    maxM = 62500000
    BlockSize = maxM // (nSmp * 3)
    print(options)
    # Disini mulai komputasi dari W nya
    #Depending on the chosen options, the function computes W in different ways:
    #If 'Supervised' mode is used, then W is constructed based on class labels (gnd field in options). 
    #Points from the same class can be connected by binary weights, weights based on the heat kernel, or cosine similarity.
    #If 'KNN' mode is used, the function computes the k-nearest neighbors for each point based on the specified metric. 
    #Weights are then assigned to these connections based on the WeightMode.
    
    #Finally, the function returns the weight matrix W, the time taken (elapse), and in some cases, the matrix M (used for LLE).

    # You will need to fill in the logic here as per your needs
    # For Supervised #
    if options["NeighborMode"].lower() == "supervised":
        Label = np.unique(options["gnd"])
        nLabel = len(Label)

        if options["bLDA"]:
            G = np.zeros((fea.shape[0], fea.shape[0]))
            for idx in Label:
                classIdx = options["gnd"] == idx
                G[classIdx, classIdx] = 1 / np.sum(classIdx)
            W = csr_matrix(G)
            elapse = time() - start_time
            return W, elapse

        if options["WeightMode"].lower() == "binary":
            print("Rest of the logic for 'binary' mode")
            G = np.zeros((nSmp*(options["k"]+1), 3))
            idNow = 0
            for idx in Label:
                classIdx = np.where(options["gnd"] == idx)[0]
                D = EuDist2(fea[classIdx], squared=False)
                idx_sorted = np.argsort(D, axis=1)
                idx_selected = idx_sorted[:, :options["k"]+1]
                
                nSmpClass = len(classIdx) * (options["k"]+1)
                G[idNow:idNow+nSmpClass, 0] = np.repeat(classIdx, options["k"]+1)
                G[idNow:idNow+nSmpClass, 1] = classIdx[idx_selected.ravel()]
                G[idNow:idNow+nSmpClass, 2] = 1
                idNow += nSmpClass

            G = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
            G = G.maximum(G.T)

            if not options["bSelfConnected"]:
                G.setdiag(0)

            W = G.copy()
            
        elif options["WeightMode"].lower() == "heatkernel":
            print("Rest of the logic for 'heatkernel' mode")
            G = np.zeros((nSmp*(options["k"]+1), 3))
            idNow = 0
            for idx in Label:
                classIdx = np.where(options["gnd"] == idx)[0]
                D = EuDist2(fea[classIdx], squared=False)
                dump, idx_sorted = np.sort(D, axis=1), np.argsort(D, axis=1)
                idx_selected = idx_sorted[:, :options["k"]+1]
                dump_selected = dump[:, :options["k"]+1]
                
                heat_kernel_weight = np.exp(-dump_selected / (2 * options["t"]**2))
                
                nSmpClass = len(classIdx) * (options["k"]+1)
                G[idNow:idNow+nSmpClass, 0] = np.repeat(classIdx, options["k"]+1)
                G[idNow:idNow+nSmpClass, 1] = classIdx[idx_selected.ravel()]
                G[idNow:idNow+nSmpClass, 2] = heat_kernel_weight.ravel()
                idNow += nSmpClass

            G = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
            G = G.maximum(G.T)

            if not options["bSelfConnected"]:
                G.setdiag(0)

            W = G.copy()

        elif options["WeightMode"].lower() == "cosine":
            # Normalize the data if required
            # if not options["bNormalized"]:
            #     feaNorm = np.linalg.norm(fea, axis=1)
            #     fea = fea / np.maximum(feaNorm[:, np.newaxis], 1e-12)
            if not options.get("bNormalized", False):
                fea_norm = np.linalg.norm(fea, axis=1, keepdims=True)
                fea = np.divide(fea, fea_norm, where=fea_norm!=0)

            G = np.zeros((nSmp*(options["k"]+1), 3))
            idNow = 0
            for idx in Label:
                classIdx = np.where(options["gnd"] == idx)[0]
                D = fea[classIdx].dot(fea[classIdx].T)
                
                # We're looking for highest cosine similarity, so we sort in descending order
                dump, idx_sorted = np.sort(-D, axis=1), np.argsort(-D, axis=1)
                idx_selected = idx_sorted[:, :options["k"]+1]
                dump_selected = -dump[:, :options["k"]+1]
                
                nSmpClass = len(classIdx) * (options["k"]+1)
                G[idNow:idNow+nSmpClass, 0] = np.repeat(classIdx, options["k"]+1)
                G[idNow:idNow+nSmpClass, 1] = classIdx[idx_selected.ravel()]
                G[idNow:idNow+nSmpClass, 2] = dump_selected.ravel()
                idNow += nSmpClass

            G = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
            G = G.maximum(G.T)

            if not options["bSelfConnected"]:
                G.setdiag(0)

            W = G.copy()

        else:
            raise ValueError("WeightMode does not exist!")
    # For KNN #
    if options["NeighborMode"].lower() == "knn" and options["k"] > 0:
        if options["Metric"].lower() == "euclidean":
            G = np.zeros((nSmp * (options["k"] + 1), 3))
            
            # Here's where you might use some sort of logic like distance calculations,
            # matrix manipulations, etc., to populate G.
            # This is just a placeholder.
            
            W = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
        else:
            # Implement the cosine logic here
            pass  # Placeholder logic
    # For KNN #
    if options["NeighborMode"].lower() == "knn" and options["k"] > 0:
        # Using Euclidean distance for KNN
        if options["Metric"].lower() == "euclidean":
            # Get k nearest neighbors using sklearn's NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=options["k"]+1, algorithm='ball_tree', metric='euclidean').fit(fea)
            distances, indices = nbrs.kneighbors(fea)
            
            # Convert neighbors into a sparse graph representation
            rows = np.repeat(np.arange(nSmp), options["k"]+1)
            cols = indices.ravel()
            
            # Depending on the weight mode, we'll compute the weights differently
            if options["WeightMode"].lower() == "binary":
                weights = np.ones_like(cols)
            elif options["WeightMode"].lower() == "heatkernel":
                t = options.get("t", 1.0)  # get the t parameter
                weights = np.exp(-distances.ravel() / (2 * t ** 2))
            else:
                raise ValueError('Unsupported WeightMode for Euclidean metric!')
            
            # Create sparse adjacency matrix
            G = csr_matrix((weights, (rows, cols)), shape=(nSmp, nSmp))
            G = G.maximum(G.T)  # Ensure the graph is symmetric

            # If self connections are not allowed, remove diagonal
            if not options.get("bSelfConnected", True):
                G.setdiag(0)
            W = G.copy()
        
        # Using Cosine distance for KNN
        elif options["Metric"].lower() == "cosine":
            # Normalize the feature vectors
            fea_norm = np.linalg.norm(fea, axis=1, keepdims=True)
            fea = np.where(fea_norm > 0, fea / fea_norm, fea)
            
            # Get k nearest neighbors based on cosine similarity
            nbrs = NearestNeighbors(n_neighbors=options["k"]+1, algorithm='brute', metric='cosine').fit(fea)
            distances, indices = nbrs.kneighbors(fea)

            # Convert neighbors into a sparse graph representation
            rows = np.repeat(np.arange(nSmp), options["k"]+1)
            cols = indices.ravel()

            # Cosine similarity to weight
            weights = 1 - distances.ravel()
            
            # Create sparse adjacency matrix
            G = csr_matrix((weights, (rows, cols)), shape=(nSmp, nSmp))
            G = G.maximum(G.T)  # Ensure the graph is symmetric

            # If self connections are not allowed, remove diagonal
            if not options.get("bSelfConnected", True):
                G.setdiag(0)

            W = G.copy()
        else:
            raise ValueError('Unsupported metric!')
    else:
        raise ValueError('Invalid options for NeighborMode or k!')


    #return part
    elapsed_time = time() - start_time
    return W.toarray(), elapsed_time  # M is not defined in the provided code.

# Usage
# fea = ...  # Some numpy array or data
# options = {"k": 5, "Metric": "euclidean", "NeighborMode": "knn", ...}
# W, elapsed_time = constructW(fea, options)
