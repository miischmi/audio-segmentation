import numpy as np
import scipy.spatial
import librosa
import matplotlib.pyplot as plt
from matplotlib import patches
import time

def compute_cost_matrix(X, Y, metric='euclidean'):
    """Compute the cost matrix of two feature sequences

     From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        X: Sequence 1
        Y: Sequence 2
        metric: Cost metric, a valid strings for scipy.spatial.distance.cdist

    Returns:
        C: Cost matrix
    """
    X, Y = np.atleast_2d(X, Y)
    C = scipy.spatial.distance.cdist(X.T, Y.T, metric=metric)
    return C

def cost_matrix_dot(X, Y):
    """Computes cost matrix via dot product

     From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        X, Y: Feature seqeuences (given as K x N and K x M matrices)

    Returns:
        C: cost matrix
    """
    return 1 - np.dot(X.T, Y)

def compute_accumulated_cost_matrix_subsequence_dtw(C):
    """Given the cost matrix, compute the accumulated cost matrix for
       subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        C: cost matrix

    Returns:
        D: Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N, M))
    D[:, 0] = np.cumsum(C[:, 0])
    D[0, :] = C[0, :]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

def compute_accumulated_cost_matrix_subsequence_dtw_21(C):
    """Given the cost matrix, compute the accumulated cost matrix for 
    subsequence dynamic time warping with step sizes {(1, 1), (2, 1), (1, 2)}

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        C: cost matrix

    Returns:
        D: Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf
    
    D[1, 2:] = C[0, :]
    
    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n+1, m+2] = C[n, m] + min(D[n-1+1, m-1+2], D[n-2+1, m-1+2], D[n-1+1, m-2+2])
    D = D[1:, 2:]
    return D

def compute_matching_function_dtw(X, Y, stepsize=2):
    """Compute matching function

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X: Query feature sequence (given as K x N matrix)
        Y: Database feature sequence (given as K x M matrix)
        stepsize: Parameter for step size condition (1 or 2)

    Returns:
        Delta: DTW-based matching function
        C: Cost matrix
        D: Accumulated cost matrix
    """
    C = cost_matrix_dot(X, Y)
    if stepsize == 1:
        D = compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
    return Delta, C, D

def mininma_from_matching_function(Delta, rho=2, tau=0.2, num=None):
    """Derives local minima positions of matching function in an iterative fashion

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        Delta: Matching function
        rho: Parameter to exclude neighborhood of a matching position for subsequent matches
        tau: Threshold for maximum Delta value allowed for matches
        num: Maximum number of matches

    Returns:
        pos: Array of local minima
    """
    Delta_tmp = Delta.copy()
    M = len(Delta)
    pos = []
    num_pos = 0
    rho = int(rho)
    if num is None:
        num = M
    while num_pos < num and np.sum(Delta_tmp < tau) > 0:
        m = np.argmin(Delta_tmp)
        pos.append(m)
        num_pos += 1
        Delta_tmp[max(0, m - rho):min(m + rho, M)] = np.inf
    pos = np.array(pos).astype(int)
    return pos
        
def compute_optimal_warping_path_subsequence_dtw(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
       subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        D: Accumulated cost matrix
        m: Index to start back tracking; if set to -1, optimal m is used

    Returns
        P: Warping path (list of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P

def compute_optimal_warping_path_subsequence_dtw_21(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 1), (2, 1), (1, 2)}

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_SubsequenceDTW.ipynb

    Args:
        D: Accumulated cost matrix
        m: Index to start back tracking; if set to -1, optimal m is used

    Returns
        P: Warping path (list of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n-1, 0)
        else:
            val = min(D[n-1, m-1], D[n-2, m-1], D[n-1, m-2])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-2, m-1]:
                cell = (n-2, m-1)
            else:
                cell = (n-1, m-2)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P

def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        pos: End positions of matches
        D: Accumulated cost matrix

    Returns:
        matches: Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches
    
def print_formatted_matches(matches, hopsize, Fs, N):
    """Converts Feature-Positions to seconds

    Args:
        matches: Array containing matches (start, end)
    
    Returns: 
        matches_sec: Array containing matches in seconds (Start, end)
    """
    matches_sec = []
    for (s,t) in matches:
        s = time.strftime('%H:%M:%S', time.gmtime(s * hopsize / Fs + (N / Fs) / 2))
        t = time.strftime('%H:%M:%S', time.gmtime(t * hopsize / Fs + (N / Fs) / 2))
        matches_sec.append([s, t])
    return print(matches_sec)

