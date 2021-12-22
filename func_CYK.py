import numpy as np
import pandas as pd

P_r = pd.read_csv("data/rules/P_r.csv").to_numpy()
P_h = pd.read_csv("data/rules/P_h.csv").to_numpy().reshape((11,11,11))
P_v = pd.read_csv("data/rules/P_v.csv").to_numpy().reshape((11,11,11))

def DataConverter(S0, S1, SE):
    """Converts images into a usable form.
       >>> DataConverter(np.zeros((1,1)), np.ones((1,1)), np.array([[1],[0],[1]]))
       array([[1],
              [0],
              [1]])
       >>> DataConverter(0, 0, 0)
       Traceback (most recent call last):
        ...
       TypeError: S0, S1 and SE must be numpy arrays
       >>> DataConverter(np.zeros((1,1)), np.ones((2,1)), np.array([[1],[0],[1]]))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of S0 and S1
       >>> DataConverter(np.zeros((1,2)), np.ones((1,2)), np.array([[1],[0],[1]]))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of SE
       >>> DataConverter(np.zeros((1,1)) + 2, np.ones((1,1)), np.array([[1],[0],[1]]))
       Traceback (most recent call last):
        ...
       ValueError: elements of S0 and S1 must be equal to 0 or 1
       >>> DataConverter(np.zeros((1,1)), np.ones((1,1)), np.array([[2],[0],[2]]))
       Traceback (most recent call last):
        ...
       ValueError: elements of SE must be equal to 0 or 1
       >>> DataConverter(np.zeros((1,2)), np.ones((1,2)), np.array([[1, 0],[0, 0],[1, 1]]))
       Traceback (most recent call last):
        ...
       ValueError: SE does not correspond to S0 and S1
    """
    if type( S0 ) != np.ndarray or type( S1 ) != np.ndarray or type( SE ) != np.ndarray:
        raise TypeError( "S0, S1 and SE must be numpy arrays" )
    if S0.shape[0] != S1.shape[0] or S0.shape[1] != S1.shape[1]:
        raise ValueError( "wrong shape of S0 and S1" )
    if SE.shape[0] != 3 * S0.shape[0] or SE.shape[1] % S0.shape[1] != 0:
        raise ValueError( "wrong shape of SE" )
    if np.min(np.isin(S0, np.array([0, 1]))) == 0 or np.min(np.isin(S1, np.array([0, 1]))) == 0:
        raise ValueError( "elements of S0 and S1 must be equal to 0 or 1" )
    if np.min(np.isin(SE, np.array([0, 1]))) == 0:
        raise ValueError( "elements of SE must be equal to 0 or 1" )
    h, w = S0.shape[0], S0.shape[1]
    c = SE.shape[1] // w
    X = np.zeros((3, c), dtype = int)

    if len(S0.shape) == 3:
        S0 = S0[:,:,0]
    if len(S1.shape) == 3:
        S1 = S1[:,:,0]
    if len(SE.shape) == 3:
        SE = SE[:,:,0]

    for i in range(3):
        for j in range(c):
            if np.all(SE[h * i:h * (i + 1),w * j:w * (j + 1)] == S0):
                X[i,j] = 0
            elif np.all(SE[h * i:h * (i + 1),w * j:w * (j + 1)] == S1):
                X[i,j] = 1
            else:
                X[i,j] = 2
    if np.min(np.isin(X, np.array([0, 1]))) == 0:
        raise ValueError( "SE does not correspond to S0 and S1" )
    return X

def CYK(X, D = {}):
    """Implements modification of CYK algorithm.
       >>> CYK(np.array([[0],[0],[0]]))[0,0]
       0
       >>> CYK(0)
       Traceback (most recent call last):
        ...
       TypeError: X must be numpy array
       >>> CYK(np.array([[2],[0],[2]]))
       Traceback (most recent call last):
        ...
       ValueError: elements of X must be equal to 0 or 1
    """
    if type( X ) != np.ndarray:
        raise TypeError( "X must be numpy array" )
    if np.min(np.isin(X, np.array([0, 1]))) == 0:
        raise ValueError( "elements of X must be equal to 0 or 1" )
    h, w = X.shape[0], X.shape[1]
    S = np.zeros((11,1), dtype = int)
    if str(X) in D:
        return D[str( X )]
    if max(h,w) == 1:
        D[str( X )] = P_r[:, X[0, 0]].reshape( (11, 1) )
        return D[str( X )]
    for j in range(1,w):
        S = np.maximum(S, np.amax(CYK(X[:,:j]) * CYK(X[:,j:]).T * P_h, axis = (1,2)).reshape((11,1)))
    for i in range(1,h):
        S = np.maximum(S, np.amax(CYK(X[:i]) * CYK(X[i:]).T * P_v, axis = (1,2)).reshape((11,1)))
    D[str( X )] = np.maximum( S, np.amax( S * P_r[:, 2:].T, axis = 0 ).reshape( (11, 1) ) )
    return D[str( X )]

def AnswerFilter(X):
    """Prints the solution for the addition problem.
       >>> AnswerFilter(np.zeros((11,1)))
       Solution is incorrect
       >>> AnswerFilter(0)
       Traceback (most recent call last):
        ...
       TypeError: X must be numpy array
       >>> AnswerFilter(np.zeros((1,1)))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of X
       >>> AnswerFilter(np.zeros((11,1)) + 2)
       Traceback (most recent call last):
        ...
       ValueError: elements of X must be equal to 0 or 1
    """
    if type( X ) != np.ndarray:
        raise TypeError( "X must be numpy array" )
    if X.shape != (11, 1):
        raise ValueError( "wrong shape of X" )
    if np.min(np.isin(X, np.array([0, 1]))) == 0:
        raise ValueError( "elements of X must be equal to 0 or 1" )
    if X[9,0] == 1:
        print("Solution is correct without overflow")
    elif X[10,0] == 1:
        print("Solution is correct with overflow")
    else:
        print("Solution is incorrect")

if __name__ == "__main__":
    import doctest
    doctest.testmod()