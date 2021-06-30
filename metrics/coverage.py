"""Metrics for measuring coverage of configuration space"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from HMSM import models

def expected_cover_time(T : ndarray):
    """expected_coverage_time.
    Gets an upper bound on the expected time it would take a naive simulation to visit each state 
    in the provided HMSM.
    We use the bound from [1]_ (Theorem 5)) which shows that :math:`E(C) = O(n^2\log n /(1-\lambda_2))`
    Parameters
    ----------
    hmsm : models.BaseHierarchicalMSMTree
        hmsm

    Returns
    -------

    References
    ----------
    [1] Broder, A.Z., Karlin, A.R. Bounds on the cover time. J Theor Probab 2, 101–120 (1989). 
    """
    T = hmsm.get_full_T()
    eigvals, eigvecs = np.linalg.eig(T)
    decreasing_order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[decreasing_order]
    eigvecs = eigvecs[decreasing_order]
    lambda_2 = eigvals[1]
    n = len(lambda_2)
    pi = np.sort(eigvecs[0])
    partial_inverse_sum = lambda i : 1/(np.sum(pi[:i+1]))

    summand_1 = 2*n*np.log(n)
    summand_2 = -n*np.log(pi[0])
    summand_3 = np.sum([partial_inverse_sum(i) for i in range(n)])

    return (summand_1 + summand_2 + summand_3)/(1-lambda_2)

