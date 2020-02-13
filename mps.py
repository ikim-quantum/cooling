#######################################
# MPS library for algorithmic cooling #
# Galit Anikeeva, Isaac Kim           #
# 2/13/2020                           #
# Use it as you wish. Just don't sue  #
# us.                                 #
#######################################
import numpy as np

class MPS():
    """
    Matrix Product State class

    Attr:
        l(int): Length of the chain
        ds(list of ints): A list of local dimensions.
                          The length must be l.
        Ds_left(list of ints): A list of "left" bond
                               dimensions.
        Ds_right(list of ints): A list of "right" bond
                                dimensions.
        matrices(list of ndarrays): A list of matrices

    Data format for the matrix: Recall that a matrix
    product state is defined by a set of matrices. If
    we fix the local (physical) dimension to be d, for
    each site we can associate d different matrices. Each
    of the matrices are D x D matrices, where D is the 
    bond dimension of the underlying MPS. We can access
    each of these matrices as follows. Suppose we want
    to pick the k'th matrix of the ith site. Then one 
    can just use

    >>> a.matrices[i][k]

    where a is an instance of the MPS class.
    Instead, if one wants to see all the matrices, one 
    can use
    
    >>> a.matrices[i]

    which will return a (d x D x D) numpy ndarray.
    """
    def __init__(self, l, d, D):
        """
        Initialize a matrix product state with length
        l, ds=[d,...,d], Ds_left=Ds_right=[D,...,D].
        """
        self.l = l
        self.ds = [d] * l
        self.Ds_left = [D] * l
        self.Ds_right = [D] * l
        self.matrices = []

        # Initialize the matrices to zero matrices
        for i in range(l):
            self.matrices.append(np.zeros([self.ds[i],
                                           self.Ds_left[i],
                                           self.Ds_right[i]]))

    def norm(self):
        """
        Returns the norm of the MPS.

        Returns:
            double: Norm of the MPS.
        """
