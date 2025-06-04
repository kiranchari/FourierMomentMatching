"""
Matrix square root. matsqrt() was added as a version without grads. use sqrtm if you need grads
original script source: https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
"""

import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg
import math

BLOCK_DIAG_THRESHOLD = 10000 # 3072 # maximum size to do full matsqrt or inverse, use block diag cov above this. 3072 = 3*32*32 (CIFAR10 size)

def matsqrt(input, block_diag=False):
    """
    input is torch.Tensor (square matrix)
    does not compute gradients. use sqrtm to compute grads
    """
    if type(input) != list:
        w = input.shape[0]
        device = input.device

        m = input.detach().cpu().numpy().astype(np.float_)
        del input # to make space

        print('==> Sqrt matrix size:', m.shape)

    if block_diag:
        if type(input) == list:
            print('Square-rooting list of blocks')
            device = input[0].device
            input = [m.detach().cpu().numpy().astype(np.float_) for m in input]
            input = [scipy.linalg.sqrtm(m).real for m in input]
            input = [torch.from_numpy(m).to(device) for m in input]
            return input

        else:
            print('Block diagonal SQRT')
            n = w/block_diag
            n = math.ceil(n)

            start, stop = 0, block_diag

            print('Total num of diagonal blocks:', n)
            print('Block size:', block_diag)

            for _ in range(n):
                if _ == n-1:
                    stop = w # last block

                # print(start, stop)
                # sqrt diagonal blocks.
                m[start:stop, start:stop] = scipy.linalg.sqrtm(m[start:stop, start:stop]).real

                # set rest of row block to 0            
                m[start:stop, :start] = 0
                m[start:stop, stop:] = 0
                
                start = stop
                stop += block_diag

            m = torch.from_numpy(m).to(device)
            return m

    else:
        print('Full SQRT')
        m = scipy.linalg.sqrtm(m).real
        m = torch.from_numpy(m).to(device)
        return m

def inverse(input, block_diag=False):
    """
    Note: this function modifies the input in place (in the block diag case)

    input is torch.Tensor (square matrix)
    CIFAR10 input is: 3,072 x 3,072
    ImageNet: 150,528 x 150,528

    """
    if type(input) != list:
        w,h=input.shape[0], input.shape[1]
        print('INVERSE func input width:', w,h)

    if block_diag:
        if type(input) == list:
            print('Inverting List of Blocks')
            # input already a list of blocks
            input = [_.inverse() for _ in input]
            return input

        else:
            print('Block DIAG Inverse')
            n = w/block_diag
            n = math.ceil(n)

            start, stop = 0, block_diag

            print('Total num of diagonal blocks:', n)
            print('Block size:', block_diag)

            for _ in range(n):
                if _ == n-1:
                    stop = w # last block

                # print(start, stop)
                # invert diagonal blocks.
                input[start:stop, start:stop] = input[start:stop, start:stop].inverse()
        
                # set rest of row block to 0            
                input[start:stop, :start] = 0
                input[start:stop, stop:] = 0
                
                start = stop
                stop += block_diag

            return input
        
    else:
        print('Full Inverse')
        return input.inverse()

def get_diagonal_sub_matrices(mat, block_size):
    """
    mat is coloring matrix (D,D)
    """
    print('Matrix shape', mat.shape)
    w,h=mat.shape[0], mat.shape[1]
    sub_matrices = []

    n = w/block_size
    n = math.ceil(n)

    start, stop = 0, block_size

    for _ in range(n):
        if _ == n-1:
            stop = w # last block

        sub_matrices.append(mat[start:stop, start:stop])
            
        start = stop
        stop += block_size

    return sub_matrices

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

sqrtm = MatrixSquareRoot.apply

def main():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = (k.t().matmul(k)).requires_grad_()
    test = gradcheck(sqrtm, (pd_mat,))
    print(test)


if __name__ == '__main__':
    main()
