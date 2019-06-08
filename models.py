import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

vgg_bn = models.vgg16_bn(pretrained = True)
class VGG_graph_matching(nn.Module):
    def __init__(self):
        super(VGG_graph_matching, self).__init__()
        self.features1 = nn.Sequential(
            *list(vgg_bn.features.children())[:33]
        )
        self.features2 = nn.Sequential(
            *list(vgg_bn.features.children())[33:43]
        )
                # => TODO: lambda init for trivial test
        self.lam = nn.Parameter(torch.ones(1024, 1024))

    def forward(self, im_1, mask_1=None, im_2 = None, mask_2 = None):
        
        x_1 = self.features1(im_1)

        x_2 = self.features2(x_1)
        if mask_1 is None:
            F1 = x_1
            U1 = x_2
        else:
            F1 = x_1[:,:, mask_1[0]]
            U1 =  x_2[:,:, mask_1[1]]
        if im_2 is None:
            return U1, F1
        
        else:
            x_21 = self.features1(im_2)
            
            x_22 = self.features2(x_21)
            if mask_2 is None:
               F2 = x_21
               U2 = x_22
            else:
               F2 = x_21[:,:, mask_2[0]]
               U2 =  x_22[:,:, mask_2[1]]
            
            test = torch.from_numpy(np.asarray([[1,0,1,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]]))
            [G, H] = self.buildGraphStructure(test)
            
            M = self.affinityMatrix_forward(F1, F2, U1, U2, G, G, H, H) #TODO: Build appropriate graph structure before using this
            v = self.powerIteration_forward(M)
            print(v.shape)
            #S = self.biStochastic_forward(v, G.shape[0], G.shape[0])
            d = self.voting_forward(S, P) #Have to get P still
            return d
    def kronecker(self, matrix1, matrix2):
        return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


    def buildGraphStructure(self, A):
        """
        Arguments:
        ----------
            - A: node-to-node adjaceny matrix

        Returns:
        --------
            - G and H: node-edge incidence matrices such that: A = G*H^T

        """

        # Get number of nodes
        n = A.shape[0]

        # Count number of ones in the adj. matrix to get number of edges
        nr_edges = torch.sum(A)

        # Init G and H
        G = torch.zeros(n, nr_edges) 
        H = torch.zeros(n, nr_edges)

        # Get all non-zero entries and build G and H
        entries = (A != 0).nonzero()
        for count, (i,j) in enumerate(entries, start=0):
            G[i, count] = 1
            H[j, count] = 1

        return [G, H]
    @staticmethod
    def batch_diagonal(input):
        # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
        # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N) 
        # works in  2D -> 3D, should also work in higher dimensions
        # make a zero matrix, which duplicates the last dim of input
        dims = [input.size(i) for i in torch.arange(input.dim())]
        dims.append(dims[-1])
        output = torch.zeros(dims)
        # stride across the first dimensions, add one to get the diagonal of the last dimension
        strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
        strides.append(output.size(-1) + 1)
        # stride and copy the imput to the diagonal 
        output.as_strided(input.size(), strides ).copy_(input)
        return output    


    def affinityMatrix_forward(self, F1, F2, U1, U2, G1, G2, H1, H2):
        """
        Arguments:
        ----------
            - F1, F2: edge features of input image 1 and 2
            - U1, U2: node features of input image 1 and 2
            - G1, H1: node-edge incidence matrices of image 1
            - G2, H2: node-edge incidence matrices of image 2

        Returns:
        ----------
            - M: affinity Matrix computed according to the given formula

        """



        # (a) Build X and Y
        #     - Get ordering of edges from G and H
        #     - Extract edge features of start and end nodes and concat
        idx1_start = (G1 != 0).nonzero()[:,0]
        idx2_start = (G2 != 0).nonzero()[:,0]
        idx1_end = (H1 != 0).nonzero()[:,0]
        idx2_end = (H2 != 0).nonzero()[:,0]
        #X = torch.cat((F1[:,np.array(idx1_start)], F1[:,np.array(idx1_end)]), 2)
        #Y = torch.cat((F2[:,np.array(idx2_start)], F2[:,np.array(idx2_end)]), 2)
        X = torch.cat((F1.view(F1.shape[0],F1.shape[1],-1)[:,:,idx1_start], F1.view(F1.shape[0],F1.shape[1],-1)[:,:,idx1_end]), 1).permute(0,2,1)
        Y = torch.cat((F2.view(F2.shape[0],F2.shape[1],-1)[:,:,idx2_start], F2.view(F2.shape[0],F2.shape[1],-1)[:,:,idx2_end]), 1).permute(0,2,1)


        # (b) Calculate M_e = X * \lambda * Y^T
        M_e = torch.bmm(torch.bmm(X, self.lam.expand(X.shape[0],-1,-1)), Y.permute(0,2,1))

        # (c) Calculate M_p = U1 * U2^T
        #M_p = torch.mm(U1, U2.t())
        M_p = torch.bmm(U1.view(U1.shape[0], U1.shape[1], -1).permute(0,2,1), U2.view(U2.shape[0], U2.shape[1], -1))

        # Calculate M = [vec(M_p)] + (G_2 \kronecker G_1)[vec(M_e)](H_2 \kronecker H_1)^T
        diagM_p = self.batch_diagonal(M_p.view(M_p.shape[0],-1))       
        diagM_e = self.batch_diagonal(M_e.view(M_e.shape[0],-1))    
        M = diagM_p + torch.bmm(torch.bmm(self.kronecker(G2, G1).expand(M_p.shape[0],-1,-1), diagM_e), self.kronecker(H2, H1).expand(M_e.shape[0],-1,-1).permute(0, 2,1))

        return M

    def powerIteration_forward(self, M, N = 100):
        """
        Arguments:
        ----------
            - M: affinity matrix

        Returns:
        --------
            - v*: optimal assignment vector (computed using power iterations) 
        """

        # Init starting v
        v = torch.ones(M.shape[0], M.shape[2],1)
        v_old = torch.ones(M.shape[0], M.shape[2],1) * 1000.
        print(torch.norm(v, 2, dim =0).shape)

        # Perform N iterations: v_k+1 = M*v_k / (||M*v_k||_2) 
        for i in range(N):
            v = torch.bmm(M, v)
            v = v / torch.norm(v, 2, dim = 1)
            print(torch.norm(v - v_old))
            v_old = v.clone()

        print(v[0])
        return v    


    def biStochastic_forward(self, v, n, m, N = 1):
        """
        Arguments:
        ----------
            - v:     optimal assignment vector
            - n, m:  dimension of nodes of image 1 and image 2

        Returns:
        --------
            - S:    double stochastic confidence matrix S

        """

        # Reshape the assignment vector to matrix form
        S = v.view(n,m)

        # Perform N iterations: S_k+1 = ...., S_k+2 = ...
        for i in range(N):
            S = torch.mm(S, torch.mm(torch.ones(1,n),S).inverse())
            S = torch.mm(torch.mv(S, torch.ones(m,1)).inverse(), S)

        return S

    def voting_forward(self, S, P, alpha = 200., th = 10):
        """
        Arguments:
        ----------
            S - confidence map obtained form bi-stochastic layer
            P - Position matrix (m x 2)
            alpha - scaling factor
            th - number of pixels to be set as threshold beyond which confidence levels are set to zero.
        Returns:
        --------
            - d: displacement vector

        """
        S_ = alpha*S
        #TODO: Apply th
        P_ = torch.mm(F.softmax(S, dim = -1), P) 
        d = torch.zeros(P.shape)
        for i in range(P.shape[0]):
            d[:, i] = P_ -  P[:, i]

        return d