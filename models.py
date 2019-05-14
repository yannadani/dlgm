import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        self.lam = nn.Parameter(torch.ones(20, 2, device = device))
    def forward(self, x_1, mask_1, x_2 = None, mask_2 = None):
        
        x_1 = self.features1(x_1)
        U1 = x_1[:,:, mask_1[0]]
        x_1 = self.features2(x_1)
        F1 =  x_1[:,:, mask_1[1]]
        
        if x_2 is None:
            return U1, F1
        
        else:
            x_2 = self.features1(x_2)
            U2 = x_2[:,:, mask_2[0]]
            x_2 = self.features2(x_2)
            F2 =  x_2[:,:, mask_2[1]]
            
            test = torch.from_numpy(np.asarray([[1,0,1],[1,1,1],[1,0,0]]))
            [G, H] = buildGraphStructure(test)
            
            M = self.affinityMatrix_forward(F1, F2, U1, U2, G, G, H, H) #TODO: Build appropriate graph structure before using this
            v = self.powerIteration_forward(M)
            S = self.biStochastic_forward(v, G.shape[0], G.shape[0])
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
        X = torch.cat((F1[np.array(idx1_start)], F1[np.array(idx1_end)]), 1)
        Y = torch.cat((F2[np.array(idx2_start)], F2[np.array(idx2_end)]), 1)

        # (b) Calculate M_e = X * \lambda * Y^T
        M_e = torch.mm(torch.mm(X, self.lam), Y)

        # (c) Calculate M_p = U1 * U2^T
        M_p = torch.mm(U1, U2.t())

        # Calculate M = [vec(M_p)] + (G_2 \kronecker G_1)[vec(M_e)](H_2 \kronecker H_1)^T
        diagM_p = torchl.diag(M_p.view(M_p.numel()))
        diagM_e = torchl.diag(M_e.view(M_e.numel()))
        M = diagM_p + torch.mm(torch.mm(kronecker(G2, G1), diagM_e), kronecker(H2, H1).t())

        return M

    def powerIteration_forward(self, M, N = 1):
        """
        Arguments:
        ----------
            - M: affinity matrix

        Returns:
        --------
            - v*: optimal assignment vector (computed using power iterations) 
        """

        # Init starting v
        v = torch.ones(M.shape[1],1)

        # Perform N iterations: v_k+1 = M*v_k / (||M*v_k||_2) 
        for i in range(N):
            v = torch.mv(M, v)
            v = v / torch.norm(v, 2)

        return tensor_v    


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