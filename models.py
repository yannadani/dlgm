import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class VGG_graph_matching(nn.Module):
    def __init__(self):
        super(VGG_graph_matching, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)   

        self.score = nn.Conv2d(512, 64, 1)  

        self.upscore32 = nn.ConvTranspose2d(64, 64, 64, stride=32,
                                          bias=False)  
        self.upscore16 = nn.ConvTranspose2d(64, 64, 32, stride=16,
                                          bias=False)  

        self.lam1 = nn.Parameter(torch.ones(64, 64))
        self.lam2 = nn.Parameter(torch.ones(64, 64))


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            #if isinstance(m, nn.Conv2d):
            #    m.weight.data.zero_()
            #    if m.bias is not None:
            #        m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self):
        vgg16 = models.vgg16(pretrained = True)   # Enabling/Disabling Pretrained-Version of VGG
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1
                ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)

    def apply_forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        feat1 = self.relu4_2(self.conv4_2(h))

        h = self.relu4_3(self.conv4_3(feat1))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))

        x_1 = self.upscore16(self.score(feat1))
        x_1 = x_1[:, :, 9:9 + x.size()[2], 9:9 + x.size()[3]].contiguous() 


        x_2 = self.upscore32(self.score(h))
        x_2 = x_2[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return x_1, x_2



    def forward(self, im_1, mask_1=None, im_2 = None, mask_2 = None):
        """
        FORWARD PASS - IMPLEMENTATION
        """
        
        x_1, x_2 = self.apply_forward(im_1)

        if mask_1 is None:
            F1 = x_1
            U1 = x_2
        else:
            F1 = x_1[:,:, mask_1[0]]
            U1 =  x_2[:,:, mask_1[1]]

        if im_2 is None:
            return U1, F1
        
        else:
            x_21, x_22 = self.apply_forward(im_2)
            if mask_2 is None:
               F2 = x_21
               U2 = x_22
            else:
               F2 =  x_21[:,:, mask_2[0]]
               U2 =  x_22[:,:, mask_2[1]]
            F1, U1, F2, U2 = F.normalize(F1), F.normalize(U1), F.normalize(F2), F.normalize(U2)
            
            # Load affinity matrix from CSV
            # - eye --> eye-matrix (WORKING)
            # - 9pt-stencil --> complete 9pt stencil (NOT WORKING => MEMORY ISSUES)
            # - 9pt-stencil-upper --> upper diag + diag of 9pt stencil (WORKING)
            graphStructure = "eye"
            A = torch.from_numpy(pd.read_csv(graphStructure + '.csv', header=None).values) 

            # Build Graph Structure based on given affinity matrix
            [G, H] = self.buildGraphStructure(A)

            # Compute Forward pass using building blocks from paper
            M = self.affinityMatrix_forward(F1, F2, U1, U2, G, G, H, H) 
            v = self.powerIteration_forward(M)
            #S = self.biStochastic_forward(v, G.shape[0], G.shape[0])  # Disable Bi-Stochastic Layer -> not necessary for optical flow
            d = self.voting_flow_forward(v)

            return d


    def kronecker(self, matrix1, matrix2):
        """
        Arguments:
        ----------
            - matrix1: batch-wise stacked matrices1
            - matrix2: batch-wise stacked matrices2

        Returns:
        --------
            - Batchwise Kronecker product between matrix1 and matrix2

        """
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
        nr_edges = torch.sum(A).to(torch.int32).item()

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
        #output = torch.zeros(dims, device=self.device)
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
        #     - Extract edge features of start and end nodes
        idx1_start = (G1 != 0).nonzero()[:,0]
        idx2_start = (G2 != 0).nonzero()[:,0]
        idx1_end = (H1 != 0).nonzero()[:,0]
        idx2_end = (H2 != 0).nonzero()[:,0]


        # (b) Concat feature vectors (as described in paper)
        X = torch.cat((F1.view(F1.shape[0],F1.shape[1],-1)[:,:,idx1_start], F1.view(F1.shape[0],F1.shape[1],-1)[:,:,idx1_end]), 1).permute(0,2,1)
        Y = torch.cat((F2.view(F2.shape[0],F2.shape[1],-1)[:,:,idx2_start], F2.view(F2.shape[0],F2.shape[1],-1)[:,:,idx2_end]), 1).permute(0,2,1)

        # (c) Calculate M_e = X * \lambda * Y^T
        lam = F.relu(torch.cat((torch.cat((self.lam1, self.lam2), dim = 1), torch.cat((self.lam2, self.lam1), dim = 1))))
        M_e = torch.bmm(torch.bmm(X, lam.expand(X.shape[0],-1,-1)), Y.permute(0,2,1))

        # (d) Calculate M_p = U1 * U2^T
        M_p = torch.bmm(U1.view(U1.shape[0], U1.shape[1], -1).permute(0,2,1), U2.view(U2.shape[0], U2.shape[1], -1))

        # (e) Calculate M = [vec(M_p)] + (G_2 \kronecker G_1)[vec(M_e)](H_2 \kronecker H_1)^T
        diagM_p = self.batch_diagonal(M_p.view(M_p.shape[0],-1))    
        diagM_e = self.batch_diagonal(M_e.view(M_e.shape[0],-1))  
        M = diagM_p + torch.bmm(torch.bmm(self.kronecker(G2, G1).expand(M_p.shape[0],-1,-1), diagM_e), self.kronecker(H2, H1).expand(M_e.shape[0],-1,-1).permute(0, 2,1))
        return M

    def powerIteration_forward(self, M, N = 10):
        """
        Arguments:
        ----------
            - M: affinity matrix

        Returns:
        --------
            - v*: optimal assignment vector (computed using power iterations) 
        """


        # Init starting v
        v = torch.ones(M.shape[0], M.shape[2], 1)

        # Perform N iterations: v_k+1 = M*v_k / (||M*v_k||_2) 
        for i in range(N):
            v = F.normalize(torch.bmm(M, v), dim=1)

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


    def voting_flow_forward(self, v, alpha=1., th = 10):
        """
        Arguments:
        ----------
            v: optimal assignment vector (output from power iteration)
            alpha: scale value in softmax
            th: threshold value

        Returns:
        --------
            d: displacement vector

        """

        n = int(np.sqrt(v.shape[1]))
        n_ = int(np.sqrt(n))

        # Calculate coordinate arrays
        i_coords, j_coords = np.meshgrid(range(n_), range(n_), indexing='ij')
        [P_y, P_x] = torch.from_numpy(np.array([i_coords, j_coords]))
        P_x = P_x.view(1, n, -1).expand(v.shape[0],-1, -1).to(torch.float32)
        P_y = P_y.view(1, n, -1).expand(v.shape[0],-1, -1).to(torch.float32)

        # Perform displacement calculation
        S = alpha * v.view(v.shape[0], n, -1)
        S_ = F.softmax(S, dim = -1)
        P_x_ = torch.bmm(S_, P_x)
        P_y_ = torch.bmm(S_, P_y) 
        d_x = P_x_ - P_x
        d_y = P_y_ - P_y
        d = torch.cat((d_x, d_y), dim=2)
        
        return d


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
        P_ = torch.bmm(F.softmax(S, dim = -1), P.expand(S_.shape[0], -1 , -1)) 
        d = torch.zeros(P.shape)
        for i in range(P.shape[0]):
            d[:, i] = P_ -  P[:, i]

        return d
