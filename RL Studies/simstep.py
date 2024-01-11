import torch



def simStep(self, W, A, B, C, D, h, learning_rates, mdW, iter):
        with torch.no_grad():
            for i in range(dim_recurrent):
                for j in range(dim_recurrent):
                    zeta = torch.sqrt(
                        torch.sum(torch.square(abs(W[j] * h - torch.dot(W[j], h) / dim_recurrent))) / dim_recurrent)
                    phi = mdW[i][j] / iter if iter != 0 else 0
                    dwij = learning_rates[i][j] * (A[i][j] * zeta + B[i][j] * phi + C[i][j] * h[i] * h[j] * + D[i][j])
                    W[i][j] = W[i][j] + dwij
                    mdW[i][j] = mdW[i][j] + dwij
            return W, mdW

def newSimStep(self, W, A, B, C, D, h, learning_rates, mdW, iter):
    with torch.no_grad():
        zeta = torch.std(W * h, dim=0)
        phi = mdW / iter if iter != 0 else 0
        dwij = learning_rates * (A * zeta + B * phi + C * h * h * + D)
        W = W + dwij
        mdW = mdW + dwij
        return W, mdW