import numpy as np
from numba import njit

    
@njit
def hebbian_update_A(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx] * o0[i] * o1[j]  

        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx] * o1[i] * o2[j] 
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx] * o2[i] * o3[j] 

        return weights1_2, weights2_3, weights3_4
    


@njit
def hebbian_update_AD(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][0] * o0[i] * o1[j] + heb_coeffs[idx][1] 

        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][0] * o1[i] * o2[j] + heb_coeffs[idx][1]  
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][0] * o2[i] * o3[j] + heb_coeffs[idx][1] 


        return weights1_2, weights2_3, weights3_4
    
@njit
def hebbian_update_AD_lr(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += (heb_coeffs[idx][0] * o0[i] * o1[j] + heb_coeffs[idx][1]) *  heb_coeffs[idx][2] 

        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += (heb_coeffs[idx][0] * o1[i] * o2[j] + heb_coeffs[idx][1]) *  heb_coeffs[idx][2]   
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += (heb_coeffs[idx][0] * o2[i] * o3[j] + heb_coeffs[idx][1]) *  heb_coeffs[idx][2] 


        return weights1_2, weights2_3, weights3_4



@njit
def hebbian_update_ABC(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                      + heb_coeffs[idx][1] * o0[i] 
                                      + heb_coeffs[idx][2]         * o1[j])  

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                      + heb_coeffs[idx][1] * o1[i] 
                                      + heb_coeffs[idx][2]         * o2[j])  
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                      + heb_coeffs[idx][1] * o2[i] 
                                      + heb_coeffs[idx][2]         * o3[j])  

        return weights1_2, weights2_3, weights3_4


@njit
def hebbian_update_ABC_lr(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1        
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j])

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j])  
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j])  

        return weights1_2, weights2_3, weights3_4

@njit
def hebbian_update_ABCD(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1        
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j])

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j])  
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j])  
                
        return weights1_2, weights2_3, weights3_4
    
    
@njit
def hebbian_update_ABCD_lr_D_in(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
    heb_offset = 0
    ## Layer 1
    for i in range(weights1_2.shape[1]):
        for j in range(weights1_2.shape[0]):
            idx = (weights1_2.shape[0] - 1) * i + i + j
            weights1_2[:, i][j] += heb_coeffs[idx][3] * (heb_coeffs[idx][0] * o0[i] * o1[j]
                                                         + heb_coeffs[idx][1] * o0[i]
                                                         + heb_coeffs[idx][2] * o1[j] + heb_coeffs[idx][4])

    heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
    # Layer 2
    for i in range(weights2_3.shape[1]):
        for j in range(weights2_3.shape[0]):
            idx = heb_offset + (weights2_3.shape[0] - 1) * i + i + j
            weights2_3[:, i][j] += heb_coeffs[idx][3] * (heb_coeffs[idx][0] * o1[i] * o2[j]
                                                         + heb_coeffs[idx][1] * o1[i]
                                                         + heb_coeffs[idx][2] * o2[j] + heb_coeffs[idx][4])

    heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
    # Layer 3
    for i in range(weights3_4.shape[1]):
        for j in range(weights3_4.shape[0]):
            idx = heb_offset + (weights3_4.shape[0] - 1) * i + i + j
            weights3_4[:, i][j] += heb_coeffs[idx][3] * (heb_coeffs[idx][0] * o2[i] * o3[j]
                                                         + heb_coeffs[idx][1] * o2[i]
                                                         + heb_coeffs[idx][2] * o3[j] + heb_coeffs[idx][4])

    return weights1_2, weights2_3, weights3_4
    
@njit
def hebbian_update_ABCD_lr_D_out(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
       
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j])  + heb_coeffs[idx][4]

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j])  + heb_coeffs[idx][4]
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j])  + heb_coeffs[idx][4]

        return weights1_2, weights2_3, weights3_4

@njit
def hebbian_update_ABCD_lr_D_in_and_out(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
       
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j]  + heb_coeffs[idx][4]) + heb_coeffs[idx][5]

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j]  + heb_coeffs[idx][4]) + heb_coeffs[idx][5]
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j]  + heb_coeffs[idx][4]) + heb_coeffs[idx][5]

        return weights1_2, weights2_3, weights3_4

@njit
def myclip(val, min, max):
    if val < min: return min
    if val > max: return max
    return val

@njit
def hebbian_update_new(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3, prev_dw1_2, prev_dw2_3, prev_dw3_4, iters):
       
        heb_offset = 0
        # Layer 1
        cur_dw1_2 = np.zeros(weights1_2.shape)   
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                # dw = heb_coeffs[idx][3] * (heb_coeffs[idx][0] * np.std(weights1_2[j]*o0)/100
                #                                            + heb_coeffs[idx][1] * (prev_dw1_2[:,i][j] / max(iters,1))
                #                                            + heb_coeffs[idx][2] * o0[i] * o1[j] + heb_coeffs[idx][4] )

                dw = heb_coeffs[idx][3] * (heb_coeffs[idx][1] * (prev_dw1_2[:,i][j] / max(iters,1)) + heb_coeffs[idx][2] * o0[i] * o1[j] + heb_coeffs[idx][4])
                # print(heb_coeffs[idx][0] * np.std(weights1_2[j]*o0))
                # print(weights1_2[j] * o0)
                # print(dw)

                cur_dw1_2[:,i][j] = dw
                weights1_2[:,i][j] += dw
        prev_dw1_2 += cur_dw1_2

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        cur_dw2_3 = np.zeros(weights2_3.shape) 
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                # dw = heb_coeffs[idx][3] * (heb_coeffs[idx][0] * np.std(weights2_3[j]*o1)/100
                #                                            + heb_coeffs[idx][1] * (prev_dw2_3[:,i][j] / max(iters,1))
                #                                            + heb_coeffs[idx][2] * o1[i] * o2[j] + heb_coeffs[idx][4] )
                dw = heb_coeffs[idx][3] * (heb_coeffs[idx][1] * (prev_dw2_3[:,i][j] / max(iters,1))
                                                           + heb_coeffs[idx][2] * o1[i] * o2[j] + heb_coeffs[idx][4])

                cur_dw2_3[:,i][j] = dw
                weights2_3[:,i][j] += dw
        prev_dw2_3 += cur_dw2_3
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        cur_dw3_4 = np.zeros(weights3_4.shape)
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                # dw = heb_coeffs[idx][3] * (heb_coeffs[idx][0] * np.std(weights3_4[j]*o2)/10
                #                                            + heb_coeffs[idx][1] * (prev_dw3_4[:,i][j] / max(iters,1))
                #                                            + heb_coeffs[idx][2] * o2[i] * o3[j] + heb_coeffs[idx][4] )
                dw = heb_coeffs[idx][3] * (heb_coeffs[idx][1] * (prev_dw3_4[:,i][j] / max(iters,1)) + heb_coeffs[idx][2] * o2[i] * o3[j] + heb_coeffs[idx][4])

                cur_dw3_4[:,i][j] = dw
                weights3_4[:,i][j] += dw
        prev_dw3_4 += cur_dw3_4

        return weights1_2, weights2_3, weights3_4, prev_dw1_2, prev_dw2_3, prev_dw3_4