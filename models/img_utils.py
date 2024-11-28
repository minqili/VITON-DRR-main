
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def mls_affine_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """
    Affine deformation

    Parameters
    ----------
    vy, vx: ndarray
        coordinate grid, generated by np.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points, in (y, x) formats
    q: ndarray
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon
    
    Return
    ------
        A deformed image.
    """

    # Change (x, y) to (row, col)
    q = np.ascontiguousarray(q.astype(np.int16))
    p = np.ascontiguousarray(p.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    phat = phat.reshape(ctrls, 2, 1, grow, gcol)                                        # [ctrls, 2, 1, grow, gcol]
    phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                                       # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwp = np.zeros((2, 2, grow, gcol), np.float32)
    for i in range(ctrls):
        pTwp += phat[i] * reshaped_w[i] * phat1[i]
    del phat1

    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False                
    except np.linalg.linalg.LinAlgError:                
        flag = True             
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf                
        reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
    
    mul_left = reshaped_v - pstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = np.multiply(reshaped_w, phat, out=phat)                                 # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)                             # [ctrls, grow, gcol, 2, 1]
    out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]                            # [ctrls, grow, gcol, 1, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right, out=out_A)    # [ctrls, grow, gcol, 1, 1]
    A = A.reshape(ctrls, 1, grow, gcol)                                                 # [ctrls, 1, grow, gcol]
    del mul_right, reshaped_mul_right, phat

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
    del w, reshaped_w

    # Get final image transfomer -- 3-D array
    transformers = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        transformers += A[i] * (reshaped_q[i] - qstar)
    transformers += qstar
    del A

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers.astype(np.int16)


def mls_similarity_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """ Similarity deformation
    
    Parameters
    ----------
    vx, vy: ndarray
        coordinate grid, generated by np.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points, in (y, x) formats
    q: ndarray
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon
    
    Return
    ------
        A deformed image.
    """
    q = np.ascontiguousarray(q.astype(np.int16))
    p = np.ascontiguousarray(p.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    
    mu = np.zeros((grow, gcol), np.float32)
    for i in range(ctrls):
        mu += w[i] * (phat[i] ** 2).sum(0)
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]

    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
    
    # Get final image transfomer -- 3-D array
    temp = np.zeros((grow, gcol, 2), np.float32)
    for i in range(ctrls):
        neg_phat_verti = phat[i, [1, 0]]                                                # [2, grow, gcol]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)              # [1, 2, grow, gcol]
        mul_left = np.concatenate((reshaped_phat[i], reshaped_neg_phat_verti), axis=0)  # [2, 2, grow, gcol]
        
        A = np.matmul((reshaped_w[i] * mul_left).transpose(2, 3, 0, 1), 
                      mul_right.transpose(2, 3, 0, 1))                                  # [grow, gcol, 2, 2]
    
        qhat = reshaped_q[i] - qstar                                                    # [2, grow, gcol]
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)            # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        temp += np.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)                      # [grow, gcol, 2]

    transformers = temp.transpose(2, 0, 1) / reshaped_mu  + qstar                       # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0

    return transformers.astype(np.int16)


def mls_rigid_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
    """ Rigid deformation
    
    Parameters
    ----------
    vx, vy: ndarray
        coordinate grid, generated by np.meshgrid(gridX, gridY)
    p: ndarray
        an array with size [n, 2], original control points, in (y, x) formats
    q: ndarray
        an array with size [n, 2], final control points, in (y, x) formats
    alpha: float
        parameter used by weights
    eps: float
        epsilon
    
    Return
    ------
        A deformed image.
    """
    q = np.ascontiguousarray(q.astype(np.int16))
    p = np.ascontiguousarray(p.astype(np.int16))

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
    w /= np.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(2, 2, grow, gcol)                            # [2, 2, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
    
    temp = np.zeros((grow, gcol, 2), np.float32)
    for i in range(ctrls):
        phat = reshaped_p[i] - pstar                                                    # [2, grow, gcol]
        reshaped_phat = phat.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]
        reshaped_w = w[i].reshape(1, 1, grow, gcol)                                     # [1, 1, grow, gcol]
        neg_phat_verti = phat[[1, 0]]                                                   # [2, grow, gcol]
        neg_phat_verti[1] = -neg_phat_verti[1]
        reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)              # [1, 2, grow, gcol]
        mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=0)     # [2, 2, grow, gcol]
        
        A = np.matmul((reshaped_w * mul_left).transpose(2, 3, 0, 1), 
                        reshaped_mul_right.transpose(2, 3, 0, 1))                       # [grow, gcol, 2, 2]

        qhat = reshaped_q[i] - qstar                                                    # [2, grow, gcol]
        reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)            # [grow, gcol, 1, 2]

        # Get final image transfomer -- 3-D array
        temp += np.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)                      # [grow, gcol, 2]

    temp = temp.transpose(2, 0, 1)                                                      # [2, grow, gcol]
    normed_temp = np.linalg.norm(temp, axis=0, keepdims=True)                           # [1, grow, gcol]
    normed_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                       # [1, grow, gcol]
    transformers = temp / normed_temp * normed_vpstar  + qstar                          # [2, grow, gcol]
    nan_mask = normed_temp[0] == 0

    # Replace nan values by interpolated values
    nan_mask_flat = np.flatnonzero(nan_mask)
    nan_mask_anti_flat = np.flatnonzero(~nan_mask)
    transformers[0][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[0][~nan_mask])
    transformers[1][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[1][~nan_mask])

    # Remove the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > grow - 1] = 0
    transformers[1][transformers[1] > gcol - 1] = 0
    
    return transformers.astype(np.int16)
