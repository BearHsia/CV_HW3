import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    u = np.pad(u,((0,0),(0,1)),constant_values=1)
    v = np.pad(v,((0,0),(0,1)),constant_values=1)
    A = np.zeros((2*N,9))
    # TODO: 1.forming A
    for i in range(N):
        A[2*i,3:6] = -v[i,2]*u[i,:]
        A[2*i,6:9] = v[i,1]*u[i,:]
        A[1+2*i,0:3] = v[i,2]*u[i,:]
        A[1+2*i,6:9] = -v[i,0]*u[i,:]
    # TODO: 2.solve H with A
    _, _, vh = np.linalg.svd(A, full_matrices=True,compute_uv=True)
    #print(np.linalg.norm(vh[-1,:]))
    #print(np.linalg.norm(vh[-2,:]))
    #norm1 = vh[-1,:] / np.linalg.norm(vh[-1,:])
    H = np.reshape(vh[-1,:], (3,3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    nx = range(xmin,xmax)
    ny = range(ymin,ymax)
    
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xv, yv = np.meshgrid(nx, ny)
    zv = np.ones((ymax-ymin,xmax-xmin))
    c = np.stack((xv,yv,zv))
    dp = np.reshape(c,(3,(ymax-ymin)*(xmax-xmin)))
    #print(dp.shape)
    #print(dp)
    #print(H)
    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        Hinvdp = np.matmul(H_inv,dp)
        Hinvdp[0,:] = Hinvdp[0,:]/Hinvdp[2,:]
        Hinvdp[1,:] = Hinvdp[1,:]/Hinvdp[2,:]
        Hinvdp[2,:] = Hinvdp[2,:]/Hinvdp[2,:]
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        Hinvdp = Hinvdp.astype(np.int)
        dp = dp.astype(np.int)
        xmin_valid = Hinvdp[0,:]>=0
        xmax_valid = Hinvdp[0,:]<w_src
        ymin_valid = Hinvdp[1,:]>=0
        ymax_valid = Hinvdp[1,:]<h_src
        mask = xmin_valid & xmax_valid & ymin_valid & ymax_valid
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        Hinvdp_valid = Hinvdp[:,mask]
        dp_valid = dp[:,mask]
        # TODO: 6. assign to destination image with proper masking
        dst[dp_valid[1,:],dp_valid[0,:],:] = src[Hinvdp_valid[1,:],Hinvdp_valid[0,:],:]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        Hdp = np.matmul(H,dp)
        Hdp[0,:] = Hdp[0,:]/Hdp[2,:]
        Hdp[1,:] = Hdp[1,:]/Hdp[2,:]
        Hdp[2,:] = Hdp[2,:]/Hdp[2,:]
        #print(Hdp)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        Hdp = Hdp.astype(np.int)
        dp = dp.astype(np.int)
        xmin_valid = Hdp[0,:]>=0
        xmax_valid = Hdp[0,:]<w_dst
        ymin_valid = Hdp[1,:]>=0
        ymax_valid = Hdp[1,:]<h_dst
        mask = xmin_valid & xmax_valid & ymin_valid & ymax_valid
        # TODO: 5.filter the valid coordinates using previous obtained mask
        Hdp_valid = Hdp[:,mask]
        dp_valid = dp[:,mask]
        # TODO: 6. assign to destination image using advanced array indicing
        dst[Hdp_valid[1,:],Hdp_valid[0,:],:] = src[dp_valid[1,:],dp_valid[0,:],:]

    return dst
