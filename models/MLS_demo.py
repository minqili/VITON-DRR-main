import numpy as np
from models.img_utils import mls_similarity_deformation
import os


path = os.getcwd()

def To_warp(image):

    data = get_point_data()
    p = np.hstack((data[:,1].reshape(-1,1),data[:,0].reshape(-1,1)))
    q = np.hstack((data[:,3].reshape(-1,1),data[:,2].reshape(-1,1)))
    
    height, width,_ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)
    
    affine = mls_similarity_deformation(vy, vx, p, q, alpha=2)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(affine)]
    return aug1


def To_warp_mask(image):

    data = get_point_data()
    p = np.hstack((data[:, 1].reshape(-1, 1), data[:, 0].reshape(-1, 1)))
    q = np.hstack((data[:, 3].reshape(-1, 1), data[:, 2].reshape(-1, 1)))


    height, width = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    affine = mls_similarity_deformation(vy, vx, p, q, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(affine)]
    return aug1
def get_point_data():
    point = []
    point_temp = []
    with open(path + "\\models\\txtdata\\data01.txt", 'r+', encoding='utf-8') as f:
        s = [i[:-1].split(',') for i in f.readlines()]
        for i in s:
            for j in i:
                if j =='':
                    break
                # point_temp.append(float(j))
                point_temp.append(int(round(float(j), 0)))
        point = np.vstack(point_temp)
    point = np.array(point).reshape(-1, 4)
    return point


