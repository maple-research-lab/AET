import numpy
import sys
from PIL import Image
import random

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

all_coeffs = []
for kk in range(500000):
    width, height = (32, 32)
    center = (32 * 0.5 + 0.5, 32 * 0.5 + 0.5)
    
    shift = [float(random.randint(-4, 4)) for ii in range(8)]
    scale = random.uniform(0.8,1.2)
    
    pts = [((0-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
                ((width-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
                ((width-center[0])*scale+center[0], (height-center[1])*scale+center[1]),
                ((0-center[0])*scale+center[0], (height-center[1])*scale+center[1])]
    
    rotation = random.randint(0,3)
    pts = [pts[(ii+rotation)%4] for ii in range(4)]
    pts = [(pts[ii][0]+shift[2*ii], pts[ii][1]+shift[2*ii+1]) for ii in range(4)]
    
    coeffs = find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
            )
    all_coeffs += [coeffs]

all_coeffs = numpy.asarray(all_coeffs, dtype='float32')
mean = numpy.mean(all_coeffs, axis=0)
std = numpy.std(all_coeffs, axis=0)
print(mean)
print(std)
        