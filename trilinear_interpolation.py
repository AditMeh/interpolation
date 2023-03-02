import matplotlib.pyplot as plt
import numpy as np


def mesh_to_voxel(inp_stream, pts):
    vol = np.zeros((32, 32, 32))

    pts = []
    for _ in range(pts):
        point = [float(p) for p in f.readline().strip("\n").split(" ")]
        pts.append(point)

    min = 
        

if __name__ == "__main__":
    f = open("chair_0635.off", "r")
    f.readline()
    pts = int(f.readline().strip("\n").split(" ")[0])
    vox = mesh_to_voxel(f, pts)

# # prepare some coordinates
# x, y, z = np.indices((32, 32, 32))

# # draw cuboids in the top left and bottom right corners, and a link between them

# cube1 = (x < 3) & (y < 3) & (z < 3)
# cube2 = (x >= 5) & (y >= 5) & (z >= 5)
# link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# # combine the objects into a single boolean array
# voxelarray = cube1 | cube2 | link

# # set the colors of each object
# colors = np.empty(voxelarray.shape, dtype=object)
# colors[link] = 'red'
# colors[cube1] = 'blue'
# colors[cube2] = 'green'

# # and plot everything
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

# plt.show()
