import numpy as np
import torch
def reshape_output(output, anchor_number):
    new_output = output.view(anchor_number, output.size(1)//anchor_number, output.size(2), output.size(3))  # reshape
    #new_output = new_output.permute(2, 3, 0, 1)
    return new_output


def separate_output(output, class_number):
    offset_x = output[:,0]
    offset_y = output[:,1]
    relative_width = output[:,2]
    relative_height = output[:,3]
    objectness = output[:, 4]
    classes = output[:,5:(5+class_number)]
    return [offset_x, offset_y, relative_width, relative_height, objectness, classes ]

def get_x_grid(grid_size, img_size):
    grid = np.zeros((grid_size,grid_size), dtype="float32")
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            grid[i, j] = j/grid_size
    return grid * img_size

def get_y_grid(grid_size, img_size):
    grid = np.zeros((grid_size,grid_size), dtype="float32")
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            grid[i, j] = i/grid_size
    return grid * img_size
