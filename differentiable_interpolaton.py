import torch
import cv2
import numpy as np


def compute_density(x, density_grid, input_img_shape, activation):

    h, w = list(input_img_shape.shape)
    density_h, density_w = list(density_grid.shape)
    # First compute where the X would be belong in the downscaled image

    prev_y, prev_x = ((x[:, 0] + 1) / h) * \
        density_h, ((x[:, 1] + 1) / w) * density_w

    closest_x, farthest_x = torch.floor(prev_x).to(
        torch.int64), torch.ceil(prev_x).to(torch.int64)
    closest_y, farthest_y = torch.floor(prev_y).to(
        torch.int64), torch.ceil(prev_y).to(torch.int64)

    closest_x = torch.where(closest_x == farthest_x, closest_x - 1, closest_x)
    closest_y = torch.where(closest_y == farthest_y, closest_y - 1, closest_y)

    interpolate_x_top = (farthest_x - prev_x) / (farthest_x - closest_x) * density_grid[(closest_y - 1), (closest_x - 1)] + \
        (prev_x - closest_x) / (farthest_x - closest_x) * \
        density_grid[(closest_y - 1), (farthest_x - 1)]

    interpolate_x_bot = (farthest_x - prev_x) / (farthest_x - closest_x) * density_grid[farthest_y - 1, closest_x - 1] + \
        (prev_x - closest_x) / (farthest_x - closest_x) * \
        density_grid[farthest_y - 1, farthest_x - 1]

    interpolate_y = (farthest_y - prev_y) / (farthest_y - closest_y) * interpolate_x_top + \
                    (prev_y - closest_y) / \
        (farthest_y - closest_y) * interpolate_x_bot

    if activation:
        interpolate_y = torch.nn.Softplus()(interpolate_y)  # softplus like mip-nerf

    return interpolate_y


def run_optimization(img, iters, activation):
    density_grid = torch.randn(
        (img.shape[0]//STRIDE, img.shape[1]//STRIDE), requires_grad=True)

    xs = torch.arange(img.shape[0])
    ys = torch.arange(img.shape[1])

    h_mesh, w_mesh = torch.meshgrid(xs, ys, indexing='ij')

    points = torch.stack([h_mesh, w_mesh], dim=-1).reshape(-1, 2)

    optimizer = torch.optim.Adam(params=[density_grid], lr=1)
    for i in range(iters):
        result = compute_density(points, density_grid, img, activation=activation).reshape(
            img.shape[0], img.shape[1])
        loss = torch.nn.MSELoss()(img, result)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return result, density_grid


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


STRIDE = 5
img = cv2.imread("square.png", 0)
img = (img > 100).astype(np.uint8) * 255

assert (img.shape[0] == img.shape[1])
# assert ((img.shape[0] % STRIDE) == 0 and (img.shape[1] % STRIDE) == 0)

img = torch.FloatTensor(img)

output_linear, grid_linear = run_optimization(
    img, iters=2000, activation=False)

output_nonlinear, grid_nonlinear = run_optimization(
    img, iters=2000, activation=True)

cv2.imwrite("output_linear.png", (rescale(
    output_linear.cpu().detach().numpy())*255).astype(np.uint8))

cv2.imwrite("output_nonlinear.png", (rescale(
    output_nonlinear.cpu().detach().numpy())*255).astype(np.uint8))

cv2.imwrite("density_grid_linear.png", (rescale(
    grid_linear.cpu().detach().numpy())*255).astype(np.uint8))

cv2.imwrite("density_grid_nonlinear.png", (rescale(
    grid_nonlinear.cpu().detach().numpy())*255).astype(np.uint8))
