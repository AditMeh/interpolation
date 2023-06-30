import numpy as np
import cv2

def interpolate(img, new_height, new_width):
    old_height, old_width = img.shape


    new_img = np.zeros(shape = (new_height, new_width))


    for x in range(new_height):
        for y in range(new_width):
            
            prev_y, prev_x = ((y + 1) / new_height) * old_height, ((x + 1) / new_width) * old_width 
            
            print(prev_x, prev_y)
            # Get box coords 
            closest_x, farthest_x = int(np.floor(prev_x)), int(np.ceil(prev_x))
            closest_y, farthest_y = int(np.floor(prev_y)), int(np.ceil(prev_y))

            if closest_x == farthest_x:
                closest_x -= 1
            if closest_y == farthest_y:
                closest_y -=1


            interpolate_x_top = (farthest_x - prev_x)/ (farthest_x - closest_x) * img[closest_y - 1][closest_x - 1] + \
                            (prev_x - closest_x) / (farthest_x - closest_x) * img[closest_y - 1][farthest_x - 1]



            interpolate_x_bot = (farthest_x - prev_x)/ (farthest_x - closest_x) * img[farthest_y - 1][closest_x - 1] + \
                            (prev_x - closest_x)/ (farthest_x - closest_x) * img[farthest_y - 1][farthest_x - 1]


            interpolate_y = (farthest_y - prev_y)/ (farthest_y - closest_y) * interpolate_x_top + \
                            (prev_y - closest_y)/ (farthest_y - closest_y) * interpolate_x_bot  

            new_img[y][x] = interpolate_y

    return new_img
if __name__ == "__main__":
    img = cv2.imread("images/dog.png", 0)
    print(img.shape)
    new_img = interpolate(img, 1300, 1300)
    print(new_img.shape)
    cv2.imwrite("images/resized.png", new_img)