import torch
import numpy as np

from neurender.rasterize import get_line_coeffs


def rasterize_face_backward(img, target_img, face):
    a1, b1, c1, a2, b2, c2, a3, b3, c3 = get_line_coeffs(face)
    coeffs = face.new([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    img_diff = img - target_img
    height, width = img_diff.shape
    total_gradients = face.new(3, 2).fill_(0)
    for y in range(height):
        for x in range(width):
            for point in face:
                if img_diff[y, x] == 0:
                    continue
                else:
                    # Get two closest points on face
                    # distance_line_1 = (a1 * x + b1 * y + c1
                    #                    ).abs() / np.sqrt(a1**2 + b1**2)
                    # distance_line_2 = (a2 * x + b2 * y + c2
                    #                    ).abs() / np.sqrt(a2**2 + b2**2)
                    # distance_line_3 = (a3 * x + b3 * y + c3
                    #                    ).abs() / np.sqrt(a3**2 + b3**2)
                    line_dist_num = torch.mm(
                        coeffs, torch.Tensor([[x, y, 1]]).transpose(
                            1, 0)).abs().transpose(1, 0)
                    line_dist_denum = torch.sqrt(coeffs[:, 0]**2 + coeffs[:, 1]
                                                 **2)
                    line_dists = line_dist_num / line_dist_denum
                    dist_val, dist_idx = line_dists.min(1)
                    if dist_idx.item() == 0:
                        direction = face.new([-b1, a1])
                    elif dist_idx.item() == 1:
                        direction = face.new([-b2, a2])
                    elif dist_idx.item() == 2:
                        direction = face.new([-b3, a3])
                    contribution = (
                        1 / dist_val) * direction / direction.norm()
                    # pixel in face
                    if img[y, x] == 1:
                        contribution = -contribution
                    import pdb
                    pdb.set_trace()

                # pixel outside of face
            else:
                pass


if __name__ == "__main__":
    face = torch.Tensor([[1, 2], [1, 7], [5, 2]])
    image_height = 12
    image_width = 10
    target_img = torch.zeros(image_height, image_width)
    target_img[0, :3] = 1
    target_img[1, :2] = 1
    target_img[2, :1] = 1
    print(target_img)
    img = rasterize_face_forward(
        face, image_height=image_height, image_width=image_width)
    rasterize_face_backward(img, target_img, face)
    print(img)
