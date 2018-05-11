import torch


def get_line_coeff(pt_1, pt_2):
    if pt_1[0] == pt_2[0]:
        a = 1
        b = 0
        c = -pt_1[0].item()
        return a, b, c
    if pt_1[1] == pt_2[1]:
        a = 0
        b = 1
        c = -pt_1[1].item()
        return a, b, c
    a = pt_1[1] - pt_2[1]
    b = pt_2[0] - pt_1[0]
    c = pt_2[1] * pt_1[0] - pt_1[1] * pt_2[0]
    return a.item(), b.item(), c.item()


def get_line_coeffs(face):
    a1, b1, c1 = get_line_coeff(face[1], face[0])
    a2, b2, c2 = get_line_coeff(face[2], face[1])
    a3, b3, c3 = get_line_coeff(face[0], face[2])
    return a1, b1, c1, a2, b2, c2, a3, b3, c3


def rasterize_face_forward(face,
                           image_height=10,
                           image_width=10,
                           background=0,
                           face_color=1):
    img = torch.zeros(image_height, image_width)
    min_y, min_x = face.min(0)[0]
    max_y, max_x = face.max(0)[0]
    a1, b1, c1, a2, b2, c2, a3, b3, c3 = get_line_coeffs(face)
    for y in range(int(min_y.item()), int(max_y.item())):
        for x in range(int(min_x.item()), int(max_x.item())):
            if a1 * y + b1 * x + c1 >= 0:
                if a2 * y + b2 * x + c2 >= 0:
                    if a3 * y + b3 * x + c3 >= 0:
                        img[y, x] = 1
    return img


def order_points(pts):
    """
    Return in order idxs of leftmost middle and rightmos points
    """
    assert pts.shape[0] == 3
