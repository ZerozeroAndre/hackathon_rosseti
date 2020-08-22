import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

show_imgs = True
img_width = 800
img_height = 600

color2range = {
    'yellow':{
        "l_h": 21,
        "l_s": 67,
        "l_v": 59,
        "u_h": 29,
        "u_s": 255,
        "u_v": 255
    },
    'green':{
        "l_h": 73,
        "l_s": 42,
        "l_v": 0,
        "u_h": 100,
        "u_s": 255,
        "u_v": 112
    },
    'red':{
        "l_h": 0,
        "l_s": 40,
        "l_v": 0,
        "u_h": 9,
        "u_s": 255,
        "u_v": 255
    },
}

color2code = (
    ['red', (0, 0, 255)],
    ['green', (0, 255, 0)],
    ['yellow', (0, 255, 255)]
)

color2code = (
    ['red', (0, 0, 255)],
    ['green', (0, 255, 0)],
    ['yellow', (0, 255, 255)]
)


def load_images_rgb(path_to_dir):
    images = []

    for filename in reversed(sorted(os.listdir(path_to_dir))):
        image_bgr = cv2.imread(os.path.join(path_to_dir, filename), )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_res = cv2.resize(image_rgb, (img_width, img_height))
        # image_hsv = cv2.cvtColor(image_rgb_res, cv2.COLOR_RGB2HSV)
        images.append(image_rgb_res)

    return images


def get_mask_bitwise(img_hsv, color_range):
    lower_color = np.array([color_range['l_h'], color_range['l_s'], color_range['l_v']])
    upper_color = np.array([color_range['u_h'], color_range['u_s'], color_range['u_v']])

    mask = cv2.inRange(img_hsv, lower_color, upper_color)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return result

class CenterPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class BounedRect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x1 + x2
        self.y2 = y1 + y2


class Сap:
    def __init__(self, center, bound_rect):
        self.center_point = CenterPoint(*center)
        self.bounded_rect = BounedRect(*bound_rect)

    def __repr__(self):
        return "center ({0}, {1})\nbounded box (({2}, {3}), ({4}, {5}))".format(
            self.center_point.x,
            self.center_point.y,
            self.bounded_rect.x1,
            self.bounded_rect.y1,
            self.bounded_rect.x2,
            self.bounded_rect.y2
        )


def get_caps(contours):
    caps = []
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        centers, radius = cv2.minEnclosingCircle(contours_poly)
        caps.append(Сap(centers, boundRect))
    return caps


def get_pair_cap(caps):
    caps = sorted(caps, key=lambda cap: cap.center_point.y)
    tempt_caps = []
    found_point_indices = []

    for i, point1 in enumerate(caps):
        # print('i', i)
        # print('point1.center_point.x: ', point1.center_point.x)

        if i in found_point_indices:
            continue

        for j, point2 in enumerate(caps):
            if i == j:
                continue

            if j in found_point_indices:
                continue

            # print('j', j)
            # print('point2.bounded_rect.x1 ', point2.bounded_rect.x1)
            # print('point2.bounded_rect.x2 ', point2.bounded_rect.x2)
            if ((point2.bounded_rect.x1 < point1.bounded_rect.x1) and \
                (point1.bounded_rect.x1 < point2.bounded_rect.x2)) or \
                    ((point2.bounded_rect.x1 < point1.bounded_rect.x2) and \
                     (point1.bounded_rect.x2 < point2.bounded_rect.x2)):
                tempt_caps.append((point1, point2))
                found_point_indices.append(i)
                found_point_indices.append(j)
                # print('find pairs: ', i, j)
                break

            # print()

    return tempt_caps


def get_lower_cap(pairs_caps):
    lower_cap = pairs_caps[0][0]
    for pair_cap in pairs_caps:
        for p in pair_cap:
            if p.center_point.y > lower_cap.center_point.y:
                lower_cap = p
    return lower_cap


def remove_caps_not_belong_fork(pairs_caps):
    if len(pairs_caps) == 2:
        return pairs_caps
    lower_cap = get_lower_cap(pairs_caps)
    caps_belonging_fork = []
    for i, pair_cap in enumerate(pairs_caps):
        if pair_cap[0] != lower_cap and pair_cap[1] != lower_cap:
            caps_belonging_fork.append(pair_cap)
    return caps_belonging_fork


def fun_left_sorted_pairs_caps(caps_belonging_fork):
    sorted_pairs_points = []

    for i, pairs in enumerate(caps_belonging_fork):
        if pairs[0].center_point.x > pairs[1].center_point.x:
            sorted_pairs_points.append((pairs[1], pairs[0]))
        else:
            sorted_pairs_points.append((pairs[0], pairs[1]))
    return sorted_pairs_points


def fun_up_sorted_pairs_caps(caps_belonging_fork):
    sorted_pairs_points = []

    for i, pairs in enumerate(caps_belonging_fork):
        if pairs[0].center_point.y > pairs[1].center_point.y:
            sorted_pairs_points.append((pairs[1], pairs[0]))
        else:
            sorted_pairs_points.append((pairs[0], pairs[1]))
    return sorted_pairs_points


#def draw_pairs(img_show, sorted_pairs_caps, color=(255,0,0)):
#    img_show = img_rgb.copy()
 #   for pairs in sorted_pairs_caps:
 #       # print(pairs)
 #       colorr = color2code[color]
 #       cv2.line(img_show, (int(pairs[0].center_point.x), int(pairs[0].center_point.y)), (int(pairs[1].center_point.x), int(pairs[1].center_point.y)), colorr, 3)
 #   return img_show


def bounded_points(upper_caps):
    left_point = int(upper_caps[0].center_point.x)
    right_point = int(upper_caps[0].center_point.x)
    upper_point = int(upper_caps[0].center_point.y)
    fotter_point = int(upper_caps[0].center_point.y)

    for p in upper_caps:
        if p.center_point.y > fotter_point:
            fotter_point = int(p.center_point.y)
        if p.center_point.y < upper_point:
            upper_point = int(p.center_point.y)

        if p.center_point.x > right_point:
            right_point = int(p.center_point.x)
        if p.center_point.x < left_point:
            left_point = int(p.center_point.x)

    return upper_point, fotter_point, left_point, right_point


def vertical_cut(img):
    y_virtical_cut = int(img.shape[0] / 2)
    img_upper = img[0:y_virtical_cut,:,:]
    img_lower = img[y_virtical_cut:img.shape[0], :,:]
    return img_upper, img_lower


def horizont_cut(img):
    horizontal_parts = []
    horizontal_slide = int(img.shape[1]/3)

    for i in range(3):
        cut_img=img[:,i*horizontal_slide:(i+1)*horizontal_slide,:]
        horizontal_parts.append(cut_img)

    return horizontal_parts



# forks
def get_fork_status_from_image(image):
    fork_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    result_connected_status = {"fork_0": True, "fork_1": True, "fork_2": True}
    # взяли верхнюю часть
    upper_part, lower_cut = vertical_cut(fork_image)

    # cv2_imshow(upper_part)
    # поделили изображение на 3 части, в каждой части изображена вилка
    horizontal_parts = horizont_cut(upper_part)

    for i, part_bgr in enumerate(horizontal_parts):

        color_name = color2code[i][0]
        color_code = color2code[i][1]

        part_hsv = cv2.cvtColor(part_bgr.copy(), cv2.COLOR_BGR2HSV)
        mask_bitwise = get_mask_bitwise(part_hsv.copy(), color2range[color_name])

        # blur = cv2.medianBlur(mask_bitwise, 5)

        blur = cv2.GaussianBlur(mask_bitwise, (17, 17), 0)
        kernel = np.ones((5, 5), np.uint8)

        dilation = cv2.dilate(blur, kernel, iterations=2)

        gray = cv2.cvtColor(dilation, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        # cv2_imshow(binary)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        caps = get_caps(contours)
        pairs_cap = get_pair_cap(caps)
        caps_belonging_fork = remove_caps_not_belong_fork(pairs_cap)
        sorted_pairs_caps = fun_up_sorted_pairs_caps(caps_belonging_fork)

        # print(sorted_pairs_caps)
        upper_caps = [sorted_pairs_caps[0][0], sorted_pairs_caps[1][0]]
        upper_cap = upper_caps[0]

        upper_point, fotter_point, left_point, right_point = bounded_points(upper_caps)
        crop_coonection_caps = part_bgr[upper_point:fotter_point, left_point:right_point, :]

        left_sorted_pairs_caps = fun_left_sorted_pairs_caps(caps_belonging_fork)
        lefter_cap = left_sorted_pairs_caps[0][0]

        template_connector = np.zeros(crop_coonection_caps.shape[0:2])
        main_diagonal = True
        if upper_cap.center_point != lefter_cap.center_point:
            # print("left_top2right_futter_diag")
            template_connector = cv2.line(template_connector,
                                          (0, 0),
                                          (template_connector.shape[1], template_connector.shape[0]),
                                          (255, 255, 255),
                                          4)
        else:
            # print("left_futter2right_upper_diag")
            main_diagonal = False
            template_connector = cv2.line(template_connector,
                                          (0, template_connector.shape[0]),
                                          (template_connector.shape[1], 0),
                                          (255, 255, 255),
                                          4)

        gray = cv2.cvtColor(crop_coonection_caps.copy(), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        value = np.sum(binary * template_connector)

        # print(value)
        treshold_is_connect = 30886875 / 2
        print('treshold_is_connect:', treshold_is_connect)
        if value > treshold_is_connect:
            # print('Connected')
            for pairs in sorted_pairs_caps:
                cv2.line(part_bgr,
                         (int(pairs[0].center_point.x), int(pairs[0].center_point.y)),
                         (int(pairs[1].center_point.x), int(pairs[1].center_point.y)),
                         (0, 255, 0),
                         3)
                # cv2.circle(part_bgr, (int(pairs[0].center_point.x), int(pairs[0].center_point.y)), 4, color_code, 4)
                # cv2.circle(part_bgr, (int(pairs[1].center_point.x), int(pairs[1].center_point.y)),4, color_code, 4)

            if main_diagonal:
                cv2.line(part_bgr[upper_point:fotter_point, left_point:right_point, :],
                         (0, 0),
                         (template_connector.shape[1], template_connector.shape[0]),
                         (0, 255, 0),
                         7)
            else:
                cv2.line(part_bgr[upper_point:fotter_point, left_point:right_point, :],
                         (0, template_connector.shape[0]),
                         (template_connector.shape[1], 0),
                         (0, 255, 0),
                         7)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            part_bgr = cv2.putText(part_bgr, 'Connect id: {0}'.format(i), org, font,
                                   0.5, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            for pairs in sorted_pairs_caps:
                cv2.line(part_bgr,
                         (int(pairs[0].center_point.x), int(pairs[0].center_point.y)),
                         (int(pairs[1].center_point.x), int(pairs[1].center_point.y)),
                         (0, 0, 255),
                         3)

            # print('Disconnect')
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            part_bgr = cv2.putText(part_bgr, 'Disconnect id: {0}'.format(i), org, font,
                                   0.5, (0, 0, 255), 2, cv2.LINE_AA)

            result_connected_status['fork_{0}'.format(i)] = False

        # if show_imgs:
        #     print('\npart_bgr')
        #     cv2_imshow(part_bgr)
        #     print('\ndilation')
        #     cv2_imshow(dilation)
        #     print('\ncrop_coonection_caps')
        #     cv2_imshow(crop_coonection_caps)
        #     print('\ntemplate_connector')
        #     cv2_imshow(template_connector)
        #     print('\ncurrent binary connector')
        #     cv2_imshow(binary)

    return result_connected_status, fork_image