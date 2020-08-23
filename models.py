import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

show_imgs = True
img_width = 800
img_height = 600
import imutils

min_radius_treshold = 40

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

        print(sorted_pairs_caps)
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


# def get_result(img):
#     result_is_on = False
#
#     output = img.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
#     device = None
#     # ensure at least some circles were found
#     if circles is not None:
#         # convert the (x, y) coordinates and radius of the circles to integers
#         circles = np.round(circles[0, :]).astype("int")
#         # loop over the (x, y) coordinates and radius of the circles
#         for (x, y, r) in circles:
#             if r < 60:
#                 continue
#             device = img.copy()[y - r: y + r, x - r: x + r, :]
#             cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#             cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#
#     hsv_min = np.array((2, 28, 65), np.uint8)
#     hsv_max = np.array((26, 238, 255), np.uint8)
#     img1 = device.copy()
#
#     hsv = cv2.cvtColor(device, cv2.COLOR_BGR2HSV)
#     thresh = cv2.inRange(hsv, hsv_min, hsv_max)
#     contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img1, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
#
#     center_contour = None
#     for i, c in enumerate(contours):
#         contours_poly = cv2.approxPolyDP(c, 3, True)
#         boundRect = cv2.boundingRect(contours_poly)
#         centers, radius = cv2.minEnclosingCircle(contours_poly)
#         center_contour = centers
#
#     x_center_device = int(img1.shape[1] / 2)
#     y_center_device = int(img1.shape[0] / 2)
#
#     result_image = img1.copy()
#     line_thickness = 2
#     cv2.line(result_image, (y_center_device, 0), (y_center_device, img1.shape[0]), (0, 255, 0),
#              thickness=line_thickness)
#     cv2.line(result_image, (0, x_center_device), (img1.shape[1], x_center_device), (0, 255, 0),
#              thickness=line_thickness)
#     if center_contour[0] < x_center_device and y_center_device < center_contour[1]:
#         cv2.line(result_image, (x_center_device, y_center_device), (int(center_contour[0]), int(center_contour[1])),
#                  (0, 255, 0), thickness=line_thickness + 1)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         org = (20, 20)
#         part_bgr = cv2.putText(result_image, 'On'.format(i), org, font,
#                                0.5, (0, 255, 0), 2, cv2.LINE_AA)
#         result_is_on = True
#     else:
#         cv2.line(result_image, (x_center_device, y_center_device), (int(center_contour[0]), int(center_contour[1])),
#                  (0, 0, 255), thickness=line_thickness + 1)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         org = (20, 20)
#         part_bgr = cv2.putText(result_image, 'Off'.format(i), org, font,
#                                0.5, (0, 0, 255), 2, cv2.LINE_AA)
#         result_is_on = False
#
#     return result_is_on, result_image
#
SHOW = True

# https://drive.google.com/file/d/1_Y9P2OjRnhDCrxlqrVzuvD1AdE15TJ-l/view?usp=sharing
PATH_TO_INSCRIPTION_IS_ON_TEMPLATE = 'C:/Users/andre/Notebooks/rosseti/templates/temp_is_on.png'
PATH_TO_INSCRIPTION_IS_OFF_TEMPLATE = 'C:/Users/andre/Notebooks/rosseti/templates/temp_is_off.png'


def get_inscription_localization(img, inscription_template):
    image = img.copy()
    template = cv2.cvtColor(inscription_template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.0, 2)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 128), 2)
    return (startX, startY, endX, endY)

def cut_device(img_region_device):
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((179, 255, 53), np.uint8)

    try:
        hsv = cv2.cvtColor(img_region_device, cv2.COLOR_BGR2HSV)
    except:
        return (None, None, None)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _= cv2.drawContours(img_region_device, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
    if len(contours) == 0:
        return (None, None, None)

    center_contour = None
    result_centers = None
    result_boundRect = None
    result_radius = 0

    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        centers, radius = cv2.minEnclosingCircle(contours_poly)
        if radius > result_radius:
            result_radius = radius
            result_centers = centers
            result_boundRect = boundRect

    return result_boundRect, result_centers, result_radius


def get_pointer_center(img_device):
    hsv_min = np.array((2, 28, 65), np.uint8)
    hsv_max = np.array((26179, 255, 255), np.uint8)

    hsv = cv2.cvtColor(img_device, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _= cv2.drawContours(img_region_device, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)

    if len(contours) == 0:
        return (None, None, None)

    result_center = None
    result_radius = 0

    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        centers, radius = cv2.minEnclosingCircle(contours_poly)
        if radius > result_radius:
            result_radius = radius
            result_center = centers

    return result_center


def get_result(img_rgb):
    status_is_on = True

    template_is_on = cv2.imread(PATH_TO_INSCRIPTION_IS_ON_TEMPLATE)
    start1_x, start1_y, end1_x, end1_y = get_inscription_localization(img_rgb, template_is_on)

    template_is_off = cv2.imread(PATH_TO_INSCRIPTION_IS_OFF_TEMPLATE)
    start2_x, start2_y, end2_x, end2_y = get_inscription_localization(img_rgb, template_is_off)

    ration = np.abs(end1_x - start2_x)

    start_y = start1_y - int(ration * 0.40)
    if start_y < 0:
        start_y = 0
    end_y = end1_y + int(ration * 0.1)
    if end_y > img_rgb.shape[0]:
        end_y = img_rgb.shape[0]

    img_region_device = img_rgb[start_y: end_y,
                        end1_x: start2_x,
                        :].copy()

    boundRect_device, center_device, radius_device = cut_device(img_region_device)

    if boundRect_device is None or center_device is None or radius_device is None:
        return None

    img_device = img_region_device[boundRect_device[1]: boundRect_device[3],
                 boundRect_device[0]: boundRect_device[2],
                 :].copy()

    center_pointer = get_pointer_center(img_region_device)

    if center_pointer is None:
        return None

    if center_pointer[0] > center_device[0]:
        status_is_on = False

    if SHOW:
        print('\ntemplate_is_on')
        # cv2_imshow(img_rgb[start1_y:end1_y, start1_x:end1_x])
        # print('\ntemplate_is_off')
        # cv2_imshow(img_rgb[start2_y:end2_y, start2_x:end2_x])
        # print('\nimg_region_device')
        # cv2_imshow(img_region_device)
        # print('\nimg_device')
        # cv2_imshow(img_device)
        # print('\ncenter_device')
        # cv2_imshow(
        #     cv2.line(img_device, (int(center_device[0]), 0), (int(center_device[0]), img_device.shape[0]), (0, 255, 0),
        #              thickness=3))

        print('\npointer')
        # cv2_imshow(cv2.line(img_device,
        #                     (int(center_device[0]), int(center_device[1])),
        #                     (int(center_pointer[0]), int(center_pointer[1])),
        #                     (0, 255, 0),
        #                     thickness=3))
        print("STATUS", status_is_on)

    return status_is_on


# path_to_dir = '/content/drive/My Drive/Colab Notebooks/data/circle/images'
# for filename in reversed(sorted(os.listdir(path_to_dir))):
#     path_to_img = os.path.join(path_to_dir, filename)
#
#     image_bgr = cv2.imread(path_to_img)
#     status = get_result(image_bgr)
#
#     if status is None:
#         print('Датчик не распознан')
#     elif status:
#         print('Вкл')
#     else:
#         print('Откл')