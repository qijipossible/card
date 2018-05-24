# coding=utf8
import cv2
import numpy as np


def _process_input(input_str):
    assert isinstance(input_str, str), "input not str"
    lines = input_str.split("\n")
    for line in lines:
        if line.isspace():
            continue
        if line.startswith("DataImg"):
            image_raw = line.split(":")[1]
        elif line.startswith("Width"):
            width = line.split(":")[1]
        elif line.startswith("Height"):
            height = line.split(":")[1]
        elif line.startswith("ObjectScanTypeIndex"):
            object_index = line.split(":")[1]
        elif line.startswith("QuetionTypeScanIndex"):
            question_index = line.split(":")[1]
        elif line.startswith("ParamData"):
            params = line.split(":")[1]
        elif line.startswith("ParamLen"):
            param_len = line.split(":")[1]
    return image_raw, width, height, object_index, question_index, params, param_len


def _parse_image(image_raw, height, width):
    assert image_raw is not None, "image is None"
    if isinstance(image_raw, str):
        image_raw = image_raw.split(",")
    image = np.array(image_raw).astype(np.uint8)
    if len(image.shape) == 1:
        image = np.reshape(image, (height, width))
    return image


def _parse_single_local_param(local_param_raw):
    tmp = []
    local_param = local_param_raw.split(" ")
    assert len(local_param) == 19, "invalid length of local_param:" + local_param_raw
    tmp.append(int(local_param[0]))
    tmp.append(int(local_param[1]))
    tmp.append(((int(local_param[2]), int(local_param[3])), (int(local_param[4]), int(local_param[5]))))
    tmp.append(((int(local_param[6]), int(local_param[7])), (int(local_param[8]), int(local_param[9]))))
    tmp.append((int(local_param[10]), int(local_param[11]), int(local_param[12]), int(local_param[13]),
                int(local_param[14]), int(local_param[15])))
    tmp.append(int(local_param[16]))
    tmp.append(int(local_param[17]))
    tmp.append(int(local_param[18]))
    return tmp


def _parse_params(params, param_len):
    assert isinstance(params, str) and params is not None, "params wrong"
    assert isinstance(param_len, int), "invalid param length"
    assert param_len == len(params), "inconsistent param length"
    params = params.split("#")
    global_param = params[0]
    global_param = global_param.split(" ")
    global_param = [int(x) for x in global_param]
    local_params = params[1:]
    local_params = [_parse_single_local_param(one_local_param) for one_local_param in local_params if
                    one_local_param != ""]
    for i, param in enumerate(local_params):
        assert len(param) == 8
        assert i + 1 == param[0], "local_param index wrong :" + str(param[0])
    return global_param, local_params


def _where_is(image, mark):
    match_1 = cv2.matchTemplate(image, mark, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_1)[2]  
    return relative_loc[1], relative_loc[0]


def _calibrate(image, global_params, local_params):
    x_range = global_params[1]
    y_range = global_params[0]
    origin_list = []
    current_list = []
    bias_list = []  
    for param in local_params:
        if param[1] == 1:
            origin_list.append((param[3][0][1], param[3][0][0]))
            mark = np.zeros((param[3][1][1] - param[3][0][1], param[3][1][0] - param[3][0][0]), dtype=np.uint8)
            search_image = image[param[3][0][1] - x_range:param[3][1][1] + x_range, param[3][0][0] - y_range:param[3][1][0] + y_range]
            x_bias, y_bias = _where_is(search_image, mark)
            x_bias = x_range - x_bias
            y_bias = y_range - y_bias
            current_list.append((param[3][0][1] - x_bias, param[3][0][0] - y_bias))
            bias_list.append((x_bias, y_bias))
            if len(bias_list) >= 2:
                break
    delta_x = current_list[1][0] - current_list[0][0]
    delta_y = current_list[1][1] - current_list[0][1]
    cross = np.sqrt(np.sum(np.square([delta_x, delta_y])))
    alpha = max(delta_x, delta_y) / cross
    beta = min(delta_x, delta_y) / cross
    if delta_x > delta_y:
        beta = -beta
    matrix = np.array([[alpha, beta, (1-alpha)*current_list[0][1]-beta*current_list[0][0] + bias_list[0][1]], [-beta, alpha, (1-alpha)*current_list[0][0]+beta*current_list[0][1] + bias_list[0][0]]], dtype=np.float32)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return image


class Reader(object):
    global_params = None
    local_params = None
    std_image = None
    std_ans = None

    def __init__(self, string):
        image_raw, width, height, object_index, question_index, params, param_len = _process_input(string)
        height = int(height)
        width = int(width)
        self.std_image = _parse_image(image_raw, height, width)
        param_len = int(param_len)
        self.global_params, self.local_params = _parse_params(params, param_len)
        

    @staticmethod
    def surround(image, h, w):
        image = _parse_image(image, height=h, width=w)
        THRESHx = h // 10 * 255
        THRESHy = w // 10 * 255
        x1 = y1 = 0
        x2 = h
        y2 = w
        image = cv2.bitwise_not(image)
        for i in range(h):
            if sum(image[i, :]) > THRESHx:
                x1 = i
                break
        for i in range(h):
            if sum(image[-i-1, :]) > THRESHx:
                x2 = image.shape[0] - i + 1
                break
        for i in range(w):
            if sum(image[:, i]) > THRESHy:
                y1 = i
                break
        for i in range(w):
            if sum(image[:, -i-1]) > THRESHy:
                y2 = image.shape[1] - i + 1
                break
        return x1, y1, x2, y2

    @staticmethod
    def _read_block(image, n, m, t, coefficient=None):
        answer_list = []
        h, w = len(image), len(image[0])
        h1 = int(h / n)
        w1 = int(w / m)
        if coefficient is None:
            coefficient = 0.815

        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 4)))
        thresh = int(h1 * w1 * coefficient)

        for j in range(n):
            this_image = image[j * h1: (j + 1) * h1, :]
            this_image_list = [this_image[:, k * w1:(k + 1) * w1] for k in range(m)]
            blackest = [sum(map(sum, x)) // 255 for x in this_image_list]
            choice_index = [blackest.index(x) for x in blackest if x < thresh]
            this_image = [["w" + str(W) + "w" + str(H) for W in range(w)] for H in range(h*2)]
            this_ans = ""
            for one_choice_index in choice_index:
                this_ans += chr(ord('A') + one_choice_index)
            if this_ans != "":
                answer_list.append(this_ans)
            else:
                answer_list.append('0')

            del this_image
        return answer_list

    @staticmethod
    def read_block(image, h, w, n, m, t, coefficient=None):
        return Reader._read_block(_parse_image(image, h, w), n, m, t, coefficient)

    def _read_one(self, image):
        ans = []
        image = _calibrate(image, self.global_params, self.local_params)
        for param in self.local_params:
            if param[1] == 4:
                block_image = image[param[3][0][1]:param[3][1][1], param[3][0][0]:param[3][1][0]]
                ans.extend(self.read_block(block_image, param[4][0], param[4][1], param[4][2]))
        return ans

    def read_one(self, image, h, w):
        return self._read_one(_parse_image(image, h, w))
