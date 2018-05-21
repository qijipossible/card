# coding=utf8
import cv2
import numpy as np

import utils

ERROR = "ERROR"

# 以下为解析输入参数的方法们
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

    image_raw = image_raw.split(",")
    image = np.array(image_raw).astype(np.uint8)
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


# 以下为校正图像的方法们
def _where_is(image, mark):

    match_1 = cv2.matchTemplate(image, mark, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_1)[2]  # max:2
    # utils.draw_show(image, relative_loc[0], relative_loc[1], relative_loc[0]+5, relative_loc[1]+10)
    return relative_loc[1], relative_loc[0]


def _calibrate(image, global_params, local_params):
    x_range = global_params[1]
    y_range = global_params[0]
    origin_list = []  # 标记点左上角的坐标
    current_list = []
    bias_list = []  # bias_list是每个标记点相对模板位置偏移的距离，如(1,2)表示当前图像需要向下平移1向右平移2
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
    # 单点：平移
    matrix = np.array([[1, 0, bias_list[0][0]], [0, 1, bias_list[0][1]]], dtype=np.float32)
    # 双点：平移+旋转
    delta_x = current_list[1][0] - current_list[0][0]
    delta_y = current_list[1][1] - current_list[0][1]
    cross = np.sqrt(np.sum(np.square([delta_x, delta_y])))
    alpha = delta_y / cross
    beta = delta_x / cross
    matrix = np.array([[alpha, beta, (1-alpha) * current_list[0][0] - beta * current_list[0][1]],
                       [-beta, alpha, beta * current_list[0][0] + (1 - alpha) * current_list[0][1]]], dtype=np.float32)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    utils.draw_show(image, origin_list[0][0], origin_list[0][1], origin_list[0][0]+5, origin_list[0][1]+10)
    return image


class Reader(object):
    global_params = None
    local_params = None
    std_image = None
    std_ans = None

    def __init__(self, string):
        """
        初始化类，参数为输入的模板字符串
        :param string: 标准格式输入
        """
        image_raw, width, height, object_index, question_index, params, param_len = _process_input(string)
        height = int(height)
        width = int(width)
        self.std_image = _parse_image(image_raw, height, width)
        param_len = int(param_len)
        self.global_params, self.local_params = _parse_params(params, param_len)
        # self.std_ans = self.read_one(self.std_image)

    @staticmethod
    def surround(image):
        h, w = image.shape
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
    def read_block(image, n, m, type, coefficient=None):
        """
        输入一个题块，识别填涂的选项并返回一个答案list，每一个元素都是字符，除了ABCD之外还有字符0，代表该题未填涂
        :param image: 输入的题块图像，是一个二维数组
        :param n: 题块中题目的数量
        :param m: 题块中题目的选项数量
        :param type: "1"代表单选, "2"代表多选
        :param coefficient: 判断填涂与否的阈值
        :return: 识别结果字符list
        """
        answer_list = []
        h, w = len(image), len(image[0])
        h1 = int(h / n)  # 每个选项的高
        w1 = int(w / m)  # 每个选项的宽
        if coefficient is None:
            coefficient = 0.815  # 这里的参数代表未填涂选项白色像素占比阈值，过高会导致填涂被识别为未填涂，过低会导致未填涂被识别为填涂

        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 4)))
        thresh = int(h1 * w1 * coefficient)
        # utils.show(image)
        for j in range(n):
            this_image = image[j * h1: (j + 1) * h1, :]  # 切出每道题的区域
            this_image_list = [this_image[:, k * w1:(k + 1) * w1] for k in range(m)]  # 切出该题的每个选项区域并列表
            blackest = [sum(map(sum, x)) // 255 for x in this_image_list]  # 通过计算每个选项的像素值的和，来确定填涂的是哪个选项
            choice_index = [blackest.index(x) for x in blackest if x < thresh]  # 填涂的项比较黑，所以和是最小的
            this_ans = ""
            for one_choice_index in choice_index:
                this_ans += chr(ord('A') + one_choice_index)
            if this_ans != "":
                answer_list.append(this_ans)
            else:
                answer_list.append('0')
                # print("INFO: blank detected:" + str(len(answer_list)))
        return answer_list

    def read_one(self, image):
        ans = []
        image = _calibrate(image, self.global_params, self.local_params)
        for param in self.local_params:
            if param[1] == 4:
                # x1, y1, x2, y2 = self.surround(image[param[2][0][1]:param[2][1][1], param[2][0][0]:param[2][1][0]])
                block_image = image[param[3][0][1]:param[3][1][1], param[3][0][0]:param[3][1][0]]
                # utils.show(block_image)
                ans.extend(self.read_block(block_image, param[4][0], param[4][1], param[4][2]))
        return ans
