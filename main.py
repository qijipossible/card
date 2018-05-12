import cv2
import numpy as np
import os
import time

import utils

root_path = ".\\in\\"

W = 1400
H = 990

ERROR = "ERROR"


def read_image(file_name):
    image = cv2.imread(root_path + file_name)
    return image


def preprocess(image):
    """
    预处理，包括二值化、标准化
    :param image:
    :return:一个标准大小(W,H)的二值化图像
    """
    # origin = image.copy()
    h, w, t = image.shape
    if w < h:
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 1001, 20)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    image = cv2.resize(image, (W, H))
    return image


def find_triangle(image, type, size):
    """
    找到image中特定三角形的中心位置
    :param image: 给定的image，是三角附近区域而非整图
    :param type: 三角形的类型，分为四种，1代表左上，2代表右上，3代表左下，4代表右下
    :param size: 三角形的大概边长
    :return: (x, y)
    """
    pass#todo


def find_2_triangles(image, template):
    pass # todo


def find_4_triangles(image):
    """
    找到四个角上的三角形
    :param image: 标准的整张卷面图像
    :return:
    """
    origin = image.copy()
    # paint = cv2.cvtColor(origin, cv2.COLOR_GRAY2BGR)
    t_length = 14
    area_depth = 300

    t1 = np.zeros((t_length, t_length), dtype=np.uint8)
    for i in range(t_length):
        for j in range(i):
            t1[i, j] += 255
    area1 = image[:area_depth, 350:350 + area_depth]
    match_1 = cv2.matchTemplate(area1, t1, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_1)[2]
    loc1 = (relative_loc[0] + 350, relative_loc[1])
    # cv2.rectangle(paint, loc1, (loc1[0]+t_length, loc1[1]+t_length), (0,0,255), 2)

    t2 = np.zeros((t_length, t_length), dtype=np.uint8)
    for i in range(t_length):
        for j in range(i):
            t2[i, j] += 255
    area2 = image[:area_depth, -area_depth:]
    match_2 = cv2.matchTemplate(area2, t2, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_2)[2]
    loc2 = (relative_loc[0] + W - area_depth, relative_loc[1])
    # cv2.rectangle(paint, loc2, (loc2[0]+t_length, loc2[1]+t_length), (0,0,255), 2)

    t3 = np.zeros((t_length, t_length), dtype=np.uint8) + 255
    for i in range(t_length):
        for j in range(i):
            t3[i, j] -= 255
    area3 = image[-area_depth:, 0:area_depth]
    match_3 = cv2.matchTemplate(area3, t3, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_3)[2]
    loc3 = (relative_loc[0], relative_loc[1] + H - area_depth)
    # cv2.rectangle(paint, loc3, (loc3[0]+t_length, loc3[1]+t_length), (0,0,255), 2)

    t4 = np.zeros((t_length, t_length), dtype=np.uint8)
    for i in range(t_length):
        for j in range(t_length - i):
            t4[i, j] += 255
    area4 = image[-area_depth:, -area_depth:]
    match_4 = cv2.matchTemplate(area4, t4, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_4)[2]
    loc4 = (relative_loc[0] + W - area_depth, relative_loc[1] + H - area_depth)
    # cv2.rectangle(paint, loc4, (loc4[0]+t_length, loc4[1]+t_length), (0,0,255), 2)

    # utils.show(paint)

    current_coordinates = np.float32([loc1, loc2, loc3, loc4])
    expected_coordinates = np.float32([(475, 40), (1360, 40), (60, 920), (1340, 920)])  # 人工确定的四个点的相对位置
    return current_coordinates, expected_coordinates


def find_lines(image):
    paint = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 找到最中间两根完整的竖线，确定其构成的长方形的位置
    image = cv2.Canny(image, 50, 150)
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))  # 去掉canny后的图中的横线
    image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 10)))  # 增强竖线
    lines = cv2.HoughLinesP(image, 1, np.pi / 720, 170, maxLineGap=15, minLineLength=500)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(paint, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # 接下来，筛选出来所有在两条所求竖线位置的线段，然后求均值，以确定两条线的位置，以确定四个端点的位置
    the_line1 = []
    the_line2 = []
    # 手动圈定两条线所在的范围
    line1_left = W / 5
    line1_right = W / 2
    line_up = H / 3
    line_down = H * 2 / 3
    line2_left = W / 2
    line2_right = W * 4 / 5
    # 手动将位于两条线所在范围的线段挑出来
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 < y2:  # 一般情况下找到的线是x1,y1点在下，x2,y2点在上，如果相反则调过来
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if y2 < line_up and y1 > line_down:
                if line1_left < x2 < line1_right and line1_left < x1 < line1_right:
                    the_line1.append([x1, y1, x2, y2])
                    cv2.line(paint, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if line2_left < x2 < line2_right and line2_left < x1 < line2_right:
                    the_line2.append([x1, y1, x2, y2])
                    cv2.line(paint, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # utils.show(image, paint)
    # 计算两条线的首尾位置
    x2s, y2s, x1s, y1s = list(zip(*the_line1))
    x4s, y4s, x3s, y3s = list(zip(*the_line2))
    x1, y1, x2, x3, y3, x4 = tuple(map(min, [x1s, y1s, x2s, x3s, y3s, x4s]))
    y2, y4 = tuple(map(max, [y2s, y4s]))
    current_coordinates = np.float32([[x1, y1], [x3, y3], [x2, y2], [x4, y4]])
    left, right, top, bottom = 495, 910, 50, 920  # 人工确定的四个点的相对位置
    expected_coordinates = np.float32([[left, top], [right, top], [left, bottom], [right, bottom]])
    return current_coordinates, expected_coordinates


def find_coordinates(image):
    return find_4_triangles(image)


def cut(image):
    """
    裁剪出卷面部分并规则化(W,H)
    :param image:标准大小(W,H)的二值化原图
    :return:一个标准大小(W,H)的二值化卷面部分
    """
    origin = image.copy()
    current_coordinates, expected_coordinates = find_coordinates(image)

    # 透视变化
    transform = cv2.getPerspectiveTransform(current_coordinates, expected_coordinates)
    result = cv2.warpPerspective(origin, transform, (W, H))
    # utils.show(result)
    return result


def choice(image):
    """
    自动识别卷面图像并生成题块的参数。
    :param image: 标准化卷面图像
    :return: 客观题结果
    """
    left, right, top, bottom = 55, 485, 320, 910  # 人工确定的选择题部分位置
    image = image[top:bottom, left:right]
    # utils.show(image)
    origin = image.copy()
    # paint = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 18)))  # 扩散成一大块一大块
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))  # 平滑填缝
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)[1]  # 反色
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # utils.show(image)
    cnts = cv2.findContours(image, cv2.cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 找块

    blocks = []
    block = [[]] * 13  # 生成一个长度为13的list

    # 找到13个大块
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)  # 描块
        if w > 60 and h > 20:  # 满足条件的块
            blocks.append([x, y, w, h])
    if len(blocks) != 13:
        print("WARNING: error finding all 13 blocks")
    # 以下为按照xy坐标排序，按顺序分别对应到13个块
    blocks.sort(key=lambda tmp: (tmp[0]))
    block[0], block[2], block[5], block[8], block[11] = sorted(blocks[:5], key=lambda z: (z[1]))
    block[3], block[6], block[9] = sorted(blocks[5:8], key=lambda z: (z[1]))
    block[1], block[12] = sorted(blocks[8:10], key=lambda z: (z[1]))
    block[4], block[7], block[10] = sorted(blocks[10:13], key=lambda z: (z[1]))

    for index, each_block in enumerate(block):
        num = 4  # num为选项个数
        if index == 1:
            num = 7
        if index == 12:
            num = 6
        each_block.extend([5, num])
    return read_all_blocks(origin, block)


def read_all_blocks(image, blocks):
    """
    识别所有客观题
    :param image: 整张卷子的图像
    :param blocks: 所有题块的参数，是一个list，每个元素都是一个list，包含六个元素分别是题块的x,y,w,h,n,m，其中n是题块中题目数量，m是选项个数
    :return: 识别结果，一个字符list
    """
    answer_list = []
    for index, each_block in enumerate(blocks):
        x, y, w, h, n, m = each_block
        this_image = image[y:y + h, x:x + w]
        answer_list.extend(read_block(this_image, n, m))
    return answer_list


def read_block(image, n, m, type, thresh=None):
    """
    输入一个题块，识别填涂的选项并返回一个答案list，每一个元素都是字符，除了ABCD之外还有字符0，代表该题未填涂
    :param image: 输入的题块图像，是一个二维数组
    :param n: 题块中题目的数量
    :param m: 题块中题目的选项数量
    :param type: "1"代表单选, "2"代表多选
    :param thresh: 判断填涂与否的阈值
    :return: 识别结果字符list
    """
    answer_list = []
    h, w = len(image), len(image[0])
    h1 = int(h / n)  # 每个选项的高
    w1 = int(w / m)  # 每个选项的宽
    if thresh is None:
        # todo 这里可以加上图形学处理
        thresh = int(h1*w1*0.71 * 255)  # 这里的参数代表未填涂选项白色像素占比阈值，过高会导致填涂被识别为未填涂，过低会导致未填涂被识别为填涂
    for j in range(n):
        this_image = image[j * h1: (j + 1) * h1, :]  # 切出每道题的区域
        this_image_list = [this_image[:, k * w1:(k + 1) * w1] for k in range(m)]  # 切出该题的每个选项区域并列表
        blackest = [sum(map(sum, x)) for x in this_image_list]  # 通过计算每个选项的像素值的和，来确定填涂的是哪个选项
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


def process_input(input_str):
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


def process_output():
    pass


def parse_image(image_raw, height, width):
    assert image_raw is not None, "image is None"

    image_raw = image_raw.split(",")
    image = np.array(image_raw).astype(np.uint8)
    image = np.reshape(image, (height, width))
    return image


def parse_single_local_param(local_param_raw):
    tmp = []
    local_param = local_param_raw.split(" ")
    assert len(local_param) == 19, "invalid length of local_param:" + local_param_raw
    tmp.append(int(local_param[0]))
    tmp.append(int(local_param[1]))
    tmp.append(((int(local_param[2]), int(local_param[3])), (int(local_param[4]), int(local_param[5]))))
    tmp.append(((int(local_param[6]), int(local_param[7])), (int(local_param[8]), int(local_param[9]))))
    tmp.append((int(local_param[10]), int(local_param[11]), int(local_param[12]), int(local_param[13]), int(local_param[14]), int(local_param[15])))
    tmp.append(int(local_param[16]))
    tmp.append(int(local_param[17]))
    tmp.append(int(local_param[18]))
    return tmp


def parse_params(params, param_len):
    assert isinstance(params, str) and params is not None, "params wrong"
    assert isinstance(param_len, int), "invalid param length"
    assert param_len == len(params), "inconsistent param length"

    params = params.split("#")
    global_param = params[0]
    local_params = params[1:]
    print(local_params[-1])
    if local_params[-1] == "":
        local_params = local_params[:-1]
    local_params = [parse_single_local_param(one_local_param) for one_local_param in local_params if one_local_param != ""]

    for i, param in enumerate(local_params):
        assert len(param) == 8
        assert i+1 == param[0], "local_param index wrong :"+str(param[0])
    return global_param, local_params


def read(image_raw, width, height, object_index, question_index, params, param_len):

    width = int(width)
    height = int(height)

    image = parse_image(image_raw, height, width)
    param_len = int(param_len)
    global_param, local_params = parse_params(params, param_len)

    answers = []
    for param in local_params:
        block_image = image[param[3][0][1]:param[3][1][1], param[3][0][0]:param[3][1][0]]
        if param[1] == 4:
            answers.extend(read_block(block_image, param[4][0], param[4][1], param[4][2]))
    # print(answers)
    # utils.show(image)
    return answers

    # find_2_triangles(image, params)
    # read_all_blocks(image, blocks=params.blocks)
    # # read_others()
    # process_output()


def single_test():
    with open("test.txt", 'r') as f:
        lines = f.read()
    read(*process_input(lines))


    # file = '135.jpg'
    # img = read_image(file)
    # img = preprocess(img)
    # img = cut(img)
    # ans = choice(img)
    # for i, x in enumerate(ans):
    #     print(str(i + 1) + ':' + x + ' ')


def test():
    with open('result.txt', 'w') as f:
        f.write("文件名\t总耗时\t除去I/O操作耗时\t结果\n")
    files = [x for x in os.listdir(root_path)]
    for file in files:
        # file = '135.jpg'
        print('Processing ' + file)
        time1 = time.clock()
        with open(root_path+file, 'r') as f:
            lines = f.read()
        # ori = img.copy()

        time2 = time.clock()
        # time.sleep(5)
        ans = read(*process_input(lines))
        # print(ans)
        time3 = time.clock()

        # cv2.imwrite("out\\" + file, cv2.resize(img, (4200, 2970)))
        with open('result.txt', 'a') as f:
            time4 = time.clock()
            f.write(file + '\t' + str(time4 - time1) + '\t' + str(time3 - time2) + '\t')
            for i, x in enumerate(ans):
                f.write(str(i + 1) + ':' + x + ' ')
            f.write('\n')
            # break


if __name__ == "__main__":
    test()
    # single_test()
