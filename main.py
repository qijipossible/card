#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本程序用于识别特定模板的机读卡的选择题部分。

程序分为三个阶段：
1.图像预处理，主要就是让不同亮度水平的图片变成标准的能看清的二值化图片。
2.图像裁剪，主要是裁剪出卷面部分，便于输出和后续的处理。
3.选择题部分的识别，正确找到选择题部分，并识别。

对于不同质量的输入和不同的要求，在保证输出质量的前提下，程序可分为两种情况：
1.输入情况较为良好，可采用较为严格的识别方法，简单、速度快；
2.输入情况较为恶劣，如果输入的图像比例、亮度、位置都不理想，则采用较为宽松的识别方法，为保证鲁棒性程序会更为复杂，识别速度也会较低。
"""
import cv2
import numpy as np
import os
import time
import utils


W = 1400
H = 990
UNFILLED_DIFF_THRESH = 5000
CHOICE_SUM_THRESH = 24500

IF_ALL_SINGLE_CHOICE = True


def read_image(file_name):
    image = cv2.imread(in_path + file_name)
    return image


def preprocess(image):
    """
    预处理，包括二值化、标准化
    :param image:
    :return:一个标准大小(W,H)的二值化图像
    """
    # origin = image.copy()
    h, w, t = image.shape

    thresh_block_size = int(h / 6) * 2 + 1
    if h >= 1440:
        thresh_c = 34
    else:
        thresh_c = 20

    if w < h:
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh_block_size, thresh_c)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # utils.show(image)
    image = cv2.resize(image, (W, H))

    return image


def find_triangles(image):
    """
    找到四个角上的三角形
    :param image:
    :return:
    """
    t_length = 14  # 三角形模板的边长
    area_depth = 300  # 寻找三角形的区域与图片边缘的距离
    topleft_offset = 350

    # paint = cv2.cvtColor(origin, cv2.COLOR_GRAY2BGR)
    t1 = np.zeros((t_length, t_length), dtype=np.uint8)
    for i in range(t_length):
        for j in range(i):
            t1[i, j] += 255
    area1 = image[:area_depth, topleft_offset:topleft_offset + area_depth]
    match_1 = cv2.matchTemplate(area1, t1, cv2.TM_SQDIFF)
    relative_loc = cv2.minMaxLoc(match_1)[2]
    loc1 = (relative_loc[0] + topleft_offset, relative_loc[1])
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
    # utils.show(image)

    # 找到最中间两根完整的竖线，确定其构成的长方形的位置
    image = cv2.Canny(image, 50, 150)
    # utils.show(image)
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6)))  # 去掉canny后的图中的横线
    # utils.show(image)
    image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 18)))  # 增强竖线
    # utils.show(image)
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))  # 把竖线变瘦，减少误判
    # utils.show(image)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 23)))  # 让线完整
    # utils.show(image)
    lines = cv2.HoughLinesP(image, 1, np.pi / 1000, 194, maxLineGap=5, minLineLength=400)  # 这个函数是玄学
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(paint, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # 接下来，筛选出来所有在两条所求竖线位置的线段，然后求均值，以确定两条线的位置，以确定四个端点的位置
    the_line1 = []
    the_line2 = []
    # 手动圈定两条线所在的范围
    line1_left = W / 4
    line1_right = W / 2
    line_up = H / 2
    line_down = H / 2
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
    assert len(the_line1) > 0, 'cannot find mark'
    assert len(the_line2) > 0, 'cannot find mark'
    # 计算两条线的首尾位置
    x2s, y2s, x1s, y1s = list(zip(*the_line1))
    x4s, y4s, x3s, y3s = list(zip(*the_line2))
    x1, y1, x2, x3, y3, x4 = tuple(map(min, [x1s, y1s, x2s, x3s, y3s, x4s]))
    y2, y4 = tuple(map(max, [y2s, y4s]))
    current_coordinates = np.float32([[x1, y1], [x3, y3], [x2, y2], [x4, y4]])
    left, right, top, bottom = 495, 910, 50, 920  # 人工确定的四个点的相对位置
    expected_coordinates = np.float32([[left, top], [right, top], [left, bottom], [right, bottom]])
    return current_coordinates, expected_coordinates


def find_cut(image):
    """
    裁剪出卷面部分并规则化(W,H)
    :param image:标准大小(W,H)的二值化原图
    :return:一个标准大小(W,H)的二值化卷面部分
    """
    origin = image.copy()
    current_coordinates, expected_coordinates = find_lines(image)

    # 透视变化
    transform = cv2.getPerspectiveTransform(current_coordinates, expected_coordinates)
    result = cv2.warpPerspective(origin, transform, (W, H))

    return result


def choice(image):
    left, right, top, bottom = 30, 485, 320, 930  # 人工确定的选择题部分位置
    image = image[top:bottom, left:right]
    # utils.show(image)
    origin = image.copy()
    paint = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))  # 降噪
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 18)))  # 扩散成一大块一大块
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))  # 再次降噪，去除枝丫
    # utils.show(image)
    image = cv2.threshold(image, 247, 255, cv2.THRESH_BINARY_INV)[1]  # 反色
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))  # 填缝
    cnts = cv2.findContours(image, cv2.cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 找块

    blocks = []
    block = [0] * 13  # 生成一个长度为13的list

    # 找到13个大块
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)  # 描块
        w -= 3
        h -= 5
        if 70 < w < 190 and 70 < h < 115:  # 满足条件的块
            blocks.append([x, y, w, h])
            cv2.rectangle(paint, (x,y), (x+w,y+h), (0,0,255))
    # utils.show(image, paint)
    assert len(blocks) == 13, "cannot find all 13 blocks"

    # 以下为按照xy坐标排序，按顺序分别对应到13个块
    blocks.sort(key=lambda tmp: (tmp[0]))
    block[0], block[2], block[5], block[8], block[11] = sorted(blocks[:5], key=lambda z: (z[1]))
    block[3], block[6], block[9] = sorted(blocks[5:8], key=lambda z: (z[1]))
    block[1], block[12] = sorted(blocks[8:10], key=lambda z: (z[1]))
    block[4], block[7], block[10] = sorted(blocks[10:13], key=lambda z: (z[1]))

    answer_list = []
    for i, each_block in enumerate(block):
        x, y, w, h = each_block
        for j in range(5):
            num = 4  # num为选项个数
            if i == 1:
                num = 7
            if i == 12:
                num = 6
            w1 = int(w / num)  # 每个选项的宽
            h1 = int(h / 5)  # 每个选项的高
            image2 = 255 - origin  # 反色，便于后面计算和
            this_image = image2[y+j*h1:y+(j+1)*h1, x:x+w]  # 切出每道题的区域
            this_image_list = [this_image[:, k*w1:(k+1)*w1] for k in range(num)]  # 切出该题的每个选项区域并列表
            sum_list = [sum(map(sum, x)) for x in this_image_list]  # 通过计算每个选项的像素值的和，来确定填涂的是哪个选项
            # print(sum_list)
            # utils.show(this_image)
            filled_list = [x for x in sum_list if x > CHOICE_SUM_THRESH]  # 去掉没有填涂的项，这里的阈值可调
            #TODO 这里可以弄成两个阈值，像canny那样
            answer = ''
            if len(filled_list) == 0:
                answer = '0'
                print("INFO: blank detected:" + str(len(answer_list)+1))
            else:
                if IF_ALL_SINGLE_CHOICE:
                    filled_list = [max(filled_list)]
                for filled in filled_list:
                    answer += chr(ord('A') + sum_list.index(filled))
            answer_list.append(answer)
            if len(answer) > 1:
                print("INFO: multi-answer detected:" + str(len(answer_list)+1))
    return answer_list


if __name__ == "__main__":
    in_path = ".\\in2\\"
    out_path = ".\\out2\\"

    with open('result.txt', 'w') as f:
        f.write("文 件 名\t总耗时\t除I/O耗时\t结果\n")
    files = [x for x in os.listdir(in_path) if os.path.splitext(x)[1] == '.jpg' or os.path.splitext(x)[1] == '.JPG']
    for file in files:
        # file = '12.jpg'  # 30 45 47
        print('Processing ' + file)
        time1 = time.clock()
        img = read_image(file)
        # ori = img.copy()

        time2 = time.clock()
        # time.sleep(5)
        img = preprocess(img)
        try:
            img = find_cut(img)
            # utils.show(img)
            ans = choice(img)
        except AssertionError as e:
            print("ERROR: " + str(e))
            utils.show(img)
            with open('result.txt', 'a') as f:
                f.write(file + '\t' + "错误：无法识别")
                f.write('\n')
            continue
        # print(ans)
        time3 = time.clock()

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        cv2.imwrite(out_path + file, cv2.resize(img, (4200, 2970)))
        time4 = time.clock()
        with open('result.txt', 'a') as f:
            f.write(file + '\t' + str(round(time4 - time1, 3)) + '\t' + str(round(time3 - time2, 3)) + '\t')
            for i, x in enumerate(ans):
                f.write(str(i + 1) + ':' + x + ' ')
            f.write('\n')
        # break
