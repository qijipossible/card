from main import *
import cv2
import matplotlib.pyplot as plt
import os
import time
import utils

in_path = ".\\in\\"
out_path = ".\\out\\"


def retrive_image(the_tiff, verbose=False):
    """ Read image from TIFF and return it as an array. """
    if the_tiff.IsTiled():
        bits = the_tiff.GetField('BitsPerSample')
        sample_format = the_tiff.GetField('SampleFormat')
        typ = the_tiff.get_numpy_type(bits, sample_format)
        return the_tiff.read_tiles(typ)
    else:
        width = the_tiff.GetField('ImageWidth')
        height = the_tiff.GetField('ImageLength')
        arr = np.empty((height, width), 'uint8')
        size = arr.nbytes

        ReadStrip = the_tiff.ReadEncodedStrip

        pos = 0
        strip_num = the_tiff.NumberOfStrips()
        for strip in range(strip_num):
            elem = ReadStrip(strip, arr.ctypes.data, max(size/8 - pos, 0))
            pos += elem
        return arr


def read_image(file_path):
    image = plt.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # utils.show(image)
    return image


def single_test():
    # with open("test.txt", 'r') as f:
    #     lines = f.read()
    # reader = Reader(lines)
    # utils.show(reader.std_image)
    # print(reader.std_ans)

    with open("test.txt", 'r') as f:
        lines = f.read()
    reader = Reader(lines)
    time1 = time.clock()
    image = read_image("test.tif")
    ans = reader.read_one(image)
    time2 = time.clock()
    print(ans)
    print(time2-time1)


def test():
    subjects = [x for x in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, x))]
    for subject_file in subjects:
        subject_name = os.path.splitext(subject_file)[0]
        print('Processing ' + subject_name)

        time1 = time.clock()
        with open(in_path+subject_file, 'r') as f:
            lines = f.read()
        reader = Reader(lines)
        time2 = time.clock()

        with open(out_path + subject_file, 'w') as f:
            f.write("加载模板耗时：" + str(time2 - time1))
            f.write("\n文件名\t总耗时\t除去I/O操作耗时\t结果\n")

        files = [x for x in os.listdir(os.path.join(in_path, subject_name))]
        for file in files:
            time3 = time.clock()
            file_path = os.path.join(in_path, subject_name, file)
            image = read_image(file_path)
            time4 = time.clock()
            # time.sleep(5)
            ans = reader.read_one(image)
            time5 = time.clock()
            print(ans)

            # cv2.imwrite("out\\" + file, cv2.resize(img, (4200, 2970)))
            with open(out_path + subject_file, 'a') as f:
                f.write(os.path.splitext(file)[0] + '\t' + str(time5 - time3) + '\t' + str(time5 - time4) + '\t')
                for i, x in enumerate(ans):
                    f.write(str(i + 1) + ':' + x + ' ')
                f.write('\n')
                # break


if __name__ == "__main__":
    test()
    # single_test()
