'''
Created on June 18, 2020

@author: fmoya
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import csv_reader
import csv_save
from sliding_window import sliding_window
import pickle
#import scipy.interpolate

#from attributes import Attributes

from IMUSequenceContainer import IMUSequenceContainer

# timeStamp,packetCounter, AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ, Q0,Q1,Q2,Q3, Vbat
# data [L, T, R]

FOLDER_PATH = "/vol/actrec/DFG_Project/2019/Motionminers_Dataset/DFG-Data/"

headers = ['Time', 'Class', 'AccX_L', 'AccY_L', 'AccZ_L', 'GyrX_L', 'GyrY_L', 'GyrZ_L',
           'MagX_L', 'MagY_L', 'MagZ_L', 'AccX_T', 'AccY_T', 'AccZ_T', 'GyrX_T', 'GyrY_T',
           'GyrZ_T', 'MagX_T', 'MagY_T', 'MagZ_T', 'AccX_R', 'AccY_R', 'AccZ_R', 'GyrX_R',
           'GyrY_R', 'GyrZ_R', 'MagX_R', 'MagY_R', 'MagZ_R']

classes = {	0: "Background", 1: "Ignore", 2: "Walking", 3: "Standing",
               4: "Handling", 5: "Driving", 6: "Sitting"}

mapping_classes_training = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
NUM_CLASSES = 6

label_dict = {'HAR': {0:    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      1:    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      3:    [2,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0],
                      31:	[2,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0],
                      312: 	[2,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0],
                      313:	[2,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0],
                      314: 	[2,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0],
                      315:	[1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0,0],
                      32:	[1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0],
                      35:	[5,1,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0],
                      37: 	[1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0],
                      38:	[3,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0],
                      39:	[6,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0],
                      4:	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      41:	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      411:	[4,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      412:	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      413:	[4,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      42:	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      421: 	[4,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      422: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      423: 	[4,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      43: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      431: 	[4,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      432: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      433: 	[4,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0],
                      44: 	[1,0,0,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
                      441: 	[5,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
                      442: 	[1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
                      45:	[1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
                      451: 	[1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
                      452: 	[1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
                      5: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      51: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      511: 	[4,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      512: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      513: 	[4,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      53: 	[4,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      54: 	[1,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0],
                      55: 	[1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0],
                      7: 	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      72: 	[4,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      721: 	[4,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      722:	[4,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      79: 	[4,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      791: 	[4,0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      792: 	[4,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      793: 	[4,0,0,1,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0],
                      8: 	[1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0],
                      85: 	[1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0],
                      86: 	[0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0],
                      861:	[0,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0],
                      862: 	[0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,0,0],
                      863: 	[0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0,0,0],
                      9: 	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      91: 	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                      92: 	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]}
              }


def visualize_motionminers(data):
    fig = plt.figure()
    axis_list = []
    plot_list = []
    axis_list.append(fig.add_subplot(211))
    axis_list.append(fig.add_subplot(212))

    plot_list.append(axis_list[0].plot([], [], '-r', label='T', linewidth=0.15)[0])
    plot_list.append(axis_list[0].plot([], [], '-b', label='L', linewidth=0.15)[0])
    plot_list.append(axis_list[0].plot([], [], '-g', label='R', linewidth=0.15)[0])
    plot_list.append(axis_list[1].plot([], [], '-r', label='Class', linewidth=0.20)[0])


    #  AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ
    # data [T, 28] with L [:, 1:10] T [:, 10:19] R [:, 19:]
    # 1,    2,      3,    4,    5,    6,    7,   8,     9
    # AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ
    # 10,    11,   12,    13,   14,  15,   16,   17,   18
    # AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ
    # 19,    20,   21,    22,   23,  24,   25,   26,   27
    # AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ
    data_range_init = 0
    data_range_end = data.shape[0]
    time_x = np.arange(data_range_init, data_range_end)

    print("Range init {} end {}".format(data_range_init, data_range_end))
    T = np.linalg.norm(data[data_range_init:data_range_end, 11:14], axis=1)
    L = np.linalg.norm(data[data_range_init:data_range_end, [2, 3, 4]], axis=1)
    R = np.linalg.norm(data[data_range_init:data_range_end, [20, 21, 22]], axis=1)
    Class = data[data_range_init:data_range_end, 1]
    Class = [label_dict['HAR'][c] for c in Class]
    #Class = label_dict['HAR'][Class]

    #Arms = L * R
    plot_list[0].set_data(time_x, T)
    plot_list[1].set_data(time_x, L)
    plot_list[2].set_data(time_x, R)
    plot_list[3].set_data(time_x, Class)

    axis_list[0].relim()
    axis_list[0].autoscale_view()
    axis_list[0].legend(loc='best')

    axis_list[1].relim()
    axis_list[1].autoscale_view()
    axis_list[1].legend(loc='best')

    fig.canvas.draw()
    plt.show()
    # plt.pause(2.0)
    return


def get_annotations_real(path):
    '''
    gets data from csv file
    data contains 3 columns, start, end and label

    returns a numpy array

    @param path: path to file
    '''

    annotation_original = np.loadtxt(path, delimiter=';')
    annotation = np.copy(annotation_original)
    annotation[0, 0] -= annotation_original[0, 0]

    return annotation


def get_annotated_data_real(path, skiprows = 1):
    '''
    gets data from csv file
    data contains 3 columns, start, end and label

    returns a numpy array

    @param path: path to file
    '''

    annotation_original = np.loadtxt(path, delimiter=',', skiprows=skiprows)
    return annotation_original


def save_data(data, filename):

    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        #spamwriter.writerow(headers)
        for d in data:
            spamwriter.writerow(d)

    return

def denoise_data(data):

    negative_times = data[:, 0] >= 0
    data = data[negative_times]

    removetimes = np.ones((data.shape[0], 1))

    for t in range(1, data.shape[0] - 1):
        if (data[t, 0] - data[t - 1, 0]) < 9 or (data[t, 0] - data[t - 1, 0]) > 10:
            removetimes[t] = 0

    data = data[np.where(removetimes[:, 0] == 1)[0]]

    return data


def mapping_annotation_func(annotation):

    mapping_annotation = [label_dict['HAR'][ann] for ann in annotation]

    return np.array(mapping_annotation)


def extend_annotation(annotation, data):

    time_idx = np.where(data[:, 0] <= annotation[-1, 1])[0]
    data = data[time_idx]

    #time = np.arange(data[0, 0], data[-1, 0] + 10, 10)

    ext_annotations = np.zeros((data.shape[0], 1))
    time_ext_annotations = np.zeros((data.shape[0], 1))

    for ann_idx, ann in enumerate(annotation):
        #print(ann_idx, ann[2])
        #print(ann[0], ann[1])
        idxs = (data[:, 0] >= ann[0]) * (data[:, 0] < ann[1])
        idxs_pos = np.where(idxs == 1)[0]
        #print(idxs_pos[0], idxs_pos[-1])
        #print(data[idxs_pos[0], 0], data[idxs_pos[-1], 0])
        #print("\n")

        ext_annotations[idxs_pos] = ann[2]
        time_ext_annotations[idxs_pos, 0] = data[idxs_pos, 0]

        if np.sum((time_ext_annotations[idxs_pos, 0] >= ann[0]) * (time_ext_annotations[idxs_pos, 0] < ann[1])) == \
                np.sum(idxs == 1):
            continue
        else:
            print("Error with times")

    return data, ext_annotations


def get_data_real(IMUcontainer, imu_file_path, person):
    ser_file = IMUSequenceContainer.read_exls3(path=imu_file_path)

    # timeStamp,packetCounter, AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ, Q0,Q1,Q2,Q3, Vbat
    # data [L, T, R]
    data = IMUSequenceContainer.get_data()
    print("Original ", data.shape)

    annotation = get_annotations_real(imu_file_path + person +'_activitylabels_extract.csv')

    data = denoise_data(data)
    print("Denoised ", data.shape)
    data, ext_annotations = extend_annotation(annotation, data)

    data_annotated = np.zeros((data.shape[0], data.shape[1] + 1))
    data_annotated[:, 0] = data[:, 0]
    data_annotated[:, 1] = ext_annotations[:, 0]
    data_annotated[:, 2:] = data[:, 1:]

    return data_annotated, ext_annotations


def get_data_extract_motionminers():
    #dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers/2019/Data_Real/DFG-Data/"
    dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/DFG-Data/"
    persons = ["MS01__P01-1", "MS01__P01-2", "MS01__P02-1", "MS01__P02-2",
               "MS01__P03-5", "MS02__P01-5", "MS02__P02-4", "MS02__P03-1",
               "MS02__P03-2", "MS02__P03-3"]

    for person in persons:
        imu_file_path = dataset_path + person + "/"
        data, annotations = get_data_real(IMUSequenceContainer, imu_file_path, person)
        save_data(data, imu_file_path + person + '_data.csv')

        print("Data size {}".format(data.shape))

    return


def get_labels_motionminers():
    dataset_path = "/vol/actrec/DFG_Project/2019/Motionminers_Dataset/DFG-Data/"
    #dataset_path = "/Users/fmoya/Documents/Dok/DFG_Project/DFG-Data/"
    persons = ["MS01__P01-1", "MS01__P01-2", "MS01__P02-1", "MS01__P02-2",
               "MS01__P03-5", "MS02__P01-5", "MS02__P02-4", "MS02__P03-1",
               "MS02__P03-2", "MS02__P03-3"]

    for pe in persons:
        imu_file_path = dataset_path + pe + '/' + pe + '_data.csv'
        print(imu_file_path)

        data = get_annotated_data_real(imu_file_path, skiprows=1)
        annotations = mapping_annotation_func(data[:, 1])
        save_data(annotations, dataset_path + pe + '/' + pe + '_labels.csv')


    #visualize_motionminers(data)
    return

def statistics_measurements():
    '''
    creates files for each of the sequences extracted from a file
    following a sliding window approach


    returns a numpy array

    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    '''

    persons = ["MS01__P01-1", "MS01__P01-2", "MS01__P02-1", "MS01__P02-2",
                "MS01__P03-5", "MS02__P01-5", "MS02__P02-4", "MS02__P03-1",
                "MS02__P03-2", "MS02__P03-3"]

    accumulator_measurements = np.empty((0, 27))
    for P in persons:
        file_name_data = "{}/{}_data.csv".format(P, P)
        file_name_label = "{}/{}_labels.csv".format(P, P)
        print("------------------------------\n{}\n{}".format(file_name_data, file_name_label))
        try:
            # getting data
            data = csv_reader.reader_data(FOLDER_PATH + file_name_data)
            data_x = data[:, 2:]
            accumulator_measurements = np.append(accumulator_measurements, data_x, axis=0)
            print("\nFiles loaded")
        except:
            print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))

    try:
        max_values = np.max(accumulator_measurements, axis=0)
        min_values = np.min(accumulator_measurements, axis=0)
        mean_values = np.mean(accumulator_measurements, axis=0)
        std_values = np.std(accumulator_measurements, axis=0)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    return max_values, min_values, mean_values, std_values


def norm_motionminers_real(data):

    mean_values = np.array([-2158.247781475599, 855.390755963325, 1389.4675813896226, 30.955428699441157,
                            6.700558238662165, -22.262269242179915, 2468.306712385996, -1341.474143620317,
                            -1150.9497615477298, 11.212617589578134, -3615.691366601724, -303.6048153056071,
                            -3.6936435585433536, 25.792026400888272, -1.6807350787039403, -144.2625391404705,
                            3261.3816818239684, 185.4163216453658, 2080.8597940893587, 711.4344334952643,
                            1526.4707456468716, 1.8147758403105247, -13.757557233141402, -0.7271561241321074,
                            -2283.7609492148913, -171.1391101933215, -1318.0302728777704])
    mean_values = np.reshape(mean_values, [1, 27])

    std_values = np.array([1852.0063567250488, 2188.5663620188247, 2151.444400261148, 934.8641122210573,
                           1597.1159856259112, 1057.2860746311146, 2388.125855279013, 2749.563422655149,
                           2213.161234657168, 921.0891362969842, 1134.7120552945705, 1476.0313538420967,
                           331.8760093184819, 619.5409993141205, 311.6784978431961, 2176.0962957638712,
                           2039.0205194433688, 1777.7449273443692, 1882.3700322895595, 2188.449667811695,
                           2221.5492336793204, 950.0355434899441, 1509.7002668544928, 1071.9455887036731,
                           2924.6853702636913, 2755.637179167859, 2443.644113640929])
    std_values = np.reshape(std_values, [1, 27])

    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0

    #data_norm = (data - mean_array) / std_array

    return data_norm


def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    '''
    Performs the sliding window approach on the data and the labels

    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window

    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    '''

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    count_l = 0
    idy = 0
    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:

        # Label from the middle
        if False:
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        else:

            # Label according to mode
            try:
                data_y_labels = []
                for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                    labels = np.zeros((20)).astype(int)
                    count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    attrs = np.sum(sw[:, 1:], axis=0)
                    attrs[attrs > 0] = 1
                    labels[0] = idy
                    labels[1:] = attrs
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)

################
# Generate data
#################
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None,
                  identity_bool = False, usage_modus = 'train'):
    '''
    creates files for each of the sequences extracted from a file
    following a sliding window approach


    returns a numpy array

    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    '''

    recordings = ["MS01__P01-1", "MS01__P01-2", "MS01__P02-1", "MS01__P02-2",
                "MS01__P03-5", "MS02__P01-5", "MS02__P02-4", "MS02__P03-1",
                "MS02__P03-2", "MS02__P03-3"]

    persons = {"MS01__P01-1": 0, "MS01__P01-2": 0, "MS01__P02-1": 1, "MS01__P02-2": 1,
                "MS01__P03-5": 2, "MS02__P01-5": 0, "MS02__P02-4": 1, "MS02__P03-1": 2,
                "MS02__P03-2": 2, "MS02__P03-3": 2}

    counter_seq = 0
    hist_classes_all = np.zeros((NUM_CLASSES))
    counter_file_label = -1

    g, ax_x = plt.subplots(2, sharex=False)
    #line3, = ax_x[0].plot([], [], '-b', label='blue')
    #line4, = ax_x[1].plot([], [], '-b', label='blue')
    for P in recordings:
        if P not in ids:
            print("\n6 No Person in expected IDS {}".format(P))
        else:
            try:
                file_name_data = "{}/{}_data.csv".format(P, P)
                file_name_label = "{}/{}_labels.csv".format(P, P)
                print("\n{}\n{}".format(file_name_data, file_name_label))
                try:
                    # getting data
                    data = csv_reader.reader_data(FOLDER_PATH + file_name_data)
                    print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_data))
                    data_x = data[:, 2:]
                    data_y = data[:, 1]
                    print("\nFiles loaded")
                except:
                    print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
                    continue

                try:
                    # Getting labels and attributes
                    labels = csv_reader.reader_labels(FOLDER_PATH + file_name_label)

                    # Deleting rows containing the "background" class
                    #class_labels = np.where(labels[:, 0] == 0)[0]
                    #data_x = np.delete(data_x, class_labels, 0)
                    #labels = np.delete(labels, class_labels, 0)

                    # Deleting rows containing the "ignore" class
                    class_labels = np.where(labels[:, 0] == 1)[0]
                    data_x = np.delete(data_x, class_labels, 0)
                    labels = np.delete(labels, class_labels, 0)

                    new_classes_labels = [mapping_classes_training[cl] for cl in labels[:, 0]]
                    labels[:, 0] = np.array(new_classes_labels)


                except:
                    print(
                        "2 In generating data, Error getting the data {}".format(FOLDER_PATH
                                                                                   + file_name_data))
                    continue

                try:

                    """
                    # Graphic Vals for X in T
                    line3.set_ydata(data_x[:, 0].flatten())
                    line3.set_xdata(range(len(data_x[:, 0].flatten())))
                    ax_x[0].relim()
                    ax_x[0].autoscale_view()
                    plt.draw()
                    plt.pause(2.0)
                    """

                    data_x = norm_motionminers_real(data_x)

                    """
                    line4.set_ydata(data_x[:, 0].flatten())
                    line4.set_xdata(range(len(data_x[:, 0].flatten())))
                    ax_x[1].relim()
                    ax_x[1].autoscale_view()
                    plt.draw()
                    plt.pause(2.0)
                    """

                except:
                    print("\n3  In generating data, Plotting {}".format(FOLDER_PATH + file_name_data))
                    continue

                try:
                    # checking if annotations are consistent
                    if data_x.shape[0] == data_x.shape[0]:
                        # Sliding window approach
                        print("\nStarting sliding window")
                        X, y, y_all = opp_sliding_window(data_x, labels.astype(int), sliding_window_length,
                                                         sliding_window_step, label_pos_end=False)
                        print("\nWindows are extracted")

                        # Statistics

                        hist_classes = np.bincount(y[:, 0], minlength=NUM_CLASSES)
                        hist_classes_all += hist_classes
                        print("\nNumber of seq per class {}".format(hist_classes))
                        print("\nTotal Number of seq per class {}".format(hist_classes_all))

                        counter_file_label += 1

                        for f in range(X.shape[0]):
                            try:
                                # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                                seq = np.require(seq, dtype=np.float)

                                obj = {"data": seq, "label": y[f], "labels": y_all[f],
                                       "identity": persons[P], "label_file": counter_file_label}
                                file_name = open(os.path.join(data_dir,
                                                              'seq_{0:07}.pkl'.format(counter_seq)), 'wb')

                                pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
                                counter_seq += 1

                                sys.stdout.write(
                                    '\r' +
                                    'Creating sequence file number {} with id {}'.format(f, counter_seq))
                                sys.stdout.flush()

                                file_name.close()

                            except:
                                raise ('\nError adding the seq {} from {} \n'.format(f, X.shape[0]))

                        print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_data))

                        del data
                        del data_x
                        del X
                        del labels
                        del class_labels

                    else:
                        print("\n4 Not consisting annotation in  {}".format(file_name_data))
                        continue

                except:
                    print("\n5 In generating data, No created file {}".format(FOLDER_PATH + file_name_data))
                print("-----------------\n{}\n{}\n-----------------".format(file_name_data, file_name_label))
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')

    print("\nFinal Number of seq per class {}".format(hist_classes_all))
    print("\nFinal Number of filess {}".format(counter_file_label))
    return


def generate_CSV(csv_dir, type_file, data_dir):
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:07}.pkl'.format(n))

    np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')

    return f

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:07}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:07}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return f

def create_dataset_motionminers(identity_bool = False):
    persons = ["MS01__P01-1", "MS01__P01-2", "MS01__P02-1", "MS01__P02-2",
                "MS01__P03-5", "MS02__P01-5", "MS02__P02-4", "MS02__P03-1",
                "MS02__P03-2", "MS02__P03-3"]

    #train_ids = ["MS01__P02-1", "MS01__P03-5", "MS02__P02-4", "MS02__P03-1", "MS02__P03-2"]
    
    train_ids = ["MS01__P02-1", "MS01__P03-5", "MS02__P02-4", "MS02__P03-1"] 
    train_final_ids = ["MS01__P02-1", "MS01__P02-2", "MS01__P03-5", "MS02__P02-4", "MS02__P03-1",
                       "MS02__P03-2", "MS02__P03-3"]
    val_ids = ["MS01__P02-2", "MS02__P03-3"]
    test_ids = ["MS01__P01-1", "MS01__P01-2", "MS02__P01-5"]

    # general_statistics(train_ids)

    base_directory = '/data/fmoya/HAR/datasets/motionminers_real100_p/'

    proportion = 80
    data_dir_train = base_directory + 'sequences_train_{}/'.format(proportion)
    #data_dir_val = base_directory + 'sequences_val/'
    #data_dir_test = base_directory + 'sequences_test/'

    if identity_bool:
        generate_data(all_data, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train,
                      identity_bool=identity_bool, usage_modus='train')
        generate_data(all_data, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val,
                      identity_bool=identity_bool, usage_modus='val')
        generate_data(all_data, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test,
                      identity_bool=identity_bool, usage_modus='test')
    else:
        generate_data(train_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train)
        #generate_data(val_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val)
        #generate_data(test_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test)

    generate_CSV(base_directory, "train_{}.csv".format(proportion), data_dir_train)
    #generate_CSV(base_directory, "val.csv", data_dir_val)
    #generate_CSV(base_directory, "test.csv", data_dir_test)
    #generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return

def half_subject_norm():
    
    persons = ["MS01__P02-1", "MS01__P03-5", "MS01__P02-2", "MS02__P03-3"]

    accumulator_measurements = np.empty((0, 27))
    for P in persons:
        file_name_data = "{}/{}_data.csv".format(P, P)
        file_name_label = "{}/{}_labels.csv".format(P, P)
        print("------------------------------\n{}\n{}".format(file_name_data, file_name_label))
        try:
            # getting data
            data = csv_reader.reader_data(FOLDER_PATH + file_name_data)
            data_x = data[:, 2:]
            accumulator_measurements = np.append(accumulator_measurements, data_x, axis=0)
            print("\nFiles loaded")
        except:
            print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))

    try:
        max_values = np.max(accumulator_measurements, axis=0)
        min_values = np.min(accumulator_measurements, axis=0)
        mean_values = np.mean(accumulator_measurements, axis=0)
        std_values = np.std(accumulator_measurements, axis=0)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    print('maxvalues') 
    print(max_values) 
    print('min values')
    print(min_values) 
    print('mean values')
    print(mean_values) 
    print('std values')
    print(std_values)
    return

if __name__ == '__main__':
    IMUSequenceContainer = IMUSequenceContainer()

    #save_data(data, imu_file_path + 'MS01__P02-1_data.csv')

    # MotionMiners
    # get_data_extract_motionminers()
    # get_labels_motionminers()

    # statistics_measurements()
    #create_dataset_motionminers(identity_bool=False)
    half_subject_norm()

    print("Done")
