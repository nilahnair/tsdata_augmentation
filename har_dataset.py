from torch.utils.data import Dataset
import polars as pl
from pathlib import Path
from collections import OrderedDict
import random

class HARDataset(Dataset):
    def __init__(self, path, dataset_name = 'mbientlab', window_length = 200, window_stride = 25, split = 'train', transform = None, target_transform = None, augmenation_probability = 0):
        self.path = path
        self.dataset_name = dataset_name
        self.window_length = window_length
        self.window_stride = window_stride
        self.split = split # TODO: sample by split, different per dataset, look at individual preporccessing
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation_probabiblity = augmenation_probability

        self.recordings = __prepare_dataframe__(self.path, self.dataset_name, self.split).with_row_count()

        self.index = self.__build_index__(self.recordings, __get_separating_cols__(self.dataset_name))


    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        start = self.index[index]
        stop = start + self.window_length
        sub_frame = self.recordings[start:stop]

        labels = sub_frame['class']
        label = sub_frame['class'].value_counts(sort=True)[0]['class'][0]
        sub_frame = sub_frame.select(__get_data_col_names__(self.dataset_name))
        sub_frame = sub_frame.to_numpy()
        sub_frame = sub_frame[None,:] # add dummy dimension for code compatibility 

        if self.transform:
            if random.choices(population=[True, False], weights=[self.augmentation_probabiblity, 1-self.augmentation_probabiblity])[0]:
                sub_frame = self.transform(sub_frame) # TODO: rn, applied on window, not implemented as intended on whole sequence
        # if self.target_transform:
        #     label = self.target_transform(label)

        return {
            'data': sub_frame,
            'label': label,
            'labels': labels.to_numpy()}

    def __build_index__(self, df, unique_cols):
        index = []
        if unique_cols:
            groups = df.group_by(unique_cols, maintain_order=True)
            for group, data in groups:
                        # start from first in subframe, go till end of subframe - window_length   , by steps of window stride
                index += list(range(data[0]['row_nr'][0], data[-1]['row_nr'][0] - self.window_length, self.window_stride)) # last [0] to get to value
        else:
            print('Building index on the whole dataframe')
            index += list(range(df[0]['row_nr'][0], df[-1]['row_nr'][0] - self.window_length, self.window_stride)) # last [0] to get to value

        return index

def __get_separating_cols__(dataset_name):
    '''
    Return columns per dataset that identify a recording session 
    to prohibit sampling window stretching over multiple files
    '''

    match dataset_name:
        case 'mocap':
            return ['logistic_scenario', 'subject', 'recording_number', 'annotator','execution']
        case 'mbientlab':
            return ['logistic_scenario', 'subject', 'recording_number']
        case 'motionsense':
            # return ['class_name_full', 'subject']
            return None # for motionsense, index is built on the whole dataframe
        case 'sisfall':
            return ['class', 'subject']
        case 'mobiact':
            return ['class', 'subject']
        case _:
            raise ValueError(f'Unique column names for {dataset_name=} not defined.')

def __get_data_col_names__(dataset_name):
    '''
    Return column names per dataset that identify data channels.
    '''

    match dataset_name:
        case 'mocap':
            return ['head_RX',
                    'head_RY',
                    'head_RZ',
                    'head_TX',
                    'head_TY',
                    'head_TZ',
                    'head_end_RX',
                    'head_end_RY',
                    'head_end_RZ',
                    'head_end_TX',
                    'head_end_TY',
                    'head_end_TZ',
                    'L_collar_RX',
                    'L_collar_RY',
                    'L_collar_RZ',
                    'L_collar_TX',
                    'L_collar_TY',
                    'L_collar_TZ',
                    'L_elbow_RX',
                    'L_elbow_RY',
                    'L_elbow_RZ',
                    'L_elbow_TX',
                    'L_elbow_TY',
                    'L_elbow_TZ',
                    'L_femur_RX',
                    'L_femur_RY',
                    'L_femur_RZ',
                    'L_femur_TX',
                    'L_femur_TY',
                    'L_femur_TZ',
                    'L_foot_RX',
                    'L_foot_RY',
                    'L_foot_RZ',
                    'L_foot_TX',
                    'L_foot_TY',
                    'L_foot_TZ',
                    'L_humerus_RX',
                    'L_humerus_RY',
                    'L_humerus_RZ',
                    'L_humerus_TX',
                    'L_humerus_TY',
                    'L_humerus_TZ',
                    'L_tibia_RX',
                    'L_tibia_RY',
                    'L_tibia_RZ',
                    'L_tibia_TX',
                    'L_tibia_TY',
                    'L_tibia_TZ',
                    'L_toe_RX',
                    'L_toe_RY',
                    'L_toe_RZ',
                    'L_toe_TX',
                    'L_toe_TY',
                    'L_toe_TZ',
                    'L_wrist_RX',
                    'L_wrist_RY',
                    'L_wrist_RZ',
                    'L_wrist_TX',
                    'L_wrist_TY',
                    'L_wrist_TZ',
                    'L_wrist_end_RX',
                    'L_wrist_end_RY',
                    'L_wrist_end_RZ',
                    'L_wrist_end_TX',
                    'L_wrist_end_TY',
                    'L_wrist_end_TZ',
                    # 'lower_back_RX', # not used in normalization, therefore drop
                    # 'lower_back_RY',
                    # 'lower_back_RZ',
                    # 'lower_back_TX',
                    # 'lower_back_TY',
                    # 'lower_back_TZ',
                    'R_collar_RX',
                    'R_collar_RY',
                    'R_collar_RZ',
                    'R_collar_TX',
                    'R_collar_TY',
                    'R_collar_TZ',
                    'R_elbow_RX',
                    'R_elbow_RY',
                    'R_elbow_RZ',
                    'R_elbow_TX',
                    'R_elbow_TY',
                    'R_elbow_TZ',
                    'R_femur_RX',
                    'R_femur_RY',
                    'R_femur_RZ',
                    'R_femur_TX',
                    'R_femur_TY',
                    'R_femur_TZ',
                    'R_foot_RX',
                    'R_foot_RY',
                    'R_foot_RZ',
                    'R_foot_TX',
                    'R_foot_TY',
                    'R_foot_TZ',
                    'R_humerus_RX',
                    'R_humerus_RY',
                    'R_humerus_RZ',
                    'R_humerus_TX',
                    'R_humerus_TY',
                    'R_humerus_TZ',
                    'R_tibia_RX',
                    'R_tibia_RY',
                    'R_tibia_RZ',
                    'R_tibia_TX',
                    'R_tibia_TY',
                    'R_tibia_TZ',
                    'R_toe_RX',
                    'R_toe_RY',
                    'R_toe_RZ',
                    'R_toe_TX',
                    'R_toe_TY',
                    'R_toe_TZ',
                    'R_wrist_RX',
                    'R_wrist_RY',
                    'R_wrist_RZ',
                    'R_wrist_TX',
                    'R_wrist_TY',
                    'R_wrist_TZ',
                    'R_wrist_end_RX',
                    'R_wrist_end_RY',
                    'R_wrist_end_RZ',
                    'R_wrist_end_TX',
                    'R_wrist_end_TY',
                    'R_wrist_end_TZ',
                    'root_RX',
                    'root_RY',
                    'root_RZ',
                    'root_TX',
                    'root_TY',
                    'root_TZ']
        case 'mbientlab':
            return [
                'LA_AccelerometerX',
                'LA_AccelerometerY',
                'LA_AccelerometerZ',
                'LA_GyroscopeX',
                'LA_GyroscopeY',
                'LA_GyroscopeZ',
                'LL_AccelerometerX',
                'LL_AccelerometerY',
                'LL_AccelerometerZ',
                'LL_GyroscopeX',
                'LL_GyroscopeY',
                'LL_GyroscopeZ',
                'N_AccelerometerX',
                'N_AccelerometerY',
                'N_AccelerometerZ',
                'N_GyroscopeX',
                'N_GyroscopeY',
                'N_GyroscopeZ',
                'RA_AccelerometerX',
                'RA_AccelerometerY',
                'RA_AccelerometerZ',
                'RA_GyroscopeX',
                'RA_GyroscopeY',
                'RA_GyroscopeZ',
                'RL_AccelerometerX',
                'RL_AccelerometerY',
                'RL_AccelerometerZ',
                'RL_GyroscopeX',
                'RL_GyroscopeY',
                'RL_GyroscopeZ',
            ]
        case 'motionsense':
            return [
                # 'attitude.roll',
                # 'attitude.pitch',
                # 'attitude.yaw',
                'gravity.x',
                'gravity.y',
                'gravity.z',
                'rotationRate.x',
                'rotationRate.y',
                'rotationRate.z',
                'userAcceleration.x',
                'userAcceleration.y',
                'userAcceleration.z'
                ]
        case 'sisfall':
            return ['accel_adxl_x',
                    'accel_adxl_y',
                    'accel_adxl_z',
                    'rot_x',
                    'rot_y',
                    'rot_z',
                    'accel_mma_x',
                    'accel_mma_y',
                    'accel_mma_z'
                    ]
        case 'mobiact':
            return [
                'acc_x',
                'acc_y',
                'acc_z',
                'gyro_x',
                'gyro_y',
                'gyro_z',
                'azimuth',
                'pitch',
                'roll']
        case _:
            raise ValueError(f'Unique column names for {dataset_name=} not defined.')


def __prepare_dataframe__(path, dataset_name, split):
    match dataset_name:
        case 'mocap':
            return __prepare_mocap__(path, split)
        case 'mbientlab':
            return __prepare_mbientlab__(path, split)
        case 'motionsense':
            return __prepare_motionsense__(path, split)
        case 'sisfall':
            return __prepare_sisfall__(path, split)
        case 'mobiact':
            return __prepare_mobiact__(path, split)
        case _:
            raise ValueError(f'DataFrame preparations for {dataset_name=} not implemented.')
        

def __prepare_mbientlab__(path, split):
    print(f'Preparing DataFrame for Mbientlab {split}')
    all_files = sorted(Path(path).glob('**/*.csv'))
    sample_files = list(filter(lambda f: 'labels' not in str(f), all_files))
    label_files =    list(filter(lambda f: 'labels'     in str(f), all_files))
    files = list(zip(sample_files, label_files))

    ids = {
        'train':    ["S07", "S08", "S09", "S10"],
        'val':      ["S11", "S12"],
        'test':     ["S13", "S14"]
    }

    # for normalization later
    mean_values = pl.DataFrame([-0.6018319,   0.234877,    0.2998928,   1.11102944,  0.17661719, -1.41729978,
                    0.03774093,  1.0202137,  -0.1362719,   1.78369919,  2.4127946,  -1.36437627,
                    -0.96302063, -0.0836716,   0.13035097,  0.08677377,  1.088766,    0.51141513,
                    -0.61147614, -0.22219321,  0.41094977, -1.45036893,  0.80677986, -0.1342488,
                    -0.02994514, -0.999678,   -0.22073192, -0.1808128,  -0.01197039,  0.82491874]) \
                    .transpose(column_names=__get_data_col_names__('mbientlab'))

    std_values = pl.DataFrame([1.17989719,   0.55680584,   0.65610454,  58.42857495,  74.36437559,
                    86.72291263,   1.01306,      0.62489802,   0.70924608,  86.47014857,
                    100.6318856,   61.02139095,   0.38256693,   0.21984504,   0.32184666,
                    42.84023413,  24.85339931,  18.02111335,   0.44021448,   0.51931148,
                    0.45731142,  78.58164965,  70.93038919,  76.34418105,   0.78003314,
                    0.32844988,   0.54919488,  26.68953896,  61.04472454,  62.9225945]) \
                    .transpose(column_names=__get_data_col_names__('mbientlab'))

    min_df = mean_values.with_columns(
        [pl.col(c) - 2 * std_values[c] for c in set(mean_values.columns).intersection(std_values.columns)]
    )
    max_df = mean_values.with_columns(
        [pl.col(c) + 2 * std_values[c] for c in set(mean_values.columns).intersection(std_values.columns)]
    )


    recordings = []
    for sfile, lfile in files:
        logistic_scenario, subject, recording_number = sfile.stem.split('_')

        # skip subjects according to split id list
        if subject not in ids[split]:
            continue

        logistic_scenario = int(logistic_scenario[1:])
        identity = int(subject[1:]) - 1 # same as original preprocessing
        recording_number = int(recording_number[1:])

        df = pl.concat(
            (pl.read_csv(sfile, truncate_ragged_lines=True, ignore_errors=True),
             pl.read_csv(lfile, truncate_ragged_lines=True, ignore_errors=True)),
             how='horizontal')
        df = df.with_columns([
            pl.lit(logistic_scenario).alias('logistic_scenario'),
            pl.lit(subject).alias('subject'),
            pl.lit(identity).alias('identity'),
            pl.lit(recording_number).alias('recording_number')])
    
        # fix col names
        if 'Class' in df.columns:
            df = df.rename({'Class': 'class'})
        
        df = df.filter(pl.col('class') != 7) # drop samples of activity 7

        # cast columns to smaller datatypes
        df = df.with_columns([
            pl.col('class').cast(pl.UInt8),
            pl.col('logistic_scenario').cast(pl.UInt8),
            pl.col('identity').cast(pl.UInt8),
            pl.col('recording_number').cast(pl.UInt8),
            pl.col('I-A_GaitCycle').cast(pl.Boolean),
            pl.col('I-B_Step').cast(pl.Boolean),
            pl.col('I-C_StandingStill').cast(pl.Boolean),
            pl.col('II-A_Upwards').cast(pl.Boolean),
            pl.col('II-B_Centred').cast(pl.Boolean),
            pl.col('II-C_Downwards').cast(pl.Boolean),
            pl.col('II-D_NoIntentionalMotion').cast(pl.Boolean),
            pl.col('II-E_TorsoRotation').cast(pl.Boolean),
            pl.col('III-A_Right').cast(pl.Boolean),
            pl.col('III-B_Left').cast(pl.Boolean),
            pl.col('III-C_NoArms').cast(pl.Boolean),
            pl.col('IV-A_BulkyUnit').cast(pl.Boolean),
            pl.col('IV-B_HandyUnit').cast(pl.Boolean),
            pl.col('IV-C_UtilityAux').cast(pl.Boolean),
            pl.col('IV-D_Cart').cast(pl.Boolean),
            pl.col('IV-E_Computer').cast(pl.Boolean),
            pl.col('IV-F_NoItem').cast(pl.Boolean),
            pl.col('V-A_None').cast(pl.Boolean),
            pl.col('VI-A_Error',).cast(pl.Boolean) 
            ])

        # normalization
        df = df.with_columns(
            [(pl.col(c) -  min_df[c]) / (max_df[c] - min_df[c])  for c in set(df.columns).intersection(min_df.columns)]
        )

        df = df.with_columns(
            pl.col(__get_data_col_names__('mbientlab')).clip(0.0, 1.0)
        )
        recordings.append(df)
    
        
    # concat into big df
    recordings = pl.concat(recordings, how='vertical')

    return recordings

def __prepare_mocap__(path, split):
    print(f'Preparing DataFrame for MoCap {split}')
    all_files = sorted(Path(path).glob('**/*.csv'))
    sample_files = list(filter(lambda f: 'labels' not in str(f), all_files))
    label_files =    list(filter(lambda f: 'labels'     in str(f), all_files))
    files = list(zip(sample_files, label_files))

    ids = {
        'train': ["S01", "S02", "S03", "S04", "S07", "S08", "S09", "S10", "S15", "S16"],
        'val':   ["S05", "S11", "S12"],
        'test':  ["S06", "S13", "S14"]
    }

    # for normalization later
    max_df = pl.DataFrame([392.85,    345.05,    311.295,    460.544,   465.25,    474.5,     392.85,
                        345.05,    311.295,   574.258,   575.08,    589.5,     395.81,    503.798,
                        405.9174,  322.9,     331.81,    338.4,     551.829,   598.326,   490.63,
                        667.5,     673.4,     768.6,     560.07,    324.22,    379.405,   193.69,
                        203.65,    159.297,   474.144,   402.57,    466.863,   828.46,    908.81,
                        99.14,    482.53,    381.34,    386.894,   478.4503,  471.1,     506.8,
                        420.04,    331.56,    406.694,   504.6,     567.22,    269.432,   474.144,
                        402.57,    466.863,   796.426,   863.86,    254.2,     588.38,    464.34,
                        684.77,    804.3,     816.4,     997.4,     588.38,    464.34,    684.77,
                        889.5,     910.6,    1079.7,     392.0247,  448.56,    673.49,    322.9,
                        331.81,    338.4,     528.83,    475.37,    473.09,    679.69,    735.2,
                        767.5,     377.568,   357.569,   350.501,   198.86,    197.66,    114.931,
                        527.08,    412.28,    638.503,   691.08,    666.66,    300.48,    532.11,
                        426.02,    423.84,    467.55,    497.1,     511.9,     424.76,    348.38,
                        396.192,   543.694,   525.3,     440.25,    527.08,    412.28,    638.503,
                        729.995,   612.41,    300.33,    535.94,    516.121,   625.628,   836.13,
                        920.7,     996.8,     535.94,    516.121,   625.628,   916.15,   1009.5,
                        1095.6,    443.305,   301.328,   272.984,   138.75,    151.84,    111.35]) \
                        .transpose(column_names=__get_data_col_names__('mocap'))

    min_df = pl.DataFrame([-382.62, -363.81, -315.691, -472.2, -471.4, -152.398,
                        -382.62, -363.81, -315.691, -586.3, -581.46, -213.082,
                        -400.4931, -468.4, -409.871, -336.8, -336.2, -104.739,
                        -404.083, -506.99, -490.27, -643.29, -709.84, -519.774,
                        -463.02, -315.637, -405.5037, -200.59, -196.846, -203.74,
                        -377.15, -423.992, -337.331, -817.74, -739.91, -1089.284,
                        -310.29, -424.74, -383.529, -465.34, -481.5, -218.357,
                        -442.215, -348.157, -295.41, -541.82, -494.74, -644.24,
                        -377.15, -423.992, -337.331, -766.42, -619.98, -1181.528,
                        -521.9, -581.145, -550.187, -860.24, -882.35, -645.613,
                        -521.9, -581.145, -550.187, -936.12, -982.14, -719.986,
                        -606.395, -471.892, -484.5629, -336.8, -336.2, -104.739,
                        -406.6129, -502.94, -481.81, -669.58, -703.12, -508.703,
                        -490.22, -322.88, -322.929, -203.25, -203.721, -201.102,
                        -420.154, -466.13, -450.62, -779.69, -824.456, -1081.284,
                        -341.5005, -396.88, -450.036, -486.2, -486.1, -222.305,
                        -444.08, -353.589, -380.33, -516.3, -503.152, -640.27,
                        -420.154, -466.13, -450.62, -774.03, -798.599, -1178.882,
                        -417.297, -495.1, -565.544, -906.02, -901.77, -731.921,
                        -417.297, -495.1, -565.544, -990.83, -991.36, -803.9,
                        -351.1281, -290.558, -269.311, -159.9403, -153.482, -162.718]) \
                        .transpose(column_names=__get_data_col_names__('mocap'))

    recordings = []
    for sfile, lfile in files:
        logistic_scenario, subject, recording_number, annotator, execution = sfile.stem.split('_')[:-2]

        # skip subjects according to split id list
        if subject not in ids[split]:
            continue 

        logistic_scenario = int(logistic_scenario[1:])
        subject = int(subject[1:])
        recording_number = int(recording_number[1:])
        annotator = int(annotator[1:])
        execution = int(execution[1:])

        df = pl.concat(
            (pl.read_csv(sfile, truncate_ragged_lines=True, ignore_errors=True),    #TODO: remove flags once the data is clean
             pl.read_csv(lfile, truncate_ragged_lines=True, ignore_errors=True)),   #TODO: remove flags once the data is clean
             how='horizontal')
        df = df.with_columns([
            pl.lit(logistic_scenario).alias('logistic_scenario'),
            pl.lit(subject).alias('subject'),
            pl.lit(recording_number).alias('recording_number'),
            pl.lit(annotator).alias('annotator'),
            pl.lit(execution).alias('execution')
            ])

    
        # fix col names
        if 'Class' in df.columns:
            df = df.rename({'Class': 'class'})
        if 'classlabel' in df.columns:
            df = df.rename({'classlabel': 'label'})

        df = df.select(
            pl.all().map_alias(lambda col_name: col_name.replace(' ', '_'))
        )

        # drop samples of activity 7
        df = df.filter(pl.col('class') != 7)

        # cast columns to smaller datatypes
        df = df.with_columns([
            pl.col('sample').cast(pl.UInt32),
            pl.col('label').cast(pl.UInt16),
            pl.col('class').cast(pl.UInt8),
            pl.col('logistic_scenario').cast(pl.UInt8),
            pl.col('subject').cast(pl.UInt8),
            pl.col('recording_number').cast(pl.UInt8),
            pl.col('annotator').cast(pl.UInt8),
            pl.col('execution').cast(pl.UInt8),
            pl.col('I-A_GaitCycle').cast(pl.Boolean),
            pl.col('I-B_Step').cast(pl.Boolean),
            pl.col('I-C_StandingStill').cast(pl.Boolean),
            pl.col('II-A_Upwards').cast(pl.Boolean),
            pl.col('II-B_Centred').cast(pl.Boolean),
            pl.col('II-C_Downwards').cast(pl.Boolean),
            pl.col('II-D_NoIntentionalMotion').cast(pl.Boolean),
            pl.col('II-E_TorsoRotation').cast(pl.Boolean),
            pl.col('III-A_Right').cast(pl.Boolean),
            pl.col('III-B_Left').cast(pl.Boolean),
            pl.col('III-C_NoArms').cast(pl.Boolean),
            pl.col('IV-A_BulkyUnit').cast(pl.Boolean),
            pl.col('IV-B_HandyUnit').cast(pl.Boolean),
            pl.col('IV-C_UtilityAux').cast(pl.Boolean),
            pl.col('IV-D_Cart').cast(pl.Boolean),
            pl.col('IV-E_Computer').cast(pl.Boolean),
            pl.col('IV-F_NoItem').cast(pl.Boolean),
            pl.col('V-A_None').cast(pl.Boolean),
            pl.col('VI-A_Error',).cast(pl.Boolean) 
            ])


        # normalization
        df = df.with_columns(
            [(pl.col(c) -  min_df[c]) / (max_df[c] - min_df[c])  for c in set(df.columns).intersection(min_df.columns)]
        )

        df = df.with_columns(
            pl.col(__get_data_col_names__('mocap')).clip(0.0, .99)
        )

        recordings.append(df)
    
    # concat into big df
    recordings = pl.concat(recordings, how='vertical')

    return recordings

def __prepare_motionsense__(path, split):
    print(f'Preparing DataFrame for MotionSense {split}')
    all_files = sorted(Path(path).glob('**/*.csv'))

    mean_values = pl.DataFrame([
        0.04213359,  0.75472223, -0.13882479,
        0.00532117,  0.01458119,  0.01276031,
        -0.00391064,  0.0442438,   0.03927177]). \
        transpose(column_names=__get_data_col_names__('motionsense'))
    std_values = pl.DataFrame([
        0.33882991, 0.33326483, 0.42832299,
        1.29291558, 1.22646988, 0.80804086,
        0.32820886, 0.52756613, 0.37621195]). \
        transpose(column_names=__get_data_col_names__('motionsense'))

    min_df = mean_values.with_columns(
        [pl.col(c) - 2 * std_values[c] 
            for c in set(mean_values.columns).intersection(std_values.columns)]
    )
    max_df = mean_values.with_columns(
        [pl.col(c) + 2 * std_values[c]
            for c in set(mean_values.columns).intersection(std_values.columns)]
    )

    print(f'Parsing {len(all_files)} files...')
    recordings = []
    for i, file in enumerate(all_files):
        subject = int(file.stem.split('_')[1])
        class_name_full = str(file.parent.name)
        class_name = class_name_full.split('_')[0]
        
        df = pl.read_csv(file)
        df = df.drop("") # drop index col  because of duplicates
        # add cls and subject cols
        df = df.with_columns([
            pl.lit(class_name_full).alias('class_name_full'),
            pl.lit(class_name).alias('class_name'),
            pl.lit(subject).alias('subject')
        ])
        # subselect split first 70% for train, next 15% for val, next 15% for test
        total_rows = df.shape[0]
        val_start_row = round(total_rows * 0.7)
        test_start_row = round(total_rows * 0.85)

        match split:
            case 'train':
                df = df[0:val_start_row]
            case 'val':
                df = df[val_start_row:test_start_row]
            case 'test':
                df = df[test_start_row:]

        df = df.with_columns([
            pl.lit(class_name).alias('class_name'),
            pl.lit(subject).alias('subject'),
        ])

        # normalization
        df = df.with_columns(
            [(pl.col(c) -  min_df[c]) / (max_df[c] - min_df[c])  for c in set(df.columns).intersection(min_df.columns)]
        )

        df = df.with_columns(
            pl.col(__get_data_col_names__('motionsense')).clip(0.0, 1)
        )

        recordings.append(df)

    recordings = pl.concat(recordings, how='vertical')

    # map activity names to ids
    # idea: get unique classes, sort, take index as id, apply as map
    all_activities = ["dws","ups", "wlk", "jog", "std", "sit"]
    activities_map = {a: i for i, a in enumerate(all_activities)}
    recordings = recordings.with_columns(
        pl.col('class_name').map_dict(activities_map).alias('class').cast(pl.Int32)
    )

    return recordings

def __prepare_sisfall__(path, split):
    print(f'Preparing DataFrame for SisFall {split}')
    # All data directories start with S
    # Don't consider falling classes
    all_files = sorted(Path(path).glob('S*/D*.txt'))

    #some filtering required D06, D13, D18, D19 #TODO keep D11 or not?
    all_files = list(filter(lambda p: not any(discard_activity in str(p) for discard_activity in ['D06', 'D13', 'D18', 'D19']) , all_files))

    # TODO keep, remove or fix?
    # all_files = list(filter(lambda p: 'SA15/D17_SE15' not in str(p), all_files)) 

    mean_values = pl.DataFrame([3.16212380, -220.821147, -37.2032848, -4.97325388, 34.9530823, -7.05977257, -0.394311490, -864.960077, -98.0097123]).transpose(column_names=__get_data_col_names__('sisfall'))
    std_values  = pl.DataFrame([76.42413571, 133.73065249, 108.80401481, 664.20882435, 503.17930668, 417.85844231, 296.16101639, 517.27540723, 443.30238268]).transpose(column_names=__get_data_col_names__('sisfall'))

    min_df = mean_values.with_columns(
        [pl.col(c) - 2 * std_values[c] for c in set(mean_values.columns).intersection(std_values.columns)]
    )
    max_df = mean_values.with_columns(
        [pl.col(c) + 2 * std_values[c] for c in set(mean_values.columns).intersection(std_values.columns)]
    )


    print(f'Parsing {len(all_files)} files...')
    recordings = []
    for i, file in enumerate(all_files):
        class_name, subject, recording = file.stem.split('_')
        subject = file.parent.name # override possible labeling error in SA15/D17_SE15...
        # cls = int(class_name[1:]) # infer class at the end because of skipped activities
        # subject = int(subject[2:])
        #TODO: map subject name to number == identity label 
        recording = int(recording[1:])

        df = pl.read_csv(
            file,
            separator=',',
            # eol_char=';',
            has_header=False,
            # ignore_errors=True,
            schema=OrderedDict([
                ('accel_adxl_x', pl.String),
                ('accel_adxl_y', pl.String),
                ('accel_adxl_z', pl.String),
                ('rot_x',        pl.String),
                ('rot_y',        pl.String),
                ('rot_z',        pl.String),
                ('accel_mma_x',  pl.String),
                ('accel_mma_y',  pl.String),
                ('accel_mma_z',  pl.String)
                ])).fill_null(strategy='forward')
        df = df.with_columns(pl.col('accel_mma_z').str.strip_chars(';').alias('accel_mma_z')) # remove ; from last col
        df = df.select(pl.all().map_batches(lambda col: col.str.strip_chars(' '))) # remove spaces
        if df[-1]['accel_adxl_x'][0] == '':
            df[-1, 'accel_adxl_x'] = df[-2, 'accel_adxl_x'] 
        # df = df.cast(pl.Int32)
        try:
            df = df.cast(pl.Int32)
        except:
            print(df)

        df = df.with_columns([
            # pl.lit(cls).alias('class'),
            pl.lit(class_name).alias('class_name'),
            pl.lit(subject).alias('subject'),
            pl.lit(recording).alias('recording')
        ])

        recordings.append(df)

    recordings = pl.concat(recordings, how='vertical')

    # map Dxx to class ids, pay attention to jumps in the list of class_names
    # idea: get unique classes, sort, take index as id, apply as map
    all_activities = recordings['class_name'].unique().sort()
    activities_map = {a: i for i, a in enumerate(all_activities)}
    recordings = recordings.with_columns(
        pl.col('class_name').map_dict(activities_map).alias('class').cast(pl.Int32)
    )

    dfs_by_split = []
    for group_name, data in recordings.group_by(__get_separating_cols__(dataset_name='sisfall'), maintain_order=True):
        # subselect split first 70% for train, next 15% for val, next 15% for test
        total_rows = data.shape[0]
        val_start_row = round(total_rows * 0.7)
        test_start_row = round(total_rows * 0.85)

        match split:
            case 'train':
                dfs_by_split.append(data[0:val_start_row])
            case 'val':
                dfs_by_split.append(data[val_start_row:test_start_row])
            case 'test':
                dfs_by_split.append(data[test_start_row:])

    recordings = pl.concat(dfs_by_split, how='vertical')

    # normalzation
    df = df.with_columns(
        [(pl.col(c) -  min_df[c]) / (max_df[c] - min_df[c])  for c in set(df.columns).intersection(min_df.columns)]
    )

    df = df.with_columns(
        pl.col(__get_data_col_names__('sisfall')).clip(0.0, 1)
    )

    return recordings

def __prepare_mobiact__(path, split):
    print(f'Preparing DataFrame for MobiAct {split}')
    # All data directories start with S
    # Don't consider falling classes
    all_files = sorted(Path(path).glob('**/*.csv'))

    # keep only:
    all_activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
    all_files = filter(lambda p: any(activity in str(p.parent) for activity in all_activities), all_files)

    # remove subjects
    all_files = list(filter(lambda p: not any(str(identity) in p.name.split('_')[1] for identity in [13, 14, 15, 17, 28, 30, 31, 34, 57]), all_files))


    mean_values = pl.DataFrame([
        0.265079537, 7.13106528, 0.387973281,
        -0.0225606363, -0.00302826137,  0.0131514254,
        178.629895, -67.8435738, 2.02923485]).transpose(column_names=__get_data_col_names__('mobiact'))
    std_values  = pl.DataFrame([
        3.49444826, 6.70119943, 3.31003981,
        1.1238746, 1.12533643, 0.72129725,
        105.81241608, 58.62837783, 17.58456297]).transpose(column_names=__get_data_col_names__('mobiact'))

    min_df = mean_values.with_columns(
        [pl.col(c) - 2 * std_values[c] for c in set(mean_values.columns).intersection(std_values.columns)]
    )
    max_df = mean_values.with_columns(
        [pl.col(c) + 2 * std_values[c] for c in set(mean_values.columns).intersection(std_values.columns)]
    )

    print(f'Parsing {len(all_files)} files...')
    recordings = []
    for i, file in enumerate(all_files):
        class_name, subject, recording, _ = file.name.split('_')

        df = pl.read_csv(file)
        df = df.with_columns([
            pl.lit(class_name).alias('class_name'),
            pl.lit(subject).alias('subject').cast(pl.UInt16),
            pl.lit(recording).alias('recording').cast(pl.UInt16)
        ])

        recordings.append(df)
    
    recordings = pl.concat(recordings, how='vertical')
    
    # pick label == class per row or class_name == class per folder
    class_col_name = 'class_name'
    #all_activities = recordings[class_col_name].unique().sort() # don't query, defined above, preserve order
    activities_map = {a: i for i, a in enumerate(all_activities)}
    recordings = recordings.with_columns(
        pl.col(class_col_name).map_dict(activities_map).alias('class').cast(pl.Int16)
    )

    dfs_by_split = []
    for group_name, data in recordings.group_by(__get_separating_cols__(dataset_name='sisfall'), maintain_order=True):
        # subselect split first 70% for train, next 15% for val, next 15% for test
        total_rows = data.shape[0]
        val_start_row = round(total_rows * 0.7)
        test_start_row = round(total_rows * 0.85)

        match split:
            case 'train':
                dfs_by_split.append(data[0:val_start_row])
            case 'val':
                dfs_by_split.append(data[val_start_row:test_start_row])
            case 'test':
                dfs_by_split.append(data[test_start_row:])

    recordings = pl.concat(dfs_by_split, how='vertical')

    # normalzation
    df = df.with_columns(
        [(pl.col(c) -  min_df[c]) / (max_df[c] - min_df[c])  for c in set(df.columns).intersection(min_df.columns)]
    )

    df = df.with_columns(
        pl.col(__get_data_col_names__('mobiact')).clip(0.0, 1)
    )

    return recordings
        

def __check_ragged_cols__(path='/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/LARa_dataset_mocap/'):
    import csv
    import numpy as np

    print(f'Checking all csv files in {path} for ragged columns')

    filelist = list(Path(path).rglob('*.csv'))
    num_files = len(filelist)

    broken_files = []
    for i, file in enumerate(filelist):
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            lengths_list = [len(row) for row in reader]
            unique_lengths, inverse, counts = np.unique(lengths_list, return_counts=True, return_inverse=True)
            if len(unique_lengths) == 1:
                print(f'[{i}/{num_files}] OK - {str(file)}')
                continue
            else:
                print(f'[{i}/{num_files}] ERROR - {str(file)}')
                print(unique_lengths, inverse, counts)
                broken_files.append((file, unique_lengths, inverse, counts))

    print('########### REPORT ###########')
    if len(broken_files) == 0:
        print('Congratulations! All rows in all files have the same length!')
    else:
        for i, (file, unique_lengths, inverse, counts) in enumerate(broken_files):
            print(f'ERROR {i}\n')
            print(f'Error in {file=}')
            most_counts_idx = np.argmax(counts)
            correct_length = unique_lengths[most_counts_idx]
            print(f'Most rows are {correct_length} long. Assuming this as correct.')

            incorrect_mask = (inverse != most_counts_idx)
            incorrect_rows = np.where(incorrect_mask)[0]
            print(f'The following lines have errors {incorrect_rows}')


def __check_same_col_names__(path='/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/LARa_dataset_mocap/', glob='*labels.csv', col_idx=0, var1 = 'class', var2 = 'Class'):
    import csv
    import numpy as np

    print(f'Checking all csv files in {path} for consistent column names {var1} vs {var2}')

    filelist = list(Path(path).rglob(glob))
    num_files = len(filelist)

    var1_count = 0
    var1_files = []
    var2_count = 0
    var2_files = []
    for i, file in enumerate(filelist):
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            fst_row = next(reader)
            class_col_name = fst_row[col_idx]

            counter_msg = f'[{i}/{num_files}] '
            msg = f' in file {file}'

            if class_col_name == var1:
                var1_count += 1
                var1_files.append(file)
            elif class_col_name == var2:
                var2_count +=1
                var2_files.append(file)
            else:
                print(f'Neither {var1} nor {var2} ...')

            print(counter_msg + class_col_name + msg)

    print('########### REPORT ###########')
    if var1_count == 0 or var2_count == 0:
        print('Congratulations! Class col names are all the same')
    else:
        majority_name =  var2 if var2_count > var1_count else var1
        print(f'Majority of files use: {majority_name}. {var1_count} {var1} vs {var2_count} {var2}')
        print('Here is a list of minority files:')
        minority_list = var1_files if var2_count > var1_count else var2_files
        for f in minority_list:
            print(str(f))