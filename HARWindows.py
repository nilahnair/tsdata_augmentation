'''
Created on May 18, 2019

@author: fmoya
'''

import os
import numpy as np
from tqdm import tqdm
from random import choices

from torch.utils.data import Dataset

import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

import pandas as pd
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class HARWindows(Dataset):
    '''
    classdocs
    '''


    def __init__(self, config, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.harwindows = pd.read_csv(csv_file)
        self.root_dir = root_dir
        #self.transform = transform

    def __len__(self):
        return len(self.harwindows)

    def __getitem__(self, idx):
        '''
        get single item

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        '''
        window_name = os.path.join(self.root_dir, self.harwindows.iloc[idx, 0])

        f = open(window_name, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()

        X = data['data']
        y = data['label']
        Y = data['labels']
        
        if self.config['usage_modus'] == 'train' and self.config['augmentations']=='none':
            X = X
        elif self.config['usage_modus']== 'train' and choices(population=["On", "Off"], weights=[0.45, 0.55])[0]=="On":
            if self.config['augmentations']=='time_warp':
                #this one or the other one check the difference
                #X, Y = self._time_warp_speed(X, Y, 100)
                X = self._time_warp(X)
            elif self.config['augmentations']=='time_warp_seed':
                X, Y = self._time_warp_speed(X, Y, 100)
            elif self.config['augmentations']=='jittering':
                X = self.jittering(X, sigma = 0.05)
            elif self.config['augmentations']=='scaling':
                X = self._scaling(X)
            elif self.config['augmentations']=='flipping':
                X = self.flipping(X)
            elif self.config['augmentations']=='magnitude_warping':
                X = self.magnitude_warping(X)
            elif self.config['augmentations']=='permutation':
                X = self.permutation(X)
            elif self.config['augmentations']=='slicing':
                X = self.slicing(X)
            elif self.config['augmentations']=='window_warping':
                X = self.window_warping(X)
            elif self.config['augmentations']=='tilt':
                X = self.tilt(X)
            elif self.config['augmentations']=='spawner':
                X = self.spawner(X)
            
            
            

        #identity = data['identity']
        #label_file = data['label_file']
        
        #label_file = data['label_file']
        '''
        if 'identity' in data.keys():
            i = data['identity']
        
            window_data = {"data" : X, "label" : y, "labels" : Y, "identity": i, "label_file": label_file}
        else:
            window_data = {"data": X, "label": y, "labels": Y, "label_file": label_file}
        '''
        window_data = {"data": X, "label": y, "labels": Y}
        return window_data
        
    def _random_curve(self, window_len: int, sigma=0.05, knot=4):
        """
        Generates a random cubic spline with mean value 1.0.
        This curve can be used for smooth, random distortions of the data, e.g., used for time warping.

        Note: According to T. Um, a cubic splice is not the best approach to generate random curves.
        Other aprroaches, e.g., Gaussian process regression, Bezier curve, etc. are also reasonable.

        :param window_len: Length of the data window (for example, 100 frames), the curve will have this length
        :param sigma: sigma of the curve, the spline deviates from a mean of 1.0 by +- sigma
        :param knot: Number of anchor points
        :return: A 1d cubic spline
        """

        random_generator = np.random.default_rng()

        xx = (np.arange(0, window_len, (window_len - 1) / (knot + 1))).transpose()
        yy = random_generator.normal(loc=1.0, scale=sigma, size=(knot + 2, 1))
        x_range = np.arange(window_len)
        cs_x = CubicSpline(xx, yy)
        return cs_x(x_range).flatten()



    def _time_warp(self, sample: np.ndarray) -> np.ndarray:
        """
        Computes a time warping that using a random cubic curve

        :param sample: sample of shape [1, time, channels]
        :return: augmented sample
        """
        window_len = sample.shape[1]
        num_samples = sample.shape[2]

        time_warp_scale = 0.05

        #
        # Generate new time sampling values using a random curve
        # Generate curve, accumulate timestamps
        #
        timesteps = self._random_curve(window_len, sigma=time_warp_scale)
        tt_cum = np.cumsum(timesteps, axis=0)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = (window_len - 1) / tt_cum[-1]
        tt_cum = tt_cum * t_scale

        #
        # Resample
        #
        x_range = np.arange(window_len)
        resampled = np.zeros(sample.shape)
        for s_i in range(num_samples):
            resampled[0, :, s_i] = np.interp(x_range, tt_cum, sample[0, :, s_i].flatten())
            # Clamp first and last value
            resampled[0, 0, s_i] = resampled[0, 0, s_i]
            resampled[0, -1, s_i] = resampled[0, -1, s_i]

        # Return the warped sample
        return resampled


    def _time_warp_speed(self, sample: np.ndarray, annotations: np.ndarray, new_window_len: int):
        """
        Computes a time warping that using a random cubic curve

        :param sample: sample of shape [1, time, channels]
        :return: augmented sample
        """
        #length_rdm = (np.random.randint(low=90, high=sample.shape[1], size=1)) // 2
        length_rdm = (np.random.randint(low=90, high=100, size=1)) // 2
        middle = sample.shape[1] // 2
        new_range_rdm = np.arange(middle - length_rdm, middle + length_rdm)
        data = sample[:, new_range_rdm, :]

        window_len = data.shape[1]
        num_samples = data.shape[2]

        #
        # Generate new time sampling values using a random curve
        # Generate curve, accumulate timestamps
        #
        t_sep = (window_len - 1) / (new_window_len - 1)
        range_int = np.arange(0, window_len, t_sep)
        range_int = range_int[:new_window_len]

        #
        # Resample
        #
        x_range = np.arange(window_len)
        resampled = np.zeros((1, new_window_len, data.shape[2]))
        for s_i in range(num_samples):
            resampled[0, :, s_i] = np.interp(range_int, x_range, data[0, :, s_i].flatten())
            # Clamp first and last value
            resampled[0, 0, s_i] = resampled[0, 0, s_i]
            resampled[0, -1, s_i] = resampled[0, -1, s_i]

        middle = sample.shape[1] // 2
        new_range_rdm = np.arange(middle - (new_window_len // 2), middle + (new_window_len // 2))
        new_range_rdm = new_range_rdm[:new_window_len]

        # Return the warped sample
        return resampled, annotations[new_range_rdm, :]
    
    # Working
    def jittering(self, sample: np.ndarray, sigma = 0.05):
        # https://arxiv.org/pdf/1706.00527.pdf
        return sample + np.random.normal(loc=0., scale=sigma, size=sample.shape)

    # Working
    def _scaling(self, sample:np.ndarray):
        # https://arxiv.org/pdf/1706.00527.pdf
        sigma=0.1
        factor = np.random.normal(loc=1., scale=sigma, size=(sample.shape[0],sample.shape[2])) #TODO: check if indices are the right ones
        augmentedData = np.multiply(sample, factor[:,np.newaxis,:])
        return augmentedData

    # Working
    def flipping(self, sample:np.ndarray):
        return sample[:, :, ::-1]

    # Working
    def magnitude_warping(self, sample:np.ndarray):
        from scipy.interpolate import CubicSpline
        sigma = 0.2
        knot = 4
        orig_steps = np.arange(sample.shape[1])
    
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(sample.shape[0], knot+2, sample.shape[2]))
        warp_steps = (np.ones((sample.shape[2],1))*(np.linspace(0, sample.shape[1]-1., num=knot+2))).T
        ret = np.zeros_like(sample)
        for i, pat in enumerate(sample):
            warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(sample.shape[2])]).T
            ret[i] = pat * warper
        return ret

    # Working
    def permutation(self, sample:np.ndarray):
        max_segments=5
        seg_mode="equal"
    
        orig_steps = np.arange(sample.shape[1])
    
        #num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
        num_segs = [max_segments]
    
        augmentedData = np.zeros_like(sample)
        for i, pat in enumerate(sample):
            if num_segs[i] > 1:
                if seg_mode == "random":
                    split_points = np.random.choice(sample.shape[1]-2, num_segs[i]-1, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                augmentedData[i] = pat[warp]
            else:
                augmentedData[i] = pat
        return augmentedData

    def slicing(self, sample:np.ndarray):
        """
        Augments a multivariate time-series by slicing and stretching.

        Parameters:
            - sample (numpy.ndarray): The input data array of shape (1, 200, 126).
            - slice_fraction (float): Fraction of the time-series to slice out for stretching.

        Returns:
            - numpy.ndarray: The augmented data array of shape (1, 200, 126).
        """
        slice_fraction=0.2

        # # Validate the shape of the data
        # if data.shape != (1, 200, 126):
            #     raise ValueError("Data must be of shape (1, 200, 126)")

        # Validate slice_fraction
        if slice_fraction <= 0 or slice_fraction >= 1:
            raise ValueError("slice_fraction must be between 0 and 1, exclusive.")

        # Determine slice size
        time_points = sample.shape[1]
        sensor_amount = sample.shape[2]
        slice_size = int(time_points * slice_fraction)

        # Randomly select the start index for slicing
        start_idx = np.random.randint(0, time_points - slice_size)

        # Extract slice
        sliced_data = sample[:, start_idx:start_idx + slice_size, :]

        # Stretch the slice to original time-series length
        stretched_data = np.zeros((1, time_points, sensor_amount))

        for sensor in range(sensor_amount):
            stretched_data[0, :, sensor] = np.interp(
                np.linspace(0, slice_size - 1, time_points),
                np.arange(slice_size),
                sliced_data[0, :, sensor])

        # Initialize an array to store the normalized time-series
        normalized_array = np.zeros_like(stretched_data)

        # Normalize each time-series
        for j in range(126):
            min_val = np.min(stretched_data[0, j, :])
            max_val = np.max(stretched_data[0, j, :])
        
            # Check for the case where all values are the same (max_val = min_val)
            if max_val == min_val:
                normalized_array[0, j, :] = 0  # or any constant value in [0, 1]
            else:
                normalized_array[0, j, :] = (stretched_data[0, j, :] - min_val) / (max_val - min_val)

        # `normalized_array` now contains the normalized time-series.
        return normalized_array

    # Working
    def time_warping(self, sample:np.ndarray):
        from scipy.interpolate import CubicSpline
        sigma = 0.2
        knot = 4
        orig_steps = np.arange(sample.shape[1])
        
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(sample.shape[0], knot+2, sample.shape[2]))
        warp_steps = (np.ones((sample.shape[2],1))*(np.linspace(0, sample.shape[1]-1., num=knot+2))).T
    
        ret = np.zeros_like(sample)
        for i, pat in enumerate(sample):
            for dim in range(sample.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (sample.shape[1]-1)/time_warp[-1]
                ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, sample.shape[1]-1), pat[:,dim]).T
    
        return ret

    # Working
    def window_warping(self, sample:np.ndarray):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        window_ratio=0.2
        scales=[0.5, 2.]

        sample.reshape((126,200))

        warp_scales = np.random.choice(scales, sample.shape[0])
        warp_size = np.ceil(window_ratio*sample.shape[1]).astype(int)
        window_steps = np.arange(warp_size)
        
        window_starts = np.random.randint(low=1, high=sample.shape[1]-warp_size-1, size=(sample.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)
            
        ret = np.zeros_like(sample)
        for i, pat in enumerate(sample):
            for dim in range(sample.shape[2]):
                start_seg = pat[:window_starts[i],dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
                end_seg = pat[window_ends[i]:,dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))                
                ret[i,:,dim] = np.interp(np.arange(sample.shape[1]), np.linspace(0, sample.shape[1]-1., num=warped.size), warped).T
        return ret.reshape((1,200,126))

    def tilt(self, sample:np.ndarray):
        x = sample.reshape((1,126,200))

        # Generate time points
        time_points = np.linspace(0, 199, 200)

        # Define the angle of rotation in degrees
        angle_degrees = 0.05  # Replace with the desired angle
        angle_radians = np.radians(angle_degrees)

        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

        # Initialize an array to store the rotated time-series
        rotated_array = np.zeros_like(x)

        # Rotate each time-series
        for j in range(126):
            for i in range(200):
                point = np.array([time_points[i], sample[0, j, i]])
                rotated_point = np.dot(rotation_matrix, point)
                rotated_array[0, j, i] = rotated_point[1]

        # `rotated_array` now contains the rotated time-series.
        #return rotated_array

        # Initialize an array to store the normalized time-series
        normalized_array = np.zeros_like(rotated_array)

        # Normalize each time-series
        for j in range(126):
            min_val = np.min(rotated_array[0, j, :])
            max_val = np.max(rotated_array[0, j, :])
        
            # Check for the case where all values are the same (max_val = min_val)
            if max_val == min_val:
                normalized_array[0, j, :] = 0  # or any constant value in [0, 1]
            else:
                normalized_array[0, j, :] = (rotated_array[0, j, :] - min_val) / (max_val - min_val)

        # `normalized_array` now contains the normalized time-series.
        return normalized_array.reshape((1,200,126))

    #working
    def spawner(self, sample:np.ndarray, labels, sigma=0.05, verbose=0):
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
        # use verbose=-1 to turn off warnings
        # use verbose=1 to print out figures
    
        import dtw as dtw            
        random_points = np.random.randint(low=1, high=sample.shape[1]-1, size=sample.shape[0])
        window = np.ceil(sample.shape[1] / 10.).astype(int)
        orig_steps = np.arange(sample.shape[1])
        l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
        ret = np.zeros_like(sample)
        for i, pat in enumerate(tqdm(sample)):
            # guarentees that same one isnt selected
            choices = np.delete(np.arange(sample.shape[0]), i)
            # remove ones of different classes
            choices = np.where(l[choices] == l[i])[0]
            if choices.size > 0:     
                random_sample = sample[np.random.choice(choices)]
                # SPAWNER splits the path into two randomly
                path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
                path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
                combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
                if verbose:
                    print(random_points[i])
                    dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, return_flag = dtw.RETURN_ALL, slope_constraint="symmetric", window=window)
                    dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                    dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
                mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
                for dim in range(sample.shape[2]):
                    ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, sample.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
            else:
                if verbose > -1:
                    print("There is only one pattern of class %d, skipping pattern average"%l[i])
                ret[i,:] = pat
        return self.jittering(ret, sigma=sigma)
