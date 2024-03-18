import sys
import numpy as np
from tqdm import tqdm
import torch
import math


def get_augmentation(augmentation):
    if isinstance(augmentation, str):
        return getattr(sys.modules[__name__], augmentation)
    elif isinstance(augmentation, list):
        augmentation_res = []
        for t in augmentation:
            augmentation_res.append(getattr(sys.modules[__name__], t))
        return augmentation_res

# Augmentations start here

# Working
def jittering(x, sigma = 0.03):
    #using this
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

# Working
def scaling(x):
    #using this
    # https://arxiv.org/pdf/1706.00527.pdf
    sigma=0.04
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2])) #TODO: check if indices are the right ones
    augmentedData = np.multiply(x, factor[:,np.newaxis,:])
    return augmentedData

# Working
def flipping(x):
    #using this
    rand_val=np.flip(x,1)
    x=np.array(rand_val)
    return x
# Working
def magnitude_warping(x):
    #using this
    from scipy.interpolate import CubicSpline
    sigma = 0.1
    knot = 4
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ret

# Working
def permutation(x):
    #using this
    max_segments=5
    seg_mode="equal"
    
    orig_steps = np.arange(x.shape[1])
    
    #num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    num_segs = [max_segments]
    
    augmentedData = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            augmentedData[i] = pat[warp]
        else:
            augmentedData[i] = pat
    return augmentedData

def slicing(data):
    #using this
    """
    Augments a multivariate time-series by slicing and stretching.

    Parameters:
    - data (numpy.ndarray): The input data array of shape (1, 200, 126).
    - slice_fraction (float): Fraction of the time-series to slice out for stretching.

    Returns:
    - numpy.ndarray: The augmented data array of shape (1, 200, 126).
    """
    slice_fraction=0.5

    # # Validate the shape of the data
    # if data.shape != (1, 200, 126):
    #     raise ValueError("Data must be of shape (1, 200, 126)")

    # Validate slice_fraction
    if slice_fraction <= 0 or slice_fraction >= 1:
        raise ValueError("slice_fraction must be between 0 and 1, exclusive.")

    # Determine slice size
    time_points = data.shape[1]
    sensor_amount = data.shape[2]
    slice_size = int(time_points * slice_fraction)

    # Randomly select the start index for slicing
    start_idx = np.random.randint(0, time_points - slice_size)

    # Extract slice
    sliced_data = data[:, start_idx:start_idx + slice_size, :]

    # Stretch the slice to original time-series length
    stretched_data = np.zeros((1, time_points, sensor_amount))

    for sensor in range(sensor_amount):
        stretched_data[0, :, sensor] = np.interp(
            np.linspace(0, slice_size - 1, time_points),
            np.arange(slice_size),
            sliced_data[0, :, sensor]
        )

    # Initialize an array to store the normalized time-series
    normalized_array = np.zeros_like(stretched_data)

    # Normalize each time-series
    for j in range(time_points):
        min_val = np.min(stretched_data[0, j, :])
        max_val = np.max(stretched_data[0, j, :])
        
        # Check for the case where all values are the same (max_val = min_val)
        if max_val == min_val:
            normalized_array[0, j, :] = 0  # or any constant value in [0, 1]
        else:
            normalized_array[0, j, :] = (stretched_data[0, j, :] - min_val) / (max_val - min_val)

    # `normalized_array` now contains the normalized time-series.
    return normalized_array


# # Possibly not working as intended
# def slicing(x):
#     # https://halshs.archives-ouvertes.fr/halshs-01357973/document
#     x = x.reshape((1,200,126))

#     reduce_ratio = 0.9
#     target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
#     if target_len >= x.shape[1]:
#         return x
#     starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
#     ends = (target_len + starts).astype(int)
#     print(starts)
#     print(ends)
#     ret = np.zeros_like(x)
#     for i, pat in enumerate(x):
#         for dim in range(x.shape[2]):
#             ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
#     return ret

# Working
def time_warping(x):
    #using this
    from scipy.interpolate import CubicSpline
    sigma = 0.06
    knot = 4
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    
    return ret

# Working
def window_warping(x):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    window_ratio=0.2
    scales=[0.5, 2.]
    channel=x.shape[2]
    length=x.shape[1]
    x.reshape((1,channel,length))

    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret.reshape((1,length,channel))

def tilt(x):
    #using this
    channel=x.shape[2]
    length=x.shape[1]
    x = x.reshape((1,channel,length))

    # Generate time points
    time_points = np.linspace(0, 199, length)

    # Define the angle of rotation in degrees
    angle_degrees = 0.02  # Replace with the desired angle
    angle_radians = np.radians(angle_degrees)

    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Initialize an array to store the rotated time-series
    rotated_array = np.zeros_like(x)

    # Rotate each time-series
    for j in range(channel):
        for i in range(length):
            point = np.array([time_points[i], x[0, j, i]])
            rotated_point = np.dot(rotation_matrix, point)
            rotated_array[0, j, i] = rotated_point[1]

    # `rotated_array` now contains the rotated time-series.
    #return rotated_array

    # Initialize an array to store the normalized time-series
    normalized_array = np.zeros_like(rotated_array)

    # Normalize each time-series
    for j in range(channel):
        min_val = np.min(rotated_array[0, j, :])
        max_val = np.max(rotated_array[0, j, :])
        
        # Check for the case where all values are the same (max_val = min_val)
        if max_val == min_val:
            normalized_array[0, j, :] = 0  # or any constant value in [0, 1]
        else:
            normalized_array[0, j, :] = (rotated_array[0, j, :] - min_val) / (max_val - min_val)

    # `normalized_array` now contains the normalized time-series.
    return normalized_array.reshape((1,length,channel))

#working
#def spawner(x, labels, sigma=0.05, verbose=0):
def spawner(x, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    #l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        #choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:     
            random_sample = x[np.random.choice(choices)]
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
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            #if verbose > -1:
            #    print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = pat
    return jittering(ret, sigma=sigma)

'''
def windowslicing(x):
        #for 4 dimensional input - including batch
        slice_fraction=0.5

        # # Validate the shape of the data
        # if data.shape != (1, 200, 126):
        #     raise ValueError("Data must be of shape (1, 200, 126)")

        # Validate slice_fraction
        if slice_fraction <= 0 or slice_fraction >= 1:
            raise ValueError("slice_fraction must be between 0 and 1, exclusive.")

        # Determine slice size
        time_points = x.shape[1]
        sensor_amount = x.shape[2]
        slice_size = int(time_points * slice_fraction)
                        
        ret_b=np.zeros_like(x)
        for i, pat in enumerate(x):
            # Randomly select the start index for slicing
            start_idx = np.random.randint(0, time_points - slice_size)

            # Extract slice
            sliced_data = pat[:, start_idx:start_idx + slice_size, :]

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
                for j in range(x[1]):
                    min_val = np.min(stretched_data[0, j, :])
                    max_val = np.max(stretched_data[0, j, :])
        
                # Check for the case where all values are the same (max_val = min_val)
                if max_val == min_val:
                    normalized_array[0, j, :] = 0  # or any constant value in [0, 1]
                else:
                    normalized_array[0, j, :] = (stretched_data[0, j, :] - min_val) / (max_val - min_val)

                # `normalized_array` now contains the normalized time-series.
                ret_b[i]=normalized_array 
        return ret_b
'''

def vertical_flip(x):
    x=((x-0.5)*-1)+0.5
    return x

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return ret

# Proposed

def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]
            
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                            
            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret

def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")

def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]
        
        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]
                        
            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.9+0.1*warp_amount[i]/max_warp)[0]
    return ret

def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def freq_mix(x,rate=0.5):
   
    #x_f = np.fft.rfft(x,dim=1)
    x_f = np.fft.rfft(x)
        
    m =(x_f.shape).uniform_() < rate
    amp = abs(x_f)
    _,index = amp.sort(dim=1, descending=True)
    dominant_mask = index > 2
    m = np.bitwise_and(m,dominant_mask)
    freal = x_f.real.masked_fill(m,0)
    fimag = x_f.imag.masked_fill(m,0)
        
    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2= x[b_idx]
    #x2_f = np.fft.rfft(x2,dim=1)
    x2_f = np.fft.rfft(x2)

    m = np.bitwise_not(m)
    freal2 = x2_f.real.masked_fill(m,0)
    fimag2 = x2_f.imag.masked_fill(m,0)

    freal += freal2
    fimag += fimag2

    x_f = np.complex(freal,fimag)
        
    #x = np.fft.irfft(x_f,dim=1)
    x = np.fft.irfft(x_f)
    return x

def resampling_random(x):
    import random
    M = random.randint(1, 3)
    N = random.randint(0, M - 1)
    assert M > N, 'the value of M have to greater than N'

    timesetps = x.shape[1]

    for i in range(timesetps - 1):
        x1 = x[:, i * (M + 1), :]
        x2 = x[:, i * (M + 1) + 1, :]
        for j in range(M):
            v = np.add(x1, np.subtract(x2, x1) * (j + 1) / (M + 1))
            x = np.insert(x, i * (M + 1) + j + 1, v, axis=1)
    length_inserted = x.shape[1]
    num = x.shape[0]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    x_selected=x[0,index_selected,:][np.newaxis,]
    for k in range(1,num):
        start = random.randint(0, length_inserted - timesetps * (N + 1))
        index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
        x_selected = np.concatenate((x_selected,x[k,index_selected,:][np.newaxis,]),axis=0)
    return x_selected

def magnify(x):
    lam = np.random.randint(11,14)/10
    return np.multiply(x,lam)

def spectral_pooling(x, pooling_number = 0):
        '''
        Carry out a spectral pooling.
        torch.rfft(x, signal_ndim, normalized, onesided)
        signal_ndim takes into account the signal_ndim dimensions stranting from the last one
        onesided if True, outputs only the positives frequencies, under the nyquist frequency

        @param x: input sequence
        @return x: output of spectral pooling
        '''
        # xpool = F.max_pool2d(x, (2, 1))

        x = x.permute(0, 2, 1)

        # plt.figure()
        # f, axarr = plt.subplots(5, 1)

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[0].plot(x_plt[0], label='input')

        #fft = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)
        fft = torch.fft.rfft(x, norm="forward")
        #if self.config["storing_acts"]:
        #    self.save_acts(fft, "x_LA_fft")
        # fft2 = torch.rfft(x, signal_ndim=1, normalized=False, onesided=False)

        # fft_plt = fft[0, 0].to("cpu", torch.double).detach()
        # fft_plt = torch.norm(fft_plt, dim=2)
        # axarr[1].plot(fft_plt[0], 'o', label='fft')

        #x = fft[:, :, :, :int(fft.shape[3] / 2)]
        x = fft[ :, :, :int(math.ceil(fft.shape[2] / 2))]
        #if self.config["storing_acts"]:
        #    self.save_acts(x, "x_LA_fft_2")

        # fftx_plt = x[0, 0].to("cpu", torch.double).detach()
        # fftx_plt = torch.norm(fftx_plt, dim=2)
        # axarr[2].plot(fftx_plt[0], 'o', label='fft')

        # x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True)
        x = torch.fft.irfft(x, norm="forward")
        #if self.config["storing_acts"]:
        #    self.save_acts(x, "x_LA_ifft")

        #x = x[:, :, :self.pooling_Wx[pooling_number]]
        #if self.config["storing_acts"]:
        #    self.save_acts(x, "x_LA_ifft_pool")

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], label='input')

        x = x.permute(0, 2, 1)

        # fft2_plt = fft2[0, 0].to("cpu", torch.double).detach()
        # fft2_plt = torch.norm(fft2_plt, dim=2)
        # print(fft2_plt.size(), 'max: {}'.format(torch.max(fft2_plt)), 'min: {}'.format(torch.min(fft2_plt)))
        # axarr[4].plot(fft2_plt[0], 'o', label='fft')

        # xpool = xpool.permute(0, 1, 3, 2)
        # x_plt = xpool[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], label='input')

        # plt.waitforbuttonpress(0)
        # plt.close()


        return x