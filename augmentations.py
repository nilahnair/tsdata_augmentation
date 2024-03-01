import sys
import numpy as np
from tqdm import tqdm

def get_augmentation(augmentation):
    if isinstance(augmentation, str):
        return getattr(sys.modules[__name__], augmentation)
    elif isinstance(augmentation, list):
        transforms = []
        for t in augmentation:
            augmentation.append(getattr(sys.modules[__name__], augmentation))
        return # TODO is there a Compose without torchvision?!

# Augmentations start here

# Working
def jittering(x, sigma = 0.05):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

# Working
def scaling(x):
    # https://arxiv.org/pdf/1706.00527.pdf
    sigma=0.1
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2])) #TODO: check if indices are the right ones
    augmentedData = np.multiply(x, factor[:,np.newaxis,:])
    return augmentedData

# Working
def flipping(x):
    return x[:, :, ::-1]

# Working
def magnitude_warping(x):
    from scipy.interpolate import CubicSpline
    sigma = 0.2
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
    """
    Augments a multivariate time-series by slicing and stretching.

    Parameters:
    - data (numpy.ndarray): The input data array of shape (1, 200, 126).
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
    from scipy.interpolate import CubicSpline
    sigma = 0.2
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

    x.reshape((126,200))

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
    return ret.reshape((1,200,126))

def tilt(x):
    x = x.reshape((1,126,200))

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
            point = np.array([time_points[i], x[0, j, i]])
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
def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
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
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = pat
    return jittering(ret, sigma=sigma)