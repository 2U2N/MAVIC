"""Frame-level and video-level measurements used in MAVIC v1."""
import gc
import os
import cv2
import numpy as np
import mediapipe as mp
import pywt
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import circvar
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from .assets import asset

def calculate_statistics_for_function(keyframe_dir, func):
    """
    Wrapper function to calculate statistics for a given image processing function over images in a directory.

    Parameters:
    keyframe_dir (str): The directory containing keyframe images.
    func (function): The image processing function to apply.

    Returns:
    tuple: Mean, standard deviation, and volatility of the function results.
    """
    results = []
    
    # Iterate through each file in the directory
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'):  # Handle multiple image formats
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = func(image)
                results.append(result)

    # Calculate mean and standard deviation
    mean = np.mean(results)
    sd = np.std(results)

    # Calculate volatility as the standard deviation of differences between adjoining results
    if len(results) > 1:
        differences = np.diff(results)
        volatility = np.std(differences)
    else:
        volatility = 0  # Not enough keyframes to calculate volatility
    
    return mean, sd, volatility
    
#############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#############################################################################################################

# Colour category
#############################################################################################################
#### color percentage:

# Calculate color
def calculate_color_percentage(image, lower_bound, upper_bound):
    """
    Calculate the percentage of pixels within the given HSV range.
    """
    # Create a mask for the pixels within the given color range
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Calculate the percentage of pixels that match the color range
    color_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    percentage = (color_pixels / total_pixels) * 100
    
    return percentage

def calculate_all_colors(image):
    """
    Calculate the percentage of multiple colors in an image (black, blue, brown, gray, green, orange, pink, purple, red, white, yellow).
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
        'gray': (np.array([0, 0, 30]), np.array([180, 50, 230])),
        'white': (np.array([0, 0, 230]), np.array([180, 25, 255])),
        'red': [(np.array([0, 50, 50]), np.array([10, 255, 255])), (np.array([170, 50, 50]), np.array([180, 255, 255]))],
        'green': (np.array([35, 50, 50]), np.array([85, 255, 255])),
        'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
        'yellow': (np.array([25, 50, 50]), np.array([35, 255, 255])),
        'orange': (np.array([10, 50, 50]), np.array([25, 255, 255])),
        'brown': (np.array([10, 100, 20]), np.array([20, 255, 200])),
        'pink': (np.array([145, 50, 50]), np.array([170, 255, 255])),
        'purple': (np.array([120, 50, 50]), np.array([160, 255, 255]))
    }
    
    color_percentages = {}

    for color, ranges in color_ranges.items():
        if isinstance(ranges, list):
            # For colors like red with two ranges, calculate for both ranges and sum the results
            percentage = sum(calculate_color_percentage(hsv_image, low, high) for low, high in ranges)
        else:
            percentage = calculate_color_percentage(hsv_image, ranges[0], ranges[1])
        color_percentages[color] = percentage
    
    return color_percentages

def calculate_color_percentages_statistics(keyframe_dir):
    #result vectors
    Black_percentages = []
    Gray_percentages = []
    White_percentages = []
    Red_percentages = []
    Green_percentages = []
    Blue_percentages = []
    Yellow_percentages = []
    Orange_percentages = []
    Brown_percentages = []
    Pink_percentages = []
    Purple_percentages = []

    
    color_lists = [
    Black_percentages,
    Gray_percentages,
    White_percentages,
    Red_percentages,
    Green_percentages,
    Blue_percentages,
    Yellow_percentages,
    Orange_percentages,
    Brown_percentages,
    Pink_percentages,
    Purple_percentages
    ]

    # Iterate through each file in the directory
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = calculate_all_colors(image) # apply function
                for i, val in enumerate(result.values()): # Get each percentage
                    color_lists[i].append(val)

    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [0,0,0,0,0,0,0,0,0,0,0]
    sd = [0,0,0,0,0,0,0,0,0,0,0]
    volatility = [0,0,0,0,0,0,0,0,0,0,0]
    
    for i, colour in enumerate(color_lists):
        means[i] = np.mean(colour)
        sd[i] = np.std(colour)
        if len(colour) > 1:
            differences = np.diff(colour)
            volatility[i] = np.std(differences)
            
    return means, sd, volatility  

#############################################################################################################
###### color wamth

def calculate_color_warmth(keyframes_folder):
    warmth_ratios = []
    
    # Calculate pixel warmth ratio for each keyframe
    for filename in sorted(os.listdir(keyframes_folder)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframes_folder, filename)
            hsv_image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2HSV)  # load keyframes in hsv color space
            hue = hsv_image[:, :, 0]
            warm_pixels = np.count_nonzero((hue < 30) | (hue > 110))
            total_pixels = hue.size
            warm_ratio = warm_pixels / total_pixels
            warmth_ratios.append(warm_ratio)
    
    # Calculate mean and standard deviation
    mean_warmth = np.mean(warmth_ratios)
    std_dev_warmth = np.std(warmth_ratios)
    
    # Calculate volatility as the standard deviation of differences between adjoining keyframes
    if len(warmth_ratios) > 1:
        differences = np.diff(warmth_ratios)
        volatility = np.std(differences)
    else:
        volatility = 0  # Not enough keyframes to calculate volatility
    
    return mean_warmth, std_dev_warmth, volatility

#############################################################################################################
##### HSV 

def calculate_hsv_statistics(keyframes_dir):
    """
    Calculate HSV statistics for all images in a directory.
    Returns the mean of means, mean of standard deviations, and their volatility based on differences.
    """
    means_value = []
    means_saturation = []
    sds_value = []  # Changed from std_devs to sds
    sds_saturation = []  # Changed from std_devs to sds

    # List all files in the specified directory
    for filename in sorted(os.listdir(keyframes_dir)):
        if filename.lower().endswith('.png'):  # Check for image files
            filepath = os.path.join(keyframes_dir, filename)
            image = cv2.imread(filepath)  # Read the image
            
            if image is not None:
                # Convert the image to HSV color space
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Extract the channels
                saturation_channel = hsv_image[:, :, 1]
                value_channel = hsv_image[:, :, 2]

                # Calculate means and standard deviations
                mean_value = np.mean(value_channel)
                mean_saturation = np.mean(saturation_channel)
                sd_value = np.std(value_channel)  # Changed from std_dev_value to sd_value
                sd_saturation = np.std(saturation_channel)  # Changed from std_dev_saturation to sd_saturation

                # Append values to the lists
                means_value.append(mean_value)
                means_saturation.append(mean_saturation)
                sds_value.append(sd_value)  # Append to sds
                sds_saturation.append(sd_saturation)  # Append to sds

    # Calculate overall statistics
    mean_of_means_value = np.mean(means_value) if means_value else 0
    mean_of_means_saturation = np.mean(means_saturation) if means_saturation else 0
    mean_of_sds_value = np.mean(sds_value) if sds_value else 0  # Changed from mean_of_std_devs_value
    mean_of_sds_saturation = np.mean(sds_saturation) if sds_saturation else 0  # Changed from mean_of_std_devs_saturation

    # Standard deviations of the means
    sd_of_means_value = np.std(means_value) if means_value else 0
    sd_of_means_saturation = np.std(means_saturation) if means_saturation else 0
    sd_sds_value = np.std(sds_value) if sds_value else 0  # Changed from sd_std_dev_value
    sd_sds_saturation = np.std(sds_saturation) if sds_saturation else 0  # Changed from sd_std_dev_saturation

    # Volatility (based on the differences)
    if len(means_value) > 1:
        value_differences = np.diff(means_value)
        volatility_means_value = np.std(value_differences)
    else:
        volatility_means_value = 0  # Not enough means to calculate volatility

    if len(means_saturation) > 1:
        saturation_differences = np.diff(means_saturation)
        volatility_means_saturation = np.std(saturation_differences)
    else:
        volatility_means_saturation = 0  # Not enough means to calculate volatility

    if len(sds_value) > 1:
        sd_differences_value = np.diff(sds_value)
        volatility_sd_value = np.std(sd_differences_value)  # Changed from volatility_std_dev_value
    else:
        volatility_sd_value = 0  # Not enough sd to calculate volatility

    if len(sds_saturation) > 1:
        sd_differences_saturation = np.diff(sds_saturation)
        volatility_sd_saturation = np.std(sd_differences_saturation)  # Changed from volatility_std_dev_saturation
    else:
        volatility_sd_saturation = 0  # Not enough sd to calculate volatility

    return (
        mean_of_means_value, mean_of_means_saturation,
        mean_of_sds_value, mean_of_sds_saturation,
        sd_of_means_value, sd_of_means_saturation,
        sd_sds_value, sd_sds_saturation,
        volatility_means_value, volatility_means_saturation,
        volatility_sd_value, volatility_sd_saturation
    )

def get_hue_circular_variance(image):
    #image = deepcopy(image)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract hue values
    hue_values = hsv_image[:, :, 0]
    
    # Calculate circular variance using scipy's circvar
    hue_radians = np.deg2rad(hue_values * 2)  # Multiply by 2 to scale to 0-360
    
    circular_variance = circvar(hue_radians, high=np.pi, low=-np.pi)
    
    return circular_variance

def calculate_hue_circular_variance_statistics(keyframe_dir):
    hue_circular_variance_val = []
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            hue_circular_variance_val.append(get_hue_circular_variance(image))

    mean = np.mean(hue_circular_variance_val)
    sd = np.std(hue_circular_variance_val)

    if len(hue_circular_variance_val) > 1:
        differences = np.diff(hue_circular_variance_val)
        volatility = np.std(differences)
    else:
        volatility = 0
    
    return mean, sd, volatility

#############################################################################################################
#### Emotion #################

def extract_hsv_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate mean, standard deviation, and variance for each channel: Hue, Saturation, and Value (Brightness)
    hue_mean = np.mean(hsv_image[:, :, 0])
    hue_std = np.std(hsv_image[:, :, 0])
    
    saturation_mean = np.mean(hsv_image[:, :, 1])
    saturation_std = np.std(hsv_image[:, :, 1])
    
    value_mean = np.mean(hsv_image[:, :, 2])  # Brightness
    value_std = np.std(hsv_image[:, :, 2])
    
    return {
        'hue_mean': hue_mean,
        'hue_std': hue_std,
        'saturation_mean': saturation_mean,
        'saturation_std': saturation_std,
        'value_mean': value_mean,
        'value_std': value_std
    }

def calculate_valence(hsv_features):
    """
    This is a simplified valence function based on image color properties.
    Valence often refers to how pleasant/unpleasant or happy/sad the image makes people feel.
    We assume that brighter and more saturated colors have higher valence.
    """
    valence = 0.5 * hsv_features['value_mean'] + 0.3 * hsv_features['saturation_mean'] - 0.2 * hsv_features['hue_mean']
    return valence

# Main function to calculate valence for an input image
def calculate_image_valence(image):
    #image = deepcopy(image)
    
    # Extract color features
    hsv_features = extract_hsv_features(image)
    
    # Calculate valence using the predefined formula
    valence = calculate_valence(hsv_features)
    
    return valence

def calculate_valence_statistics(keyframe_dir):
    valences = []
    # Calculate vallence for each keyframe
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            valences.append(calculate_image_valence(image))

    # Calculate mean and standard deviation
    mean_valence = np.mean(valences)
    sd_valence = np.std(valences)
    
    # Calculate volatility as the standard deviation of differences between adjoining keyframes
    if len(valences) > 1:
        differences = np.diff(valences)
        volatility = np.std(differences)
    else:
        volatility = 0  # Not enough keyframes to calculate volatility
    
    return mean_valence, sd_valence, volatility

# Function to calculate dominance
def calculate_dominance(hsv_features):
    """
    The dominance calculation is typically related to the image's intensity, contrast, and saturation.
    A higher value for brightness and saturation may indicate a higher dominance value.
    
    This is a simplified approach based on common features used to calculate emotional dominance in images.
    """
    # Adjust the coefficients based on empirical data or your specific use case
    dominance = 0.4 * hsv_features['value_mean'] + 0.4 * hsv_features['saturation_mean'] - 0.2 * hsv_features['hue_mean']
    
    return dominance

# Main function to process an image and calculate its dominance score
def get_image_dominance(image):
    #image = deepcopy(image)
    
    # Extract HSV features
    hsv_features = extract_hsv_features(image)
    
    # Calculate dominance based on HSV features
    dominance_score = calculate_dominance(hsv_features)
    
    return dominance_score


def calculate_dominance_statistics(keyframe_dir):
    dominance_val = []
    # Calculate vallence for each keyframe
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            dominance_val.append(get_image_dominance(image))

    # Calculate mean and standard deviation
    mean = np.mean(dominance_val)
    sd = np.std(dominance_val)
    
    # Calculate volatility as the standard deviation of differences between adjoining keyframes
    if len(dominance_val) > 1:
        differences = np.diff(dominance_val)
        volatility = np.std(differences)
    else:
        volatility = 0  # Not enough keyframes to calculate volatility
    
    return mean, sd, volatility

def calculate_arousal(hsv_features):
    """
    The arousal calculation typically relates to brightness, contrast, and saturation.
    Higher saturation and brightness levels can indicate higher arousal.
    
    This is a simplified approach based on common image features used to calculate emotional arousal.
    """
    # Adjust the coefficients based on empirical data or your specific use case
    arousal = 0.5 * hsv_features['value_mean'] + 0.3 * hsv_features['saturation_mean'] + 0.2 * hsv_features['value_std']
    
    return arousal

# Main function to process an image and calculate its arousal score
def get_image_arousal(image):
    #image = deepcopy(image)
    
    # Extract HSV features
    hsv_features = extract_hsv_features(image)
    
    # Calculate arousal based on HSV features
    arousal_score = calculate_arousal(hsv_features)
    
    return arousal_score

# Function to calculate arousal statistics
def calculate_arousal_statistics(keyframe_dir):
    arousal_val = []
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            arousal_val.append(get_image_arousal(image))

    mean = np.mean(arousal_val)
    sd = np.std(arousal_val)

    if len(arousal_val) > 1:
        differences = np.diff(arousal_val)
        volatility = np.std(differences)
    else:
        volatility = 0
    
    return mean, sd, volatility

#############################################################################################################
# colorfulness

def calculate_colorfulness(image):
    """
    Calculate the colorfulness of an image using the Hasler and Süsstrunk (2003) algorithm.
    """
    #image = deepcopy(image)

    # Convert the image to the RGB color space (OpenCV loads images in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split the image into R, G, B channels
    (R, G, B) = cv2.split(image)
    
    # Compute rg = R - G and yb = 0.5 * (R + G) - B
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    
    # Compute the mean and standard deviation of both rg and yb
    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)
    
    # Combine the standard deviations and means of both channels
    std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
    
    # The colorfulness metric
    colorfulness = std_root + (0.3 * mean_root)
    
    return colorfulness

def calculate_colorfulness_statistics(keyframe_dir):
    return calculate_statistics_for_function(keyframe_dir, calculate_colorfulness)

#############################################################################################################
# clarity
def calculate_clarity_statistics(keyframes_folder):
    clarity_ratios = []
    
    # Calculate clarity ratio for each keyframe
    for filename in sorted(os.listdir(keyframes_folder)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframes_folder, filename)
            hsv_image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2HSV)  # load keyframes in hsv color space
            brightness = hsv_image[:, :, 2]
            threshold = 0.7 * 255  # 70% of the highest possible brightness
            clarity_pixels = np.count_nonzero(brightness > threshold)
            total_pixels = brightness.size
            clarity_ratio = clarity_pixels / total_pixels
            clarity_ratios.append(clarity_ratio)
    
    # Calculate mean and standard deviation
    mean_clarity = np.mean(clarity_ratios)
    std_dev_clarity = np.std(clarity_ratios)
    
    # Calculate volatility as the standard deviation of differences between adjoining keyframes
    if len(clarity_ratios) > 1:
        differences = np.diff(clarity_ratios)
        volatility = np.std(differences)
    else:
        volatility = 0  # Not enough keyframes to calculate volatility
    
    return mean_clarity, std_dev_clarity, volatility

#############################################################################################################
# avg wavelet coeficient value

def calculate_wavelet_mean(channel, level=3, wavelet='db1'):
    # Perform wavelet transform at the specified level
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    
    # Flatten the approximation coefficients
    flattened_coeffs = coeffs[0].flatten()
    
    # Flatten and add all detail coefficients
    for detail_level in coeffs[1:]:
        flattened_coeffs = np.hstack((flattened_coeffs, detail_level[0].flatten(), detail_level[1].flatten(), detail_level[2].flatten()))
    
    # Calculate and return the mean of the coefficients
    return np.mean(flattened_coeffs)

def calculate_hsv_wavelet_means(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the Hue, Saturation, and Value channels
    hue_channel = hsv_image[:, :, 0] / 255.0
    saturation_channel = hsv_image[:, :, 1] / 255.0
    value_channel = hsv_image[:, :, 2] / 255.0
    
    # Calculate the mean of wavelet coefficients for each channel
    hue_wavelet_mean = calculate_wavelet_mean(hue_channel)
    saturation_wavelet_mean = calculate_wavelet_mean(saturation_channel)
    value_wavelet_mean = calculate_wavelet_mean(value_channel)
    
    return hue_wavelet_mean, saturation_wavelet_mean, value_wavelet_mean

def calculate_hsv_wavelet_mean_statistics(keyframe_dir):
    hue = []
    saturation = []
    value = []
    hsv = [hue, saturation, value]
        
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = calculate_hsv_wavelet_means(image) # apply function
                for i, val in enumerate(result):
                    hsv[i].append(val)

    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [0,0,0]
    sd = [0,0,0]
    volatility = [0,0,0]
    
    for i, item in enumerate(hsv):
        means[i] = np.mean(item)
        sd[i] = np.std(item)
        if len(item) > 1:
            differences = np.diff(item)
            volatility[i] = np.std(differences)
            
    return means, sd, volatility 

#############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#############################################################################################################

### texture category

#############################################################################################################
### GLCM
def calculate_glcm_features(image):
    #image = deepcopy(image)
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the channels
    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]
    
    # Quantize the hue-, saturation- and value-values to 8-bit values for GLCM (scikit-image works best with 8-bit images)
    hue_quantized = (hue_channel / 255.0 * 15).astype(np.uint8)
    saturation_quantized = (saturation_channel / 255.0 * 15).astype(np.uint8)
    value_quantized = (value_channel / 255.0 * 15).astype(np.uint8)
    
    # Compute the GLCM (Gray Level Co-occurrence Matrix)
    hue_glcm = graycomatrix(hue_quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    saturation_glcm = graycomatrix(saturation_quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    value_glcm = graycomatrix(value_quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    
    # Calculate contrast, correlation, energy and homogeneity from GLCM
    hue_contrast = graycoprops(hue_glcm, 'contrast')[0, 0]
    hue_correlation = graycoprops(hue_glcm, 'correlation')[0, 0]
    hue_energy = graycoprops(hue_glcm, 'energy')[0, 0]
    hue_homogeneity = graycoprops(hue_glcm, 'homogeneity')[0, 0]
    saturation_contrast = graycoprops(saturation_glcm, 'contrast')[0, 0]
    saturation_correlation = graycoprops(saturation_glcm, 'correlation')[0, 0]
    saturation_energy = graycoprops(saturation_glcm, 'energy')[0, 0]
    saturation_homogeneity = graycoprops(saturation_glcm, 'homogeneity')[0, 0]
    value_contrast = graycoprops(value_glcm, 'contrast')[0, 0]
    value_correlation = graycoprops(value_glcm, 'correlation')[0, 0]
    value_energy = graycoprops(value_glcm, 'energy')[0, 0]
    value_homogeneity = graycoprops(value_glcm, 'homogeneity')[0, 0]
    
    return (hue_contrast, hue_correlation, hue_energy, hue_homogeneity), (saturation_contrast, saturation_correlation, saturation_energy, saturation_homogeneity), (value_contrast, value_correlation, value_energy, value_homogeneity)

def calculate_glcm_statistics(keyframe_dir):
    hsv = [
    [[], [], [], []],  # hue: contrast, correlation, energy, homogeneity
    [[], [], [], []],  # saturation: contrast, correlation, energy, homogeneity
    [[], [], [], []]   # value: contrast, correlation, energy, homogeneity
    ]
        
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = calculate_glcm_features(image) # apply function
                for i in range(3):
                    for j in range(4):
                        hsv[i][j].append(result[i][j])

    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []]]
    sd = [
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []]]
    volatility = [
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []]]
    
    for i in range(3):
        for j in range(4):
            means[i][j] = np.mean(hsv[i][j])
            sd[i][j] = np.std(hsv[i][j])
            if len(hsv[i][j]) > 1:
                differences = np.diff(hsv[i][j])
                volatility[i][j] = np.std(differences)
            else:
                volatility[i][j] = 0
            
    return means, sd, volatility    

#############################################################################################################
# Gray Distribution Entropy
def calculate_gray_distribution_entropy(image):
    #image = deepcopy(image)
    
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
    
    # Normalize the gray values to 0-255
    gray_image = (gray_image * 255).astype(np.uint8)
    
    # Calculate the histogram of the grayscale values
    histogram, _ = np.histogram(gray_image, bins=256, range=(0, 256), density=True)
    
    # Calculate the entropy
    entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))  # Add eps to avoid log(0)
    
    return entropy

def calculate_gray_distribution_entropy_statistics(keyframe_dir):
    return calculate_statistics_for_function(keyframe_dir, calculate_gray_distribution_entropy)

#############################################################################################################
# bluriness / laplace variance
def calculate_blurriness_statistics(keyframes_folder):
    blurriness_values = []
    
    # Calculate Laplacian variance for each keyframe to assess blurriness
    for filename in sorted(os.listdir(keyframes_folder)):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(keyframes_folder, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # load keyframes in grayscale
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()  # calculate Laplacian variance
            blurriness_values.append(laplacian_var)
    
    # Calculate mean, median, and standard deviation
    mean_blurriness = np.mean(blurriness_values)
    median_blurriness = np.median(blurriness_values)
    std_dev_blurriness = np.std(blurriness_values)
    
    # Calculate volatility as the standard deviation of differences between adjoining keyframes
    if len(blurriness_values) > 1:
        differences = np.diff(blurriness_values)
        volatility = np.std(differences)
    else:
        volatility = 0  # Not enough keyframes to calculate volatility
    
    return mean_blurriness, median_blurriness, std_dev_blurriness, volatility

#############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#############################################################################################################

### composition category

#############################################################################################################
# low depth of field

def calculate_low_dof_channel(channel, image, mask=None):
    """
    Calculate the Low DOF for a specific channel (hue, saturation, or value).
    This function computes the sharpness for the specified channel using a mask.
    """
    # Apply the mask to the channel
    if mask is not None:
        channel = cv2.bitwise_and(channel, channel, mask=mask)
    
    # Calculate the sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(channel, cv2.CV_64F).var()
    
    return laplacian_var

def calculate_low_dof_hsv(image):
    """
    Calculate the Low Depth of Field (DOF) based on hue, saturation, and value
    by comparing the sharpness in the inner and outer regions of the image.
    """
    # Convert the image to the HSV color space and extract the channels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]
    
    # Define masks for the inner region and the outer region
    height, width = hue_channel.shape
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y) // 2  # Define radius for the central region
    
    # Create a circular mask for the inner region
    inner_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(inner_mask, (center_x, center_y), radius, 255, -1)
    
    # The outer region is the complement of the inner region
    outer_mask = cv2.bitwise_not(inner_mask)
    
    # Calculate the sharpness for hue in the inner and outer regions
    inner_sharpness_hue = calculate_low_dof_channel(hue_channel, image, mask=inner_mask)
    outer_sharpness_hue = calculate_low_dof_channel(hue_channel, image, mask=outer_mask)
    low_dof_hue = outer_sharpness_hue - inner_sharpness_hue
    
    # Calculate the sharpness for saturation in the inner and outer regions
    inner_sharpness_saturation = calculate_low_dof_channel(saturation_channel, image, mask=inner_mask)
    outer_sharpness_saturation = calculate_low_dof_channel(saturation_channel, image, mask=outer_mask)
    low_dof_saturation = outer_sharpness_saturation - inner_sharpness_saturation
    
    # Calculate the sharpness for value (brightness) in the inner and outer regions
    inner_sharpness_value = calculate_low_dof_channel(value_channel, image, mask=inner_mask)
    outer_sharpness_value = calculate_low_dof_channel(value_channel, image, mask=outer_mask)
    low_dof_value = outer_sharpness_value - inner_sharpness_value
    
    return low_dof_hue, low_dof_saturation, low_dof_value

def calculate_low_dof_statistics(keyframe_dir):
    hsv = [[], [], []]
        
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = calculate_low_dof_hsv(image) # apply function
                for i in range(3):
                    hsv[i].append(result[i])

    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [[], [], []]
    sd = [[], [], []]
    volatility = [[], [], []]

    for i in range(3):
        means[i] = np.mean(hsv[i])
        sd[i] = np.std(hsv[i])
        if len(hsv[i]) > 1:
            differences = np.diff(hsv[i])
            volatility[i] = np.std(differences)
        else:
            volatility[i] = 0
            
    return means, sd, volatility   

#############################################################################################################
# rule of thirds
def calculate_rule_of_thirds(image, channel):
    """
    Calculate the average value of a specified channel (saturation or value)
    in the central region (middle rectangle) of an image based on the rule of thirds grid,
    and compare it to the overall image's average.
    """
    # Get image dimensions
    height, width = channel.shape

    # Define the coordinates for the central third region (middle rectangle in rule of thirds)
    x_start = width // 3
    x_end = 2 * width // 3
    y_start = height // 3
    y_end = 2 * height // 3

    # Extract the central third region
    central_region = channel[y_start:y_end, x_start:x_end]

    # Calculate the average in the central region and the overall image
    avg_central = np.mean(central_region)
    avg_overall = np.mean(channel)

    return avg_central

def calculate_rule_of_thirds_saturation_and_value(image):
    """
    Calculate the Rule of Thirds for both Saturation and Value (brightness).
    """
    # Convert the image to HSV color space to access the saturation and value channels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]  # Extract the saturation channel
    value_channel = hsv_image[:, :, 2]       # Extract the value (brightness) channel

    # Calculate Rule of Thirds - Saturation
    avg_saturation_central = calculate_rule_of_thirds(image, saturation_channel)

    # Calculate Rule of Thirds - Value
    avg_value_central = calculate_rule_of_thirds(image, value_channel)

    return avg_saturation_central, avg_value_central

def calculate_rule_of_thirds_statistics(keyframe_dir):
    sv = [[],[]]
        
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = calculate_rule_of_thirds_saturation_and_value(image) # apply function
                for i in range(2):
                        sv[i].append(result[i])

    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [0,0]
    sd = [0,0]
    volatility = [0,0]

    for i in range(2):
        means[i] = np.mean(sv[i])
        sd[i] = np.std(sv[i])
        if len(sv[i]) > 1:
            differences = np.diff(sv[i])
            volatility[i] = np.std(differences)
            
    return means, sd, volatility 
#############################################################################################################
# edge pixels

def calculate_edge_points(image):
    
    # Apply the Canny edge detector
    edges = cv2.Canny(image, 100, 200)
    
    # Calculate the number of edge points
    num_edge_points = np.sum(edges > 0)
    
    return num_edge_points

def calculate_edge_points_statistics(keyframe_dir):
    return calculate_statistics_for_function(keyframe_dir, calculate_edge_points)


#############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#############################################################################################################

### content category

#############################################################################################################
# people

# Number and region of people

def count_people_in_image(image, confidence_threshold=0.5, net=None):
    if net is None:
        # Load the pre-trained MobileNet SSD model and the corresponding configuration file
        net = cv2.dnn.readNetFromCaffe(
            str(asset('deploy.prototxt')),
            str(asset('mobilenet_iter_73000.caffemodel'))
        )
    
    # Load the image and get its dimensions
    #image = deepcopy(image)
    (h, w) = image.shape[:2]

    # Prepare the image as input for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Perform detection
    detections = net.forward()

    # Initialize a counter for people detected
    person_count = 0

    # Initialize int for largest confidence and a list for confidences 
    largest_confidence = 0
    confidences = list()

    # Initialize int for biggest bounding box and lsit for size of bounding boxes
    biggest_bb = 0
    bbs = list()
    largest_bb_ratio = 0
    bbs_image_ratio = 0
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections (confidence threshold can be adjusted)
        if confidence > confidence_threshold:
            # Extract the index of the class label
            idx = int(detections[0, 0, i, 1])

            # If the detected object is a person (class index 15)
            if idx == 15:
                person_count += 1
                
                # update largest confidence if aplicable
                if confidence > largest_confidence:
                    largest_confidence = confidence

                # add confidence to confidences vector
                confidences.append(confidence)

                # Optionally, you can draw a bounding box around the detected person
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence:.2f}', (startX + 10, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # calculate the size of the bounding box
                w_bb, h_bb = endX - startX, endY - startY
                bb_size = w_bb * h_bb
                
                # update largest bounding box if apliccable
                if bb_size > biggest_bb:
                    biggest_bb = bb_size
                    
                # append bounding box size to bbs
                bbs.append(bb_size)

    # calculate area and find ratios
    area = w * h
    largest_bb_ratio = biggest_bb/area
    bbs_image_ratio = np.sum(bbs)/area
    if len(confidences)==0:
        confidences = [0,0]
        
    # Optionally, save the image with detections
    #cv2.imwrite('output_image.jpg', image)

    return person_count, image, largest_confidence, np.mean(confidences), largest_bb_ratio, bbs_image_ratio

def calculate_nb_persons_statistics(keyframe_dir):
    person_counts = [] 
    largest_confidence = []
    mean_conf = []
    largest_bbs = []
    bbs_image_ratio = []
    
    # Load the pre-trained MobileNet SSD model and the corresponding configuration file
    net = cv2.dnn.readNetFromCaffe(
        str(asset('deploy.prototxt')),
        str(asset('mobilenet_iter_73000.caffemodel'))
    )

    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = count_people_in_image(image, net = net) # apply function
                # remove nans if any
                person_counts.append(0 if np.isnan(result[0]) else result[0])
                largest_confidence.append(0 if np.isnan(result[2]) else result[2])
                mean_conf.append(0 if np.isnan(result[3]) else result[3])
                largest_bbs.append(0 if np.isnan(result[4]) else result[4])
                bbs_image_ratio.append(0 if np.isnan(result[5]) else result[5])

    results = [person_counts, largest_confidence, mean_conf, largest_bbs, bbs_image_ratio]
    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [np.mean(L) if len(L) > 0 else 0 for L in results]
    sd = [np.std(L) if len(L) > 0 else 0 for L in results]
    volatility = [np.std(np.diff(L)) if len(L) > 1 else 0 for L in results]

    return means, sd, volatility

#############################################################################################################
# faces

def detect_faces_yunet(image, face_detector=None):
    model_input_size = (300, 300)
    if face_detector is None:
        face_detector = cv2.FaceDetectorYN_create(
            str(asset('face_detection_yunet_2023mar.onnx')),  # Path to the ONNX model
            "",  # No configuration file
            model_input_size,  # Input size of the model
            score_threshold=0.5,  # Confidence threshold
            nms_threshold=0.3,  # Non-maximum suppression threshold
            top_k=5000,  # Keep top 5000 proposals before NMS
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,  # Use OpenCV backend
            target_id=cv2.dnn.DNN_TARGET_CPU  # Run on CPU
        )

    height, width, _ = image.shape
    resized_image = cv2.resize(image, model_input_size)
    faces = face_detector.detect(resized_image)

    face_count = 0
    largest_confidence = 0
    confidences = []
    biggest_bb = 0
    bbs = []

    if faces[1] is not None:
        for face in faces[1]:
            face_count += 1
            box = face[:4].astype(int)
            confidence = face[-1]
            if confidence > largest_confidence:
                largest_confidence = confidence
            confidences.append(confidence)
            box = [
                int(box[0] * width / model_input_size[0]), 
                int(box[1] * height / model_input_size[1]),
                int(box[2] * width / model_input_size[0]), 
                int(box[3] * height / model_input_size[1])
            ]
            bb_size = box[2] * box[3]
            if bb_size > biggest_bb:
                biggest_bb = bb_size
            bbs.append(bb_size)
            
    area = width * height
    largest_bb_ratio = (biggest_bb / area) if area > 0 else 0
    bb_image_ratio = (np.sum(bbs) / area) if area > 0 else 0
    
    if not confidences:
        confidences = [np.nan]  # Ensures mean calculation returns NaN if empty

    return face_count, largest_confidence, np.mean(confidences), largest_bb_ratio, bb_image_ratio

def calculate_nb_faces_statistics(keyframe_dir):
    faces_counts = [] 
    largest_confidence = []
    mean_conf = []
    largest_bbs = []
    bbs_image_ratio = []

    model_input_size = (300, 300)
    face_detector = cv2.FaceDetectorYN_create(
        str(asset('face_detection_yunet_2023mar.onnx')),  # Path to the ONNX model
        "",  # No configuration file
        model_input_size,  # Input size of the model
        score_threshold=0.5,  # Confidence threshold
        nms_threshold=0.3,  # Non-maximum suppression threshold
        top_k=5000,  # Keep top 5000 proposals before NMS
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,  # Use OpenCV backend
        target_id=cv2.dnn.DNN_TARGET_CPU  # Run on CPU
    )
    
    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            if image is not None: 
                result = detect_faces_yunet(image, face_detector = face_detector)
                faces_counts.append(0 if np.isnan(result[0]) else result[0])
                largest_confidence.append(0 if np.isnan(result[1]) else result[1])
                mean_conf.append(0 if np.isnan(result[2]) else result[2])
                largest_bbs.append(0 if np.isnan(result[3]) else result[3])
                bbs_image_ratio.append(0 if np.isnan(result[4]) else result[4])

    results = [faces_counts, largest_confidence, mean_conf, largest_bbs, bbs_image_ratio]
    means = [np.mean(L) if L else np.nan for L in results]
    sd = [np.std(L) if L else np.nan for L in results]
    volatility = [np.std(np.diff(L)) if len(L) > 1 else np.nan for L in results]

    return means, sd, volatility

#############################################################################################################
# hands

def initialize_hand_detector(mode=True, maxHands=50, modelComplexity=1,
                             detectionCon=0.5, trackCon=0.5):
    """
    Initializes the MediaPipe Hands solution.
    Returns the hands object and drawing utilities.
    """
    mp_hands = mp.solutions.hands
    hand_detector = mp_hands.Hands(
        static_image_mode=mode,
        max_num_hands=maxHands,
        model_complexity=modelComplexity,
        min_detection_confidence=detectionCon,
        min_tracking_confidence=trackCon)
    return hand_detector


def find_hands(img, hand_detector):
    """
    Processes an image and finds hands.
    Returns the processed image and the results.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(img_rgb)

    return results


def find_positions(img, results, handNo=0, draw=True):
    """
    Finds the positions of landmarks on the specified hand.
    Returns a list of landmark positions [id, x, y].
    """
    if not results.multi_hand_landmarks:
        return None

    h, w, c = img.shape
    hands = []
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hands.append([])
        for landmark_no, landmark in enumerate(hand_landmarks.landmark):
            if landmark.x <= 1 and landmark.x >= 0 and landmark.y <= 1 and landmark.y >= 0:
                hands[hand_no].append((landmark.x*w, landmark.y*h))

    return hands


def hand_number_and_ratio(img, draw=False, hand_detector=None):
    if hand_detector is None:
        hand_detector = initialize_hand_detector(mode=True)

    # Detect hands in the image
    results = find_hands(img, hand_detector)

    # Get landmark positions
    hands = find_positions(img, results, draw=True)
    
    max_area = 0
    hands_bb = []
    
    if not hands:
        return 0, 0, 0

    all_hands = []
    for hand in hands:
        all_hands += hand
        for i, p in enumerate(hand):
            if i == 0:
                bbox_dims = [p[0], p[0], p[1], p[1]]
            else:
                if p[0] < bbox_dims[0]:
                    bbox_dims[0] = p[0]
                if p[0] > bbox_dims[1]:
                    bbox_dims[1] = p[0]
                if p[1] < bbox_dims[2]:
                    bbox_dims[2] = p[1]
                if p[1] > bbox_dims[3]:
                    bbox_dims[3] = p[1]
        width = bbox_dims[1] - bbox_dims[0]
        height = bbox_dims[3] - bbox_dims[2]
        area = width*height
        hands_bb.append(area)
        if area > max_area:
            max_area = area
            
        if draw:
            rect = patches.Rectangle((bbox_dims[0], bbox_dims[2]), width, height, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
    if draw:
        xs = [hand[0] for hand in all_hands]
        ys = [hand[1] for hand in all_hands]

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.scatter(xs, ys, s=4)
    
    h, w, c = img.shape
    largest_hand_to_screen_ratio = max_area/(h*w)
    sum_hand_to_screen_ratio = sum(hands_bb)/(h*w)
    number_of_hands = len(hands)

    return number_of_hands, largest_hand_to_screen_ratio, sum_hand_to_screen_ratio

def calculate_nb_hands_statistics(keyframe_dir):
    hand_counts = [] 
    largest_bbs = []
    bbs_image_ratio = []

    # Initialize hand detector here
    hand_detector = initialize_hand_detector()

    for filename in sorted(os.listdir(keyframe_dir)):
        if filename.lower().endswith('.png'): 
            filepath = os.path.join(keyframe_dir, filename)
            image = cv2.imread(filepath)
            
            # Calculate the result using the passed function and append to the list
            if image is not None:  # Ensure image is loaded successfully
                result = hand_number_and_ratio(image, hand_detector=hand_detector) # apply function
                hand_counts.append(result[0])
                largest_bbs.append(result[1])
                bbs_image_ratio.append(result[2])

                del image  # Explicitly delete image to free memory
                gc.collect()  # Trigger garbage collection

    results = [hand_counts, largest_bbs, bbs_image_ratio]
    # Calculate means, standard deviation and volatility as the standard deviation of differences between adjoining results
    means = [np.mean(L) if len(L) > 0 else 0 for L in results]
    sd = [np.std(L) if len(L) > 0 else 0 for L in results]
    volatility = [np.std(np.diff(L)) if len(L) > 1 else 0 for L in results]

    return means, sd, volatility

#############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#############################################################################################################

