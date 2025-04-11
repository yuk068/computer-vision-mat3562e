import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, generic_filter
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.ndimage import convolve
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.signal import convolve2d
import warnings

GRADIENT_KERNELS = {
    'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'sobel_y': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'prewitt_y': np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    'roberts_x': np.array([[1, 0], [0, 1]]),
    'roberts_y': np.array([[0, -1], [-1, 0]])
}

# ==============================================================================
# == I/O, Display, and Basic Utilities ==
# ==============================================================================

def read_image(path, mode='RGB'):
    """
    Reads an image from a specified path and converts it to the desired color mode.

    Args:
        path (str): The file path to the image.
        mode (str): The target color mode. Options: 'RGB' (default), 'GRAYSCALE',
                    'HSV', 'LAB', 'CMY'. Case-insensitive.

    Returns:
        np.ndarray or None: The loaded image as a NumPy array in the specified
                            color space, or None if the image cannot be read.
                            RGB, HSV, LAB, CMY are 3-channel, GRAYSCALE is 1-channel.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        warnings.warn(f"Could not read image from path: {path}")
        return None

    mode = mode.upper()
    if mode == 'GRAYSCALE':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif mode == 'RGB':
        # OpenCV reads in BGR, convert to RGB for standard consistency
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mode == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif mode == 'CMY':
        # Convert BGR to RGB first, then calculate CMY
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = 255 - img_rgb
    else:
        warnings.warn(f"Unsupported mode: {mode}. Returning RGB image.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Default to RGB if mode unknown

    return img

def read_image_from_txt(path):
    """
    Reads image data from a text file where pixel values are space-separated.

    Args:
        path (str): The file path to the text file.

    Returns:
        np.ndarray: The image data as a NumPy array of uint8 type.
    """
    data = np.loadtxt(path)
    return data.astype(np.uint8)

def show_image(img, cmap=None, size=(6, 6)):
    """
    Displays an image using matplotlib.

    Args:
        img (np.ndarray): The image to display.
        cmap (str, optional): The colormap to use for grayscale images.
                               Defaults to None (matplotlib default).
        size (tuple): The figure size (width, height) in inches. Defaults to (6, 6).
    """
    plt.figure(figsize=size)
    if len(img.shape) == 2:
        # Display grayscale image
        plt.imshow(img, cmap='gray' if cmap is None else cmap)
    else:
        # Display color image
        plt.imshow(img, cmap=cmap)
    plt.axis('off') # Hide axes
    plt.show()

def show_all_channels(img):
    """
    Displays each channel of a multi-channel image separately.
    If the image is grayscale, it displays the single channel.

    Args:
        img (np.ndarray): The input image (can be grayscale or color).
    """
    if len(img.shape) == 2:
        # If grayscale, just show the single channel
        show_image(img, cmap='gray')
        return

    num_channels = img.shape[2]
    fig, axs = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
    # Handle case where there's only one channel after potentially incorrect input
    if num_channels == 1:
        axs = [axs]
        
    for i in range(num_channels):
        axs[i].imshow(img[:, :, i], cmap='gray') # Show each channel in grayscale
        axs[i].set_title(f'Channel {i+1}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# == Color Space and Normalization Functions ==
# ==============================================================================

def convert_color(img, mode='RGB'):
    """
    Converts an image (assumed to be RGB) to a different color space.

    Args:
        img (np.ndarray): The input image, assumed to be in RGB format.
        mode (str): The target color mode. Options: 'GRAYSCALE', 'HSV', 'LAB', 'CMY'.
                    Case-insensitive. If mode is 'RGB' or unsupported, returns the original image.

    Returns:
        np.ndarray: The image converted to the specified color space.
    """
    mode = mode.upper()
    if mode == 'GRAYSCALE':
        # Convert RGB to Grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if mode == 'HSV':
        # Convert RGB to HSV
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if mode == 'LAB':
        # Convert RGB to LAB
        return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    if mode == 'CMY':
        # Calculate CMY from RGB
        return 255 - img
    # Return original if mode is RGB or not recognized
    return img

def normalize(img):
    """
    Normalizes image pixel values from [0, 255] to [0.0, 1.0].

    Args:
        img (np.ndarray): The input image (uint8).

    Returns:
        np.ndarray: The normalized image (float32).
    """
    return img.astype(np.float32) / 255.0

def denormalize(img):
    """
    Denormalizes image pixel values from [0.0, 1.0] back to [0, 255].
    Clips values to ensure they are within the valid range.

    Args:
        img (np.ndarray): The input normalized image (float).

    Returns:
        np.ndarray: The denormalized image (uint8).
    """
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def true_grayscale_normalize(img):
    """
    Converts an RGB image to grayscale and then normalizes it to [0.0, 1.0].

    Args:
        img (np.ndarray): The input RGB image.

    Returns:
        np.ndarray: The normalized grayscale image (float32).
    """
    gray = convert_color(img, 'GRAYSCALE')
    return gray.astype(np.float32) / 255.0

def gamma_correction(img, gamma=1.0):
    """
    Applies gamma correction to an image.

    Args:
        img (np.ndarray): Input image (uint8 or float).
        gamma (float): Gamma value (>1 darkens, <1 lightens).

    Returns:
        np.ndarray: Gamma corrected image.
    """
    if img.dtype != np.float32:
        img = normalize(img)
    corrected = np.power(img, gamma)
    return denormalize(corrected)

def apply_color_correction(img, factors=(0.299, 0.587, 0.114)):
    """
    Applies simple color correction using weighted RGB factors
    (typically aligned with human brightness perception).

    Args:
        img (np.ndarray): RGB image.
        factors (tuple): Tuple of 3 values (R, G, B scale factors).

    Returns:
        np.ndarray: Color corrected image.
    """
    img = img.astype(np.float32)
    corrected = img * np.array(factors, dtype=np.float32)
    return np.clip(corrected, 0, 255).astype(np.uint8)

# ==============================================================================
# == Point Processing and Histogram Functions ==
# ==============================================================================

def point_process(img, func, normalize_output=False):
    """
    Applies a function pixel-wise to an image.

    Args:
        img (np.ndarray): The input image.
        func (callable): A function that takes a single pixel value (or array of values)
                         and returns a transformed value (or array).
        normalize_output (bool): If True, clips the output to [0, 1]. If False (default),
                                 clips to [0, 255] and converts to uint8.

    Returns:
        np.ndarray: The image after applying the point process.
    """
    out = func(img)
    if normalize_output:
        # Clip to [0, 1] for normalized outputs
        return np.clip(out, 0.0, 1.0)
    else:
        # Clip to [0, 255] and convert to uint8 for standard image range
        return np.clip(out, 0, 255).astype(np.uint8)

def show_histogram(img, separate=False):
    """
    Displays the histogram of an image.

    Args:
        img (np.ndarray): The input image.
        separate (bool): If True and the image is color, displays histograms
                         for each color channel separately. Otherwise, displays
                         a single histogram (for grayscale or combined color).
                         Defaults to False.
    """
    plt.figure(figsize=(6, 4))
    min_val = np.min(img)
    max_val = np.max(img)

    if len(img.shape) == 2 or not separate:
        # Grayscale image or combined histogram for color image
        plt.hist(img.ravel(), bins=256, range=(min_val, max_val), color='black', histtype='step')
    else:
        # Separate histograms for RGB channels
        colors = ['r', 'g', 'b']
        for i, c in enumerate(colors):
            channel_data = img[:, :, i]
            channel_min = np.min(channel_data)
            channel_max = np.max(channel_data)
            plt.hist(channel_data.ravel(), bins=256, range=(channel_min, channel_max), color=c, alpha=0.7, label=f'Channel {c.upper()}', histtype='step')
        plt.legend()
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def show_cdf(img, separate=False):
    """
    Displays the Cumulative Distribution Function (CDF) of an image's histogram.

    Args:
        img (np.ndarray): The input image.
        separate (bool): If True and the image is color, displays CDFs
                         for each color channel separately. Otherwise, displays
                         a single CDF (for grayscale or combined color).
                         Defaults to False.
    """
    plt.figure(figsize=(6, 4))

    def compute_cdf(channel):
        """Helper function to compute CDF for a single channel."""
        min_val = np.min(channel)
        max_val = np.max(channel)
        # Handle case where min == max (flat image)
        if min_val == max_val:
            hist = np.zeros(256, dtype=np.int64)
            hist[int(min_val)] = channel.size
            bins = np.arange(257)
        else:
             hist, bins = np.histogram(channel.ravel(), bins=256, range=(min_val, max_val))

        cdf = hist.cumsum()
        # Avoid division by zero if CDF is flat
        if cdf[-1] == 0:
             return bins[:-1], np.zeros_like(cdf, dtype=float)

        cdf_normalized = cdf / cdf[-1]
        return bins[:-1], cdf_normalized

    if len(img.shape) == 2 or not separate:
        # Grayscale image or combined CDF for color image
        x, y = compute_cdf(img)
        plt.plot(x, y, color='black')
    else:
        # Separate CDFs for RGB channels
        colors = ['r', 'g', 'b']
        for i, c in enumerate(colors):
            x, y = compute_cdf(img[:, :, i])
            plt.plot(x, y, color=c, label=f'Channel {c.upper()}')
        plt.legend()

    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.5)
    plt.ylim(0, 1.05) # Ensure y-axis goes from 0 to 1
    plt.show()


def show_histogram_with_median(img):
    """
    Displays the histogram of an image and marks the median value.

    Args:
        img (np.ndarray): The input image (grayscale or color). If color,
                          the histogram is computed over all pixels.

    Returns:
        float: The median pixel value of the image.
    """
    plt.figure(figsize=(6, 4))
    values = img.ravel() # Flatten the image to get all pixel values
    min_val = np.min(values)
    max_val = np.max(values)
    plt.hist(values, bins=256, range=(min_val, max_val), color='gray', alpha=0.7)

    # Calculate and plot median
    median_val = np.median(values)
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

    plt.title('Histogram with Median')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()
    return median_val

def histogram_match(source, reference):
    """
    Matches the histogram of a source image to a reference image.
    Assumes images are uint8, grayscale.

    Args:
        source (np.ndarray): The source image (grayscale, uint8).
        reference (np.ndarray): The reference image (grayscale, uint8).

    Returns:
        np.ndarray: The source image with its histogram matched to the reference (uint8).
    """
    # Ensure images are grayscale
    if len(source.shape) > 2:
        warnings.warn("Source image is not grayscale. Converting to grayscale.")
        source = convert_color(source, 'GRAYSCALE')
    if len(reference.shape) > 2:
        warnings.warn("Reference image is not grayscale. Converting to grayscale.")
        reference = convert_color(reference, 'GRAYSCALE')

    src_flat = source.ravel()
    ref_flat = reference.ravel()

    # Compute histograms and CDFs for source and reference
    src_hist, _ = np.histogram(src_flat, bins=256, range=(0, 256), density=True)
    ref_hist, _ = np.histogram(ref_flat, bins=256, range=(0, 256), density=True)

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    # Create the lookup table (LUT)
    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_idx in range(256):
        # Find the closest intensity in reference CDF for the current source CDF value
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
            ref_idx += 1
        lut[src_idx] = ref_idx

    # Apply the LUT to the source image
    matched_img = lut[source]
    return matched_img

# ==============================================================================
# == Thresholding and Basic Image Operations ==
# ==============================================================================

def threshold(img, value):
    """
    Applies binary thresholding to an image.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        value (int or float): The threshold value. Pixels above this value become 1 (or 255),
                              others become 0.

    Returns:
        np.ndarray: The thresholded binary image (uint8, values 0 or 1).
    """
    # Convert to grayscale if necessary for consistent thresholding
    if len(img.shape) == 3:
        img_proc = convert_color(img, 'GRAYSCALE')
    else:
        img_proc = img

    # Apply threshold: pixels > value become 1, others 0
    binary_img = (img_proc > value).astype(np.uint8)
    return binary_img

def color_filter(img, channel, func):
    """
    Applies a function to a specific color channel of an image.

    Args:
        img (np.ndarray): The input color image (assumed 3 channels, e.g., RGB).
        channel (int): The index of the channel to modify (0, 1, or 2).
        func (callable): A function to apply to the selected channel. It should
                         take the channel data (2D array) and return the modified channel data.

    Returns:
        np.ndarray: The image with the specified channel modified, clipped to [0, 255] uint8.
                    Returns the original image if it's grayscale.
    """
    if len(img.shape) == 2:
        warnings.warn("Input image is grayscale. Color filter cannot be applied.")
        return img
    if channel < 0 or channel >= img.shape[2]:
         warnings.warn(f"Invalid channel index {channel}. Image has {img.shape[2]} channels.")
         return img

    out = img.copy()
    # Apply the function to the selected channel and ensure output is valid
    modified_channel = func(out[:, :, channel])
    out[:, :, channel] = np.clip(modified_channel, 0, 255).astype(np.uint8)
    return out

def apply_mask(img, mask):
    """
    Applies a binary mask to an image. Regions where mask is 0 become black.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        mask (np.ndarray): The binary mask (0s and 1s or 0s and 255s).
                           Must have the same height and width as the image.
                           If grayscale, it will be broadcast to color images.

    Returns:
        np.ndarray: The masked image (uint8).
    """
    # Normalize mask to 0s and 1s
    mask_norm = mask.astype(np.float32) / mask.max() if mask.max() > 0 else mask.astype(np.float32)

    # Ensure mask has the same number of dimensions as image for broadcasting
    if len(mask_norm.shape) == 2 and len(img.shape) == 3:
        mask_norm = mask_norm[:, :, np.newaxis] # Add channel dimension

    # Apply mask by element-wise multiplication
    masked_img = (img.astype(np.float32) * mask_norm).astype(np.uint8)
    return masked_img


def invert_binary(img):
    """
    Inverts a binary image (0 becomes 1/255, 1/255 becomes 0).

    Args:
        img (np.ndarray): The input binary image (containing values 0 and 1, or 0 and 255).

    Returns:
        np.ndarray: The inverted binary image (same type and max value as input).
    """
    max_val = img.max()
    if max_val <= 1:
        # Handles masks with 0 and 1
        return 1 - img
    else:
        # Handles masks with 0 and 255
        return 255 - img

# ==============================================================================
# == Image Quality Metrics ==
# ==============================================================================

def compute_psnr(img1, img2):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image (must have same shape as img1).

    Returns:
        float or None: The PSNR value, or None if shapes mismatch.
                       Higher PSNR generally indicates better quality/similarity.
    """
    if img1.shape != img2.shape:
        warnings.warn("Images must have the same shape to compute PSNR.")
        return None
    # Determine the data range (max possible pixel value)
    data_range = np.iinfo(img1.dtype).max if np.issubdtype(img1.dtype, np.integer) else img1.max() - img1.min()
    if data_range == 0: # Avoid division by zero if image is flat
         return float('inf') # Or consider returning a very large number or None

    return peak_signal_noise_ratio(img1, img2, data_range=data_range)

def compute_ssim(img1, img2):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image (must have same shape as img1).

    Returns:
        float or None: The SSIM value (between -1 and 1), or None if shapes mismatch.
                       Value closer to 1 indicates higher similarity.
    """
    if img1.shape != img2.shape:
        warnings.warn("Images must have the same shape to compute SSIM.")
        return None

    # Determine the data range
    data_range = np.iinfo(img1.dtype).max if np.issubdtype(img1.dtype, np.integer) else img1.max() - img1.min()
    if data_range == 0: # Handle flat images
        return 1.0 if np.array_equal(img1, img2) else 0.0 # Or another appropriate value

    if len(img1.shape) == 2:
        # Grayscale SSIM
        return structural_similarity(img1, img2, data_range=data_range)
    else:
        # Color SSIM - Use multichannel=True
        # Note: skimage<0.19 requires channel_axis=-1 instead of multichannel=True
        try:
             # For scikit-image >= 0.19
            return structural_similarity(img1, img2, multichannel=True, data_range=data_range)
        except TypeError:
             # For scikit-image < 0.19
             return structural_similarity(img1, img2, channel_axis=-1, data_range=data_range)


def compute_mse(img1, img2):
    """
    Computes the Mean Squared Error (MSE) between two images.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image (must have same shape as img1).

    Returns:
        float or None: The MSE value, or None if shapes mismatch.
                       Lower MSE indicates better similarity.
    """
    if img1.shape != img2.shape:
        warnings.warn("Images must have the same shape to compute MSE.")
        return None
    # Calculate MSE using float32 to avoid overflow issues
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return mse

# ==============================================================================
# == Convolution and Filtering Functions ==
# ==============================================================================

def convolve(img, kernel, pad=0, stride=1):
    """
    Performs 2D convolution on a grayscale image using basic loops.
    Converts color images to grayscale first.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        kernel (np.ndarray): The convolution kernel (2D array).
        pad (int): Amount of zero-padding to add around the image border. Defaults to 0.
        stride (int): The step size for moving the kernel across the image. Defaults to 1.

    Returns:
        np.ndarray: The convolved image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    k_h, k_w = kernel.shape
    i_h, i_w = img_gray.shape

    # Apply padding
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    else:
        img_padded = img_gray

    p_h, p_w = img_padded.shape

    # Calculate output dimensions
    out_h = (p_h - k_h) // stride + 1
    out_w = (p_w - k_w) // stride + 1

    # Initialize output array
    out = np.zeros((out_h, out_w), dtype=np.float64) # Use float64 for intermediate sums

    # Perform convolution using loops
    for y in range(out_h):
        for x in range(out_w):
            # Extract the region of interest
            row_start = y * stride
            row_end = row_start + k_h
            col_start = x * stride
            col_end = col_start + k_w
            region = img_padded[row_start:row_end, col_start:col_end]
            # Apply kernel (element-wise multiplication and sum)
            out[y, x] = np.sum(region * kernel)

    # Clip and convert back to uint8
    return np.clip(out, 0, 255).astype(np.uint8)


def fast_convolve(img, kernel, pad=0, stride=1):
    """
    Performs 2D convolution using scipy.signal.convolve2d for better performance.
    Converts color images to grayscale first. Handles padding and stride.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        kernel (np.ndarray): The convolution kernel (2D array).
        pad (int): Amount of zero-padding to add around the image border. Defaults to 0.
        stride (int): The step size for moving the kernel across the image. Defaults to 1.

    Returns:
        np.ndarray: The convolved image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Apply padding
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    else:
        img_padded = img_gray

    # Perform convolution using scipy's optimized function
    # 'valid' mode means no padding is applied by convolve2d itself (we handled it)
    # and the output size is reduced accordingly.
    conv = convolve2d(img_padded, kernel, mode='valid')

    # Apply stride by subsampling the result
    if stride > 1:
        conv = conv[::stride, ::stride]

    # Clip and convert back to uint8
    return np.clip(conv, 0, 255).astype(np.uint8)


def gaussian_blur(img, size=3, strength=1.0, pad=0, stride=1):
    """
    Applies Gaussian blur using scipy.ndimage.gaussian_filter.
    Converts color images to grayscale first. Handles padding and stride.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        size (int): Base size parameter influencing the sigma of the Gaussian. Defaults to 3.
        strength (float): Multiplier for the sigma value (sigma = size * strength). Defaults to 1.0.
        pad (int): Amount of padding (mode 'reflect') before filtering. Defaults to 0.
        stride (int): Subsampling step after blurring. Defaults to 1.

    Returns:
        np.ndarray: The blurred image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Apply padding (using reflection is common for blurring)
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        img_padded = img_gray

    # Calculate sigma for the Gaussian filter
    sigma = size * strength
    # Apply Gaussian filter
    blurred = gaussian_filter(img_padded, sigma=sigma)

    # Apply stride by subsampling
    if stride > 1:
        blurred = blurred[::stride, ::stride]

    # Clip and convert back to uint8
    return np.clip(blurred, 0, 255).astype(np.uint8)


def median_blur(img, size=3, strength=1.0, pad=0, stride=1):
    """
    Applies Median blur using scipy.ndimage.median_filter. Effective for salt-and-pepper noise.
    Converts color images to grayscale first. Handles padding and stride.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        size (int): Base size of the median filter window. Defaults to 3.
        strength (float): Multiplier for the window size. Final size is odd. Defaults to 1.0.
        pad (int): Amount of padding (mode 'reflect') before filtering. Defaults to 0.
        stride (int): Subsampling step after blurring. Defaults to 1.

    Returns:
        np.ndarray: The filtered image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Apply padding
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        img_padded = img_gray

    # Calculate final filter size, ensuring it's an odd integer >= 1
    final_size = max(1, int(size * strength))
    if final_size % 2 == 0: # Ensure size is odd
        final_size += 1

    # Apply median filter
    filtered = median_filter(img_padded, size=final_size)

    # Apply stride by subsampling
    if stride > 1:
        filtered = filtered[::stride, ::stride]

    # Clip and convert back to uint8
    return np.clip(filtered, 0, 255).astype(np.uint8)


def outlier_filter(img, size=3, strength=1.0, pad=0, stride=1):
    """
    Applies a simple outlier removal filter using scipy.ndimage.generic_filter.
    Replaces pixels that differ significantly from their neighborhood median.
    Converts color images to grayscale first. Can be slow for large images/sizes.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        size (int): Base size of the filter window. Defaults to 3.
        strength (float): Multiplier for the window size. Final size is odd. Defaults to 1.0.
        pad (int): Amount of padding (mode 'reflect') before filtering. Defaults to 0.
        stride (int): Subsampling step after filtering. Defaults to 1.

    Returns:
        np.ndarray: The filtered image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Apply padding
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        img_padded = img_gray

    # Calculate final filter size, ensuring it's an odd integer >= 1
    final_size = max(1, int(size * strength))
    if final_size % 2 == 0: # Ensure size is odd
        final_size += 1

    def remove_outliers(window):
        """Function for generic_filter: replace center if it's an outlier."""
        center_index = len(window) // 2
        center_pixel = window[center_index]
        median_val = np.median(window)
        std_dev = np.std(window)
        # If standard deviation is very small, avoid division by zero or extreme sensitivity
        if std_dev < 1e-5:
             return center_pixel # Keep original pixel if neighborhood is flat
        # Replace if the difference from median is large relative to std dev
        # (Here using 1 standard deviation as threshold, could be parameterized)
        if abs(center_pixel - median_val) > std_dev:
            return median_val
        else:
            return center_pixel

    # Apply the generic filter with the outlier removal function
    filtered = generic_filter(img_padded, remove_outliers, size=final_size, mode='reflect')

    # Apply stride by subsampling
    if stride > 1:
        filtered = filtered[::stride, ::stride]

    # Clip and convert back to uint8
    return np.clip(filtered, 0, 255).astype(np.uint8)


def mean_blur(img, size=3, strength=1.0, pad=0, stride=1):
    """
    Applies Mean (box) blur using scipy.ndimage.uniform_filter.
    Converts color images to grayscale first. Handles padding and stride.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        size (int): Base size of the mean filter window. Defaults to 3.
        strength (float): Multiplier for the window size. Final size is odd. Defaults to 1.0.
        pad (int): Amount of padding (mode 'reflect') before filtering. Defaults to 0.
        stride (int): Subsampling step after blurring. Defaults to 1.

    Returns:
        np.ndarray: The blurred image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Apply padding
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        img_padded = img_gray

    # Calculate final filter size, ensuring it's >= 1
    final_size = max(1, int(size * strength))
    # Note: Uniform filter does not require odd size, but consistency might be desired
    # if final_size % 2 == 0: final_size += 1 # Optional: force odd size

    # Apply uniform (mean) filter
    blurred = uniform_filter(img_padded, size=final_size, mode='reflect')

    # Apply stride by subsampling
    if stride > 1:
        blurred = blurred[::stride, ::stride]

    # Clip and convert back to uint8
    return np.clip(blurred, 0, 255).astype(np.uint8)


def fast_outlier_filter(img, size=3, strength=1.0, pad=0, stride=1, threshold_factor=1.0):
    """
    Applies a faster outlier removal filter based on comparing pixels to the
    local median +/- threshold * local standard deviation.
    Uses optimized median and uniform filters. Converts color images to grayscale.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        size (int): Base size of the filter window. Defaults to 3.
        strength (float): Multiplier for the window size. Final size is odd. Defaults to 1.0.
        pad (int): Amount of padding (mode 'reflect') before filtering. Defaults to 0.
        stride (int): Subsampling step after filtering. Defaults to 1.
        threshold_factor (float): Multiplier for standard deviation to define outlier threshold.
                                  Defaults to 1.0. Higher values are less aggressive.

    Returns:
        np.ndarray: The filtered image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE').astype(np.float32) # Use float for calculations
    else:
        img_gray = img.astype(np.float32) # Use float for calculations

    # Apply padding
    if pad > 0:
        img_padded = np.pad(img_gray, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        img_padded = img_gray

    # Calculate final filter size, ensuring it's an odd integer >= 1
    final_size = max(1, int(size * strength))
    if final_size % 2 == 0: # Ensure size is odd for median filter
        final_size += 1

    # Calculate local median
    med = median_filter(img_padded, size=final_size, mode='reflect')

    # Calculate local standard deviation (approximation using uniform filter)
    # Variance = E[X^2] - (E[X])^2. We use median instead of mean E[X] here.
    # std = sqrt(mean((pixel - median)^2))
    abs_diff_sq = (img_padded - med)**2
    mean_abs_diff_sq = uniform_filter(abs_diff_sq, size=final_size, mode='reflect')
    std_dev = np.sqrt(mean_abs_diff_sq)

    # Identify outliers: pixels where |pixel - median| > threshold * std_dev
    outliers = np.abs(img_padded - med) > (threshold_factor * std_dev)

    # Replace outliers with the local median
    img_filtered = np.where(outliers, med, img_padded)

    # Apply stride by subsampling
    if stride > 1:
        img_filtered = img_filtered[::stride, ::stride]

    # Clip and convert back to uint8
    return np.clip(img_filtered, 0, 255).astype(np.uint8)


# ==============================================================================
# == Edge Detection Functions ==
# ==============================================================================

def apply_edge_filter(img, kx, ky):
    """
    Applies an edge detection filter using provided x and y derivative kernels (e.g., Sobel, Prewitt).
    Calculates gradient magnitude. Converts color images to grayscale first.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        kx (np.ndarray): The kernel for detecting gradients in the x-direction.
        ky (np.ndarray): The kernel for detecting gradients in the y-direction.

    Returns:
        np.ndarray: The gradient magnitude image (uint8), clipped to [0, 255].
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Compute gradients using fast convolution
    gx = fast_convolve(img_gray.astype(np.float32), kx) # Use float for intermediate calc
    gy = fast_convolve(img_gray.astype(np.float32), ky)

    # Calculate magnitude: sqrt(gx^2 + gy^2)
    magnitude = np.sqrt(gx.astype(np.float64)**2 + gy.astype(np.float64)**2) # Use float64 for precision

    # Normalize magnitude to 0-255 range (optional, but common)
    # mag_max = np.max(magnitude)
    # if mag_max > 0:
    #     magnitude = (magnitude / mag_max) * 255.0

    # Clip and convert to uint8
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def canny_edge(img, low=100, high=200):
    """
    Applies the Canny edge detection algorithm using OpenCV's implementation.
    Converts color images to grayscale first.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        low (int): The lower threshold for hysteresis procedure. Defaults to 100.
        high (int): The higher threshold for hysteresis procedure. Defaults to 200.

    Returns:
        np.ndarray: The binary edge map (uint8, values 0 or 255).
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img_gray = convert_color(img, 'GRAYSCALE')
    else:
        img_gray = img.copy()

    # Apply Canny edge detector
    edges = cv2.Canny(img_gray, low, high)
    return edges

def gradient_magnitude(dx, dy):
    """
    Calculates the magnitude of the gradient given dx and dy components.

    Args:
        dx (np.ndarray): Gradient in the x-direction.
        dy (np.ndarray): Gradient in the y-direction.

    Returns:
        np.ndarray: Gradient magnitude.
    """
    # Use float64 to avoid overflow during squaring before sqrt
    magnitude = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)
    return magnitude

def gradient_direction(dx, dy):
    """
    Calculates the direction (angle) of the gradient in degrees [0, 180).

    Args:
        dx (np.ndarray): Gradient in the x-direction.
        dy (np.ndarray): Gradient in the y-direction.

    Returns:
        np.ndarray: Gradient direction in degrees (uint8, 0-179).
    """
    # Calculate angle in radians, range [-pi, pi]
    angle_rad = np.arctan2(dy, dx)
    # Convert to degrees, range [-180, 180]
    angle_deg = np.degrees(angle_rad)
    # Map to [0, 360)
    angle_deg = (angle_deg + 360) % 360
    # Map to [0, 180) as direction is often considered symmetric
    angle_deg = angle_deg % 180
    return angle_deg.astype(np.uint8) # Store as uint8

def get_hsv_edge(img, kernel='sobel'):
    """
    Computes edge magnitude and direction, then visualizes them in HSV color space.
    Hue represents direction, Value represents magnitude. Converts color images to grayscale first.

    Args:
        img (np.ndarray): The input image (grayscale or color).
        kernel (str): The kernel type to use for gradient calculation ('sobel' or 'prewitt').
                      Defaults to 'sobel'.

    Returns:
        np.ndarray: An RGB image where color encodes edge direction and brightness encodes magnitude.
                    Returns None if an invalid kernel is specified.
    """
    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        gray = convert_color(img, 'GRAYSCALE')
    else:
        gray = img.copy()

    gray_float = gray.astype(np.float32) # Use float for gradient calculations

    # Select kernels and compute gradients
    if kernel.lower() == 'sobel':
        # Use OpenCV's Sobel for potentially optimized calculation
        dx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    elif kernel.lower() == 'prewitt':
        # Define Prewitt kernels
        kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        # Use filter2D for convolution
        dx = cv2.filter2D(gray_float, -1, kx)
        dy = cv2.filter2D(gray_float, -1, ky)
    else:
        warnings.warn("Unsupported kernel. Use 'sobel' or 'prewitt'.")
        raise ValueError("Unsupported kernel. Use 'sobel' or 'prewitt'.")
        # return None # Or raise error

    # Calculate magnitude and direction
    mag = gradient_magnitude(dx, dy)
    ang_deg = gradient_direction(dx, dy) # Angle is already uint8 [0, 180)

    # Normalize magnitude to [0, 255] for the Value channel
    mag_max = np.max(mag)
    if mag_max > 0:
        mag_normalized = np.clip((mag / mag_max) * 255, 0, 255).astype(np.uint8)
    else:
        mag_normalized = np.zeros_like(mag, dtype=np.uint8)


    # Create HSV image
    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang_deg # Hue channel represents angle (0-179 maps well to Hues)
    hsv[..., 1] = 255     # Saturation channel set to maximum
    hsv[..., 2] = mag_normalized # Value channel represents magnitude

    # Convert HSV image back to RGB for display
    rgb_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_output

def best_gradient_edge(img, ref, metric='psnr'):
    """
    Applies multiple gradient-based edge detection filters (Sobel, Prewitt, Roberts)
    and selects the best result based on PSNR or SSIM with respect to a reference edge image.

    Args:
        img (np.ndarray): Input image (color or grayscale).
        ref (np.ndarray): Reference edge image to compare against.
        metric (str): Evaluation metric ('psnr' or 'ssim').

    Returns:
        tuple: (Best edge image as np.ndarray, method name as str)
    """
    ref = ref.astype(np.uint8)
    gray = convert_color(img, 'GRAYSCALE')

    results = {}

    kernels = {
        'sobel': (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                  np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])),
        'prewitt': (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])),
        'roberts': (np.array([[1, 0], [0, -1]]),
                    np.array([[0, 1], [-1, 0]]))
    }

    for method, (kx, ky) in kernels.items():
        edge = apply_edge_filter(gray, kx, ky)
        score = peak_signal_noise_ratio(ref, edge) if metric == 'psnr' else structural_similarity(ref, edge)
        results[method] = (score, edge)

    best_method = max(results, key=lambda k: results[k][0])
    return results[best_method][1], best_method


def best_canny_edge(img, ref, metric='psnr', thresholds=[(50, 150), (75, 200), (100, 250)]):
    """
    Applies the Canny edge detector using different threshold pairs and selects the best
    result based on PSNR or SSIM with respect to a reference edge image.

    Args:
        img (np.ndarray): Input image (color or grayscale).
        ref (np.ndarray): Reference edge image to compare against.
        metric (str): Evaluation metric ('psnr' or 'ssim').
        thresholds (list of tuple): List of (low, high) threshold pairs to try.

    Returns:
        tuple: (Best edge image as np.ndarray, threshold pair as tuple)
    """
    ref = ref.astype(np.uint8)
    gray = convert_color(img, 'GRAYSCALE')

    results = {}

    for (low, high) in thresholds:
        edge = canny_edge(gray, low, high)
        score = peak_signal_noise_ratio(ref, edge) if metric == 'psnr' else structural_similarity(ref, edge)
        results[(low, high)] = (score, edge)

    best_thresh = max(results, key=lambda k: results[k][0])
    return results[best_thresh][1], best_thresh

def best_gradient_edge_auto(img):
    """
    Selects the best gradient-based edge map (Sobel, Prewitt, Roberts) using internal energy scoring.

    Args:
        img (np.ndarray): The input image (grayscale or RGB).

    Returns:
        np.ndarray: The best gradient edge image (uint8).
        str: The name of the selected method.
    """
    methods = {
        'sobel': (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                  np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])),
        'prewitt': (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])),
        'roberts': (np.array([[1, 0], [0, -1]]),
                    np.array([[0, 1], [-1, 0]]))
    }

    best_energy = -1
    best_edge = None
    best_method = None

    for name, (kx, ky) in methods.items():
        edge = apply_edge_filter(img, kx, ky)
        energy = np.sum(edge.astype(np.float32) ** 2)
        if energy > best_energy:
            best_energy = energy
            best_edge = edge
            best_method = name

    return best_edge, best_method

def best_canny_edge_auto(img, thresholds=[(50, 100), (100, 200), (75, 150), (30, 120)]):
    """
    Selects the best Canny edge result by testing multiple threshold pairs and ranking by edge energy.

    Args:
        img (np.ndarray): The input image (grayscale or RGB).
        thresholds (list of tuple): List of (low, high) threshold pairs to try.

    Returns:
        np.ndarray: The best Canny edge map.
        tuple: The best (low, high) thresholds used.
    """
    best_energy = -1
    best_edge = None
    best_thresh = None

    for low, high in thresholds:
        edge = canny_edge(img, low, high)
        energy = np.sum(edge.astype(np.float32) ** 2)
        if energy > best_energy:
            best_energy = energy
            best_edge = edge
            best_thresh = (low, high)

    return best_edge, best_thresh

# ==============================================================================
# == Noise Generation Functions ==
# ==============================================================================

def add_gaussian_noise(img, mean=0, std=10):
    """
    Adds Gaussian (normal) noise to an image.

    Args:
        img (np.ndarray): The input image.
        mean (float): The mean of the Gaussian distribution. Defaults to 0.
        std (float): The standard deviation of the Gaussian distribution.
                     Determines noise intensity. Defaults to 10.

    Returns:
        np.ndarray: The noisy image (uint8), clipped to [0, 255].
    """
    # Generate noise with the same shape as the image
    noise = np.random.normal(mean, std, img.shape)
    # Add noise to the image (convert image to float first)
    noisy_img = img.astype(np.float32) + noise
    # Clip values to the valid range [0, 255] and convert back to uint8
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, amount=0.01, salt_vs_pepper=0.5):
    """
    Adds Salt and Pepper noise (random white and black pixels) to an image.

    Args:
        img (np.ndarray): The input image.
        amount (float): The proportion of pixels to be affected by noise (0.0 to 1.0).
                        Defaults to 0.01 (1%).
        salt_vs_pepper (float): The ratio of salt (white) pixels to total noise pixels (0.0 to 1.0).
                                Defaults to 0.5 (equal amounts of salt and pepper).

    Returns:
        np.ndarray: The noisy image (uint8).
    """
    noisy = img.copy()
    num_pixels = img.size if len(img.shape) == 2 else img.shape[0] * img.shape[1]

    # Calculate number of salt pixels
    num_salt = int(amount * num_pixels * salt_vs_pepper)
    # Generate random coordinates for salt pixels
    salt_coords_rows = np.random.randint(0, img.shape[0], num_salt)
    salt_coords_cols = np.random.randint(0, img.shape[1], num_salt)
    # Set salt pixels to white (255)
    if len(img.shape) == 2:
        noisy[salt_coords_rows, salt_coords_cols] = 255
    else:
        noisy[salt_coords_rows, salt_coords_cols, :] = 255 # Set all channels to white

    # Calculate number of pepper pixels
    num_pepper = int(amount * num_pixels * (1.0 - salt_vs_pepper))
    # Generate random coordinates for pepper pixels
    pepper_coords_rows = np.random.randint(0, img.shape[0], num_pepper)
    pepper_coords_cols = np.random.randint(0, img.shape[1], num_pepper)
    # Set pepper pixels to black (0)
    if len(img.shape) == 2:
        noisy[pepper_coords_rows, pepper_coords_cols] = 0
    else:
        noisy[pepper_coords_rows, pepper_coords_cols, :] = 0 # Set all channels to black

    return noisy

def add_speckle_noise(img, strength=0.1):
    """
    Adds Speckle (multiplicative) noise to an image. Noise = img + img * gaussian_noise.

    Args:
        img (np.ndarray): The input image.
        strength (float): Controls the intensity of the noise, related to the standard
                          deviation of the underlying Gaussian noise. Defaults to 0.1.

    Returns:
        np.ndarray: The noisy image (uint8), clipped to [0, 255].
    """
    # Generate Gaussian noise (mean 0, std dev 1)
    gaussian_noise = np.random.randn(*img.shape)
    # Add multiplicative noise (convert image to float)
    noisy_img = img.astype(np.float32) + img.astype(np.float32) * gaussian_noise * strength
    # Clip values and convert back to uint8
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# ==============================================================================
# == Morphological Operations (Binary) ==
# ==============================================================================
# These functions assume binary images (values of 0 and 1 or 0 and 255).
# While they can be applied to grayscale images, their traditional interpretation
# and mathematical foundation are for binary morphology.
#
# Mathematical Notation:
#   Let A be the binary input image, and B be the structuring element (SE).
#
#   ● Erosion (A ⊖ B): Shrinks white regions.
#     Only keeps pixels where SE B fits entirely within A.
#
#   ● Dilation (A ⊕ B): Expands white regions.
#     Adds pixels to boundaries where SE B touches any foreground pixel in A.
#
#   ● Opening (A ○ B): Opening is erosion followed by dilation.
#     Smooths contour, removes small foreground noise, preserves background:
#         A ○ B = (A ⊖ B) ⊕ B
#
#   ● Closing (A ● B): Closing is dilation followed by erosion.
#     Fills small holes/gaps in foreground regions, preserves foreground:
#         A ● B = (A ⊕ B) ⊖ B
#
# Note:
# - Structuring elements (SE) are typically small matrices (e.g. 3x3 or 5x5)
#   with shapes like squares, disks, crosses, or ellipses.
# - All implementations here rely on OpenCV's efficient `cv2.morphologyEx()` or
#   `cv2.erode()` / `cv2.dilate()` functions.

def erode(img, se=np.ones((3,3), dtype=bool), iterations=1):
    """
    Performs binary erosion on an image using scipy.ndimage.

    Args:
        img (np.ndarray): The input binary image (values typically 0 and 1 or 0 and 255).
        se (np.ndarray): The structuring element (a boolean array). Defaults to a 3x3 square.
        iterations (int): Number of times to apply the erosion. Defaults to 1.

    Returns:
        np.ndarray: The eroded image (uint8). Output values will be 0 or 1.
    """
    # Convert image to boolean for scipy's binary functions
    img_bool = img.astype(bool)
    # Perform binary erosion
    eroded_bool = binary_erosion(img_bool, structure=se, iterations=iterations)
    # Convert back to uint8 (True -> 1, False -> 0)
    return eroded_bool.astype(np.uint8)

def dilate(img, se=np.ones((3,3), dtype=bool), iterations=1):
    """
    Performs binary dilation on an image using scipy.ndimage.

    Args:
        img (np.ndarray): The input binary image (values typically 0 and 1 or 0 and 255).
        se (np.ndarray): The structuring element (a boolean array). Defaults to a 3x3 square.
        iterations (int): Number of times to apply the dilation. Defaults to 1.

    Returns:
        np.ndarray: The dilated image (uint8). Output values will be 0 or 1.
    """
    # Convert image to boolean
    img_bool = img.astype(bool)
    # Perform binary dilation
    dilated_bool = binary_dilation(img_bool, structure=se, iterations=iterations)
    # Convert back to uint8
    return dilated_bool.astype(np.uint8)

def open_img(img, se=np.ones((3,3), dtype=bool), iterations=1):
    """
    Performs binary opening (erosion followed by dilation) using scipy.ndimage.
    Useful for removing small noise specks (salt noise).

    Args:
        img (np.ndarray): The input binary image (values typically 0 and 1 or 0 and 255).
        se (np.ndarray): The structuring element (a boolean array). Defaults to a 3x3 square.
        iterations (int): Number of times to apply the opening operation. Defaults to 1.

    Returns:
        np.ndarray: The opened image (uint8). Output values will be 0 or 1.
    """
    # Convert image to boolean
    img_bool = img.astype(bool)
    # Perform binary opening
    opened_bool = binary_opening(img_bool, structure=se, iterations=iterations)
    # Convert back to uint8
    return opened_bool.astype(np.uint8)

def close_img(img, se=np.ones((3,3), dtype=bool), iterations=1):
    """
    Performs binary closing (dilation followed by erosion) using scipy.ndimage.
    Useful for filling small holes (pepper noise).

    Args:
        img (np.ndarray): The input binary image (values typically 0 and 1 or 0 and 255).
        se (np.ndarray): The structuring element (a boolean array). Defaults to a 3x3 square.
        iterations (int): Number of times to apply the closing operation. Defaults to 1.

    Returns:
        np.ndarray: The closed image (uint8). Output values will be 0 or 1.
    """
    # Convert image to boolean
    img_bool = img.astype(bool)
    # Perform binary closing
    closed_bool = binary_closing(img_bool, structure=se, iterations=iterations)
    # Convert back to uint8
    return closed_bool.astype(np.uint8)

# ==============================================================================
# == Segmentation Functions ==
# ==============================================================================

def watershed_segmentation(img, blur_size=2, blur_strength=1.5, kernel_shape='ellipse',
                           morph_iter=2, dist_thresh=0.5, dilate_iter=3):
    """
    Performs image segmentation using the Watershed algorithm via OpenCV.
    Includes preprocessing steps like blurring, thresholding, and morphological ops.
    Assumes foreground objects are brighter than the background.

    Args:
        img (np.ndarray): The input image (grayscale or color). Color images are converted to grayscale.
        blur_size (int): Base size parameter for initial Gaussian blur. Defaults to 2.
        blur_strength (float): Strength parameter for initial Gaussian blur. Defaults to 1.5.
        kernel_shape (str): Shape of the structuring element for morphological operations.
                            Options: 'ellipse', 'rect', 'cross'. Defaults to 'ellipse'.
        morph_iter (int): Number of iterations for morphological opening. Defaults to 2.
        dist_thresh (float): Threshold factor (0-1) applied to the distance transform
                             to determine 'sure foreground'. Defaults to 0.5.
        dilate_iter (int): Number of iterations for dilating to find 'sure background'. Defaults to 3.

    Returns:
        np.ndarray: A binary mask (uint8, values 0 or 255) where 255 indicates segmented foreground regions.
                    Returns None if an invalid kernel shape is provided.
    """
    # 1. Preprocessing: Convert to Grayscale and Blur
    if len(img.shape) == 3:
        gray = convert_color(img, 'GRAYSCALE')
    else:
        gray = img.copy()
    # Apply Gaussian blur to reduce noise
    blurred = gaussian_blur(gray, size=blur_size, strength=blur_strength)

    # 2. Thresholding: Separate rough foreground/background using Otsu's method
    # Otsu automatically finds a good threshold value
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Note: THRESH_BINARY_INV assumes dark objects on light background. If objects are light, use THRESH_BINARY
    binary = binary.astype(np.uint8)

    # 3. Morphological Operations: Clean up the binary mask
    # Define structuring element based on shape parameter
    if kernel_shape.lower() == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    elif kernel_shape.lower() == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    elif kernel_shape.lower() == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    else:
        warnings.warn("Invalid kernel shape. Choose from: ellipse, rect, cross.")
        raise ValueError("Invalid kernel shape. Choose from: ellipse, rect, cross.")
        # return None # Or raise error

    # Opening: Remove small noise specks (erosion then dilation)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

    # 4. Identify Sure Regions: Background and Foreground
    # Sure Background: Dilate the opened image - regions definitely not foreground
    sure_bg = cv2.dilate(opened, kernel, iterations=dilate_iter)

    # Sure Foreground: Use distance transform on the opened image
    # Distance transform calculates distance from each pixel to the nearest zero pixel
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    # Threshold the distance map to get confident foreground areas (peaks in distance)
    _, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown Region: Subtract sure foreground from sure background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Marker Labeling for Watershed
    # Create markers for watershed: label connected components in the sure foreground
    _, markers = cv2.connectedComponents(sure_fg)
    # Add 1 to all labels so background (0) is distinct
    markers = markers + 1
    # Mark the unknown region with 0 - watershed will classify these
    markers[unknown == 255] = 0

    # 6. Apply Watershed Algorithm
    # Watershed needs a 3-channel image, convert original gray/color image
    if len(img.shape) == 2:
        color_for_watershed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        color_for_watershed = img.copy() # Use original color if available

    # Apply watershed algorithm
    markers = cv2.watershed(color_for_watershed, markers)

    # 7. Create Final Segmentation Mask
    # Boundaries marked by watershed become -1
    # Regions labeled > 1 are the segmented objects
    segmented_mask = np.zeros_like(gray, dtype=np.uint8)
    segmented_mask[markers > 1] = 255 # Mark segmented regions as white

    return segmented_mask