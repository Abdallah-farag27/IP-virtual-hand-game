import cv2
import numpy as np

def generate_gaussian_kernel(ksize, sigma):
    """
    Generates a Gaussian kernel manually.
    Values are calculated based on the Gaussian distribution formula.
    """
    # 1. Calculate the center index of the kernel (e.g., for 5x5, center is 2)
    center = ksize // 2
    
    # 2. Create a grid of (x, y) coordinates centered at (0,0)
    # np.mgrid creates a dense meshgrid. 
    # For ksize=5, x and y go from -2 to +2.
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    # 3. Calculate the Gaussian formula: G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2+y^2)/(2*sigma^2))
    
    # Calculate the constant part: 1 / (2 * pi * sigma^2)
    normal_const = 1 / (2.0 * np.pi * sigma**2)
    
    # Calculate the power of e: -(x^2 + y^2) / (2 * sigma^2)
    exponent = -(x**2 + y**2) / (2.0 * sigma**2)
    
    # Combine them to get the raw kernel values
    kernel = normal_const * np.exp(exponent)
    
    # 4. Normalize the kernel so the sum of all elements is 1.
    # This ensures the image doesn't get brighter or darker after blurring.
    kernel = kernel / np.sum(kernel)
    
    return kernel

def bgr_to_hsv_manual(image):
    """
    Manually converts a BGR image to HSV color space.
    """
    # 1. Convert image to float32 and normalize to range [0, 1]
    # We need floats for division and precise calculation.
    img_float = image.astype(np.float32) / 255.0
    
    # 2. Separate the Blue, Green, and Red channels
    # [..., 0] means take all height, all width, and the 0th channel (Blue)
    blue, green, red = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    
    # 3. Calculate Value (V)
    # V is simply the maximum value among R, G, B for each pixel.
    # This represents the brightness of the pixel.
    cmax = np.max(img_float, axis=2)
    
    # Finding the minimum is needed for Saturation and Hue calculation.
    cmin = np.min(img_float, axis=2)
    
    # Calculate Delta (difference between max and min)
    diff = cmax - cmin 
    
    # Initialize H, S, V arrays with zeros, same shape as input
    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax # V is already calculated as cmax
    
    # 4. Calculate Saturation (S)
    # Formula: S = Delta / Cmax (if Cmax != 0), else 0
    # We use a mask to avoid division by zero.
    non_zero_cmax = cmax > 0
    s[non_zero_cmax] = diff[non_zero_cmax] / cmax[non_zero_cmax]
    
    # 5. Calculate Hue (H)
    # Hue depends on which color channel was the maximum.
    
    # Create mask where Delta > 0 (pixels that are not grayscale)
    non_zero_diff = diff > 0
    
    # Case A: Red is the max component
    # formula: H = 60 * ((G - B) / Delta) % 6
    red_mask = (cmax == red) & non_zero_diff
    h[red_mask] = (60 * ((green[red_mask] - blue[red_mask]) / diff[red_mask]) + 360) % 360
    
    # Case B: Green is the max component
    # formula: H = 60 * ((B - R) / Delta + 2)
    green_mask = (cmax == green) & non_zero_diff
    h[green_mask] = (60 * ((blue[green_mask] - red[green_mask]) / diff[green_mask]) + 120)
    
    # Case C: Blue is the max component
    # formula: H = 60 * ((R - G) / Delta + 4)
    blue_mask = (cmax == blue) & non_zero_diff
    h[blue_mask] = (60 * ((red[blue_mask] - green[blue_mask]) / diff[blue_mask]) + 240)
    
    # 6. Scale values to fit OpenCV's 8-bit format
    # H: 0-360 degrees -> divided by 2 to fit 0-180
    # S: 0-1 float -> multiplied by 255 to fit 0-255
    # V: 0-1 float -> multiplied by 255 to fit 0-255
    h = (h / 2).astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)
    
    # 7. Merge the three channels back into one image
    return cv2.merge([h, s, v])

def histogram_equalization_manual(channel):

    # 1. Calculate Histogram
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    
    # 2. Calculate CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    
    # 3. Normalize the CDF
    cdf_m = np.ma.masked_equal(cdf, 0) 
    
    # The equalization formula: h(v) = round( [ (cdf(v) - cdf_min) / (total_pixels - cdf_min) ]* 255)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Fill masked values back with 0 and convert to uint8
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 4. Map the original pixels to their new values
    return cdf_final[channel]

def preprocess_frame(frame, width=960, height=720):
    if frame is None:
        return None
        
    # 1. Resize the frame to a standard size used by the game
    frame_resized = cv2.resize(frame, (width, height))
    
    # 2. Flip Horizontally to create a mirror effect (more natural for user)
    frame_flipped = frame_resized[:, ::-1]
    
    # 3. Apply Gaussian Blur to reduce camera noise
    gaussian_kernel = generate_gaussian_kernel(ksize=5, sigma=1.0)
    #gaussian_kernel = cv2.getGaussianKernel(5, 1.0)

    # We use filter2D to convolve the image with our kernel
    frame_blurred = cv2.filter2D(frame_flipped, -1, gaussian_kernel)
    
    # 4. Convert to HSV color space using standard OpenCV function
    hsv_frame = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
    #hsv_frame = bgr_to_hsv_manual(frame_blurred)

    # 5. Apply Histogram Equalization to the Value (Brightness) channel
    h, s, v = cv2.split(hsv_frame)
    v_eq = histogram_equalization_manual(v)
    
    # Merge the equalized V channel back with original H and S
    hsv_final = cv2.merge([h, s, v_eq])
    
    return hsv_final
