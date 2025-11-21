import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# --- Configuration ---
# Set the path once here.
IMAGE_PATH = '/Users/shivanshraj/Downloads/Images for task2/victoria1.jpg' 
# --- The Canny Algorithm Implementation (Task 1.1) ---

def my_canny_detector(image_path, blur_kernel=(5, 5), high_ratio=0.15, low_ratio=0.05):
    """
    Performs Canny Edge Detection using custom implementation steps.
    Returns: final_image, runtime, sobel_x, sobel_y, laplacian, mag_vis, nms_res, thresh_res, edge_pixel_count
    """
    
    # 1. Load and Preprocess Image - FIX 1: Use the argument
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use float for accurate calculations later (Sobel and NMS)
    img_float = gray_image.astype(np.float64) 

    start_time = time.time() # Start timing for Task 1, Part 3

    # 2. Gaussian Blur
    blur = cv2.GaussianBlur(img_float, blur_kernel, 0)
    
    # 3. Sobel Gradient and Direction Calculation
    # Note: ksize=3 is standard for Sobel in Canny; ksize=1 is unusual but kept if intentional.
    gradient_sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gradient_sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_sobel_x, gradient_sobel_y)
    direction = np.arctan2(gradient_sobel_y, gradient_sobel_x)

    # Visualization of Magnitude (Normalized for display)
    magnitude_visual = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Calculate Laplacian for visualization only (not part of the Canny process)
    # Using the floating-point blurred image for consistency
    gradient_laplacian = cv2.Laplacian(blur, cv2.CV_64F)

    # 4. Non-Maximum Suppression (NMS)
    (height, width) = magnitude.shape
    nms_result = np.zeros((height, width), dtype=np.float64)
    direction_degrees = np.degrees(direction)
    direction_degrees[direction_degrees < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = direction_degrees[i, j]
            mag = magnitude[i, j]
            
            # Simplified NMS logic (based on your provided code)
            neighbor1, neighbor2 = 0, 0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180): 
                neighbor1, neighbor2 = magnitude[i, j - 1], magnitude[i, j + 1] 
            elif (22.5 <= angle < 67.5): 
                neighbor1, neighbor2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif (67.5 <= angle < 112.5): 
                neighbor1, neighbor2 = magnitude[i - 1, j], magnitude[i + 1, j]
            elif (112.5 <= angle < 157.5): 
                neighbor1, neighbor2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if mag >= neighbor1 and mag >= neighbor2:
                nms_result[i, j] = mag
            # else: nms_result[i, j] = 0 (already handled by initialization)

    # 5. Double Thresholding
    high_threshold = nms_result.max() * high_ratio
    low_threshold = nms_result.max() * low_ratio 
    thresholded_result = np.zeros_like(nms_result)
    STRONG = 255 
    WEAK = 75     

    strong_i, strong_j = np.where(nms_result >= high_threshold)
    thresholded_result[strong_i, strong_j] = STRONG
    weak_i, weak_j = np.where((nms_result < high_threshold) & (nms_result >= low_threshold))
    thresholded_result[weak_i, weak_j] = WEAK

    # 6. Hysteresis Tracking
    final_image = np.copy(thresholded_result)
    (height, width) = final_image.shape
    
    # Simplified single-pass Hysteresis loop
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if final_image[i, j] == WEAK:
                # Check 8 neighbors for a STRONG edge (255)
                if np.any(final_image[i-1:i+2, j-1:j+2] == STRONG):
                    final_image[i, j] = STRONG
                else:
                    final_image[i, j] = 0

    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate the number of edge pixels for numerical comparison
    edge_pixel_count = np.sum(final_image == 255)

    # Return required components for testing (final image, runtime, and components for detailed vis)
    return (
        final_image.astype(np.uint8), 
        runtime, 
        gradient_sobel_x,
        gradient_sobel_y,
        gradient_laplacian.astype(np.uint8), # Ensure this is cast for display
        magnitude_visual, 
        cv2.normalize(nms_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), # Normalized NMS for visualization
        thresholded_result.astype(np.uint8), # Cast for display
        edge_pixel_count
    )

# --------------------------------------------------------------------------------
# --- TESTING AND VISUALIZATION SECTION (Tasks 1.2 & 1.3) ---

# --- Test Cases for Gaussian Blur ---
print("--- Running Test Case 1 (Low blurring) ---")
result_1, time_1, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_1 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(1, 1), high_ratio=0.15, low_ratio=0.05)
print(f"Time (Low blurring): {time_1:.4f}s, Edges: {count_1}")

print("--- Running Test Case 2 (Normal blurring) ---")
result_2, time_2, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_2 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(5,5), high_ratio=0.15, low_ratio=0.05)
print(f"Time (Normal blurring): {time_2:.4f}s, Edges: {count_2}")

print("--- Running Test Case 3 (Large blurring) ---")
result_3, time_3, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_3 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(11,11), high_ratio=0.15, low_ratio=0.05)
print(f"Time (Large blurring): {time_3:.4f}s, Edges: {count_3}")


# Test Cases involving Low thresholds and High thresholds

# --- Test 4: High Threshold (High Ratio) - THRESHOLD TEST ---
# Use Normal Blur (5,5) but aggressively high thresholds (less noise, more gaps)
print("4. High Threshold (0.3/0.1) Test:")
result_4, time_4,sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_4 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(5, 5), high_ratio=0.30, low_ratio=0.10)
print(f"   Time: {time_4:.4f}s, Edges: {count_4}")

# --- Test 5: Low Threshold (Low Ratio) - THRESHOLD TEST ---
# Use Normal Blur (5,5) but very low thresholds (more noise/connected edges)
print("5. Low Threshold (0.1/0.01) Test:")
result_5, time_5, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_5 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(5, 5), high_ratio=0.10, low_ratio=0.01)
print(f"   Time: {time_5:.4f}s, Edges: {count_5}")


# --- Test 6: Tight Threshold (Minimal Hysteresis) - NEW INSIGHT TEST ---
# High and Low thresholds are too close, limiting the ability of hysteresis to connect edges.
print("6. Tight Threshold (0.15/0.12) Test: (Minimal Hysteresis)")
result_6, time_6, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_6 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(5, 5), high_ratio=0.15, low_ratio=0.12)
print(f"   Time: {time_6:.4f}s, Edges: {count_6}")

print("7. Low Blur, Low threshold Test: ")
result_7, time_7, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_7 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(1, 1), high_ratio=0.1, low_ratio=0.01)
print(f"   Time: {time_7:.4f}s, Edges: {count_7}")

print("8. Normal Blur, Normal threshold Test: ")
result_8, time_8, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_8 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(1, 1), high_ratio=0.15, low_ratio=0.05)
print(f"   Time: {time_8:.4f}s, Edges: {count_8}")

print("9. Large Blur, High Threshold Test: ")
result_9, time_9, sobel_x, sobel_y, laplacian, mag_vis, nms_vis, thresh_res, count_9 = \
    my_canny_detector(IMAGE_PATH, blur_kernel=(15, 15), high_ratio=0.40, low_ratio=0.15)
print(f"   Time: {time_9:.4f}s, Edges: {count_9}")


# --- Test Case 9: OpenCV Comparison (Task 1.3) ---
image_cv = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if image_cv is not None:
    image_cv = cv2.resize(image_cv, (200, 200))

    print("--- Running OpenCV Comparison ---")
    cv_start = time.time()
    # Using fixed thresholds (not ratios) for OpenCV Canny
    cv_result = cv2.Canny(image_cv, 40, 100) 
    cv_time = time.time() - cv_start
    cv_count = np.sum(cv_result == 255)
    print(f"Time (OpenCV): {cv_time:.4f}s, Edges: {cv_count}")

    # --- Visualization for Report (Matplotlib) ---

    plt.figure(figsize=(9, 4)) 

    titles_blur = [
        f"1. Low Blur (1x1)    Edges: {count_1}", 
        f"2. Normal Blur (5x5)    Edges: {count_2}", 
        f"3. Large Blur (11x11)    Edges: {count_3}", 
    ]
    results_blur = [result_1, result_2, result_3]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(results_blur[i], cmap='gray')
        plt.title(titles_blur[i], fontsize=8)
        plt.axis('off')

    plt.suptitle("Task 1 Results: Canny Detector - Gaussian Blurring Analysis", fontsize=12)
    # The rect parameter maintains the small margin for the suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Blocks execution

    plt.figure(figsize=(9, 4)) 

    titles_thresh = [
        f"4. High Threshold    Edges: {count_4}", 
        f"5. Low Threshold    Edges: {count_5}", 
        f"6. Tight Threshold    Edges: {count_6}"
    ]
    results_thresh = [result_4, result_5, result_6]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(results_thresh[i], cmap='gray')
        plt.title(titles_thresh[i], fontsize=8)
        plt.axis('off')

    plt.suptitle("Task 1 Results: Canny Detector - Double Thresholding Analysis", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Blocks execution



    plt.figure(figsize=(15, 5)) 

    titles_task3 = [
        f"7. Low Blur, Low threshold    Edges: {count_7}", 
        f"8. Normal Blur, Average Threshold    Edges: {count_8}",
        f"9. Large Blur, High threshold   Edges: {count_9}", 
        f"10. Canny in-built    Edges: {cv_count}"
    ]
    results_task3 = [result_7, result_8, result_9, cv_result]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(results_task3[i], cmap='gray')
        plt.title(titles_task3[i], fontsize=8)
        plt.axis('off')

    plt.suptitle("Task 1 Results: Canny Detector", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Blocks execution


    # --- Detailed Intermediate Visualization (Debug/Explanation) ---
    # print("\nShowing intermediate steps for Baseline Test Case 1...")

    # cv2.imshow('1. Sobel X Gradient', sobel_x)
    # cv2.imshow('2. Sobel Y Gradient', sobel_y)
    # cv2.imshow('3. Laplacian', laplacian)
    # cv2.imshow('4. Gradient Magnitude (Normalized)', mag_vis)
    # cv2.imshow('5. Non-Maximum Suppression (NMS) Result', nms_vis)
    # cv2.imshow('6. Double Thresholding Result', thresh_res)
    # cv2.imshow('7. Final Canny Result (Custom)', result_1)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("Cannot run OpenCV comparison or visualizations due to image loading failure.")


# ------------Extracting ORB Keypoints---------------

# --- 1. Load the images in grayscale ---
img1_color = cv2.imread('/Users/shivanshraj/Downloads/Images for task2/victoria1.jpg')
img2_color = cv2.imread('/Users/shivanshraj/Downloads/Images for task2/victoria2.jpg')

img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# --- 2. Initialize the ORB detector ---
# Sets max features to 500 by default, using FAST corners and BRIEF descriptors.
orb = cv2.ORB_create()

# --- 3. Extract Keypoints and Descriptors (using detect and compute separately) ---

# Image 1: victoria.jpg
# a) Detect keypoints (kp1 is a list of keypoint objects)
kp1 = orb.detect(img1_gray, None)
# b) Compute descriptors (des1 is the feature vector array)
kp1, des1 = orb.compute(img1_gray, kp1)

# Image 2: victoria2.jpg
# a) Detect keypoints
kp2 = orb.detect(img2_gray, None)
# b) Compute descriptors
kp2, des2 = orb.compute(img2_gray, kp2)


# --- 4. Visualize the Detected Keypoints ---

# Draw only keypoint locations (not size and orientation) in green (0, 255, 0)
# Note: drawKeypoints returns a BGR image, even if the input was grayscale.
img_kp1 = cv2.drawKeypoints(img1_gray, kp1, None, color=(0, 255, 0), flags=0)
img_kp2 = cv2.drawKeypoints(img2_gray, kp2, None, color=(0, 255, 0), flags=0)

# Display the results
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
# Matplotlib requires RGB format, so convert BGR output of drawKeypoints
# plt.imshow(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB))
plt.imshow(img_kp1)
plt.title('victoria1.jpg: ORB Keypoints')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_kp2)
plt.title('victoria2.jpg: ORB Keypoints')
plt.axis('off')

plt.show()


# ----------Task 2.3: SIFT vs. ORB Feature Matching Comparison----------

# --- 1. Load and Prepare Images ---
    # Load images in grayscale
color_img1 = cv2.imread('/Users/shivanshraj/Downloads/Images for task2/victoria1.jpg')
color_img2 = cv2.imread('/Users/shivanshraj/Downloads/Images for task2/victoria2.jpg')

img1 = cv2.cvtColor(color_img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(color_img2, cv2.COLOR_BGR2GRAY)


# --- 2. Feature Matching Utility Function ---
"""Extracts features and performs Brute-Force matching with Lowe's Ratio Test."""

sift = cv2.SIFT_create(nfeatures = 2000)
# Detect and Compute Features
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's Ratio Test (ratio_threshold = 0.75)
good_matches_sift = []
for m, n in matches:
    if m.distance < 0.80 * n.distance:
        good_matches_sift.append([m])
            
# --- 3. SIFT and ORB Execution ---

img_sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_sift, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# ORB (Hamming Norm: Binary distance for binary descriptors)
orb = cv2.ORB_create(nfeatures = 2000)

# Detect and Compute Features (Keypoints and Descriptors)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.knnMatch(des1, des2, k=2)

good_matches_orb = []
for m, n in matches:
    # m is the best match, n is the second best
    if m.distance < 0.80 * n.distance:
        # Note: drawMatchesKnn expects a list of lists of matches
        good_matches_orb.append([m])

img_orb_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_orb, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.figure(figsize=(15, 8))
plt.suptitle(f"Task 2.3: SIFT vs. ORB Feature Matching Comparison: ", fontsize=16)

plt.subplot(2, 1, 1)
plt.imshow(img_sift_matches)
plt.title(f'SIFT Matches (Total: {len(good_matches_sift)} good matches)')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(img_orb_matches)
plt.title(f'ORB Matches (Total: {len(good_matches_orb)} good matches)')
plt.axis('off')

plt.tight_layout()
plt.show()


