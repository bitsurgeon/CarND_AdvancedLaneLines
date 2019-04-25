import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque
import cv2
import glob
import pickle
import os


# set image pipeline mode
try:
    import debug_ctrl
    debug_ctrl.IMG_MODE
except (ModuleNotFoundError, NameError):
    # default mode
    IMG_MODE = False
else:
    # external configuration
    IMG_MODE = debug_ctrl.IMG_MODE
finally:
    # init debug support
    if IMG_MODE:
        debug_folder = 'debug_outputs/'
        output_folder = 'output_images/'
        debug_cam_cal = debug_folder + 'camera_cal/'
        if not os.path.exists(debug_cam_cal):
            os.makedirs(debug_cam_cal)

# set image pipeline constants
try:
    import lane_finding_const
    lane_finding_const.src
    lane_finding_const.dst
    lane_finding_const.ym_per_pix
    lane_finding_const.xm_per_pix
    lane_finding_const.N_LP
    lane_finding_const.N_MISS
except (ModuleNotFoundError, NameError):
    # default values
    # perspective transform source points
    src = np.float32([[568, 470], [718, 470], [1110, 720], [210, 720]])
    # perspective transform destination points
    dst = np.float32([[300, 0], [980, 0], [980, 720], [300, 720]])
    # pixels space to meters
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    # queue depth and miss count for reset
    N_LP = 5
    N_MISS = 5
else:
    # external configuration
    src = lane_finding_const.src
    dst = lane_finding_const.dst
    ym_per_pix = lane_finding_const.ym_per_pix
    xm_per_pix = lane_finding_const.xm_per_pix
    N_LP = lane_finding_const.N_LP
    N_MISS = lane_finding_const.N_MISS

# init camera matrix and undistortion coefficients
mtx = None
dist = None

# init buffer that stores last N_LP good lane fitting coefficients
lane_l_q = deque([np.array([0, 0, 0])] * N_LP, maxlen=N_LP)
lane_r_q = deque([np.array([0, 0, 0])] * N_LP, maxlen=N_LP)

# count for missed frames
lane_queue_cnt = 0
bad_frame_cnt = 0

def camera_calibration(sample_images, ncol, nrow, isDebug=False):
    # prepare object points
    objp = np.zeros((ncol * nrow, 3), np.float32)
    objp[:, :2] = np.mgrid[:ncol, :nrow].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Make a list of calibration images
    images = glob.glob(sample_images)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (ncol, nrow), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if isDebug:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (ncol, nrow), corners, ret)
                cv2.imwrite(os.path.join(os.getcwd(), debug_folder, fname), img)

    # load a sample image
    img = cv2.imread(images[0])

    # get image sizes
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if isDebug:
        # save the objpoints, imgpoints, camera matrix and distortion coefficients
        cam_pickle = {}
        cam_pickle["objpoints"] = objpoints
        cam_pickle["imgpoints"] = imgpoints
        cam_pickle["mtx"] = mtx
        cam_pickle["dist"] = dist
        pickle.dump(cam_pickle, open(os.path.join(os.getcwd(), debug_folder + 'cam_pickle.p'), "wb"))

        # undistort on an image
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        # save camera calibration output image
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(os.path.join(os.getcwd(), output_folder, 'undistort.jpg'))

    return mtx, dist

def thresholded_binary(image, isDebug=False):
    # HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # HLS color threshold on S channel
    s_channel = hls[:, :, 2]
    s_thresh = (160, 255)
    s_img_binary = np.zeros_like(s_channel)
    s_img_binary[(s_channel > s_thresh[0]) & (s_channel < s_thresh[1])] = 1

    # Gradients
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # reduce noise with Gaussian Blur
    gaussian_kernel = 3
    blur_gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

    # Calculate the x and y gradients
    sobel_kernel = 3
    sobelx = np.absolute(cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * sobelx / np.max(sobelx))

    # Create a copy and apply the threshold
    sob_thresh = (20, 150)
    sob_img_binary = np.zeros_like(scaled_sobel)
    sob_img_binary[(scaled_sobel > sob_thresh[0]) & (scaled_sobel < sob_thresh[1])] = 1

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_thresh = (20, 200)
    mag_img_binary = np.zeros_like(gradmag)
    mag_img_binary[(gradmag > mag_thresh[0]) & (gradmag < mag_thresh[1])] = 1

    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(sobely, sobelx)
    dir_thresh = (0.8, 1.1)
    dir_img_binary =  np.zeros_like(absgraddir)
    dir_img_binary[(absgraddir > dir_thresh[0]) & (absgraddir < dir_thresh[1])] = 1

    # Combine the binary thresholds
    gradient_comb_img_binary = np.zeros_like(gray)
    gradient_comb_img_binary[(sob_img_binary == 1) & (mag_img_binary == 1) & (dir_img_binary == 1)] = 1

    # Combine the color transform and gradient to create a thresholded binary
    comb_img_binary = np.zeros_like(gray)
    comb_img_binary[(s_img_binary == 1) | (gradient_comb_img_binary == 1)] = 1

    if isDebug:
        # save thresolded binary image
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(image)
        ax1.set_title('Undistorted Image')
        ax2.imshow(comb_img_binary, cmap='gray')
        ax2.set_title('Thresholded Binary Image')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(os.path.join(os.getcwd(), output_folder, 'binary_combo.jpg'))

        # save detailed binary images
        f, axes = plt.subplots(2, 4, figsize=(24, 9))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Undistorted Image')
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale Image')
        axes[0, 2].imshow(s_img_binary, cmap='gray')
        axes[0, 2].set_title('Thresholded S Channel')
        axes[1, 0].imshow(sob_img_binary, cmap='gray')
        axes[1, 0].set_title('Thresholded X-Gradient')
        axes[1, 1].imshow(mag_img_binary, cmap='gray')
        axes[1, 1].set_title('Thresholded Gradient Magnitude')
        axes[1, 2].imshow(dir_img_binary, cmap='gray')
        axes[1, 2].set_title('Thresholded Gradient Direction')
        axes[1, 3].imshow(gradient_comb_img_binary, cmap='gray')
        axes[1, 3].set_title('Combined Thresholded Gradient')
        axes[0, 3].imshow(comb_img_binary, cmap='gray')
        axes[0, 3].set_title('Final Thresholded Binary')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(os.path.join(os.getcwd(), 'debug_outputs/binary_combo_detail.jpg'))

    return comb_img_binary

def perspective_transform(binary_image, src=src, dst=dst, isDebug=False):
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV
    warped = cv2.warpPerspective(binary_image, M, binary_image.shape[::-1])

    if isDebug:
        # draw line overlay in red for visual checking
        color = [255, 0, 0]
        thickness = 5

        src_t = tuple(map(tuple, src))
        color_combo = cv2.cvtColor(binary_image * 255, cv2.COLOR_GRAY2RGB)
        cv2.line(color_combo, src_t[0], src_t[3], color, thickness)
        cv2.line(color_combo, src_t[1], src_t[2], color, thickness)

        dst_t = tuple(map(tuple, dst))
        color_warped = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2RGB)
        cv2.line(color_warped, dst_t[0], dst_t[3], color, thickness)
        cv2.line(color_warped, dst_t[1], dst_t[2], color, thickness)

        # save warped images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(color_combo)
        ax1.set_title('Thresholded Binary Image')
        ax2.imshow(color_warped)
        ax2.set_title('Warped Image')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(os.path.join(os.getcwd(), output_folder, 'birds_eye.jpg'))

        # save transform matrix
        trans_pickle = {}
        trans_pickle["M"] = M
        trans_pickle["Minv"] = Minv
        pickle.dump(trans_pickle, open(os.path.join(os.getcwd(), debug_folder + 'trans_pickle.p'), "wb"))

    return warped, M, Minv

def _fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def _search_around_poly(binary_warped, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix, isDebug=False):
    # Choose the width of the margin around the previous polynomial to search
    margin = 40

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_fit_pre_avg = _lane_fit_weighted_average(lane_l_q)
    right_fit_pre_avg = _lane_fit_weighted_average(lane_r_q)

    left_lane_inds = ((nonzerox > (left_fit_pre_avg[0] * (nonzeroy ** 2) + left_fit_pre_avg[1] * nonzeroy + 
                    left_fit_pre_avg[2] - margin)) & (nonzerox < (left_fit_pre_avg[0] * (nonzeroy ** 2) + 
                    left_fit_pre_avg[1] * nonzeroy + left_fit_pre_avg[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_pre_avg[0] * (nonzeroy ** 2) + right_fit_pre_avg[1] * nonzeroy + 
                    right_fit_pre_avg[2] - margin)) & (nonzerox < (right_fit_pre_avg[0] * (nonzeroy ** 2) + 
                    right_fit_pre_avg[1] * nonzeroy + right_fit_pre_avg[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # get departure
    midpoint = np.int(binary_warped.shape[1] // 2)
    leftx_base = np.average(leftx)
    rightx_base = np.average(rightx)
    midlane = rightx_base - leftx_base
    departure = np.around((midpoint - midlane) * xm_per_pix, decimals=2)

    if isDebug:
        # Fit new polynomials
        left_fitx, right_fitx, ploty = _fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    if isDebug:
        return leftx, lefty, rightx, righty, departure, result
    else:
        return leftx, lefty, rightx, righty, departure

def _find_lane_pixels(binary_warped, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix, isDebug=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    if isDebug:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # calculate lane departure,
    # where departure <= 0 means car is on the lest of the lane center
    midlane = rightx_base - leftx_base
    departure = np.around((midpoint - midlane) * xm_per_pix, decimals=2)

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if isDebug:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 
        
        # Identify the nonzero pixels in x and y within current window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if isDebug:
        return leftx, lefty, rightx, righty, departure, out_img
    else:
        return leftx, lefty, rightx, righty, departure

def _lane_fit_weighted_average(l_q):
    # Weights for Low Pass FIR filter
    # maxlen supported is 5
    # more weights for most recent samples
    W = [[1.0], [0.6, 0.4], [0.5, 0.35, 0.15], [0.4, 0.3, 0.2, 0.1], [0.3, 0.25, 0.2, 0.15, 0.1]]

    w = W[lane_queue_cnt - 1]
    avg = 0.0
    for i in range(lane_queue_cnt):
        avg += w[i] * l_q[i]
    return avg

def _lane_fit_queue_add(ql, qr, lf, rf, tol=0.05):
    # trace latest lane fitting
    # new fitting parameters are checked against past ones
    # it is ignored if the difference is greater the the tolorance
    global lane_queue_cnt
    global bad_frame_cnt

    if lane_queue_cnt < N_LP:
        ql.appendleft(lf)
        qr.appendleft(rf)
        lane_queue_cnt += 1
    else:
        l_f_avg_pre = _lane_fit_weighted_average(ql)
        r_f_avg_pre = _lane_fit_weighted_average(qr)

        l_ok = ((l_f_avg_pre - lf) / l_f_avg_pre < tol).all()
        r_ok = ((r_f_avg_pre - rf) / r_f_avg_pre < tol).all()

        if l_ok and r_ok:
            ql.appendleft(lf)
            qr.appendleft(rf)
            bad_frame_cnt = 0
        else:
            bad_frame_cnt += 1
            if bad_frame_cnt == N_MISS:
                lane_queue_cnt -= 1
                bad_frame_cnt = 0

def fit_polynomial(binary_warped, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix, isDebug=False):
    # Find our lane pixels first
    if lane_queue_cnt == N_LP:
        if isDebug:
            leftx, lefty, rightx, righty, departure, out_img = _search_around_poly(binary_warped, isDebug=isDebug)
        else:
            leftx, lefty, rightx, righty, departure = _search_around_poly(binary_warped, isDebug=isDebug)
    else:
        if isDebug:
            leftx, lefty, rightx, righty, departure, out_img = _find_lane_pixels(binary_warped, isDebug=isDebug)
        else:
            leftx, lefty, rightx, righty, departure = _find_lane_pixels(binary_warped, isDebug=isDebug)

    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # store current fits if it is a good measurement
    _lane_fit_queue_add(lane_l_q, lane_r_q, left_fit, right_fit)

    # get weighted average
    left_fit_avg = _lane_fit_weighted_average(lane_l_q)
    right_fit_avg = _lane_fit_weighted_average(lane_r_q)
    
    # convert to meters
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )

    try:
        left_fitx = left_fit_avg[0] * ploty ** 2 + left_fit_avg[1] * ploty + left_fit_avg[2]
        right_fitx = right_fit_avg[0] * ploty ** 2 + right_fit_avg[1] * ploty + right_fit_avg[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    if isDebug:
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.figure()
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(out_img)
        plt.title('Sliding Window Lane Fitting Image')
        plt.savefig(os.path.join(os.getcwd(), output_folder, 'color_poly_fitting.jpg'))

    return ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr, departure

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix):
    # the radius of curvature at the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = np.around(((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0]), decimals=2)
    right_curverad = np.around(((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0]), decimals=2)

    # average lane curvature
    avg_curverad = np.around((left_curverad + right_curverad) / 2., decimals=2)

    return left_curverad, right_curverad, avg_curverad

def lane_overlay(undist, Minv, ploty, left_fitx, right_fitx, avg_curverad, depart, isDebug=False):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # add stats
    if depart < 0:
        l_r = 'm left  '
        depart = -depart
    else:
        l_r = 'm right '
    stat_cur = 'Radius of Curvature = ' + format(avg_curverad, '.2f') + 'm'
    stat_dep = 'Vehicle is ' + format(depart, '.2f') + l_r + 'of centre'
    cv2.putText(result, stat_cur, (50, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), lineType=1000)
    cv2.putText(result, stat_dep, (50, 120), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), lineType=1000)

    if isDebug:
        plt.figure()
        plt.imshow(result)
        plt.title('Lane Line Overlay')
        plt.savefig(os.path.join(os.getcwd(), output_folder, 'lane_line_overlay.jpg'))

    return result

def image_pipeline(image, mtx=mtx, dist=dist, isDebug=False):
    # undistort input image
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # generate thresholded binary image
    binary_combo = thresholded_binary(undist, isDebug=isDebug)

    # generate bird's-eye view
    warped, M, Minv = perspective_transform(binary_combo, isDebug=isDebug)

    # fit lane lines
    ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr, depart = fit_polynomial(warped, isDebug=isDebug)

    # radius of curvature in meters for both lane lines, and the averaged
    left_curverad, right_curverad, avg_curverad = measure_curvature_real(ploty, left_fit_cr, right_fit_cr)

    # mark lane line area back onto undistorted image
    final_image = lane_overlay(undist, Minv, ploty, left_fitx, right_fitx, avg_curverad, depart, isDebug=isDebug)

    return final_image

def process_image(image):
    # process video stream frame by frame
    # disable debugging function in video processing
    result = image_pipeline(image, mtx, dist, False)
    return result

# -------------------------- #                    
# Advanced Lane Line Finding #
# -------------------------- #

print('### Advanced Lane Line Finding ###')

## camera calibration - a once-off step for new camera
if (mtx == None) or (dist == None):
    print('calibrating camera ...')
    mtx, dist = camera_calibration('camera_cal/calibration*.jpg', 9, 6, IMG_MODE)

## single image
print('single image processing ...')

test_image = 'test_images/straight_lines1.jpg'
print('  test image: ' + test_image)
print('  debug mode: ' + str(IMG_MODE))
if IMG_MODE:
    print('  pipeline sample folder: ' + output_folder)
    print('  debuging output folder: ' + debug_folder)

# load distorted color image
image = mpimg.imread(test_image)

# test image pipeline
out = image_pipeline(image, mtx=mtx, dist=dist, isDebug=IMG_MODE)

# ## video clip
print('video clip processing ...')

# test video pipeline
output_video = 'lane_line_marked.mp4'
video_clip = VideoFileClip('sample_video.mp4')
overlay_clip = video_clip.fl_image(process_image)
overlay_clip.write_videofile(output_video, audio=False, verbose=False, logger=None) # logger="bar" to enable progress bar

print('### END ###')
