import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Open the video file
vidcap = cv2.VideoCapture("LaneVideo.mp4")

# Function to perform sliding window lane detection
def sliding_window(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Number of sliding windows
    nwindows = 9
    # Set the height of windows
    window_height = int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions for the left and right windows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

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

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window on the mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Check if there are enough pixels for fitting
    if len(lefty) >= 2:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        # Handle the case when there are not enough pixels for fitting
        left_fit = np.array([0, 0, 0])

    if len(righty) >= 2:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        # Handle the case when there are not enough pixels for fitting
        right_fit = np.array([0, 0, 0])

    return left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds
while True:  # Infinite loop for continuous video playback
    # Read a frame from the video
    success, image = vidcap.read()

    if not success:
        # If the video reading is finished, rewind the video
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue  # Continue to the next iteration to read the first frame of the video

    frame = cv2.resize(image, (640, 480))
    frame2 = cv2.resize(image, (640, 480))

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the Sobel filter
    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitudes of the gradients
    gradient_magnitude_x = np.abs(sobel_x)
    gradient_magnitude_y = np.abs(sobel_y)

    # Combine Sobel x and Sobel y gradients using the bitwise AND operator
    combined_gradient = cv2.bitwise_and(gradient_magnitude_x, gradient_magnitude_y)

    # Normalize the gradient magnitude to the range [0, 255]
    combined_gradient = cv2.normalize(combined_gradient, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit unsigned integer
    combined_gradient = np.uint8(combined_gradient)

    # Define region of interest (ROI) corner points
    tl = (220, 387)
    bl = (70, 472)
    tr = (400, 380)
    br = (538, 472)

    # Draw lines on the original frame
    cv2.line(frame, tl, tr, (0, 0, 255), 1)
    cv2.line(frame, tr, br, (0, 0, 255), 1)
    cv2.line(frame, br, bl, (0, 0, 255), 1)
    cv2.line(frame, bl, tl, (0, 0, 255), 1)

    # Define perspective transform points
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Matrix to warp the image for bird's-eye view
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Convert the warped image to grayscale
    transformed_gray = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to obtain the binary version
    _, binary_transformed = cv2.threshold(transformed_gray, 190, 255, cv2.THRESH_BINARY)

    # Perform sliding window lane detection
        # sliding_window fonksiyonunun döndürdüğü değerleri al
    left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds = sliding_window(binary_transformed)

    # Filtreleme yapmak için kullanılan değerler
    left_lane_pixels = binary_transformed[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]]
    right_lane_pixels = binary_transformed[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]]

    # Sol ve sağ bantlar için Gaussian pdf'yi uyumla
    left_mean, left_std = norm.fit(left_lane_pixels)
    right_mean, right_std = norm.fit(right_lane_pixels)

    ploty = np.linspace(0, binary_transformed.shape[0] - 1, binary_transformed.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_transformed, binary_transformed, binary_transformed)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # ...

        # Generate a polygon to illustrate the search window area (old windows)
    margin = 40
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Create an empty frame for the drawing in reverse perspective
    reverse_perspective_frame = np.zeros_like(frame)

    # Draw the lane onto the empty frame in green
    cv2.fillPoly(reverse_perspective_frame, [np.int32(left_line_pts)], (0, 255, 0))
    cv2.fillPoly(reverse_perspective_frame, [np.int32(right_line_pts)], (0, 255, 0))

    # Generate a polygon to fill the area between the left and right lines in red
    fill_margin = 10  # Margin for filling the area
    fill_pts_left = np.array([np.transpose(np.vstack([left_fitx - fill_margin, ploty]))])
    fill_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx + fill_margin, ploty])))])
    fill_pts = np.hstack((fill_pts_left, fill_pts_right))

    # Fill the area between the left and right lines in red
    cv2.fillPoly(reverse_perspective_frame, [np.int32(fill_pts)], (0, 0, 255))

    # Apply the inverse perspective transformation to get the drawing back to the original perspective
    reverse_perspective_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    reverse_perspective_frame = cv2.warpPerspective(reverse_perspective_frame, reverse_perspective_matrix, (frame.shape[1], frame.shape[0]))

    # Combine the original frame with the drawing in reverse perspective
    result = cv2.addWeighted(frame2, 1, reverse_perspective_frame, 0.3, 0)
    window_width = 50
    histogram = np.sum(binary_transformed[binary_transformed.shape[0]//2:, :], axis=0)
    y = binary_transformed.shape[0]  # İlk başlangıç yüksekliği
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    while y > 0:
        ## Left threshold
        img_left = binary_transformed[y-40:y, left_base-50:left_base+50]
        contours_left, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_left:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                left_base = left_base - 50 + cx

        ## Right threshold
        img_right = binary_transformed[y-40:y, right_base-50:right_base+50]
        contours_right, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_right:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                right_base = right_base - 50 + cx

        cv2.rectangle(binary_transformed, (left_base-50, y), (left_base+50, y-40), (255, 255, 255), 2)
        cv2.rectangle(binary_transformed, (right_base-50, y), (right_base+50, y-40), (255, 255, 255), 2)
        y -= 40
    # Display the result
    cv2.imshow("Sliding Window Lane Detection2", result)


    # Display the Sobel-filtered image
    cv2.imshow("Sobel Filtered Image", combined_gradient)

    # Display the original frame with lines
    cv2.imshow("Original Frame", frame)

    # Display the transformed binary image
    cv2.imshow("Transformed Binary Image", binary_transformed)
    # Display the histogram visualization
    histogram = np.sum(binary_transformed, axis=0)
    hist_image = np.zeros((binary_transformed.shape[0], binary_transformed.shape[1], 3), dtype=np.uint8)
    for col, value in enumerate(histogram):
        cv2.line(hist_image, (col, hist_image.shape[0]), (col, hist_image.shape[0] - value // 256),
                 color=(255, 255, 255), thickness=1)
        
    # Draw the polynomial lines on the binary image
    ploty = np.linspace(0, binary_transformed.shape[0] - 1, binary_transformed.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_transformed, binary_transformed, binary_transformed)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area (old windows)
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin=40
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, [np.int32(left_line_pts)], (0, 255, 0))
    cv2.fillPoly(window_img, [np.int32(right_line_pts)], (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Display the result
    cv2.imshow("Sliding Window Lane Detection", result)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
vidcap.release()

# Close windows
cv2.destroyAllWindows()
