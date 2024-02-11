import cv2
import numpy as np

def preprocess_image(image):
    # Gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için Gaussian Blur uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel operatörlerini uygula
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Sobel çıkışlarını birleştir
    combined_sobel = np.sqrt(sobelx**2 + sobely**2)

    # Gürültüyü filtrele
    combined_sobel = cv2.GaussianBlur(combined_sobel, (5, 5), 0)

    # Kenarları belirlemek için ikinci bir eşikleme uygula
    _, binary_image = cv2.threshold(combined_sobel, 30, 255, cv2.THRESH_BINARY)

    return binary_image

# Video yakalama
cap = cv2.VideoCapture("LaneVideo.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    def nothing(x):
        pass
    # Çerçeveyi yeniden boyutlandır
    frame = cv2.resize(frame, (640, 480))
    cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    # ROI'yi belirle
    tl = (222, 387)
    bl = (70, 472)
    tr = (400, 380)
    br = (538, 472)
    roi_points = np.array([tl, bl, br, tr], dtype=np.int32)
    roi_points = roi_points.reshape((-1, 1, 2))
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    

    # ROI'yi al ve filtre uygula
    roi = frame.copy()
    cv2.fillPoly(roi, [roi_points], (0, 0, 255))
    roi = cv2.bitwise_and(frame, roi)
    binary_image = preprocess_image(roi)

    # Görüntüyü göster
    cv2.imshow("roi", roi)
    cv2.imshow("transformedframe", transformed_frame)
    cv2.imshow("roi", roi)
    cv2.imshow("Lane Detection", binary_image)

    if cv2.waitKey(10) == 27:
        break

# Pencereyi kapat
cap.release()
cv2.destroyAllWindows()
