import cv2
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
vidcap = cv2.VideoCapture("LaneVideo.mp4")

# Kayan pencereler fonksiyonu
def sliding_window(binary_warped,sigma=2.0):
    # Görüntünün alt yarısının histogramını al
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    
    smoothed_histogram = gaussian_filter1d(histogram, sigma=1.0) #sigma değerine göre artarsa daha yayvan azarlırsa daha dik bir histogram görürüz
    # Görselleştirmek ve sonucu çizmek için çıktı görüntüsü oluştur

    # Histogramın sol ve sağ yarısındaki pikselin tepe noktalarını bul
    midpoint = len(smoothed_histogram) // 2
    leftx_base = np.argmax(smoothed_histogram[:midpoint]) #np.argmax ile sol taraftaki en yüksek piksel değerini histogramda buluyoruz
    rightx_base = np.argmax(smoothed_histogram[midpoint:]) + midpoint # ortadan böldüğümüz için sağdan 35 sütunda bulduğumuzu düşünürsek üstüne midpoint kadar basamak eklememiz gerekir ki gerçek konumu ortaya çıksın
    
    nwindows = 9
    # Pencere yüksekliğini ayarla
    window_height = int(binary_warped.shape[0] // nwindows) # binary_warped.shape[0] ile görüntünün 'M' yani yüksekliğini nwindows sayısında kareye böleriz
    # Görüntüdeki tüm sıfır olmayan piksellerin x ve y pozisyonlarını tanımla
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]) #binary formdaki görüntüde 0 ve 1 olan kısımları ayırmak için kullanılır y ekseni
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base #sağ sol lane için kayan pencerelerin başlangıç konumunu ayarlar

    # Pencere genişliğini +/- marjin kadar ayarla
    margin = 100
    # Pencereyi tekrar merkeze almak için gerekli minimum piksel sayısı
    minpix = 50 #y ekseninde 50 piksellik alan

    # sol ve sağ şerit piksel indislerini almak için boş listeler oluştur
    left_lane_inds = []
    right_lane_inds = []

    # pencereleri birer birer geç
    for window in range(nwindows): #9 pencere kadar dönüyor fonksiyon sürekli
        # x ve y sınırlarını (sağ ve sol) belirle
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height #pencereleri alttan yukarı doğru sıralar low alt yüksekliği high oencerenin üst yüksekliğini gösterir
        """""
        mesela binary_warped.shape[0] değeri 720  ise ve window_height değeri 80 ise bu kodlar bir pencerenin alt ve üst yüksekliklerini belirler. 
        örnek window değeri 0 ise, win_y_low ifadesi 720 - (0 + 1) * 80 = 640 ve win_y_high ifadesi 720 - 0 * 80 = 720 değerini alır. yani biz
        yukarıda nwindows değişkeni ile 720 pikseli 9 a böldüğümüz için windows_height değeri 720/9 dan 80 oluyor bu durumda da window değeri 
        arttıkça penceler aşağı doğru sıralanıyor
        """""
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin #margin değeriyle sol kısmın pencerelerinin sağ sol genişliğini ayarlar
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin #margin değeriyle sağ kısmın pencerelerinin sağ sol genişliğini ayarlar

        # Pencere içindeki x ve y'deki sıfır olmayan pikselleri tanımla
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0] 
        """"pencere içindeki 0 ve 1 leri belirler dizi içinde 0 olmayan koordinatları döndürür  fakat[0] dediği için satır koordinatlarını döndürür y 
        ekseni diyebiliriz.
        """""
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Bu indisleri listelere ekle
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    # İndis dizilerini birleştir
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Sol ve sağ şerit piksel pozisyonlarını çıkar
    leftx = nonzerox[left_lane_inds] #left lane indsteki x ekseninde 0 olmayan pikselleri belirler 
    lefty = nonzeroy[left_lane_inds]  #left lane indsteki y ekseninde 0 olmayan pikselleri belirler, bir nevi şeridi burda tespit ediyoruz
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]#sol ve sağ şeridin x y koordinatlarını belirler

    # polinom için yeterli piksel var mı kontrol ediyor eğer 2 den fazla piksel varsa polinom uyduruyor
    if len(lefty) >= 2:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        #polinom için yeterli piksel yoksa, [0, 0, 0] ile doldur
        left_fit = np.array([0, 0, 0])
    

    if len(righty) >= 2:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        # Fitting için yeterli piksel yoksa, [0, 0, 0] ile doldur
        right_fit = np.array([0, 0, 0])
    #şeridi tespit etmek amaçlı yeterli piksel varsa  bir polinom uydurur. bu polinom şeritteki piksellere göre uydurulur buna göre de bir şerit çizilir.
    #örnek olarak 5 farklı yerde bulunan nokta için bunları en iyi ortalayan mx+b şeklinde bir polinom uydurur. polinom derecesi artarsa benzeme artar.
    return left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds
fig, ax = plt.subplots()
while True:  
    # Videodan bir kare oku
    success, image = vidcap.read()

    if not success:
    # Eğer video okuma işlemi tamamlandıysa, videoyu başa sar
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue  # Bir sonraki tekrara geçerek videonun ilk karesini okumaya devam etq

    
    # Görüntüyü yeniden boyutlandır
    frame = cv2.resize(image, (640, 480))
    frame2 = cv2.resize(image, (640, 480))
    
    # Görüntüyü gri tonlamaya çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sobel filtresini uygula
    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=1)#x ekseni için (-1,0,1;-2,0,2;-1,0,1) kerneli ile konvolüsyon alınır
    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)#y ekseni için (1,2,1;0,0,0;1,2,1) kerneli ile konvolüsyon alınır
 
    # Gradyan büyüklüklerini hesapla
    gradient_magnitude_x = np.abs(sobel_x)
    gradient_magnitude_y = np.abs(sobel_y) #mutlak değerini alıyor büyüklüğünü öyle belirliyor.
    # Sobel x ve Sobel y gradyanlarını bitwise AND operatörü kullanarak birleştir
    combined_gradient = cv2.bitwise_and(gradient_magnitude_x, gradient_magnitude_y)#makalede söylenen and operatörüne sokuyoruz.
    # Gradyan büyüklüğünü [0, 255] aralığına normalize et
    combined_gradient = cv2.normalize(combined_gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #normalize edip uint8e çeviriyoruz işaretsiz tamsayı
    #ROI belirleme köşe noktaları
    tl = (220, 387)
    bl = (70, 472)
    tr = (400, 380)
    br = (538, 472)
   # Orijinal kare üzerine çizgiler çiz
    cv2.line(frame, tl, tr, (0, 0, 255), 1)
    cv2.line(frame, tr, br, (0, 0, 255), 1)
    cv2.line(frame, br, bl, (0, 0, 255), 1)
    cv2.line(frame, bl, tl, (0, 0, 255), 1)
    # Perspektif dönüşüm noktalarını tanımla
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Kuşbakışı görüntü için görüntüyü çevirmek için matris
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(combined_gradient, matrix, (640, 480)) 
    # sliding_window fonksiyonunun döndürdüğü değerleri al
    left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds = sliding_window(transformed_frame)
    histogram = np.sum(transformed_frame[transformed_frame.shape[0]//2:, :], axis=0)#satırları 2 ye böler ve üst yarısını alır.
    ax.clear()
    ax.plot(histogram)
    ax.set_title("Histogram")
    ax.set_xlabel("Sütunlar")
    ax.set_ylabel("Piksel Sayısı")
    plt.pause(0.00001)  
    
    window_width = 60  # Pencere genişliği

    ploty = np.linspace(0, transformed_frame.shape[0] - 1, transformed_frame.shape[0])#başlangıcı bitişi arasında belirli aralıklarla artan dizi tanımlar
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2] #sağ ve sol şeridin x koordinatlarını polinom ile hesaplar
    ############# makalede belirtilen şerit polinomunu sağ şeritten sol şeridin tepe bölgesine kaydırma işlemini yapamadım 
    ############# ikisi için ayrı polinom çarpımı yaptım.

    # Şeritleri görsel olarak göstermek ve şeritler arasındaki alanı doldurmak için işlemler
    margin = 40

    # Sol şerit için pencere alanını oluşturun
    left_line_window1 = np.vstack([left_fitx - margin, ploty]).T
    left_line_window2 = np.flipud(np.vstack([left_fitx + margin, ploty]).T)
    left_line_pts = np.concatenate((left_line_window1, left_line_window2), axis=0)
    # Sağ şerit için pencere alanını oluşturun
    right_line_window1 = np.vstack([right_fitx - margin, ploty]).T
    right_line_window2 = np.flipud(np.vstack([right_fitx + margin, ploty]).T)
    right_line_pts = np.concatenate((right_line_window1, right_line_window2), axis=0)
    """"burada yapılmak istenen olay x ve y eksenlerini yatayda genişleyen bir matris olarak en başta yatayda genişleyen bir fonksiyon olarak 
    belirlemek ve bunları dikey eksende birleştirmek ardından bu fonksiyonun transpozunu alarak bu matrixi [x + margin,y + 100] yani x değerinin 
    y konumunda denk geldiği yeri belirlemek en başta sol şeridin + margin kısmını alıyoruz ve bundan sonra sol şeridin + kısmını alarak bu matrisi de 
    alttan üste ters çeviriyoruz bu seferde [x-margin, y+100] alt alta geliyor ve aynı sırayla y değeri ve x değeri azalıyor bu sayede hem sağ hem
    sol kısmını çizebiliyoruz. 100 sayısı uydurmadır paint üzerinden çizerek anlatabilirim ya da kodu çalıştırabilirim hocam. 
    """""
    # Boş bir frame oluşturma
    reverse_perspective_frame = np.zeros_like(frame)

    # Sol ve sağ şeritleri yeşil renkle doldur
    cv2.fillPoly(reverse_perspective_frame, [np.int32(left_line_pts)], (0, 255, 0))
    cv2.fillPoly(reverse_perspective_frame, [np.int32(right_line_pts)], (0, 255, 0))

    # Şeritler arasındaki alanı kırmızı renkle doldur
    fill_pts_left = np.array([np.transpose(np.vstack([left_fitx , ploty]))])
    fill_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx , ploty])))])
    fill_pts = np.hstack((fill_pts_left, fill_pts_right))
    cv2.fillPoly(reverse_perspective_frame, [np.int32(fill_pts)], (0, 0, 255))
    

    # Perspektif dönüşüm matrisini oluştur
    reverse_perspective_matrix = cv2.getPerspectiveTransform(pts2, pts1)

    # Ters perspektife dönüşüm uygula
    reverse_perspective_frame = cv2.warpPerspective(reverse_perspective_frame, reverse_perspective_matrix, (frame.shape[1], frame.shape[0]))

    # Orijinal çerçeve ile ters perspektif çerçevesini birleştir
    result = cv2.addWeighted(frame2, 1, reverse_perspective_frame, 0.3, 0)
  
    # Sonucu göster
    cv2.imshow("Sliding Window Lane Detection", result)

    # Sobel filtresi uygulanmış görüntüyü göster
    cv2.imshow("Sobel Filtered Image", combined_gradient)

    # Ana görüntüyü göster
    cv2.imshow("Original Frame", frame)

    # Binarye çevrilmiş görüntüyü göster
    cv2.imshow("Transformed Binary Image", transformed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #q ya basınca çıkmamızı sağlar
        break

vidcap.release() #Videoyu sürekli baştan oynatır


cv2.destroyAllWindows()
