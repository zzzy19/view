import cv2
import pyzbar.pyzbar as pyzbar 



def decodeDisplay(image,image1):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
 
        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image1, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)
 
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    return image1
 
def detect():
    cap = cv2.VideoCapture(0)
 
    while True:
        # 读取当前帧
        ret, img = cap.read()
        # 转为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = decodeDisplay(gray,img)
 
        key = cv2.waitKey(5)
        cv2.namedWindow('image', 0)
        cv2.resizeWindow('image', 700, 500)
        cv2.imshow("image", im)
        if key & 0xFF == ord('q'):
            break
 
    cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()

decodeDisplay(image,image1)
