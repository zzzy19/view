import cv2
import numpy as np

def color_detection(frame):
    """
    检测图像中的红、蓝、绿三种颜色，并在检测到的颜色区域周围绘制矩形框
    
    参数:
        frame: 输入图像帧 (BGR格式)
        
    返回:
        带有检测框和标签的图像
    """
    # 转换为HSV颜色空间 (更适合颜色检测)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 定义颜色范围 (HSV格式)
    # 红色有两个范围，因为它在HSV色环的两端
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # 蓝色范围
    lower_blue = np.array([90, 120, 70])
    upper_blue = np.array([130, 255, 255])
    
    # 绿色范围
    lower_green = np.array([40, 120, 70])
    upper_green = np.array([80, 255, 255])
    
    # 创建颜色掩膜
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # 合并两个红色掩膜
    
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # 形态学操作 (去除噪声)
    kernel = np.ones((7, 7), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 处理红色物体
    for contour in red_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Red", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # 处理蓝色物体
    for contour in blue_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Blue", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # 处理绿色物体
    for contour in green_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Green", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误: 无法打开摄像头")
    exit()

print("摄像头已打开，按 'q' 键退出...")

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    
    if not ret:
        print("错误: 无法获取帧")
        break
    
    # 水平翻转帧 (使显示更自然)
    frame = cv2.flip(frame, 1)
    
    # 进行颜色识别
    result = color_detection(frame)
    
    # 显示结果帧
    cv2.imshow("Color Detection", result)
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()