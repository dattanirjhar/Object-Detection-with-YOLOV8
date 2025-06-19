import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
  ret, img= cap.read()
  cv2.imshow("Webcam Feed", img)
  
  if cv2.waitKey(1) == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()