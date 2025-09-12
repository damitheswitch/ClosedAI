import cv2
cap = cv2.VideoCapture(0)  # /dev/video0 from DroidCam stream
ret, frame = cap.read()
if ret:
    cv2.imwrite('test_frame.jpg', frame)
    print("Success! Frame saved as test_frame.jpg (your face from phone). Check with: ls -l test_frame.jpg")
else:
    print("Failed: Check if ffmpeg stream is running in other terminal.")
cap.release()
cv2.destroyAllWindows()
