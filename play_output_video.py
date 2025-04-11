import cv2

# Load the processed output video
cap = cv2.VideoCapture('output_parking.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Output Parking Video', frame)
    if cv2.waitKey(25) == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
