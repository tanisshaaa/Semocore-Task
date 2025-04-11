import cv2

# Load the first frame of the video
video_path = 'parking_lot_video.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read the video.")
    exit()

# Store drawn rectangles
slots = []
drawing = False
ix, iy = -1, -1

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame, slots

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        slots.append([min(ix, x), min(iy, y), max(ix, x), max(iy, y)])
        print(f"🟩 Slot {len(slots)}: {[min(ix, x), min(iy, y), max(ix, x), max(iy, y)]}")

# Set up the drawing window
cv2.namedWindow('Draw Parking Slots')
cv2.setMouseCallback('Draw Parking Slots', draw_rectangle)

print("🔰 Draw parking slots using the left mouse button.")
print("➡️ Press 's' to save and exit, 'r' to reset, 'q' to quit without saving.")

while True:
    cv2.imshow('Draw Parking Slots', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        frame = frame.copy()
        slots = []
        print("🔄 Reset all slots.")
        
    elif key == ord('s'):
        with open('parking_slots.py', 'w') as f:
            f.write("# Auto-generated parking slot coordinates\n")
            f.write("parking_slots = [\n")
            for slot in slots:
                f.write(f"    {slot},\n")
            f.write("]\n")
        print("✅ Slots saved to parking_slots.py")
        break

    elif key == ord('q'):
        print("❌ Quit without saving.")
        break

cv2.destroyAllWindows()
