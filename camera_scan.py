import cv2

camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not camera.isOpened():
    print("Could not open Camera 0")
    exit()

print("Camera 0 opened successfully.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    cv2.imshow("AirControl AI - Camera 0", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()