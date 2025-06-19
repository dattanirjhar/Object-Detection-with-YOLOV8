import cv2
from ultralytics import YOLO
import math
import threading
from queue import Queue
import time

# A thread-safe queue to hold frames for processing
frame_queue = Queue(maxsize=1)

# A thread-safe queue to hold results for display
results_queue = Queue(maxsize=1)

def video_capture_thread(cap):
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Capture thread: Failed to read frame, exiting.")
            break
        # If the queue is full, it means the inference thread is busy.
        # We remove the old frame and put the new one to process the latest.
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()  # Discard the old frame
            except Queue.empty:
                pass
        frame_queue.put(img)
    print("Capture thread finished.")
    cap.release()

def inference_thread(model, classNames):
    while True:
        try:
            # Wait until a frame is available
            img = frame_queue.get(timeout=1) # Use timeout to prevent indefinite blocking

            # Perform inference with a confidence threshold
            results = model(img, stream=True, conf=0.5, verbose=False)

            # Draw boxes and labels on the image
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    className = classNames[cls]
                    
                    # This print statement is useful for debugging
                    print("Confidence --->", confidence, "Class name -->", className)

                    org = [x1, y1 - 10]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.8
                    color = (255, 255, 255)
                    thickness = 2
                    cv2.putText(img, f"{className} {confidence}", org, font, fontScale, color, thickness)

            # Put the processed image in the results queue
            if not results_queue.empty():
                try:
                    results_queue.get_nowait() # Discard old result
                except Queue.Empty:
                    pass
            results_queue.put(img)

        except Queue.Empty:
            # This can happen if the capture thread stops. We can just continue.
            continue
        except Exception as e:
            # If any other error occurs, we can break the loop
            print(f"Inference thread error: {e}")
            break
    print("Inference thread finished.")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("yolo-weights/yolov8n.pt")
    classNames = model.names

    # Start the threads
    capture_thread = threading.Thread(target=video_capture_thread, args=(cap,), daemon=True)
    proc_thread = threading.Thread(target=inference_thread, args=(model, classNames), daemon=True)
    
    capture_thread.start()
    proc_thread.start()

    # --- MAIN DISPLAY LOOP ---
    latest_frame = None

    # Wait briefly for threads to start and populate queues
    while latest_frame is None:
        if not frame_queue.empty():
             latest_frame = frame_queue.get()
        else:
             time.sleep(0.1)

    while True:
        # Check if there is a new processed frame
        if not results_queue.empty():
            latest_frame = results_queue.get()

        # Display the latest available frame
        if latest_frame is not None:
            cv2.imshow("Optimized Object Detection", latest_frame)

        # Allow user to quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated gracefully.")

if __name__ == "__main__":
    main()