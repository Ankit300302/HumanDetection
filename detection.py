import cv2
import numpy as np

def load_video(source):
    """Load video from local file."""
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    return cap

def detect_humans(frame):
    """Detect humans using HOG Descriptor."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect humans in the frame
    (humans, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # Draw rectangles around detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame, humans

def initialize_trackers(frame, humans):
    """Initialize object trackers for each detected human."""
    trackers = []
    
    for (x, y, w, h) in humans:
        tracker = cv2.TrackerKCF_create()  # You can choose other algorithms like CSRT, MOSSE, etc.
        bbox = (x, y, w, h)
        tracker.init(frame, bbox)
        trackers.append(tracker)
    
    return trackers

def update_trackers(frame, trackers):
    """Update the tracker positions and draw bounding boxes."""
    human_coordinates = []
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Display the coordinates on the screen
            coord_text = f"({x}, {y}, {w}, {h})"
            cv2.putText(frame, coord_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # Collect coordinates of the bounding box
            human_coordinates.append((x, y, w, h))
    
    return frame, human_coordinates

def calculate_frame_difference(frame1, frame2):
    """Calculate the absolute difference between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate the difference between frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference to ignore minor changes
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Sum the difference values to get a single number representing the change
    return np.sum(diff_thresh)

def process_video(source):
    """Main function to process video and perform detection and tracking."""
    
    # Load the video file directly
    cap = load_video(source)
    
    if not cap:
        return
    
    trackers = []
    frame_count = 0
    prev_frame = None
    detect_interval = 10  # Default detect interval
    detect_threshold = 500000  # Threshold for scene changes (based on experimentation)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate the frame difference to determine if detection is needed
        if prev_frame is not None:
            frame_difference = calculate_frame_difference(prev_frame, frame)
            
            if frame_difference > detect_threshold:
                print(f"Scene change detected at frame {frame_count}, difference: {frame_difference}")
                detect_interval = 5  # Decrease interval if scene changes
            else:
                detect_interval = 20  # Increase interval if little motion
            
        prev_frame = frame.copy()  # Update the previous frame for the next iteration
        
        # Perform human detection every detect_interval frames or if trackers are empty
        if frame_count % detect_interval == 0 or not trackers:
            frame, humans = detect_humans(frame)
            trackers = initialize_trackers(frame, humans)
        else:
            # Update the trackers in every other frame
            frame, human_coordinates = update_trackers(frame, trackers)
            
            # Print or log the coordinates of the tracked humans
            print(f"Frame {frame_count} - Tracked human coordinates: {human_coordinates}")
        
        # Display the frame
        cv2.imshow("Frame", frame)
        
        # Wait for a short period and check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Use the provided local video path
process_video("v3.mp4")