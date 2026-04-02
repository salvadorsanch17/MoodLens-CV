import cv2
from deepface import DeepFace

# How often to run analysis (every N frames) to keep it real-time
ANALYZE_EVERY_N_FRAMES = 5

def draw_results(frame, result):
    """Draw emotion label and confidence bar on the frame."""
    dominant = result.get("dominant_emotion", "")
    emotions = result.get("emotion", {})

    # Draw dominant emotion label
    cv2.putText(
        frame, f"Mood: {dominant}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )

    # Draw emotion bars
    y = 80
    for emotion, score in sorted(emotions.items(), key=lambda x: -x[1]):
        bar_width = int(score * 2)  # scale 0-100 → 0-200 px
        cv2.rectangle(frame, (20, y), (20 + bar_width, y + 14), (0, 200, 255), -1)
        cv2.putText(
            frame, f"{emotion}: {score:.1f}%",
            (230, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        y += 22

def draw_face_box(frame, result):
    """Draw bounding box around detected face."""
    region = result.get("region", {})
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Press 'q' to quit.")

    frame_count = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        # Run DeepFace analysis every N frames
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,  # don't crash if no face found
                    silent=True,
                )
                # analyze() returns a list; take the first face
                last_result = results[0] if isinstance(results, list) else results
            except Exception as e:
                last_result = None
                cv2.putText(frame, f"Error: {e}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Overlay results from the last successful analysis
        if last_result:
            draw_face_box(frame, last_result)
            draw_results(frame, last_result)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("MoodLens — Real-time Emotion Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
