import cv2

class Visualizer:
    def __init__(self):

        self.class_names = {
            0:"person",
            2:"car",
            3:"motorcycle",
            5:"bus",
            7:"truck"
        }

    def draw(self, image, tracks):

        for obj in tracks:

            x,y,w,h = obj["x"], obj["y"], obj["width"], obj["height"]
            class_id = obj["class_id"]
            track_id = obj["track_id"]
            conf = obj["confidence"]

            name = self.class_names.get(
                class_id,
                str(class_id)
            )

            label = (
                f"{name} "
                f"ID:{track_id} "
                f"{conf:.2f}"
            )

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

            cv2.putText(
                image,
                label,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

        return image