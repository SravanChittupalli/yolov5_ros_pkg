# image_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self, source='folder', image_folder=None, publish_rate=1.0):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.bridge = CvBridge()
        self.publish_rate = publish_rate  # Frequency in Hz
        self.source = source

        if source == 'folder' and image_folder:
            # Image folder mode
            self.image_folder = image_folder
            self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg'))]
            self.current_image_index = 0
        elif source == 'webcam':
            # Webcam mode
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.get_logger().error("Failed to open webcam.")
                rclpy.shutdown()
                return
        else:
            self.get_logger().error("Invalid source. Choose 'folder' or 'webcam'.")
            rclpy.shutdown()
            return

        self.timer = self.create_timer(1.0 / publish_rate, self.publish_image)

    def publish_image(self):
        if self.source == 'folder':
            # Read image from folder
            if self.current_image_index >= len(self.image_files):
                self.current_image_index = 0  # Loop back to the first image if needed

            image_path = self.image_files[self.current_image_index]
            cv_image = cv2.imread(image_path)

            if cv_image is not None:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.publisher_.publish(ros_image)
                self.get_logger().info(f'Published image from file: {image_path}')
            else:
                self.get_logger().warn(f'Could not read image: {image_path}')

            self.current_image_index += 1

        elif self.source == 'webcam':
            # Capture image from webcam
            ret, cv_image = self.cap.read()
            if ret:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.publisher_.publish(ros_image)
                self.get_logger().info('Published image from webcam')
            else:
                self.get_logger().warn('Failed to capture image from webcam')

    def destroy_node(self):
        # Release webcam if used
        if self.source == 'webcam' and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # Parameters
    source = 'folder'  # 'folder' or 'webcam'
    image_folder = '/home/sasuke/Projects/yolov5/datasets/pallets/test/images/'  # Only needed if source is 'folder'
    publish_rate = 1.0  # Frequency in Hz

    # Initialize the node with the specified source
    node = ImagePublisher(source=source, image_folder=image_folder if source == 'folder' else None, publish_rate=publish_rate)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()