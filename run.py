from camera import Camera
import cv2
import torch
import torchvision.transforms as transforms
import timeit


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_path='fp32', map_location=device)
    nvidia_ssd_processing_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    ssd_model.to(device)
    ssd_model.eval()

    rtsp_camera = Camera('rtsp://192.168.1.180:554/stream1')

    while cv2.waitKey(33):
        start_t = timeit.default_timer()

        image = rtsp_camera.get_frame()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed_image = transform(image)
        tensor = torch.tensor(transformed_image, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            detections = ssd_model(tensor)
        results_per_input = nvidia_ssd_processing_utils.decode_results(detections)
        best_results_per_input = [nvidia_ssd_processing_utils.pick_best(results, 0.45) for results in results_per_input]

        for image_idx in range(len(best_results_per_input)):
            bboxes, classes, confidences = best_results_per_input[image_idx]
            for idx in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[idx]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

        terminate_t = timeit.default_timer()
        FPS = int(1. / (terminate_t - start_t))
        cv2.putText(image, 'FPS : %s' % (str(FPS)), (20, 20), 1, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("RTSP_IP_CAMERA_LOW_LATENCY", image)

    cv2.destroyAllWindows()
