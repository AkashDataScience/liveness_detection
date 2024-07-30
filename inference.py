import cv2
import torch
import models
import argparse
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

transforms = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ToTensorV2()])

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

CLASS_NAMES= ['Fake', 'Real']

def get_args():
    parser = argparse.ArgumentParser(description='Liveness Detection - Zalo AI Challenge Inference')
    parser.add_argument('--image', action=argparse.BooleanOptionalAction)
    parser.add_argument('--model_path', default="model.pth", type=str, help="Path of saved model")
    parser.add_argument('--data_path', type=str, help="Path of image or video to predict")
    args = parser.parse_args()
    return args

def inference_on_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    image_tensor = transforms(image=image)['image']
    image_tensor = image_tensor.unsqueeze(0)  

    with torch.no_grad():
        model.eval()
        output = model(image_tensor.to(device))
        pred=output.argmax(dim=1,keepdim=True).item()

    print("Predicted label:", CLASS_NAMES[pred])

def inference_on_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transforms(image=frame)['image']
        frame_tensor = frame_tensor.unsqueeze(0) 

        with torch.no_grad():
            model.eval()
            output = model(frame_tensor.to(device))
            pred=output.argmax(dim=1,keepdim=True).item()
        predictions.append(pred)
    
    cap.release()

    # Calculate the final prediction
    final_prediction = sum(predictions) / len(predictions)
    if final_prediction>0.5:
        final_prediction=1
    else:
        final_prediction=0
    print("Final prediction:", CLASS_NAMES[final_prediction])

def infer():
    args = get_args()
    model = models.ResNet18(num_classes=2).to(device)
    model = model.load_state_dict(args.model_path)
    if args.image:
        inference_on_image(args.data_path, model)
    else:
        inference_on_video(args.data_path, model)

if __name__ == "__main__":
    infer()