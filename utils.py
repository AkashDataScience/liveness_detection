import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_train_val_test_split(data_df):
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=1,
                                         stratify=data_df['liveness_score'])
    valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=1,
                                         stratify=test_df['liveness_score'])
    return train_df, valid_df, test_df

def extract_frames_and_create_csv(video_dir, df, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    label_data = {"filename": [], "label": []}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_fname = row["fname"]
        label = row["liveness_score"]
        video_path = os.path.join(video_dir, video_fname)

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(num_samples):
            frame_idx = int(i * num_frames / num_samples)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_name = f"{os.path.splitext(video_fname)[0]}_{i}.jpg"
                frame_path = os.path.join(images_dir, frame_name) 
                cv2.imwrite(frame_path, frame)
                label_data["filename"].append(frame_name)
                label_data["label"].append(label)

    label_df = pd.DataFrame(label_data)

    label_csv_path = os.path.join(output_dir, "labels.csv")
    label_df.to_csv(label_csv_path, index=False)

def get_classification_data(model, device, test_loader):
    model.eval()
    model.to(device)

    classified_data = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            for image, label in zip(data, target):
                image = image.unsqueeze(0)
                output = model(image)
                pred = output.argmax(dim=1, keepdim=True)
                classified_data.append((image, label, pred))

    return classified_data