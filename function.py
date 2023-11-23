import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms
import openai
import torchvision
import numpy as np

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Corrected in_channels to 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )
        self.flat = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, 7)  # Adjust the number of output units for your classification task
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)
        x = self.classifier(x)
        return x
    
@st.cache_data
def load_model():
    loaded_model = VGG16()
    loaded_model.load_state_dict(torch.load("./save/3_3_0.0001.pt",map_location=torch.device('cpu')))
    loaded_model.eval()
    return loaded_model



@st.cache_data
def complet_prompt(face, talk):
    prompt1 = "너는 이제부터 내 감정을 공감하며 상담해 주는 상담사야. 나는 너에게 나의 [표정]과 [상황]을 알려줄 거야. 공감하는 말(자세히, 이해하면서)을 50%, 해결할 수 있는 방법(노래 듣기, 다이어리 쓰기 등)을 50%의 비율로 하고, 부드러운 대화체로 1~2분 분량으로 말해줘. 상담할 때는 이모티콘을 적절히 사용해주면 좋을 것 같아. 그리고 나의 표정을 언급하면서 공감해주면 좋아. 상담할 때에는 존댓말로 상담해 줘. \n"
    prompt2 = "[표정]"
    prompt3 = "[상황]"
    return prompt1 + prompt2 + face + "\n" + prompt3 + talk

@st.cache_data
def get_response(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt}
    ])
    return response['choices'][0]['message']['content']

@st.cache_data
def get_face(image):
    face = torchvision.transforms.ToTensor()(image)
    
def change_image(tensor_img):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 흑백 변환
    transforms.Resize((48, 48)),  # 크기 조정
    transforms.ToTensor()  # Tensor로 변환
    ])
    return transform(tensor_img)