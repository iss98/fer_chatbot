import streamlit as st
import torch
import torchvision
from function import *
from PIL import Image

openai.api_key = st.secrets["api_key"]
# Streamlit app
st.title("Chatbox Example")
#dl
target_size = (48, 48)
emotion_dict = {0 : "화남", 1 : "역겨움", 2 : "두려움", 3 : "행복함", 4 : "중립", 5 : "슬픔", 6 : "놀라움"}

img_file_buffer = st.camera_input("Take a picture")
resized_img = False
if img_file_buffer is not None:
    # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
    img = Image.open(img_file_buffer)
    
    resized_img = change_image(img)
    
    st.image(img)

    st.write("표정을 확인했습니다!")

    user_input = st.text_input("Chat with the system :")

    # Handle user input and generate system responses
    with st.chat_message("user"):
        st.write(user_input)

    
    button_clicked = st.button("Click to Perform Task")
    model = load_model()

    if button_clicked:
        # resized img 1 48 48 --> 1 1 48 48
        resized_img = torch.unsqueeze(resized_img, 0) # 1 1  48 48  
        output = model(resized_img)
        output = output.detach().cpu().numpy()
        emotion = np.argmax(output)
        if user_input : 
            prompt = complet_prompt(emotion_dict[emotion], str(user_input))
            response = get_response(prompt)

            with st.chat_message("ai"):
                st.write(f"당신의 표정 : {emotion_dict[emotion]}")
                st.write(response) #출력

        if not user_input:
            st.write("대화를 시작해 주세요!")
        