import streamlit as st
import torch
from function import *
from PIL import Image

openai.api_key = st.secrets["api_key"]
# Streamlit app
st.title("너의 기분은?")
st.write('서울대학교 과학영재교육원 정보과학 사사과정 박준태 최정원 김민서 이영풍')
st.write("""1. 당신의 표정을 담은 사진(얼굴이 화면에 최대한 꽉 차게)을 찍고 먼저 ‘감정 확인하기!’ 버튼을 눌러주세요. 
2. 당신의 표정을 인식한 결과가 출력되면, 자신의 표정과 일치하지 않는다면 ‘Clear Photo’ 버튼을 이용하여 다시 찍어주세요. 
3. 원하는 인식 결과가 나왔다면, 이제 당신의 상황에 대한 간단한 문장(2~3문장)을 입력해주세요. 
4. ‘상담 시작!’ 버튼을 누르고 결과를 기다려주세요.
5. 설문지를 꼭 남겨주세요!""")

#dl
target_size = (48, 48)
emotion_dict = {0 : "화남", 1 : "역겨움", 2 : "두려움", 3 : "행복함", 4 : "중립", 5 : "슬픔", 6 : "놀라움"}

img_file_buffer = st.camera_input("사진을 찍어주세요!")
resized_img = False
if img_file_buffer is not None:
    # To read image file buffer as a 3D uint8 tensor with `torchvision.io`:
    img = Image.open(img_file_buffer)
    
    resized_img = change_image(img)
    
    st.image(img)

    #표정 확인
    feeling_clicked = st.button("감정 확인하기!")
    model = load_model()
    if feeling_clicked:
        st.balloons()
        # resized img 1 48 48 --> 1 1 48 48
        resized_img = torch.unsqueeze(resized_img, 0) # 1 1  48 48  
        output = model(resized_img)
        output = output.detach().cpu().numpy()
        emotion = np.argmax(output)
        st.write("표정을 확인했습니다!")
        st.write(f"당신의 표정 : {emotion_dict[emotion]}") 

    user_input = st.text_input("쳇봇에게 상담받고싶은 내용을 작성해주세요(예:나 오늘 너무 슬픈일이 있었어):")
    # Handle user input and generate system responses
    with st.chat_message("user"):
        st.write(user_input)

    text_clicked = st.button("상담 시작!")


    if text_clicked:
        st.balloons()
        if user_input : 
            prompt = complet_prompt(emotion_dict[emotion], str(user_input))
            response = get_response(prompt)

            with st.chat_message("ai"):
                st.write(response) #출력

        if not user_input:
            st.write("대화를 시작해 주세요!")

url = "https://forms.gle/gLP1Qm9gNkQLZ9B66"
st.subheader("[후기를 남겨주세요](%s)" % url)
        