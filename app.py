import streamlit as st
import requests
from PIL import Image

st.title("Avocado checker")

#画像アップロード
uploaded_file = st.file_uploader("Please check your avocado here!", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Avocado Image", width=250)
    
    #FASTAPIサーバーに画像を送信
    #files = {"file":("filename", uploaded_file.getvalue())}
    files = {"file":uploaded_file.getvalue()}
    #response = requests.post("https://avofinal.onrender.com/predict", files=files)

    
    #ローカル用
    response = requests.post("http://localhost:8000/predict", files=files)


    #応答の内容をコンソールに表示
    print("Response status code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error reading JSON response:", e)

    #予測結果を表示
    if response.status_code == 200:
        result = response.json()
        if result['result'] == '0':
           st.write("Wait a few days...")
        elif result['result'] == '1':
           st.write("Eat me")
        else:
            st.write("Really avocado??")
        
    else:
        st.write("Error??")

        