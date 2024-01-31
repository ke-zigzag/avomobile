import streamlit as st
import requests
from PIL import Image

st.title("Avocado checker")

upload_file = st.file_uploader("Please check your avocado here!", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="Avocado Image", width=250)
    
    #FASTAPIサーバーに画像を送信
    files = {"file":upload_file.getvalue()}
    #response = requests.post("https://avofinal.onrender.com/predict", files=files)
    #/predict

    
    #ローカル用
    response = requests.post("http://localhost:10000/predict", files=files)


    # 応答の内容をコンソールに表示
    print("Response status code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error reading JSON response:", e)

    #応答として受け取った予測結果を表示
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

        