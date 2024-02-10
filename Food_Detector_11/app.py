import pathlib
from pathlib import Path 
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b2
import tempfile
import os
import streamlit as st
import PIL
from PIL import Image


model_path = Path(r"C:\Users\Lenovo\Food_Detector_11\models")

 
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
transforms_1 = weights.transforms()
auto_transform = transforms.Compose([transforms.TrivialAugmentWide(),
                                     transforms_1])
fd_model = efficientnet_b2(weights = weights).to("cpu")
for params in fd_model.parameters():
  params.requires_grad = False

fd_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3,inplace = True),
    nn.Linear(in_features = 1408,
              out_features = 11)
)

fd_model.load_state_dict(torch.load(f= "C:/Users/Lenovo/Food_Detector_11/models/effnetb2_model_1.pth",
                                    map_location=torch.device('cpu')))

#print(next(iter(fd_model.parameters())).device)

st.title("FOOD DETECTOR 11")
st.write("Discover our Food Detector App, a PyTorch-based image classifier trained on 10,000 images across 11 food classes. With an impressive accuracy exceeding 80%, effortlessly identify and enjoy precision in recognizing a variety of dishes. Revolutionize your culinary experience with our cutting-edge technology!",)
st.write("Classes = 'Bread', 'Dairy product', 'Dessert', 'Egg','Fried food', 'Meat', 'Noodles-Pasta','Rice', 'Seafood', 'Soup', 'Vegetable-Fruit' ")
file_uploader = st.sidebar.file_uploader("upload iMAGE",type = "jpg")
def predict(img):
        transformed_image = auto_transform(img).to("cpu")
        classes = ['Bread', 'Dairy product', 'Dessert', 'Egg',
                   'Fried food', 'Meat', 'Noodles-Pasta',
                   'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']
        fd_model.eval()
        with torch.inference_mode() :
          input_img = torch.unsqueeze(transformed_image,dim = 1)
          input_image = input_img.permute(1,0,2,3)
          #print(input_image.shape)
          img_pred = fd_model(input_image)
          raw_logits = torch.softmax(img_pred,dim=1)
          pred_img_label = torch.argmax(raw_logits,dim = 1)
          #print(raw_logits)
          st.title(classes[pred_img_label])
          st.image(img)
          

container = st.container()
    
with container:
    st.write("EXAMPLES")
    with st.form(key = "my_form", clear_on_submit=True):
        img1 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/0 (1).jpg")
        img2 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/0.jpg")
        img3 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/1 (1).jpg")
        img4 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/1.jpg")
        img5 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/11.jpg")
        img6 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/12.jpg")
        img7 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/102 (1).jpg")
        img8 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/102.jpg")
        img9 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/104.jpg")
        img10 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/106.jpg")
        img11 = Image.open("C:/Users/Lenovo/Food_Detector_11/examples/107.jpg")
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        col1.image(img1,caption = "1",width = 100)
        col1.image(img2,caption ="2", width = 100)
        col2.image(img3,caption = "3", width = 100)
        col2.image(img4,caption = "4", width = 100)
        col3.image(img5,caption = "5", width = 100)
        col3.image(img6,caption = "6", width = 100)
        col4.image(img7,caption = "7", width = 100)
        col4.image(img8,caption = "8", width = 100)
        col5.image(img9,caption = "9", width = 100)
        col5.image(img10,caption = "10", width = 100)
        col6.image(img11,caption= "11" ,width = 100)
        images = [img1 ,img2 ,img3,img4,img5 ,img6 ,img7,img8 ,img9 ,img10 ,img11]
            
        num = st.selectbox(label = "select image number", options = ["1","2","3","4","5","6","7","8","9","10","11"],
                           placeholder = "select any number")
        #print(num)
        
        submit_bttn = st.form_submit_button("Submit")
        
    
    if submit_bttn:
        #st.title("Jai Shree Ram")
        predict(images[int(num)-1])
        
                      
if file_uploader :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_uploader.getvalue())
        tmp_file_path = tmp_file.name
    
    #print(tmp_file_path)
    img = Image.open(tmp_file_path)
    predict(img)

