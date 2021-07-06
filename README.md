# <h1 align = 'center'> Air Handwriting Detection And Recognition </h1>

This project is divided into two parts:

* Detection 
* Recognition 

### Execution:
Execute the air-handwriting-detection-and-recognnition.ipynb with color_selector.py in the same folder if you are running code on jupyter notebook or VS code or any local host machine with replacement of path of dataset and last_frame.png to where you have saved your dataset and last_frame.png.

If execution is done using Google Colab; Execute the detection.py with color_selector.py in the same folder of code using VS Code or prefered python editor and OCR.ipynb in google colab by uploading last_frame.png into your google drive and replacing the path of both dataset and last_frame.png.

Initially `Training` is set to `True`, Changing it to false will use the trained model but initially you have to train the model.

### **Detecting the Alphabets:**

![Screenshot (95)](https://user-images.githubusercontent.com/62256509/117626260-89328e00-b194-11eb-8320-b5f25dc9cf84.png)


### **The Output obtained:**

![mediahandler](https://user-images.githubusercontent.com/62256509/117627675-09a5be80-b196-11eb-85df-f5e419bb4766.gif)



### **Recognising the Alphabets:**

![Screenshot (97)](https://user-images.githubusercontent.com/62256509/117626368-a8c9b680-b194-11eb-9694-a99feb3010a9.png)


### **Dataset Description:**

#### **Link to Dataset** 

[A-Z Dataset](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format)

#### **Ploting Dataset**

This image shows a histogram plot of complete dataset, representing number of entries of each label.

![Screenshot (101)](https://user-images.githubusercontent.com/62256509/117628584-f515f600-b196-11eb-8191-b9e1ab48e940.png)

Ploting Images:

* Training Data
* Testing Data (Shows the predicted label)

![Screenshot (102)](https://user-images.githubusercontent.com/62256509/117632422-beda7580-b19a-11eb-805c-183779ffa422.png)

### **The output obtained:**

![Screenshot (96)](https://user-images.githubusercontent.com/62256509/117632612-e92c3300-b19a-11eb-86f7-6088d6cec561.png)

### **Confusion Matrix of testing Data:**
![image](https://user-images.githubusercontent.com/54438860/124570642-b59d1a80-de64-11eb-9c44-ee054c006242.png)

### **Confusion Matrix of training Data:**
![image](https://user-images.githubusercontent.com/54438860/124570830-dbc2ba80-de64-11eb-9abc-2d441432dbf6.png)

### **Performance Comparision Between RMSProp and ADAM optimizer:**

`test1` and `train1` are the output of the **ADAM** while `test2` and `train2` are the output of **RMSProp**

![image](https://user-images.githubusercontent.com/54438860/124571194-32c88f80-de65-11eb-8e87-327c0e6eb525.png)


### **Team Of:**

* [Abhishek chaudhary](https://github.com/chaudhary312)
* [Khushi Agrawal](https://github.com/khushi-411)
* [Prabhat Kumar](https://github.com/prabhatk579)
