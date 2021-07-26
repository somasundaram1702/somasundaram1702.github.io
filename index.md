# Portfolio

## White papers

### Real time facial recognition system on Nvidia AGX Xavier 

[![White paper](https://img.shields.io/badge/PDF-Read_paper-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/Face-on-edge-realtime-face-recognition.pdf)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/somasundaram1702/Unknown-face-recognition)

<div style="text-align: justify">A face recognition model developed and deployed on the Nvidia Jetson Xavier AGX board. The model can identify unknown and known faces with an average accuracy of 96% with well-known datasets like LFW and VGGFace2. The model is optimized using the TensorRT 
module, which improved the inference speed 10-15 times.</div>
<br>
<center><img src="images/Facerec_xavier_3.png"/></center>
<br>

---
### Extreme value machine: An algorithm for openset classification

[![White paper](https://img.shields.io/badge/PDF-Read_paper-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/EV_machine.pdf)

<div style="text-align: justify">Extreme value machine (EVM) is a state of art algorithm for open set classification tasks. This algorithm is readily available in python, however implementations in other languages like C++ & Java are not handy. In this work we have developed an EVM algorithm using Java and integrated the algorithm in our android face recognition pipeline. This paper explains construction and working of the algorithm.</div>
<br>
<center><img src="images/EVM_pic3.png"/></center>
<br>

---
## Patents
### A tube inspection system using Artificial Intelligence

<div style="text-align: justify">An inspection system for inspecting the internal surface of tubes was designed and developed. A light and efficient Convolutional Neural Network model was designed & trained to automatically identify and classify 5 different types of defects using bounding boxes. The CNN algorithm can identify defects of size ranging from 100 micron to a few millimeters. The algorithm was optimized to run on an edge device with at least 15 fps, making the inspection system work real time.</div>
<br>

[![Tube inspection system](https://img.shields.io/badge/Link-Read_patent-blue?logo=adobe-acrobat-reader&logoColor=white)](https://worldwide.espacenet.com/patent/search/family/062235809/publication/WO2019219956A1?q=somasundaram%20supriya%20sarkar%20sandvik)
<br>
<center><img src="images/blog_patent_pic.png"/></center>
<br>

---
## Udacity course: Intel Edge AI

### Designing a people counter appilication

This is a simple project to count the number of people. In this project, people enter a room from one side, read a document and leave the room on the other side. An SSD people detector model was used to count the people with 100% accuracy. 

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/somasundaram1702/people-counter-python)

<center><img src="images/people_counter.gif"/></center>

---
### AI for Smart Queue management

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-red?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1f32gj5IYIyFipoINiC8P3DvKat-WWLUK)

<div style="text-align: justify">In this work, people standing in a queue are counted. The same SSD people detection model was used to count people inside a region of interest. If too many people are identified in a single queue, they are re-directed to the other queue.</div>

<center><img src="images/smart_queue.gif"/></center>

---
### Gaze estimation algorithm

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-food-trends-facebook.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/facebook-detect-food-trends)

<div style="text-align: justify">First I build co-occurence matrices of ingredients from Facebook posts from 2011 to 2015. Then, to identify interesting and rare ingredient combinations that occur more than by chance, I calculate Lift and PPMI metrics. Lastly, I plot time-series data of identified trends to validate my findings. Interesting food trends have emerged from this analysis.</div>
<br>
<center><img src="images/fb-food-trends.png"></center>
<br>

---


---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>
