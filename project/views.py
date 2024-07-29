from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest
import numpy as np
import cv2
import base64
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

#Create your views here.
def home(request): #ketika client request, maka return hello world
     return render(request,'index.html') #render(request,'home.html') #memanggil home.html
# #name : nona itu dynamic content, jadi klo ganti value disini,
# #otomatis value di home.html yang pake {{name}} jg ikut berubah

# # def add(request):
# #     #ambil value dari textbox 
# #     val1 = request.POST['num1'] #ambil value dr form POST, klo method GET ya request.get
# #     val2 = request.POST['num2'] 
# #     res = int(val1) + int(val2) #pastiin convert ke int krn base hasil input text itu string
# #     return render(request,'result.html',{'result': res}) #mengirim variabel result dengan nilai res (hasil perhitungan)

def upload(request):
    if request.method == 'POST': #jika user mengirim data dgn method post dan ada files dgn variabel foto
        if 'photo' not in request.FILES:
            return HttpResponseBadRequest('No file uploaded')
        photo = request.FILES['photo']
        extensions = ['.jpg', '.jpeg', '.png', '.tif'] #jenis extension file yang diterima
        if not any(photo.name.lower().endswith(ext) for ext in extensions):
            return HttpResponseBadRequest('Only JPG, JPEG, PNG, and TIF files are allowed')

        try:
            #TRAINING DATASET
            dataset_dir = 'static/gender/'
            test_data = list()
            test_labels= list()
            train_data= list()
            train_labels= list()
            #JIKA SUDAH ADA MODEL YG TERSIMPAN SEBELUMNYA, TINGGAL LOADING
            #KLO BELOM ADA, BUAT MODELNYA 
            if os.path.exists('modelPredGender.pickle'):
                with open('modelPredGender.pickle', 'rb') as file:
                    classifier=pickle.load(file)
            else:
                train_dir = os.path.join(dataset_dir, 'Train')
                for class_name in os.listdir(train_dir):
                    class_dir = os.path.join(train_dir, class_name)
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ubah ke gray
                        image = cv2.GaussianBlur(image, (5,5),1)#hapus noise pada gambar
                        img_lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
                        img_sharp = image-img_lap
                        img_sharp = cv2.convertScaleAbs(img_sharp) #pertajam gambar
                        #ratain distribusi intensitas warna pada gambar
                        equalized=cv2.equalizeHist(img_sharp)
                        img_sharp = cv2.resize(equalized,(150,150), interpolation=cv2.INTER_AREA)
                        train_data.append(img_sharp.flatten())
                        train_labels.append(class_name)
                train_data = np.array(train_data)
                train_labels = np.array(train_labels)
            
                #LOADING TESTING SET
                test_dir = os.path.join(dataset_dir, 'Test')
                for class_name in os.listdir(test_dir):
                    class_dir = os.path.join(test_dir, class_name)
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ubah ke gray
                        image = cv2.GaussianBlur(image, (5,5),1)#hapus noise pada gambar
                        img_lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
                        img_sharp = image-img_lap
                        img_sharp = cv2.convertScaleAbs(img_sharp) #pertajam gambar
                        #ratain distribusi intensitas warna pada gambar
                        equalized=cv2.equalizeHist(img_sharp)
                        img_sharp = cv2.resize(equalized,(150,150), interpolation=cv2.INTER_AREA)
                        test_data.append(img_sharp.flatten())
                        test_labels.append(class_name)
                test_data = np.array(test_data)
                test_labels = np.array(test_labels)
                classifier = svm.SVC(kernel='linear')
                classifier.fit(train_data, train_labels)
                with open('modelPredGender.pickle','wb') as file:
                    pickle.dump(classifier, file)
            if os.path.exists('accuracy.pickle'):
                with open('accuracy.pickle', 'rb') as file:
                    accuracysvm=pickle.load(file)
            else:
                y_pred = classifier.predict(test_data)
                accuracysvm = accuracy_score(test_labels, y_pred)
                with open('accuracy.pickle','wb') as file:
                    pickle.dump(accuracysvm, file)
            print('Accuracy of SVM:', accuracysvm)
            #PERCOBAAN PREDIKSI GAMBAR
            nparr = np.frombuffer(photo.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #CONVERT IMAGE DR BENTUK ARRAY KEMBALI KE IMAGE SPY BISA DIPROSES
            #convert gambar ke bentuk RGB
            imgOri = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #ubah dari bgr ke gray
            imgGray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)
            #hilangkan noise pada gambar dengan gaussian blur
            imgGauss = cv2.GaussianBlur(imgGray,(5,5),1)
            #tajamkan gambar
            img_lap = cv2.Laplacian(imgGauss, cv2.CV_64F, ksize=3)
            img_sharp = imgGauss-img_lap
            img_sharp = cv2.convertScaleAbs(img_sharp)
            #ratakan distribusi intensitas warna
            imgEqualized = cv2.equalizeHist(img_sharp)
            #RESIZE IMAGE SUPAYA LEBIH ENAK BUAT DIPROSES
            imgFinal = cv2.resize(imgEqualized,(150,150),interpolation=cv2.INTER_AREA)
            predicted_class = classifier.predict([imgFinal.flatten()]) #PAKE SVM
            print(predicted_class)
            gender = predicted_class[0]


            #SIAPKAN VARIABEL YANG MAU DIKIRIM KE 1.HTML (RESULT PAGE)
            #Convert the processed image to base64
            imgOri = cv2.cvtColor(imgOri,cv2.COLOR_BGR2RGB)
            _, bufferOri = cv2.imencode('.jpg', imgOri)
            #DIENCODE DAN DIDECODE BIAR BISA KIRIM GAMBAR KE HTML HASIL
            imageOri = base64.b64encode(bufferOri).decode('utf-8')
            #GAMBAR GRAY
            _,bufferGray = cv2.imencode('.jpg',imgGray)
            imageGray = base64.b64encode(bufferGray).decode('utf-8')
            #GAMBAR GAUSSIAN
            _,bufferGaussian = cv2.imencode('.jpg',imgGauss)
            imageGaussian = base64.b64encode(bufferGaussian).decode('utf-8')
            #GAMBAR LAPLACIAN
            _,bufferLap = cv2.imencode('.jpg',img_sharp)
            imageLap = base64.b64encode(bufferLap).decode('utf-8')
            #GAMBAR HISTOGRAM EQU
            _,bufferEqu = cv2.imencode('.jpg',imgEqualized)
            imageEqualization = base64.b64encode(bufferEqu).decode('utf-8')
            _,bufferFinal = cv2.imencode('.jpg',imgFinal)
            imageFinal = base64.b64encode(bufferFinal).decode('utf-8')

            #BALIKKAN KE one.HTML
            # return render(request, 'one.html', {'gender' : gender, 'accSVM' : accuracysvm,
            #                  'original_image' : imageOri})
            return render(request, 'one.html', {'gender' : gender, 'accSVM' : accuracysvm,
                            'original_image':imageOri, 'gray_image':imageGray,
                            'gauss_image': imageGaussian,'lap_image':imageLap,
                            'equ_image':imageEqualization,'gender':gender})
        except Exception as e:
            return HttpResponseBadRequest('Error processing the image: {}'.format(str(e)))
    else:
        return HttpResponseBadRequest('No file chosen')
    return render(request, 'one.html')