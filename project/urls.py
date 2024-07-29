from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name='home'), #views.home itu berarti pesannya dia ada di views.py, home itu maksudnya pesan 
    #hello world ada di def home
    path('upload',views.upload, name='upload'),
    #path('back',views.home, name='home') #views.home itu berarti pesannya dia ada di views.py, home itu maksudnya pesan 
    #hello world ada di def home
    #urutan jalannya >> dari form action ke url pattern >> dari url pattern diarahin ke views >>
    #dari views itu nnti code nya disitu trus direturn ke page html WOW
]
