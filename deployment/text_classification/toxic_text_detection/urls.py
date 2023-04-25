
from django.urls import path

from toxic_text_detection import views


urlpatterns = [
    path('',views.home,name='home' ),
    path('imagefile',views.imagefile,name='imagefile'),
    path('textfile',views.textfile,name='textfile')
]