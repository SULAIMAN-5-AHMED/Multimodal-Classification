from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('lung-cancer/', views.lung_cancer, name='lung_cancer'),
    path('pneumonia/', views.pneumonia, name='pneumonia'),
    path('brain-tumor/', views.brain_tumor, name='brain_tumor'),
    path('dementia/', views.dementia, name='dementia'),

    # 🔽 API ENDPOINTS 🔽
    path('api/predict/lung/', views.predict_lung, name='predict_lung'),
    path('api/predict/pneumonia/', views.predict_pneumonia, name='predict_pneumonia'),
    path('api/predict/brain/', views.predict_brain, name='predict_brain'),
    path('api/predict/dementia/', views.predict_dementia, name='predict_dementia'),
]