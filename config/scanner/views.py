from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.conf import settings
import os
from .predictor import predictor


@ensure_csrf_cookie
def home(request):
    """Home page view"""
    return render(request, 'scanner/home.html')


def lung_cancer(request):
    """Lung cancer detection page"""
    return render(request, 'scanner/lung_cancer.html')


def pneumonia(request):
    """Pneumonia detection page"""
    return render(request, 'scanner/pneumonia.html')


def brain_tumor(request):
    """Brain tumor detection page"""
    return render(request, 'scanner/brain_tumor.html')


def dementia(request):
    """Dementia assessment page (placeholder)"""
    return render(request, 'scanner/dementia.html')


def predict_lung(request):
    """API endpoint for lung cancer prediction + Grad-CAM"""
    if request.method == 'POST' and request.FILES.get('ct_image'):
        ct_image = request.FILES['ct_image']
        result = predictor.predict_lung_cancer(ct_image)

        # Attach Grad-CAM if prediction succeeded
        if result.get('success'):
            ct_image.seek(0)  # Reset file pointer for second read
            gradcam = predictor.generate_gradcam(ct_image, model_key='lung')
            result['gradcam'] = gradcam

        return JsonResponse(result)
    return JsonResponse({'success': False, 'error': 'No image provided'})


def predict_pneumonia(request):
    """API endpoint for pneumonia prediction + Grad-CAM"""
    if request.method == 'POST' and request.FILES.get('ct_image'):
        ct_image = request.FILES['ct_image']
        result = predictor.predict_pneumonia(ct_image)

        # Attach Grad-CAM if prediction succeeded
        if result.get('success'):
            ct_image.seek(0)
            gradcam = predictor.generate_gradcam(ct_image, model_key='pneumonia')
            result['gradcam'] = gradcam

        return JsonResponse(result)
    return JsonResponse({'success': False, 'error': 'No image provided'})


def predict_brain(request):
    """API endpoint for brain tumor prediction + Grad-CAM"""
    if request.method == 'POST' and request.FILES.get('ct_image'):
        ct_image = request.FILES['ct_image']
        result = predictor.predict_brain_tumor(ct_image)

        # Attach Grad-CAM if prediction succeeded
        if result.get('success'):
            ct_image.seek(0)
            gradcam = predictor.generate_gradcam(ct_image, model_key='brain')
            result['gradcam'] = gradcam

        return JsonResponse(result)
    return JsonResponse({'success': False, 'error': 'No image provided'})


def predict_dementia(request):
    """API endpoint for dementia prediction (tabular data only)"""
    if request.method == 'POST':
        data = request.POST.dict()
        result = predictor.predict_dementia(data)
        return JsonResponse(result)
    return JsonResponse({'success': False, 'error': 'POST request required'})