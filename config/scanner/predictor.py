import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import cv2
from django.conf import settings


class MedicalImagePredictor:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        # Brain tumor model
        self.models['brain'] = tf.keras.models.load_model('../models/BrainTumorInceptionV3(A88L06).keras')
        # Lung Cancer Model
        self.models['lung'] = tf.keras.models.load_model('../models/LungCancer(A78L1).keras')
        # Pneumonia Model
        self.models['pneumonia'] = tf.keras.models.load_model('../models/PenumoniaA78L2.keras')
        # Dementia model (joblib)
        dementia_path = "../models/dementia_modelA90.joblib"
        if os.path.exists(dementia_path):
            self.models['dementia'] = joblib.load(dementia_path)
        print("✅ All models loaded successfully.")

    def preprocess_image(self, image_file, target_size=(224, 224)):
        img = Image.open(image_file).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img).astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)

    def _get_last_conv_layer(self, model):
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                return layer
        return None

    def generate_gradcam(self, image_file, model_key):
        """Generate Grad-CAM heatmap matching EXACT preprocessing of predict functions"""
        print(f"🔬 Generating Grad-CAM for {model_key}...")
        try:
            model = self.models[model_key]
            last_conv_layer = self._get_last_conv_layer(model)
            if last_conv_layer is None:
                return {'success': False, 'error': 'No Conv2D layer found.'}

            # 1. Load & preprocess EXACTLY like predict functions
            img_orig = Image.open(image_file).convert('RGB')
            img_resized = img_orig.resize((224, 224))
            img_array = np.array(img_resized).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 2. Build intermediate model
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output]
            )

            # 3. Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array, training=False)
                pred_class = tf.argmax(predictions[0])
                class_output = predictions[:, pred_class]
                grads = tape.gradient(class_output, conv_outputs)

            # 4. Weighted combination & normalize
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            heatmap = heatmap / (max_val + 1e-8)
            heatmap = heatmap.numpy()

            # 5. Resize & blend
            heatmap = np.uint8(255 * heatmap)
            heatmap_pil = Image.fromarray(heatmap).resize(img_orig.size, Image.BICUBIC)
            heatmap_np = np.array(heatmap_pil)

            heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = Image.blend(img_orig, Image.fromarray(heatmap_color), alpha=0.5)

            # 6. Encode
            buffered = io.BytesIO()
            overlay.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            print(f"✅ Grad-CAM generated for {model_key} (Class: {int(pred_class)})")
            return {
                'success': True,
                'gradcam_image': f'data:image/png;base64,{img_base64}',
                'predicted_class_index': int(pred_class)
            }
        except Exception as e:
            print(f"❌ Grad-CAM failed for {model_key}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict_brain_tumor(self, image_file):
        try:
            brain_tumor = {0: "No tumor", 1: "glioma", 2: "meningioma", 3: "pituitary"}
            img_array = self.preprocess_image(image_file)
            model_output = self.models['brain'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            return {
                'success': True, 'predicted_class': predicted_class,
                'tumor_type': brain_tumor[predicted_class], 'confidence': round(confidence, 2),
                'recommendation': self._get_brain_recommendation(brain_tumor[predicted_class])
            }
        except Exception as e: return {'success': False, 'error': str(e)}

    def predict_lung_cancer(self, image_file):
        try:
            lung_classes = {0: "Normal", 1: "Cancer Detected"}
            img_array = self.preprocess_image(image_file)
            model_output = self.models['lung'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            return {
                'success': True, 'predicted_class': predicted_class,
                'prediction': lung_classes[predicted_class], 'confidence': round(confidence, 2),
                'recommendation': self._get_lung_recommendation(lung_classes[predicted_class])
            }
        except Exception as e: return {'success': False, 'error': str(e)}

    def predict_pneumonia(self, image_file):
        try:
            pneu_classes = {0: "Normal", 1: "Pneumonia"}
            img_array = self.preprocess_image(image_file)
            model_output = self.models['pneumonia'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            return {
                'success': True, 'predicted_class': predicted_class,
                'prediction': pneu_classes[predicted_class], 'confidence': round(confidence, 2),
                'recommendation': self._get_pneumonia_recommendation(pneu_classes[predicted_class])
            }
        except Exception as e: return {'success': False, 'error': str(e)}

    def predict_dementia(self, form_data):
        try:
            DEMENTIA_FEATURE_ORDER = ['age', 'gender', 'educationyears', 'EF', 'PS', 'Global', 'diabetes', 'smoking', 'hypertension']
            form_data['gender'] = 1 if str(form_data.get('gender', 'male')).lower() == 'female' else 0
            form_data['hypertension'] = 1 if str(form_data.get('hypertension', 'No')).lower() == 'yes' else 0
            form_data['smoking'] = {'never-smoker': 0, 'ex-smoker': 1, 'current-smoker': 2}.get(str(form_data.get('smoking', 'never-smoker')).lower(), 0)
            for feat in ['age', 'educationyears', 'EF', 'PS', 'Global', 'diabetes']:
                try: form_data[feat] = float(form_data.get(feat, 0))
                except: form_data[feat] = 0.0
            input_array = np.array([[form_data[f] for f in DEMENTIA_FEATURE_ORDER]])
            model = self.models['dementia']
            pred_class = int(model.predict(input_array)[0])
            pred_proba = model.predict_proba(input_array)[0]
            return {
                'success': True, 'predicted_class': pred_class, 'probability': round(float(np.max(pred_proba) * 100), 2),
                'class_probabilities': {'non_demented': round(pred_proba[0]*100, 2), 'demented': round(pred_proba[1]*100, 2)},
                'input_features': {f: form_data[f] for f in DEMENTIA_FEATURE_ORDER},
                'recommendation': self._get_dementia_recommendation(pred_class, 0)
            }
        except Exception as e: return {'success': False, 'error': str(e)}

    def _get_brain_recommendation(self, t):
        return {"No tumor": "Normal findings.", "glioma": "Neurosurgery consult.", "meningioma": "Monitor/resect.", "pituitary": "Endo/neuro referral.", "Unknown": "Further eval."}.get(t, "Clinical correlation advised.")
    def _get_lung_recommendation(self, p):
        return {"Normal": "Routine follow-up.", "Cancer Detected": "Urgent oncology referral.", "Unknown": "3-month CT follow-up."}.get(p, "Clinical correlation advised.")
    def _get_pneumonia_recommendation(self, p):
        return {"Normal": "No pneumonia detected.", "Pneumonia": "Antibiotics & monitoring.", "Unknown": "CRP/WBC labs & repeat imaging."}.get(p, "Clinical evaluation advised.")
    def _get_dementia_recommendation(self, pc, c):
        return "Routine check-ups." if pc == 0 else "Neuropsych eval & imaging recommended."

predictor = MedicalImagePredictor()