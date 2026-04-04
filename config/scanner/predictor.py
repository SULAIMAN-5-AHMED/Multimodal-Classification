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
        self.models['brain'] = tf.keras.models.load_model('../models/BrainTumorInceptionV3(A89L08).keras')

        # Lung Cancer Model
        self.models['lung'] = tf.keras.models.load_model('../models/LungCancer(A78L1).keras')

        # Pneumonia Model (Uncomment when ready)
        self.models['pneumonia'] = tf.keras.models.load_model('../models/PenumoniaA78L2.keras')

        # Dementia model (joblib)
        dementia_path = "../models/dementia_modelA90.joblib"
        if os.path.exists(dementia_path):
            self.models['dementia'] = joblib.load(dementia_path)

    def preprocess_image(self, image_file, target_size=(224, 224)):
        img = Image.open(image_file).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img).astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)

    def _get_last_conv_layer(self, model):
        """Automatically detect the last Conv2D layer for Grad-CAM"""
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        return None

    def generate_gradcam(self, image_file, model_key):
        """
        Generate Grad-CAM heatmap for a given model and image.
        Returns a base64-encoded PNG overlay ready for <img src="...">
        """
        try:
            model = self.models[model_key]
            last_conv_layer = self._get_last_conv_layer(model)

            if last_conv_layer is None:
                return {'success': False, 'error': 'Could not automatically detect a Conv2D layer for Grad-CAM.'}

            # Load original image for overlay
            img_orig = Image.open(image_file).convert('RGB')

            # Preprocess for model inference
            img_resized = img_orig.resize((224, 224))
            img_array = np.array(img_resized).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Create intermediate model: outputs [conv_features, predictions]
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output]
            )

            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_class = tf.argmax(predictions[0])
                class_output = predictions[:, pred_class]
                grads = tape.gradient(class_output, conv_outputs)

            # Compute weights and heatmap
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Normalize and apply ReLU
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # Resize heatmap to original image dimensions
            heatmap = np.uint8(255 * heatmap)
            heatmap_pil = Image.fromarray(heatmap).resize(img_orig.size, Image.BICUBIC)
            heatmap_np = np.array(heatmap_pil)

            # Apply colormap and blend
            heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = Image.blend(img_orig, Image.fromarray(heatmap_color), alpha=0.4)

            # Convert to base64 PNG
            buffered = io.BytesIO()
            overlay.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                'success': True,
                'gradcam_image': f'data:image/png;base64,{img_base64}',
                'predicted_class_index': int(pred_class)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_brain_tumor(self, image_file):
        try:
            brain_tumor = {0: "No tumor", 1: "glioma", 2: "meningioma", 3: "pituitary"}
            img_array = self.preprocess_image(image_file)
            model_output = self.models['brain'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            tumor_type_label = brain_tumor[predicted_class]

            return {
                'success': True,
                'predicted_class': predicted_class,
                'tumor_type': tumor_type_label,
                'confidence': round(confidence, 2),
                'recommendation': self._get_brain_recommendation(tumor_type_label)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_lung_cancer(self, image_file):
        try:
            lung_classes = {0: "Normal", 1: "Cancer Detected"}
            img_array = self.preprocess_image(image_file)
            model_output = self.models['lung'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            prediction_label = lung_classes[predicted_class]

            return {
                'success': True,
                'predicted_class': predicted_class,
                'prediction': prediction_label,
                'confidence': round(confidence, 2),
                'recommendation': self._get_lung_recommendation(prediction_label)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_pneumonia(self, image_file):
        try:
            pneu_classes = {0: "Normal", 1: "Pneumonia"}
            img_array = self.preprocess_image(image_file)
            model_output = self.models['pneumonia'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            prediction_label = pneu_classes[predicted_class]

            return {
                'success': True,
                'predicted_class': predicted_class,
                'prediction': prediction_label,
                'confidence': round(confidence, 2),
                'recommendation': self._get_pneumonia_recommendation(prediction_label)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_dementia(self, form_data):
        try:
            DEMENTIA_FEATURE_ORDER = [
                'age', 'gender', 'educationyears', 'EF', 'PS',
                'Global', 'diabetes', 'smoking', 'hypertension'
            ]

            gender_input = str(form_data.get('gender', 'male')).lower().strip()
            form_data['gender'] = 1 if gender_input == 'female' else 0

            hypertension_input = str(form_data.get('hypertension', 'No')).lower().strip()
            form_data['hypertension'] = 1 if hypertension_input == 'yes' else 0

            smoking_map = {'never-smoker': 0, 'ex-smoker': 1, 'current-smoker': 2}
            smoking_input = str(form_data.get('smoking', 'never-smoker')).lower().strip()
            form_data['smoking'] = smoking_map.get(smoking_input, 0)

            numeric_features = ['age', 'educationyears', 'EF', 'PS', 'Global', 'diabetes']
            for feat in numeric_features:
                try:
                    form_data[feat] = float(form_data.get(feat, 0))
                except (ValueError, TypeError):
                    form_data[feat] = 0.0

            input_array = np.array([[form_data[f] for f in DEMENTIA_FEATURE_ORDER]])

            model = self.models['dementia']
            prediction_class = int(model.predict(input_array)[0])
            prediction_proba = model.predict_proba(input_array)[0]
            confidence = float(np.max(prediction_proba) * 100)

            return {
                'success': True,
                'predicted_class': prediction_class,
                'probability': round(confidence, 2),
                'class_probabilities': {
                    'non_demented': round(prediction_proba[0] * 100, 2),
                    'demented': round(prediction_proba[1] * 100, 2)
                },
                'input_features': {feat: form_data[feat] for feat in DEMENTIA_FEATURE_ORDER},
                'recommendation': self._get_dementia_recommendation(prediction_class, confidence)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _get_brain_recommendation(self, tumor_type):
        recommendations = {
            "No tumor": "No tumor detected. Normal brain imaging findings.",
            "glioma": "Neurosurgical consultation recommended. Biopsy and molecular profiling advised for grading.",
            "meningioma": "Neurosurgical evaluation recommended. Most are benign; monitoring or resection based on symptoms.",
            "pituitary": "Endocrinology and neurosurgery referral recommended. Hormonal workup and visual field testing advised.",
            "Unknown": "Inconclusive result. Recommend contrast-enhanced MRI and specialist review."
        }
        return recommendations.get(tumor_type, "Further clinical evaluation recommended.")

    def _get_lung_recommendation(self, prediction):
        recommendations = {
            "Normal": "Routine follow-up in 12 months recommended. No immediate intervention required.",
            "Cancer Detected": "Urgent referral to oncologist recommended. Biopsy and staging required.",
            "Unknown": "Inconclusive nodule characteristics. Recommend 3-month follow-up CT and pulmonology consultation."
        }
        return recommendations.get(prediction, "Clinical correlation and follow-up imaging recommended.")

    def _get_pneumonia_recommendation(self, prediction):
        recommendations = {
            "Normal": "No signs of pneumonia detected. Clinical correlation advised.",
            "Pneumonia": "Findings consistent with pneumonia. Recommend antibiotic therapy, oxygen saturation monitoring, and follow-up imaging in 2-4 weeks.",
            "Unknown": "Indeterminate opacities. Recommend clinical correlation with symptoms, CRP/WBC labs, and possible repeat imaging."
        }
        return recommendations.get(prediction, "Clinical evaluation and supportive care recommended.")

    def _get_dementia_recommendation(self, pred_class, confidence):
        if pred_class == 0:
            return "No significant cognitive decline detected. Routine neurological check-ups recommended annually."
        else:
            return "Clinical signs suggest cognitive impairment. Recommend comprehensive neuropsychological evaluation, brain imaging (MRI/CT), and caregiver support planning."


predictor = MedicalImagePredictor()