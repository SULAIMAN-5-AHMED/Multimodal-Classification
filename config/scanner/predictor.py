import os
import io
import base64
import threading
import traceback
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import cv2


class MedicalImagePredictor:
    # Max side length for the Grad-CAM overlay image to optimize browser loading
    GRADCAM_MAX_SIDE = 512

    # Preferred target layer per model. None = auto-detect the last spatial layers.
    GRADCAM_LAYER_HINTS = {'brain': 'mixed10', 'lung': None, 'pneumonia': None}

    # Number of last spatial layers to try for Grad-CAM (fallback mechanism)
    GRADCAM_MAX_CANDIDATES = 4

    def __init__(self):
        self.models = {}
        self._gradcam_cache = {}
        self._gradcam_lock = threading.Lock()
        self.load_models()

    def load_models(self):
        # Load image models
        self.models['brain'] = tf.keras.models.load_model('../models/BrainTumorInceptionV3(A88L06).keras')
        self.models['lung'] = tf.keras.models.load_model('../models/LungCancer(A86L2).keras')
        self.models['pneumonia'] = tf.keras.models.load_model('../models/Penumonia(A82L1).keras')

        # Load tabular model
        dementia_path = "../models/dementia_modelA90.joblib"
        if os.path.exists(dementia_path):
            self.models['dementia'] = joblib.load(dementia_path)

        print("✅ All models loaded successfully.")

        # Pre-build Grad-CAM models to avoid rebuilding on every request
        for key in ('brain', 'lung', 'pneumonia'):
            try:
                self._get_gradcam_model(key)
            except Exception as e:
                print(f"⚠️ Could not pre-build Grad-CAM model for {key}: {e}")

    # ================= IMAGE HELPERS =================
    @staticmethod
    def _load_rgb(image_file):
        """Open an upload as RGB and rewind the stream for reuse."""
        if hasattr(image_file, 'seek'):
            try:
                image_file.seek(0)
            except Exception:
                pass
        return Image.open(image_file).convert('RGB')

    def preprocess_image(self, image_file, target_size=(224, 224)):
        """Standard preprocessing for all image models (224x224)."""
        img = self._load_rgb(image_file)
        img = img.resize(target_size)
        img_array = np.array(img).astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)

    # ================= GRAD-CAM IMPLEMENTATION =================
    def _make_logit_model(self, model):
        """Clone model and remove final activation to output raw logits for Grad-CAM."""
        clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())

        last = clone.layers[-1]
        if hasattr(last, 'activation'):
            last.activation = tf.keras.activations.linear
        return clone

    @staticmethod
    def _spatial_output(layer):
        """Returns the 4D feature-map output if the layer is usable for Grad-CAM."""
        if isinstance(layer, tf.keras.layers.InputLayer):
            return None
        try:
            out = layer.output
        except (AttributeError, RuntimeError):
            return None
        if isinstance(out, (list, tuple)) or len(out.shape) != 4:
            return None
        if (out.shape[1] or 0) <= 1 or (out.shape[2] or 0) <= 1:
            return None
        return out

    def _candidate_layers(self, model, hint=None, max_candidates=None):
        """
        Collects the last N layers with 4D spatial outputs.
        This ensures that if the absolute last layer yields an empty heatmap (0 pixels),
        the algorithm can fall back to the 2nd last, 3rd last, etc.
        """
        max_candidates = max_candidates or self.GRADCAM_MAX_CANDIDATES
        candidates = []

        if hint:
            try:
                layer = model.get_layer(hint)
                if self._spatial_output(layer) is not None:
                    candidates.append(layer)
            except ValueError:
                pass

        # Iterate backwards to grab the last N spatial layers
        for layer in reversed(model.layers):
            if len(candidates) >= max_candidates:
                break
            if layer in candidates:
                continue
            if self._spatial_output(layer) is not None:
                candidates.append(layer)

        if not candidates:
            raise ValueError("No spatial (4D) layer found for Grad-CAM.")
        return candidates

    def _get_gradcam_model(self, model_key):
        """Build and cache a model returning [candidate feature maps..., logits]."""
        with self._gradcam_lock:
            if model_key in self._gradcam_cache:
                return self._gradcam_cache[model_key]

            logit_model = self._make_logit_model(self.models[model_key])
            layers_ = self._candidate_layers(logit_model, self.GRADCAM_LAYER_HINTS.get(model_key))

            grad_model = tf.keras.models.Model(
                inputs=logit_model.input,
                outputs=[l.output for l in layers_] + [logit_model.output]
            )
            names = [l.name for l in layers_]
            self._gradcam_cache[model_key] = (grad_model, names)
            return self._gradcam_cache[model_key]

    @staticmethod
    def _normalize_heatmap(raw, eps=1e-12):
        """Standard Grad-CAM normalization with fallback to absolute values if ReLU yields zeros."""
        raw = np.nan_to_num(np.asarray(raw, dtype='float32'), nan=0.0, posinf=0.0, neginf=0.0)
        cam = np.maximum(raw, 0.0)
        peak = float(cam.max())
        if peak > eps:
            return cam / peak, False

        # Fallback: use absolute values if standard ReLU map is completely empty
        cam = np.abs(raw)
        peak = float(cam.max())
        if peak > eps:
            return cam / peak, True

        return cam, True

    def _pick_heatmap(self, names, conv_list, grads_list):
        """Iterates through candidate layers and returns the first valid heatmap."""
        first_fallback = None
        for name, conv, grads in zip(names, conv_list, grads_list):
            if grads is None:
                continue
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            raw = tf.reduce_sum(conv[0] * pooled, axis=-1).numpy()
            cam, fallback = self._normalize_heatmap(raw)

            if not fallback:
                return name, cam, False
            if first_fallback is None and float(cam.max()) > 0:
                first_fallback = (name, cam, True)

        if first_fallback is not None:
            return first_fallback

        shape = tuple(int(d) for d in conv_list[0].shape[1:3])
        return names[0], np.zeros(shape, dtype='float32'), True

    def _compute_gradcam(self, image_file, model_key):
        """Core Grad-CAM computation and overlay generation."""
        try:
            img_orig = self._load_rgb(image_file)
            img_array = np.array(img_orig.resize((224, 224))).astype('float32') / 255.0
            img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

            grad_model, names = self._get_gradcam_model(model_key)

            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                outputs = grad_model(img_tensor, training=False)
                conv_list, logits = list(outputs[:-1]), outputs[-1]
                pred_class = int(tf.argmax(logits[0]))
                class_score = logits[:, pred_class]
            grads_list = tape.gradient(class_score, conv_list)

            layer_name, cam, used_fallback = self._pick_heatmap(names, conv_list, grads_list)
            empty = float(cam.max()) <= 0.0

            # Resize and blend heatmap on the original image
            base = img_orig.copy()
            base.thumbnail((self.GRADCAM_MAX_SIDE, self.GRADCAM_MAX_SIDE), Image.LANCZOS)
            heat_u8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
            heat_img = Image.fromarray(heat_u8).resize(base.size, Image.BICUBIC)
            heat_color = cv2.applyColorMap(np.array(heat_img), cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            overlay = Image.blend(base, Image.fromarray(heat_color), alpha=0.5)

            buffered = io.BytesIO()
            overlay.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                'success': True,
                'gradcam_image': f'data:image/png;base64,{img_base64}',
                'predicted_class_index': pred_class,
                'gradcam_layer': layer_name,
                'gradcam_fallback': used_fallback,
                'gradcam_empty': empty,
            }
        except Exception as e:
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    # ================= PUBLIC GRAD-CAM WRAPPERS =================
    def generate_gradcam_brain(self, image_file):
        return self._compute_gradcam(image_file, 'brain')

    def generate_gradcam_lung(self, image_file):
        return self._compute_gradcam(image_file, 'lung')

    def generate_gradcam_pneumonia(self, image_file):
        return self._compute_gradcam(image_file, 'pneumonia')

    def generate_gradcam(self, image_file, model_key):
        """Route to the architecture-specific Grad-CAM implementation."""
        dispatch = {
            'brain': self.generate_gradcam_brain,
            'lung': self.generate_gradcam_lung,
            'pneumonia': self.generate_gradcam_pneumonia,
        }
        handler = dispatch.get(model_key)
        if handler is None:
            return {'success': False, 'error': f'No Grad-CAM handler for model: {model_key}'}
        return handler(image_file)

    # ================= PREDICTION FUNCTIONS =================
    def predict_brain_tumor(self, image_file):
        try:
            brain_tumor = {0: "No tumor", 1: "Glioma", 2: "Meningioma", 3: "Pituitary"}
            img_array = self.preprocess_image(image_file, target_size=(224, 224))
            model_output = self.models['brain'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            return {
                'success': True, 'predicted_class': predicted_class,
                'tumor_type': brain_tumor[predicted_class], 'confidence': round(confidence, 2),
                'recommendation': self._get_brain_recommendation(brain_tumor[predicted_class])
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_lung_cancer(self, image_file):
        try:
            lung_classes = {0: "Normal No Cancer Detected", 1: "Malignant Case Detected", 2: "Benign Case Detected"}
            img_array = self.preprocess_image(image_file, target_size=(224, 224))
            model_output = self.models['lung'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            return {
                'success': True, 'predicted_class': predicted_class,
                'prediction': lung_classes[predicted_class], 'confidence': round(confidence, 2),
                'recommendation': self._get_lung_recommendation(lung_classes[predicted_class])
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_pneumonia(self, image_file):
        try:
            pneu_classes = {0: "Normal", 1: "Pneumonia"}
            img_array = self.preprocess_image(image_file, target_size=(224, 224))
            model_output = self.models['pneumonia'].predict(img_array, verbose=0)
            predicted_class = int(np.argmax(model_output[0]))
            confidence = float(np.max(model_output[0]) * 100)
            return {
                'success': True, 'predicted_class': predicted_class,
                'prediction': pneu_classes[predicted_class], 'confidence': round(confidence, 2),
                'recommendation': self._get_pneumonia_recommendation(pneu_classes[predicted_class])
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_dementia(self, form_data):
        try:
            DEMENTIA_FEATURE_ORDER = ['age', 'gender', 'educationyears', 'EF', 'PS', 'Global', 'diabetes', 'smoking',
                                      'hypertension']
            form_data['gender'] = 1 if str(form_data.get('gender', 'male')).lower() == 'female' else 0
            form_data['hypertension'] = 1 if str(form_data.get('hypertension', 'No')).lower() == 'yes' else 0
            form_data['smoking'] = {'never-smoker': 0, 'ex-smoker': 1, 'current-smoker': 2}.get(
                str(form_data.get('smoking', 'never-smoker')).lower(), 0)
            for feat in ['age', 'educationyears', 'EF', 'PS', 'Global', 'diabetes']:
                try:
                    form_data[feat] = float(form_data.get(feat, 0))
                except:
                    form_data[feat] = 0.0

            input_array = np.array([[form_data[f] for f in DEMENTIA_FEATURE_ORDER]])
            model = self.models['dementia']
            pred_class = int(model.predict(input_array)[0])
            pred_proba = model.predict_proba(input_array)[0]

            return {
                'success': True, 'predicted_class': pred_class,
                'probability': round(float(np.max(pred_proba) * 100), 2),
                'class_probabilities': {'non_demented': round(pred_proba[0] * 100, 2),
                                        'demented': round(pred_proba[1] * 100, 2)},
                'input_features': {f: form_data[f] for f in DEMENTIA_FEATURE_ORDER},
                'recommendation': self._get_dementia_recommendation(pred_class, 0)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ================= RECOMMENDATION HELPERS =================
    def _get_brain_recommendation(self, t):
        return {"No tumor": "Normal findings.", "Glioma": "Neurosurgery consult.", "Meningioma": "Monitor/resect.",
                "Pituitary": "Endo/neuro referral.", "Unknown": "Further eval."}.get(t, "Clinical correlation advised.")

    def _get_lung_recommendation(self, p):
        # Fixed keys to match the actual lung_classes dictionary outputs & corrected "Benign" typo
        return {"Normal No Cancer Detected": "Routine follow-up.",
                "Malignant Case Detected": "Urgent oncology referral.",
                "Benign Case Detected": "Urgent oncology referral", "Unknown": "3-month CT follow-up."}.get(p,"Clinical correlation advised.")

    def _get_pneumonia_recommendation(self, p):
        return {"Normal": "No pneumonia detected.", "Pneumonia": "Antibiotics & monitoring.",
                "Unknown": "CRP/WBC labs & repeat imaging."}.get(p, "Clinical evaluation advised.")

    def _get_dementia_recommendation(self, pc, c):
        return "Routine check-ups." if pc == 0 else "Neuropsych eval & imaging recommended."


predictor = MedicalImagePredictor()