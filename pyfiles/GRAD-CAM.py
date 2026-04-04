import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, class_dict=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)
    predicted_class = class_dict[pred_class]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, predicted_class, confidence

def display_gradcam(img, heatmap, alpha=0.4, pred_class=None, confidence=None):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    plt.imshow(cv2.cvtColor(superimposed_img.astype("uint8"), cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {pred_class}, Confidence: {confidence:.2f}")
    plt.axis("off")
    plt.show()

# Class dictionary
brain_tumor = {0: 'no tumor', 1: 'glioma', 2: 'meningioma', 3: 'pituitary'}

# Load model
model = load_model("../models/BrainTumorInceptionV3(A89L08).keras")

# Load and preprocess image
img = cv2.imread(r"../Datasets/BrainTumor/Testing/3/Te-pi_1.jpg")
preprocessed_image = cv2.cvtColor(cv2.resize(img,(224,224)), cv2.COLOR_BGR2RGB).astype("float32")/255.0
img_array = np.expand_dims(preprocessed_image, axis=0)

# Generate Grad-CAM
heatmap, pred_class, confidence = make_gradcam_heatmap(img_array, model, last_conv_layer_name="mixed10", class_dict=brain_tumor)
display_gradcam(img, heatmap, pred_class=pred_class, confidence=confidence)