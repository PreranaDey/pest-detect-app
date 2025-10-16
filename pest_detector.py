import customtkinter as ctk
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import datetime
import csv
import cv2
import threading # For non-blocking operations like training
from tkinter import filedialog, messagebox

# --- Global Variables ---
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 3 # This will be updated by the generator based on actual folders
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "pest_detector.h5"
LOG_FILE = "pest_log.csv"
INPUT_IMAGE_NAME = "test_leaf.jpg" # Default name for image to process
CLASS_NAMES = [] # Will be populated after model training/loading

# 2.2 Model Definition and Training Function

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, num_classes, progress_callback=None):
    global CLASS_NAMES
    print("Starting model training...")
    if progress_callback: progress_callback("status", "Initializing training...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
    except Exception as e:
        error_msg = f"Error loading dataset for training: {e}. Please ensure your 'dataset/train' folder is correctly structured with subfolders for each class, each containing images."
        print(error_msg)
        if progress_callback: progress_callback("error", error_msg)
        raise e


    CLASS_NAMES = list(train_generator.class_indices.keys())
    print(f"Detected classes: {CLASS_NAMES}")
    actual_num_classes = len(CLASS_NAMES)

    if actual_num_classes == 0:
        error_msg = "No classes detected in the training dataset. Please ensure your 'dataset/train' folder contains class subdirectories with images."
        print(error_msg)
        if progress_callback: progress_callback("error", error_msg)
        raise ValueError(error_msg)


    if actual_num_classes != num_classes:
        print(f"Warning: Expected {num_classes} classes, but detected {actual_num_classes}. Adjusting model output layer.")
        num_classes = actual_num_classes # Adjust NUM_CLASSES to match actual detected classes


    model = create_model(num_classes)

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                # Assuming 10 epochs as a base for progress bar
                current_progress = (epoch + 1) / 10
                self.model.history.history.setdefault('accuracy', []).append(logs.get('accuracy'))
                self.model.history.history.setdefault('loss', []).append(logs.get('loss'))
                self.model.history.history.setdefault('val_accuracy', []).append(logs.get('val_accuracy'))
                self.model.history.history.setdefault('val_loss', []).append(logs.get('val_loss'))

                # Update progress_callback with more details
                progress_callback("progress", current_progress, 
                                  f"Epoch {epoch+1}/10 - Loss: {logs.get('loss'):.4f}, Acc: {logs.get('accuracy'):.4f}, Val_Acc: {logs.get('val_accuracy'):.4f}")

    if progress_callback: progress_callback("status", "Training started...")
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[CustomCallback()]
        )
        model.save(MODEL_SAVE_PATH)
        print(f"Model trained and saved to {MODEL_SAVE_PATH}")
        if progress_callback: progress_callback("status", "Model trained successfully!")
        return model, CLASS_NAMES
    except Exception as e:
        error_msg = f"Error during model training: {e}. Check console for details."
        print(error_msg)
        if progress_callback: progress_callback("error", error_msg)
        raise e


def load_or_train_model(progress_callback=None):
    global CLASS_NAMES
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading pre-trained model...")
        if progress_callback: progress_callback("status", "Loading pre-trained model...")
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            # Re-derive CLASS_NAMES if loading model (assuming dataset structure is consistent)
            train_datagen = ImageDataGenerator(rescale=1./255)
            train_generator = train_datagen.flow_from_directory(
                os.path.join(DATASET_PATH, 'train'),
                target_size=IMAGE_SIZE,
                batch_size=1, # Just to get class_indices
                class_mode='categorical'
            )
            CLASS_NAMES = list(train_generator.class_indices.keys())
            print(f"Loaded model with classes: {CLASS_NAMES}")
            if progress_callback: progress_callback("status", "Model loaded successfully!")
            return model, CLASS_NAMES
        except Exception as e:
            error_msg = f"Error loading model from {MODEL_SAVE_PATH}: {e}. Training a new model instead."
            print(error_msg)
            if progress_callback: progress_callback("error", error_msg)
            # If loading fails, fall through to training a new model
            pass
    
    # If model doesn't exist or loading failed
    print("No valid pre-trained model found. Training a new model...")
    if progress_callback: progress_callback("status", "No model found. Starting training...")
    train_dir = os.path.join(DATASET_PATH, 'train')
    return train_model(train_dir, NUM_CLASSES, progress_callback)


# 2.3 Prediction and Recommendation Logic

def predict_pest(model, image_path, class_names):
    if not os.path.exists(image_path):
        return "Error: Image not found.", 0, None

    img = Image.open(image_path).resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array, verbose=0) # Suppress Keras output
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_pest = class_names[predicted_class_index]
    return predicted_pest, confidence, predictions[0]

def get_recommendation(pest_name):
    recommendations = {
        "healthy": {
            "alert": "✅ Plant is healthy. No pest found.",
            "remedy": "Continue good plant care practices."
        },
        "aphid": {
            "alert": "⚠️ Alert: Pest detected - Aphid",
            "remedy": "🧪 Remedy: Use neem oil spray every 3 days. Introduce ladybugs (natural predator). Remove infected leaves manually. Consider insecticidal soap for heavy infestations."
        },
        "armyworm": {
            "alert": "⚠️ Alert: Pest detected - Armyworm",
            "remedy": "🧪 Remedy: Apply Bacillus thuringiensis (Bt) spray, effective against caterpillars. Hand-pick larger larvae if possible. Use pheromone traps to monitor adult moths."
        },
        "leaf_miner": {
            "alert": "⚠️ Alert: Pest detected - Leaf Miner",
            "remedy": "🧪 Remedy: Prune and destroy affected leaves promptly. Use insecticidal soap or spinosad-based pesticides. Cover young plants with row covers to prevent egg-laying."
        },
        "caterpillar": {
            "alert": "⚠️ Alert: Pest detected - Caterpillar",
            "remedy": "🧪 Remedy: Hand-pick caterpillars from leaves. Apply Bt spray if infestation is severe. Encourage beneficial insects like parasitic wasps. Use natural predators if available."
        }
    }
    return recommendations.get(pest_name.lower(), {
        "alert": f"❓ Unknown Pest: {pest_name}",
        "remedy": "No specific recommendation available. Please consult a local agricultural expert."
    })

def log_prediction(image_name, detected_pest, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if not log_exists or os.stat(LOG_FILE).st_size == 0:
            writer.writerow(["Timestamp", "Image Name", "Detected Pest", "Confidence (%)"])
        writer.writerow([timestamp, image_name, detected_pest, round(confidence, 2)])
    print(f"Logged: {image_name}, {detected_pest}, {confidence:.2f}%")

# 2.4 CustomTkinter UI Application

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.model = None
        self.image_to_predict_path = INPUT_IMAGE_NAME
        self.current_image_display = None

        self.title("Smart Crop Pest Identifier")
        self.geometry("1100x750") # Increased size for better spacing
        self.minsize(800, 600) # Minimum size
        self.grid_columnconfigure((0, 1), weight=1) # Two main columns for content
        self.grid_rowconfigure(1, weight=1) # Main content row expands

        # --- Appearance Setting (for user) ---
        self.appearance_mode_var = ctk.StringVar(value="System")
        ctk.set_appearance_mode(self.appearance_mode_var.get())

        # --- Fonts (modern and consistent) ---
        self.title_font = ctk.CTkFont(family="Arial", size=30, weight="bold")
        self.heading_font = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.sub_heading_font = ctk.CTkFont(family="Arial", size=16, weight="bold")
        self.body_font = ctk.CTkFont(family="Arial", size=14)
        self.button_font = ctk.CTkFont(family="Arial", size=16, weight="bold")
        self.output_font = ctk.CTkFont(family="Consolas", size=13) # Monospace for results

        # --- Colors (using CTk default themes, but defining for clarity) ---
        self.primary_color = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        self.secondary_color = ctk.ThemeManager.theme["CTkButton"]["hover_color"]
        self.text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        self.bg_color = ctk.ThemeManager.theme["CTkFrame"]["fg_color"] # Main app background
        self.card_bg_color = ("gray90", "gray15") # Background for sections/cards

        # --- Top Header Frame ---
        self.header_frame = ctk.CTkFrame(self, fg_color=self.primary_color[0] if ctk.get_appearance_mode() == "Light" else self.primary_color[1],
                                        corner_radius=15)
        self.header_frame.grid(row=0, column=0, columnspan=2, pady=(20, 15), padx=25, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="🌿 Smart Crop Pest Identifier 🐞", font=self.title_font, text_color="white")
        self.title_label.pack(pady=12)

        # --- Main Content Frame ---
        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=25, sticky="nsew")
        self.main_content_frame.grid_columnconfigure(0, weight=2) # Image display takes more space
        self.main_content_frame.grid_columnconfigure(1, weight=1) # Controls take less
        self.main_content_frame.grid_rowconfigure(0, weight=1)

        # --- Image Display Section (Left) ---
        self.image_display_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=15, fg_color=self.card_bg_color,
                                                border_width=2, border_color=("gray70", "gray25"))
        self.image_display_frame.grid(row=0, column=0, padx=(0, 20), pady=0, sticky="nsew")
        self.image_display_frame.grid_rowconfigure(1, weight=1) # Image label expands
        self.image_display_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.image_display_frame, text="Current Leaf Image", font=self.heading_font,
                     text_color=self.text_color).grid(row=0, column=0, pady=(15,10))
        self.image_label = ctk.CTkLabel(self.image_display_frame, text="No image loaded", compound="top",
                                        font=self.body_font, text_color=("gray50", "gray40"),
                                        bg_color=("gray95", "gray10"), corner_radius=10) # Faint background for image area
        self.image_label.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0,15))
        self.image_label.bind("<Configure>", self.on_image_label_resize) # Bind resize event

        # --- Controls and Output Section (Right) ---
        self.controls_output_frame = ctk.CTkFrame(self.main_content_frame, corner_radius=15, fg_color=self.card_bg_color,
                                                  border_width=2, border_color=("gray70", "gray25"))
        self.controls_output_frame.grid(row=0, column=1, padx=(20, 0), pady=0, sticky="nsew")
        self.controls_output_frame.grid_columnconfigure(0, weight=1)
        self.controls_output_frame.grid_rowconfigure(9, weight=1) # Make output area expandable

        ctk.CTkLabel(self.controls_output_frame, text="Actions & Results", font=self.heading_font,
                     text_color=self.text_color).grid(row=0, column=0, pady=(15,10))

        # Action Buttons
        self.upload_button = ctk.CTkButton(self.controls_output_frame, text="⬆️ Upload Image", command=self.upload_image,
                                           font=self.button_font, height=45, corner_radius=10, hover_color=self.secondary_color)
        self.upload_button.grid(row=1, column=0, pady=(10,7), padx=25, sticky="ew")

        self.image_path_label = ctk.CTkLabel(self.controls_output_frame, text=f"File: {os.path.basename(self.image_to_predict_path)}",
                                             font=self.body_font, wraplength=250, text_color=("gray40", "gray50"))
        self.image_path_label.grid(row=2, column=0, pady=5, padx=25, sticky="w")

        self.predict_button = ctk.CTkButton(self.controls_output_frame, text="✨ Identify Pest", command=self.run_prediction,
                                            font=self.button_font, fg_color="#4CAF50", hover_color="#45a049",
                                            height=45, corner_radius=10) # Green for primary action
        self.predict_button.grid(row=3, column=0, pady=(10,15), padx=25, sticky="ew")

        self.train_button = ctk.CTkButton(self.controls_output_frame, text="⚙️ Re-train Model", command=self.start_retrain_thread,
                                          font=self.button_font, fg_color="#FF9800", hover_color="#fb8c00",
                                          height=45, corner_radius=10) # Orange for secondary action
        self.train_button.grid(row=4, column=0, pady=(0,20), padx=25, sticky="ew")

        # Status and Progress
        ctk.CTkLabel(self.controls_output_frame, text="System Status:", font=self.sub_heading_font).grid(row=5, column=0, pady=(5,0), padx=25, sticky="w")
        self.status_label = ctk.CTkLabel(self.controls_output_frame, text="Ready", font=self.body_font, text_color=("gray40", "gray50"))
        self.status_label.grid(row=6, column=0, pady=(0,5), padx=25, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.controls_output_frame, orientation="horizontal", height=8, corner_radius=5)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=7, column=0, pady=(5,15), padx=25, sticky="ew")
        self.progress_bar.configure(mode="determinate")

        # Prediction Results
        ctk.CTkLabel(self.controls_output_frame, text="Detection Results:", font=self.sub_heading_font).grid(row=8, column=0, pady=(10, 5), padx=25, sticky="w")

        self.output_text = ctk.CTkTextbox(self.controls_output_frame, wrap="word", height=200, font=self.output_font,
                                          corner_radius=10, border_width=1, border_color=("gray80", "gray25"))
        self.output_text.grid(row=9, column=0, pady=(0,15), padx=25, sticky="nsew")
        self.output_text.insert("0.0", "Upload an image and click 'Identify Pest' to get started.")
        self.output_text.configure(state="disabled", fg_color=("white", "gray10"), text_color=("black", "white"))

        # --- Footer (Appearance Mode Switch) ---
        self.footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.footer_frame.grid(row=2, column=0, columnspan=2, pady=(10, 15), padx=25, sticky="ew")
        self.footer_frame.grid_columnconfigure(0, weight=1)
        self.footer_frame.grid_columnconfigure(1, weight=0) # keep switch on right

        self.appearance_mode_label = ctk.CTkLabel(self.footer_frame, text="Appearance Mode:", font=self.body_font)
        self.appearance_mode_label.grid(row=0, column=0, sticky="w")
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.footer_frame, values=["Light", "Dark", "System"],
                                                               command=self.change_appearance_mode_event,
                                                               variable=self.appearance_mode_var,
                                                               font=self.body_font)
        self.appearance_mode_optionemenu.grid(row=0, column=1, sticky="e")


        # --- Initial Load ---
        self.load_image_into_ui(self.image_to_predict_path)
        self.update_status("Initializing model...", progress=0)
        self.load_model_threaded() # Start model loading/training in a thread

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def update_status(self, message, progress=None):
        self.status_label.configure(text=message)
        if progress is not None:
            self.progress_bar.set(progress)
        self.update_idletasks() # Force UI update

    def load_model_threaded(self):
        # Disable buttons during initial load/train
        self.upload_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        
        thread = threading.Thread(target=self._load_model_task)
        thread.start()

    def _load_model_task(self):
        try:
            self.model, _ = load_or_train_model(self.update_training_progress)
            self.update_status("Model Ready for predictions!", progress=1)
        except Exception as e:
            self.update_status(f"Initialization Failed: {e}", progress=0)
            messagebox.showerror("Initialization Error", f"Failed to load or train model: {e}\nPlease check your dataset structure (dataset/train/CLASS_NAME/images) and console for details.")
        finally:
            # Re-enable buttons
            self.upload_button.configure(state="normal")
            self.predict_button.configure(state="normal")
            self.train_button.configure(state="normal")

    def update_training_progress(self, type, value, status_text=None):
        # This function runs in the training thread, so safely update UI elements
        if type == "progress":
            self.after(0, self.progress_bar.set, value)
            if status_text:
                self.after(0, self.status_label.configure, {"text": status_text})
        elif type == "status":
            self.after(0, self.status_label.configure, {"text": value})
        elif type == "error":
            self.after(0, self.update_status, value, 0) # Set progress to 0 on error
            self.after(0, messagebox.showerror, "Training Error", value)


    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_to_predict_path = file_path
            self.image_path_label.configure(text=f"File: {os.path.basename(file_path)}")
            self.load_image_into_ui(file_path)
            self.update_output("Image loaded. Click 'Identify Pest' to analyze.")
            self.update_status("Image ready for prediction.")

    def load_image_into_ui(self, image_path):
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                # Keep original aspect ratio but resize to fit within label
                self.original_image = img.copy() # Store original for dynamic resizing
                self.resize_image_for_display()
            except Exception as e:
                self.image_label.configure(image=None, text=f"Error loading image: {e}", text_color="red")
                self.update_output(f"Error loading image: {e}")
                self.update_status(f"Error loading image: {e}", progress=0)
        else:
            self.image_label.configure(image=None, text="Image file not found.\nUpload an image or place 'test_leaf.jpg' in project root.", text_color="red")
            self.update_output("Error: Image file not found.")
            self.update_status("Error: Image file not found.", progress=0)

    def on_image_label_resize(self, event):
        # This function is called when the image_label widget changes size
        if hasattr(self, 'original_image') and self.original_image:
            self.resize_image_for_display()

    def resize_image_for_display(self):
        if not hasattr(self, 'original_image') or self.original_image is None:
            return

        label_width = self.image_label.winfo_width() - 20 # Account for padding
        label_height = self.image_label.winfo_height() - 20 # Account for padding

        if label_width <= 0 or label_height <= 0:
            # Fallback if label size is not yet determined
            label_width = 450
            label_height = 450

        img = self.original_image.copy()
        img.thumbnail((label_width, label_height), Image.LANCZOS)
        
        self.current_image_display = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self.image_label.configure(image=self.current_image_display, text="")


    def update_output(self, message):
        self.output_text.configure(state="normal")
        self.output_text.delete("0.0", "end")
        self.output_text.insert("0.0", message)
        self.output_text.configure(state="disabled")

    def run_prediction(self):
        if self.model is None:
            messagebox.showwarning("Model Not Ready", "The model is still initializing/loading or has not been trained yet. Please wait or re-train.")
            return

        self.update_output("Identifying pest...")
        self.update_status("Running prediction...", progress=0.5)

        if not os.path.exists(self.image_to_predict_path):
            self.update_output(f"Error: Image '{self.image_to_predict_path}' not found.\nPlease upload an image or ensure 'test_leaf.jpg' exists in the project root.")
            self.update_status("Prediction failed: Image not found.", progress=0)
            return
        
        if not CLASS_NAMES:
            messagebox.showwarning("Classes Not Found", "Model classes are not loaded. This usually happens if the model failed to train or load class names. Please try re-training the model.")
            self.update_status("Prediction failed: Model classes not found.", progress=0)
            return

        try:
            detected_pest, confidence, raw_predictions = predict_pest(self.model, self.image_to_predict_path, CLASS_NAMES)
            recommendation = get_recommendation(detected_pest)

            output_message = f"Detected: {detected_pest.replace('_', ' ').title()}\n"
            output_message += f"Confidence: {confidence:.2f}%\n\n"
            output_message += f"{recommendation['alert']}\n\n"
            output_message += f"{recommendation['remedy']}\n\n"

            output_message += "--- Detailed Confidences ---\n"
            # Sort predictions for better readability
            sorted_predictions = sorted(zip(CLASS_NAMES, raw_predictions), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_predictions:
                output_message += f"- {class_name.replace('_', ' ').title()}: {prob*100:.2f}%\n"

            self.update_output(output_message)
            log_prediction(os.path.basename(self.image_to_predict_path), detected_pest, confidence)
            self.update_status(f"Prediction complete: {detected_pest.replace('_', ' ').title()}", progress=1)

        except Exception as e:
            error_msg = f"Error during prediction: {e}. Check console for details."
            self.update_output(error_msg)
            self.update_status(f"Prediction failed: {e}", progress=0)
            messagebox.showerror("Prediction Error", error_msg)


    def start_retrain_thread(self):
        confirm = messagebox.askyesno("Confirm Re-training", 
                                      "Re-training the model can take a significant amount of time and resources. "
                                      "Are you sure you want to proceed? "
                                      "Ensure your 'dataset/train' folder is correctly structured.")
        if not confirm:
            return

        self.update_output("Re-training model... This may take some time. Please wait.")
        self.update_status("Re-training in progress...", progress=0)

        # Disable buttons during training
        self.upload_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.train_button.configure(state="disabled")

        thread = threading.Thread(target=self._retrain_task)
        thread.start()

    def _retrain_task(self):
        try:
            self.model, _ = train_model(os.path.join(DATASET_PATH, 'train'), NUM_CLASSES, self.update_training_progress)
            self.update_status("Model re-trained successfully! Ready for predictions.", progress=1)
            self.update_output("Model re-training complete. You can now use the updated model for predictions.")
        except Exception as e:
            self.update_status(f"Re-training failed: {e}", progress=0)
            messagebox.showerror("Re-training Error", f"Failed to re-train model: {e}\nPlease check your dataset structure (dataset/train/CLASS_NAME/images) and console for details.")
            self.update_output(f"Re-training failed: {e}")
        finally:
            # Re-enable buttons
            self.upload_button.configure(state="normal")
            self.predict_button.configure(state="normal")
            self.train_button.configure(state="normal")


# --- Main execution block ---
if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("green")  # Themes: "blue" (default), "green", "dark-blue"

    app = App()
    app.mainloop()