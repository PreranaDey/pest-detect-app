# Smart Crop Pest Identifier

<img width="1920" height="1080" alt="Screenshot (2)" src="https://github.com/user-attachments/assets/3bd3475b-f4c6-436f-a76f-05c069bb186c" />

## 🌿 Project Overview

The **Smart Crop Pest Identifier** is a desktop application built using Python with CustomTkinter for the GUI and TensorFlow/Keras for the deep learning model. Its primary goal is to help farmers and agricultural enthusiasts quickly identify common crop pests from leaf images, providing instant alerts and recommended remedies. This tool aims to improve crop health management, reduce crop loss, and promote sustainable farming practices.

### Key Features:

*   **Image Upload:** Easily upload leaf images for analysis.
*   **Pest Identification:** Utilizes a Convolutional Neural Network (CNN) to detect various crop pests (e.g., Aphid, Armyworm, Leaf Miner, Caterpillar) or identify a healthy leaf.
*   **Confidence Score:** Displays the model's confidence in its prediction.
*   **Remedy Recommendations:** Provides practical recommendations for managing detected pests.
*   **Model Re-training:** Allows users to re-train the model with new or updated datasets directly from the UI, improving its accuracy over time.
*   **Prediction Logging:** Maintains a log of all predictions for record-keeping.
*   **Modern UI:** A clean, intuitive, and attractive user interface built with CustomTkinter, supporting light/dark themes.
*   **Non-blocking Operations:** Model training runs in a separate thread, keeping the UI responsive.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+** (Recommended)
*   **pip** (Python package installer)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd Smart-Crop-Pest-Identifier
    ```
    *(Remember to replace `your-username/your-repo-name` with your actual GitHub repository details)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file. See the "Dependencies" section below.)*
    **OR in Local Machine**
    ```bash
    pip install tensorflow keras opencv-python pillow scikit-learn customtkinter ```
5.  **Prepare the Dataset:**
    The application expects a dataset structured as follows for training:
    ```
    dataset/
    ├── train/
    │   ├── healthy/
    │   │   ├── healthy_001.jpg
    │   │   └── ...
    │   ├── aphid/
    │   │   ├── aphid_001.jpg
    │   │   └── ...
    │   ├── armyworm/
    │   │   ├── armyworm_001.jpg
    │   │   └── ...
    │   └── ... (other pest classes)
    └── test/ (Optional, for evaluation outside the app)
        ├── healthy/
        └── ...
    ```
    *   Create a `dataset` folder in the project root.
    *   Inside `dataset`, create a `train` folder.
    *   Inside `train`, create subfolders for each class (e.g., `healthy`, `aphid`, `armyworm`, `leaf_miner`, `caterpillar`).
    *   Place relevant images in each class subfolder.
    *   A sample `test_leaf.jpg` can be placed in the project root for initial testing, or you can upload any image via the UI.

### Running the Application

Once everything is set up, run the main application script:

```bash
python main.py  # or whatever your main script is named, e.g., app.py
