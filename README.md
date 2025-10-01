# Face Recognition Lock

This project uses a Raspberry Pi and a camera to perform face recognition and unlock a solenoid lock.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Add images of authorized people to the `dataset` folder. Each person should have their own subfolder within the `dataset` folder.
2.  Run the `model_training.py` script to train the face recognition model:
    ```bash
    python model_training.py
    ```
3.  Run the `facial_recognition_hardware.py` script to start the face recognition and lock system:
    ```bash
    python facial_recognition_hardware.py
    ```
