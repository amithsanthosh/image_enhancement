A Comparative Study of Low-Light Image Enhancement Algorithms

Overview
This project aims to explore and compare various low-light image enhancement techniques using deep learning methods. Specifically, it focuses on three models: Autoencoders, Dense Autoencoders, and Generative Adversarial Networks (GANs). Each model is trained and evaluated on a dataset of low-light and corresponding enhanced images, and their performance is compared based on the loss during training and image quality after enhancement.


Run the Notebooks
The main code for training the models is located in the Jupyter notebooks.

Dependencies
- tensorflow: For model building, training, and evaluation.
- numpy: For numerical operations.
- skimage: For image processing utilities.
- matplotlib: For plotting and visualizing images and loss graphs.
- PIL and Pillow: For image handling and enhancement.
- scikit-learn: For data splitting.

Image Enhancement Workflow
Preprocessing: Load and preprocess the low-light and enhanced images by resizing them and applying any necessary image adjustments (e.g., contrast enhancement).
Convert the images to NumPy arrays and tensors suitable for training.

Model Architecture:
- Autoencoder Model: A simple convolutional autoencoder with encoding and decoding layers.
- Dense Autoencoder Model: A deeper and more complex version of the autoencoder, with additional layers and more parameters.
- GAN Model: A Generative Adversarial Network that consists of a generator and a discriminator for image enhancement.

Training: Split the dataset into training and testing sets.
Train each model (Autoencoder, Dense Autoencoder, and GAN) on the low-light images, and evaluate them based on loss metrics and image quality after enhancement.

Evaluation: After training, the models are tested on the test set, and their performance is visualized. Enhanced images are displayed alongside their original low-light counterparts.


Results
Training Loss: Loss curves for each model are plotted to observe the performance and convergence over epochs.
Visual Output: Enhanced images are displayed side-by-side with the original low-light images to evaluate the visual enhancement quality.

Contributing
If you'd like to contribute to this project, feel free to fork the repository, make improvements, and submit a pull request. We welcome contributions in the form of new models, bug fixes, optimizations, or documentation improvements.

Issues
If you encounter any issues while using this code or have suggestions for improvements, please open an issue on the GitHub repository page.


Acknowledgments
LOLDataset: The dataset used for this project was taken from the LOLDataset repository.
TensorFlow: For providing the framework used for building and training the deep learning models.
