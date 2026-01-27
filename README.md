# DigitRecognizer (0-9) trained on MNIST Dataset

DigitRecognizer is a small project that uses a neural network built using only numpy and pandas.

The neural network has a very simple architecture with one hidden layer of 10 units, achieving 85% accuracy on the training set after 500 epochs.

This project uses a simple flask server with python's inbuilt http library to serve a simple GUI.

Users will be able to draw a digit and see how their entry will be preprocessed similar to the original dataset numbers before being fed into the model for prediction.

# Brief History on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) (Wikipedia)

The MNIST was constructed sometime before summer 1994. It was constructed by mixing 128x128 binary images from SD-3 and SD-7. Specifically, they first took all images from SD-7 and divided them into a training set and a test set, each from 250 writers. This resulted in nearly 30000 images in each set. They then added more images from SD-3 until each set contained exactly 60000 images.

Each image was size-normalized to fit in a 20x20 pixel box while preserving their aspect ratio, and anti-aliased to grayscale. Then it was put into a 28x28 image by translating it until the center of mass of the pixels is in the center of the image. The details of how the downsampling proceeded was reconstructed

# Setting up environment
**Making a virtual environment**
```terminal
python3 -m venv myenv
```
**Activating virtual environment**
```terminal
source myenv/bin/activate
```
**Install requirements**
```terminal
pip install -r requirements.txt
```
**Training Model**
```terminal
python3 ./neuralnetwork.py
```
# Running the GUI
**Backend**
```terminal
python3 app.py
```
**Frontend**
```terminal
python3 -m http.server 8000
```