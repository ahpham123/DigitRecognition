# DigitRecognizer (0-9) trained on MNIST Digit Dataset

DigitRecognizer is a small project that uses a neural network built using only numpy and pandas.

The neural network has a very simple architecture with one hidden layer of 10 units, achieving 85% accuracy on the training set after 500 epochs.

This project uses a simple flask server with python's inbuilt http library to serve a simple GUI for users to draw a digit and feed it into the model to see predictions.

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