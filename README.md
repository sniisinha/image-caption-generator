## Image Caption Generator

Image Caption Generator is a deep learning project aimed at developing a tool that can automatically generate captions for images. It leverages computer vision and natural language processing techniques to create meaningful captions. Dataset used is [flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).

A swin transformer is used to extract encodings for images, following which a transformer model is used to generate captions.

Captions are generated using beam search and the model is evaluated using bleu score.

## Prerequisites

- Python: 3.10
- Libraries: TensorFlow 2.x, NumPy, scikit_learn

## Project Structure

- `dataset.py`: Implements DatasetGenerator class which generates batches of inputs.
- `extract.py`: Extracts the encodings of the images using a Swin Tranformer.
- `model.py`: Defines the model used for image captioning.
- `utils.py`: Defines functions used to extract pre-trained word embeddings, clean the captions (remove punctuation, convert to lower case), and create the vocabulary.
- `train.ipynb`: Trains and evalutes the model.
