# Meme Evolution - Data Analysis

This program provides an automated way to analyze memes by extracting text, classifying the format, determining the topic, and evaluating sentiment.

The program performs the following:
* Extracts text from meme images using Tesseract OCR.
* Classifies meme format using a ResNet50 pre-trained on ImageNet.
* Infers the topic of the meme text using zero-shot classification.
* Performs sentiment analysis on the meme text using a distilbert-base-uncased model fine-tuned for sentiment analysis.
* Analyzes all memes in a folder and provides a summary of sentiment by meme format and topic.


## Prerequisites

Before running the script, make sure you have the following Python packages installed:
* ``` opencv-python```

* ```pytesseract```

* ```numpy```

* ```torch```

* ```torchvision```

* ```transformers```

* ```Pillow```

* ```tf-keras```

*```rapidfuzz```

If you do not have these dependencies installed, follow the steps below to install them
## Installation

Install the required dependencies using pip:
```bash 
  pip install opencv-python pytesseract torch torchvision transformers Pillow numpy tf-keras rapidfuzz
```
Download Tesseract OCR from the following link: 

https://github.com/UB-Mannheim/tesseract/wiki

* Run the installer and note the installation directory (e.g., C:\Program Files\Tesseract-OCR).

* Add Tesseract to the PATH via Environment Variables

* Verify Tesseract installation by using ```tesseract --version``` in a terminal

    
## Running the Program

Clone the project

```bash
  git clone https://github.com/5ALin/MemeEvolution-CIS400.git
```

Go to the project directory

```bash
  cd MemeEvolution-CIS400
```

Run the program

```bash
  python memes.py
```


## Results

When the program runs, it will output the following information for each meme:
```
File: distracted_boyfriend.jpg
  Meme Name: Distracted Boyfriend
  Meme Format: 203
  Topic: Relationships
  Text: "When you find a better option."
  Sentiment: Negative
```
It will also print a sentiment summary at the end that will look similar to this:
```
Meme Format: Distracted Boyfriend
  Topic: Relationships
    Positive Sentiment: 15.00%
    Negative Sentiment: 85.00%

Meme Format: Expanding Brain
  Topic: Technology
    Positive Sentiment: 40.00%
    Negative Sentiment: 60.00%

Meme Format: Drake Hotline Bling
  Topic: Work
    Positive Sentiment: 75.00%
    Negative Sentiment: 25.00%
```

## Modifications

* **Known Memes:** The known_memes dictionary can be expanded with additional meme formats and their descriptions.
* **Topics:** The possible labels for topic inference (politics, work, relationships, etc.) can be modified to fit your needs.
* **Sentiment Analysis Model:** You can change the sentiment analysis model to another pre-trained model from Hugging Face if desired.
## Troubleshooting

* **Tesseract OCR Not Working**
    
    Ensure Tesseract is correctly installed and its path is added to your system    environment.Improve image quality (resize, increase contrast) to help Tesseract recognize text better.
* **File Not Found or Path Issues** 
    
    Double-check the file paths and ensure the correct image extensions are being used (.jpg, .png).

* **Slow Performance or Memory Errors**
    * Resize images before processing to reduce memory usage.
    * Use GPU acceleration with torch.device('cuda') if available.
    * Process images concurrently using multi-threading or multi-processing.
