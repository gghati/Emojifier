# Emojifier!  

![](https://img.shields.io/badge/Gaurav-Ghati-red)
![](https://img.shields.io/github/languages/top/gauravghati/Emojify)
![](https://img.shields.io/github/last-commit/gauravghati/Emojify)

Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. 
So rather than writing:
>"Congratulations on the promotion! Let's get coffee and talk. Love you!"   

The emojifier can automatically turn this into:
>"Congratulations on the promotion! 👍 Let's get coffee and talk. ☕️ Love you! ❤️"

This is a NLP project to Predict the Emojis for the Given Text, Using LSTM Structure.

## Baseline model: Emojifier

###  About Dataset emojiset

The dataset (X, Y) where:
- X contains 127 sentences (strings).
- Y contains an integer label between 0 and 4 corresponding to an emoji for each sentence.
- as shown in the below picture.

<img src="images/data_set.png" style="width:700px;height:300px;">
EMOJISET - a classification problem with 5 classes. A few examples of sentences are given here.

## Inputs and outputs to the embedding layer

* The figure shows the propagation of two example sentences through the embedding layer. 
    * Both examples have been zero-padded to a length of `max_len=5`.

<img src="images/embedding1.png" style="width:700px;height:250px;">
Embedding layer

### Overview of the LSTM model

This is the LSMT structure that we're going use for prediction:

<img src="images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
Emojifier. A 2-layer LSTM sequence classifier.

### Output Accuracy :
<img src="images/accuracy.png" style="width:700px;"> <br>
Trainset Accuracy of the Emoji Predicted Model.

### Output of the Model:
<img src="images/output.png" style="width:700px;"> <br>
Output of the Model on given String!


This Project is inspired by the Coursera's course on Sequence Models!
