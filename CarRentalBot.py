#import the dependencies

import numpy
import random
import tflearn
import json
import tensorflow

#import Natural Language Toolkit
import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer

#create words list to store words in
words = []
#create doc list to store patterns in
docs_x = []
docs_y = []
#create labels list to store in labels
labels = []


#Start the Data Preprocessing phase
with open("discussion.json") as file:
	data = json.load(file)

	for discussion in data["discussions"]:
		for pattern in discussion["patterns"]:
			
			wrd = nltk.word_tokenize(pattern)
			words.extend(wrd)
			docs_x.append(wrd)
			docs_y.append(discussion["tag"])
		if discussion["tag"] not in labels:
			labels.append(discussion["tag"])

#till here works
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
	bag = []

	wrds = [stemmer.stem(w.lower()) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

#create and train the model

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")