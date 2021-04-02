import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import random
import keras 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical


data = pd.read_csv("data/train.csv").as_matrix()

clf1 = KNeighborsClassifier() #KNN Classifier
clf2 = DecisionTreeClassifier() # Decision Tree Classifier
clf3 = RandomForestClassifier() # Random Forest Classifier
clf4 = LinearSVC() # LinearSVC


train_inputs = data[0:21000, 1:]
train_outputs = data[0:21000, 0]
test_inputs = data[21000:, 1:]
test_outputs = data[21000:, 0]

while True:
	print("\n1. KNeighbours (KNN)\n2. Decision Tree Classifier\n3. Random Forest Classifier\n4. Linear SVC\n5. Convolutional Neural Network")
	
	n = input('Enter your choice: ')
	if n == "1":
		###### KNeighbors(KNN) Classifier ########
		# Train the KNeighbors(KNN) Classifier
		clf1.fit(train_inputs, train_outputs)

		# Test of KNeighbors(KNN) Classifier
		print("\n\n******************* KNEIGHBORS(KNN) CLASSIFIER *****************************\n")

		print("\tPredicted Result\t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# Take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf1.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with KNeighbors Classifier: %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
		print("\n\n")

###### End of KNeighbors(KNN) Classifier ########

	elif n == "2":
		#Train Decision Tree Classifier
		clf2.fit(train_inputs, train_outputs)
		print("******************* DECISION TREE CLASSIFIER *****************************\n")

		print("\tPredicted Result Decition Tree \t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf2.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with Decision Tree Classifier:  %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
		print("\n\n")
##### End of Decision Tree Classifier ###########

	elif n == "3":
		##### Random Forest Classifier ######

		#Train Random Forest Classsifier
		clf3.fit(train_inputs, train_outputs)

		print("******************* RANDOM FOREST CLASSIFIER *****************************\n")
		print("\tPredicted Result \t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf3.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with Random Forest Classifier:  %.2f%%" % (correct_predictions_count / nb_tests * 100.0))

##### End of Random Forest Classifier #####
	elif n == "4":
		##### SVM ######

		#Train SVM Classsifier
		clf4.fit(train_inputs, train_outputs)

		print("******************* LINEAR SVC*****************************\n")
		print("\tPredicted Result \t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf4.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with SVM Classifier:  %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
##### End of SVM Classifier #####

########## Convolutional Neural Network ###############	
	elif n == "5":
		#from keras.datasets import mnist #importing MNIST dataset
		#(train_inputs, train_outputs), (test_inputs, test_outputs) = mnist.load_data()
		batch_size = 64
		num_classes = 10
		epochs = 1
		
		img_rows, img_cols = 28, 28


		train_inputs = train_inputs.reshape(21000,28,28,1)
		test_inputs = test_inputs.reshape(21000,28,28,1)

		print('train_inputs shape:', train_inputs.shape)
		print(train_inputs.shape[0], 'train samples')
		print(test_inputs.shape[0], 'test samples')

		train_outputs = keras.utils.to_categorical(train_outputs, num_classes)
		test_outputs = keras.utils.to_categorical(test_outputs, num_classes)

		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(num_classes, activation='softmax'))
		
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
		model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_inputs, test_outputs))
		score = model.evaluate(test_inputs, test_outputs, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		
		predictions = model.predict(test_inputs)
		print(np.argmax(np.round(predictions[0])))
		pt.imshow(test_inputs[0].reshape(28, 28), cmap = "gray")
		pt.show()
	
	elif n == "exit":
		break
	else:
		print("\nInvalid Input!!! Try Again")
	



