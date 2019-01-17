from tkinter import Tk, Label, Button, filedialog, E, W, StringVar, NSEW, messagebox
import _pickle as cPickle #cPickle in Python3 comes as a library _pickle
import pandas as pd
from gensim.models import FastText
from sklearn import preprocessing
import numpy as np
import re
from gensim.test.utils import get_tmpfile
import classes
import os

VECTOR_DIM = 100
failed_indexes = [] # failed sentences while fasttext tries to convert

def sentence_vectorizer(input_array, operand='Average'):
    if operand == 'Average':
        sentence_vector = np.average(input_array, axis=0)
    elif operand == 'Squares':
        sentence_vector = np.sqrt(np.sum(np.square(input_array), axis=0))
    elif operand == 'Both':
        sentence_vector = np.concatenate((np.average(input_array, axis=0), np.sqrt(np.sum(np.square(input_array), axis=0))), axis=0)
    return sentence_vector

# Set av variable to num if you don't want to combine multiple columns into one array.
def convert_str_array_to_numpy_vector(model, inputs, av, operation):
	no_exceptions = 0
	i = 0
	w2v_array = []
	try:
		for line in range(len(inputs)):
			question = re.sub(' +', ' ', str(inputs[line]).strip()) #[1], question = str(inputs[line][0]).strip()

			words = question.split()
			word_array = []
			for word in words:
				try:
					fasttext_word = model[word] #Throws exception on unknown words
					word_array.append(fasttext_word)
				except Exception as ex:
					no_exceptions += 1
					word_array.append(np.zeros(VECTOR_DIM))
			line_avg_vector = sentence_vectorizer(word_array, operation)
			line_avg_vector[VECTOR_DIM - 1] = av[i]
			w2v_array.append(line_avg_vector)
			i += 1
	except Exception as ex:
		print("Exception occured in convert_to_fasttext_numpy_vector: " + str(ex))
		pass

	print('# of failed word calculation: ' + str(no_exceptions))
	return w2v_array


class PredictGUI:

	WIDTH = 500
	HEIGHT = 100
	X = 300
	Y = 200

	def __init__(self, master):
		self.master = master
		self.model = None
		master.title("PredictGUI")
		self.master.geometry('{}x{}+{}+{}'.format(self.WIDTH, self.HEIGHT, self.X, self.Y))

		# ELEMENTS
		self.input_file_name = StringVar()
		self.input_file_name.set("Select input csv file: (sentence, source)")
		self.label_input_file_name = Label(self.master, textvariable=self.input_file_name)
		self.button_input_file = Button(self.master, text="Select file", command=self.select_input_file)

		self.output_file_name = StringVar()
		self.output_file_name.set("Select output csv file that will be created.")
		self.label_output_file_name = Label(self.master, textvariable=self.output_file_name)
		self.button_output_file = Button(self.master, text="Select file", command=self.select_output_file)

		self.button_predict = Button(self.master, text="Predict", command=self.predict)
		# ELEMENTS

		# LAYOUT
		self.label_input_file_name.grid(row=0, column=0, sticky=E)
		self.button_input_file.grid(row=0, column=1, sticky=W)
		self.label_output_file_name.grid(row=1, column=0, sticky=E)
		self.button_output_file.grid(row=1, column=1, sticky=W)
		self.button_predict.grid(row=3, column=0, sticky=NSEW)
		# LAYOUT

	def select_input_file(self):
		filename = filedialog.askopenfilename(parent=root, title='Choose a file', filetypes=[("CSV files", "*.csv")])
		self.input_file_name.set(filename)

	def select_output_file(self):
		filename = filedialog.asksaveasfilename(parent=root, title='Choose a file', filetypes=[("CSV files", "*.csv")], confirmoverwrite=False)
		self.output_file_name.set(filename)

	def predict(self):
		try:
			#ML stuff below.
			#load pre-existing classifier from disk
			self.fp_fasttext_model = os.getcwd() + '\\assets\\model\\fasttext\\fasttext.model'
			if(os.path.isfile(self.fp_fasttext_model)):
				if(self.model == None):
					self.model = FastText.load(get_tmpfile(self.fp_fasttext_model))
			else:
				raise Exception('Fasttext model file cannot be found at "{}"'.format(self.fp_fasttext_model))

			self.fp_classifier = os.getcwd() + '\\assets\\model\\model_gen_1_0_75803995341754.pkl'
			if (os.path.isfile(self.fp_classifier)):
				with open(self.fp_classifier, 'rb') as fid:
					self._classifier = cPickle.load(fid)
			else:
				raise Exception('Machine learning model file cannot be found at "{}"'.format(self.fp_classifier))

			# load data
			if (os.path.isfile(self.input_file_name.get())):
				self._df = pd.read_csv(self.input_file_name.get(), sep=";", skipinitialspace=True, engine='python') #engine='python' required to run in ipython
			else:
				raise Exception('Input file cannot be found at "{}"'.format(self.input_file_name.get()))

			# prepare
			try:
				self.le = preprocessing.LabelEncoder()
				self.le.fit(classes.classes)
				self._df['source'] = preprocessing.LabelEncoder().fit_transform(self._df['source'].values)
				self._df['sentence'] = convert_str_array_to_numpy_vector(self.model, self._df['sentence'].values.tolist(), self._df['source'].values.tolist(), 'Average')
			except Exception as ex:
				raise Exception('Error occured while preparing input data. Are you sure it has only (sentence,source) columns? Trace: {}'.format(str(ex)))

			# predict
			try:
				self._y_pred = self._classifier.predict(self._df['sentence'].values.tolist())
				self._y_pred = self.le.inverse_transform(self._y_pred)
			except Exception as ex:
				raise Exception('Error occured while predicting input data. Are you sure it has only (sentence,source) columns? Trace: {}'.format(str(ex)))

			# save data
			np.savetxt(self.output_file_name.get(), self._y_pred, delimiter=',', fmt='%s')
			messagebox.showinfo("Info", "Predicting is finished. File is saved to {}".format(self.output_file_name.get()))

		except Exception as e:
			messagebox.showinfo("Error", "Error occured while predicting. Send details to alozta\nDetails:\n{}".format(str(e)))

#main
print("#############################################################################################")
print("#                                     COPS PREDICTOR GUI                                    #")
print("#                                                                                           #")
print("# Requirements:                                                                             #")
print("# - Python 3.7 environment                                                                  #")
print("# - Fasttext model file under " + "%HOMEPATH%\\assets\\model\\fasttext\\fasttext.model" + "               #")
print("# - ML model file under " + "%HOMEPATH%\\assets\\model\\model_gen_1_0_75803995341754.pkl" + "            #")
print("#############################################################################################")
root = Tk()
my_gui = PredictGUI(root)
root.mainloop()