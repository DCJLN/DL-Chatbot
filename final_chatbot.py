from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import sys
import hyperparameters as hp
import preprocessing_functions as pp



if len(sys.argv) != 2:
    print("Incorrect number of arguments. Please use -train or -talk argument to specify the mode")
    sys.exit()
elif sys.argv[1] != "-train" and sys.argv[1] != "-talk":
    print("Incorrect argument. Please use -train or -talk argument to specify the mode")
    sys.exit()


########################################
#####         DATA LOADING         #####
########################################

# We load the files containing all the questions and answers
questions = open('questions.txt', 'r', encoding = 'utf-8', errors = 'ignore').read()
answers = open('answers.txt', 'r', encoding = 'utf-8', errors = 'ignore').read()
questions = questions.split('\n')
answers = answers.split('\n')

# We limit the number of questions and answers due to memory issue
down = 30000
up = 85000
questions = questions[down:up]
answers = answers[down:up]



#########################################
#####         PREPROCESSING         #####
#########################################

# We clean all the questions and answers
# by removing the abbreviations and punctuations
questions = pp.cleaner(questions)
answers = pp.cleaner(answers)

# We remove the too long questions and answers
# these could causes trouble during training
questions, answers = pp.length_filter(questions, answers, 15, 14)
print("Number of questions: {}".format(len(questions)))
print("Number of answers: {}".format(len(answers)))

# We add the special tokens <BOS> and <EOS> to the answers.
answers = pp.start_end_adder(answers)

# We build the token vocabulary
vocab_size = 3876
vocab_size = vocab_size + 3  # to take the special tokens into account that are not part of the words stats
tokenizer = pp.vocab_builder(questions + answers, vocab_size, oov_token='unk', number=False)
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= vocab_size}
vocab_size = len(tokenizer.word_index) + 1 # one more to take the PAD token (0) into account
print('Vocabulary size : {}'.format(vocab_size))


# Stats to determine the threshold used for word filtration (Tokenizer) 
TH = 6
nb_total_words = 1
occ_total_words = 0
nb_word_less_TH = 0
occ_word_less_TH = 0
for i, (word, count) in enumerate(tokenizer.word_counts.items()):
    if word != 'bos' and word != 'eos':
      occ_total_words += count
      nb_total_words += 1
      if count < TH:
          occ_word_less_TH += count
          nb_word_less_TH += 1

print("Number of words: {}".format(nb_total_words))
print("Number of words with less than {} occurences: {}".format(TH, nb_word_less_TH))
print("Percentage of words with less than {} occurences: {}%".format(TH, round((occ_word_less_TH/occ_total_words)*100, 3)))
print("Resulting: {}".format(nb_total_words-nb_word_less_TH))


# We tokenize the questions and the answers
tokenized_questions = tokenizer.texts_to_sequences(questions)
tokenized_answers = tokenizer.texts_to_sequences(answers)

# We determine the maximum lengths
max_q_length = max([len(x) for x in tokenized_questions])
print("Maximum length of questions: {}".format(max_q_length))
max_a_length = max([len(x) for x in tokenized_answers])
print("Maximum length of answers: {}".format(max_a_length))



if sys.argv[1] == "-train":
    
    # We apply padding to each questions and answers to get the encoder & decoder inputs
    encoder_input = pp.padder(tokenized_questions, max_length=max_q_length)
    decoder_input = pp.padder(tokenized_answers, max_length=max_a_length)
    
    # We create the decoder targets by removing special tokens <BOS>, by applying
    # padding and finally by converting each word in an one-hot encoding
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = pp.padder(tokenized_answers, max_a_length)
    decoder_target = to_categorical(padded_answers, num_classes=vocab_size)



#########################################
#####  Pre-Trained GloVe Embedding  #####
#########################################

# We load the whole embedding into memory
embeddings_index = {}
with open('glove.6B.200d.txt', encoding='utf-8') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

# we create a matrix of one embedding for each word in the training dataset
embedding_matrix = np.zeros((vocab_size, 200))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# We create the embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=200, trainable=True)
embedding_layer.build((None,))
embedding_layer.set_weights([embedding_matrix])



##########################################
#####        SEQ2SEQ TRAINING        #####
##########################################

# --- Encoder part ---
# We instantiate keras tensor
encoder_inputs = Input(shape=(max_q_length,))
# We create an embedding layer that will convert word index into dense vector of fixed size (hidden_dim)
encoder_embedding = embedding_layer(encoder_inputs)
# We create 2 stacked Long Short-Term Memory layer
# The LSTM layer return output, hidden state and cell state
encoder_LSTM_1 = LSTM(units=hp.HIDDEN_DIM, return_state=True, return_sequences=True, dropout=hp.DROPOUT)
encoder_output_1, encoder_h_1, encoder_c_1 = encoder_LSTM_1(encoder_embedding)
encoder_LSTM_2 = LSTM(units=hp.HIDDEN_DIM, return_state=True, return_sequences=False, dropout=hp.DROPOUT)
_, encoder_h_2, encoder_c_2 = encoder_LSTM_2(encoder_output_1)
encoder_states = [encoder_h_1, encoder_c_1, encoder_h_2, encoder_c_2]

# --- Decoder part ---
# We instantiate keras tensor
decoder_inputs = Input(shape=(max_a_length,))
# We create an embedding layer that will convert word index into dense vector of fixed size (hidden_dim)
decoder_embedding = embedding_layer(decoder_inputs)
# We create 2 stacked Long Short-Term Memory layer
# The LSTM layer return output, hidden state and cell state
decoder_LSTM_1 = LSTM(units=hp.HIDDEN_DIM, return_state=True, return_sequences=True, dropout=hp.DROPOUT)
decoder_output_1, decoder_h_1, decoder_c_1 = decoder_LSTM_1(decoder_embedding, initial_state=[encoder_h_1, encoder_c_1])
decoder_LSTM_2 = LSTM(units=hp.HIDDEN_DIM, return_state=True, return_sequences=True, dropout=hp.DROPOUT)
decoder_output_2, decoder_h_2, decoder_c_2 = decoder_LSTM_2(decoder_output_1, initial_state=[encoder_h_2, encoder_c_2])
# We connect the decoder to the output layer (using Softmax activation function)
dense_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
outputs = dense_layer(decoder_output_2)

# We put the encoder and decoder together to form the seq2seq model
model = Model([encoder_inputs, decoder_inputs], outputs)
# We build and config the training seq2seq model
opt = Adam(learning_rate=hp.LEARNING_RATE)
model.compile(optimizer=opt, loss=hp.LOSS_FUNCTION, metrics=['accuracy'])
model.summary()



if sys.argv[1] == "-train":
    
    ########################################
    #####           TRAINING           #####
    ########################################

    # We train the network and save the weights
    history = model.fit([encoder_input, decoder_input], 
                        decoder_target,
                        validation_split=0.2,
                        batch_size=hp.BATCH_SIZE, 
                        epochs=hp.EPOCHS, 
                        shuffle=True)
    
    # We save the weights
    model.save('weights_{}_{}_adam_2layers_GLOVE_200.h5'.format(hp.EPOCHS, hp.BATCH_SIZE))
    
    # We save the history of the training
    history_df = pd.DataFrame(history.history)
    with open('training_history_{}_{}_adam_2layers_GLOVE_200.csv'.format(hp.EPOCHS, hp.BATCH_SIZE), 'w') as h_csv:
        history_df.to_csv(h_csv)
    print("weights and training data have been saved.")


elif sys.argv[1] == "-talk":
    
    # If we just want to talk with the chatbot, we load the weights
    model = model.load_weights('weights_{}_{}_adam_2layers_GLOVE_200.h5'.format(hp.EPOCHS, hp.BATCH_SIZE))
    print("Model has been uploaded.")
    
    
    
    ###########################################
    #####        SEQ2SEQ INFERRING        #####
    ###########################################
    # In this case we have two separate models because the encoder states will be feed to the decoder only
    # at the first time step. After that, the first word generated by the decoder and the corresponding cell
    # state and hidden state are given as input to the decoder itself and so on. 
    
    # --- Encoder part ---
    # We create the encoder model which will produce the first states used by the decoder
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # --- Decoder part ---
    # We instantiate the Keras tensor. The states that will be given to the decoder
    decoder_input_h_1 = Input(shape=(hp.HIDDEN_DIM,))
    decoder_input_c_1 = Input(shape=(hp.HIDDEN_DIM,))
    decoder_input_h_2 = Input(shape=(hp.HIDDEN_DIM,))
    decoder_input_c_2 = Input(shape=(hp.HIDDEN_DIM,))
    decoder_states_inputs = [decoder_input_h_1, decoder_input_c_1, 
                             decoder_input_h_2, decoder_input_c_2]
    
    # The LSTM layers return output, hidden state and cell state
    decoder_output, h_1, c_1 = decoder_LSTM_1(decoder_embedding, initial_state=decoder_states_inputs[:2])
    decoder_output, h_2, c_2 = decoder_LSTM_2(decoder_output, initial_state=decoder_states_inputs[-2:])
    decoder_states = [h_1, c_1, h_2, c_2]
    # We connect the decoder to the final output layer
    outputs = dense_layer(decoder_output)
    # We create the decoder model
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [outputs] + decoder_states)
    decoder_model.summary()
    
    
    
    ############################################
    #####           CHATBOT LOOP           #####
    ############################################
    
    # Fucntion to convert the format user's input sequences
    def converter(sentence):
        sentence = pp.cleaning(sentence)
        words = sentence.split()
        tokens_seq = []
        for word in words:
            result = tokenizer.word_index.get(word, '')
            if result != '':
                tokens_seq.append(result)
        return pad_sequences([tokens_seq],
                             maxlen=max_q_length,
                             padding='post')
    
    
    for _ in range(100):
        states_values = encoder_model.predict(converter(input('Enter question : ')))
        
        # start with a target sequence of size 1 containing the special token 'bos'   
        target_seq = np.zeros((1, 1)) # one line, one column
        target_seq[0, 0] = tokenizer.word_index['bos']
        
        flag = False
        translated_sentence = ''
        
        while not flag:
            # We give the one-word target sequence (special token <go>) and
            # the states vectors produced by the encoder to the decoder to
            # allow it to produce a prediction for the next word of the sequence.
            o, h1, c1, h2, c2 = decoder_model.predict([target_seq] + states_values)
            
            predicted_word_index = np.argmax(o[0, -1, :])
            predicted_word = None
            
            # We look for the word in our dictionary and stop the process if the
            # special token <eos> is produced or if the maximum length of answer
            # is exceeded
            for word, index in tokenizer.word_index.items():
                if predicted_word_index == index:
                    if word != 'eos':
                      translated_sentence += ' {}'.format(word)
                    predicted_word = word
                    
            if predicted_word == 'eos' or len(translated_sentence.split()) > max_a_length:
                flag = True
                
            # We prepare the next iteration
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = predicted_word_index
            states_values = [h1, c1, h2, c2]
        
        print(translated_sentence)
