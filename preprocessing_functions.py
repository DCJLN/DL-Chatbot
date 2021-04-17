from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def cleaning(text):
    # punctuations = '''()+=-[]{};:"`\,<>./?@#$%^&*_~|'''
    # not_inc: $%^&*_~
    punctuations = '''[-()\"#/@;:<>{}`+=~|.!?,]'''
    _text = text.lower()
    
    _text = _text.replace("--", " ")
    _text = _text.replace("  ", " ")
    
    # We remove all the non necessary symbols
    for character in _text:
        if character in punctuations:
            _text = _text.replace(character, "")
    
    _text = _text.replace("i'm", "i am")
    _text = _text.replace("he's", "he is")
    _text = _text.replace("she's", "she is")
    _text = _text.replace("it's", "it is")
    _text = _text.replace("'re", " are")

    _text = _text.replace("isn't", "is not")
    _text = _text.replace("wasn't", "was not")
    _text = _text.replace("aren't", "are not")
    _text = _text.replace("weren't", "were not")
    
    _text = _text.replace("that's", "that is")
    _text = _text.replace("what's", "what is")
    _text = _text.replace("where's", "where is")
    
    _text = _text.replace("who's", "who is")
    _text = _text.replace("when's", "when is")
    _text = _text.replace("how's", "how is")
    _text = _text.replace("there's", "there is")
    
    _text = _text.replace("'ve", " have")
    _text = _text.replace("'d", " would")
    _text = _text.replace("'ll", " will")
    
    _text = _text.replace("haven't", "have not")
    _text = _text.replace("hasn't", "has not")
    _text = _text.replace("hadn't", "had not")
    _text = _text.replace("wouldn't", "would not")
    _text = _text.replace("won't", "will not")
    _text = _text.replace("can't", "cannot")
    _text = _text.replace("couldn't", "could not")
    _text = _text.replace("don't", "do not")
    _text = _text.replace("doesn't", "does not")
    _text = _text.replace("didn't", "did not")
    _text = _text.replace("shouldn't", "should not")
    
    _text = _text.replace("'bout", " about")
    _text = _text.replace("'til", "until")
    
    return _text


def cleaner(lines):
    _lines = []
    for i in range(len(lines)):
        _lines.append(cleaning(lines[i]))
    return _lines


def length_filter(questions, answers, max_length_q, max_length_a):
    _questions = []
    _answers = []
    for i in range(len(questions)):
        if len(questions[i].split()) <= max_length_q and len(answers[i].split()) <= max_length_a:
            _questions.append(questions[i])
            _answers.append(answers[i])
    return _questions, _answers


def start_end_adder(answers):
    special_tokens = ['<BOS>', '<EOS>']
    _answers = []
    for i in range(len(answers)):
        _answers.append(special_tokens[0] + ' ' + answers[i] + ' ' + special_tokens[1])
    return _answers


def vocab_builder(text, num_words, oov_token, number):
    if number == False:
        to_filter = '0123456789!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    else:
        to_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(num_words=num_words, filters=to_filter, oov_token=oov_token)
    tokenizer.fit_on_texts(text)
    return tokenizer


def padder(lines, max_length):
    # max_length = max([len(i) for i in lines])
    padded_lines = pad_sequences(lines, 
                                 maxlen=max_length, 
                                 padding='post')
    return padded_lines