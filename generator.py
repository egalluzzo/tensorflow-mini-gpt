import tensorflow as tf
from tensorflow import keras
import numpy as np

class TextGenerator:
    """A class to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        model: Model, the model to use for text generation
        maxlen: Integer, the number of input tokens
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
    """

    def __init__(
            self,
            model: tf.keras.Model,
            maxlen: int,
            index_to_word: dict[int, str],
            top_k=10,
    ):
        self.model = model
        self.maxlen = maxlen
        self.index_to_word = index_to_word
        self.k = top_k

        # Generate the reverse lookup.
        self.word_to_index = {}
        for index, word in enumerate(index_to_word):
            self.word_to_index[word] = index

    def _sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)
    
    def _tokenize(self, text):
        return [self.word_to_index.get(word, 1) for word in text.split()]

    def _detokenize(self, tokens):
        return " ".join(
            [self.index_to_word[number] for number in tokens]
        )

    def generate_text(
            self,
            prompt: str,
            max_tokens: int):
        """
        Arguments:
            prompt: str, the prompt with which to start generating text
            max_tokens: Integer, the number of tokens to be generated after prompt.
        """
        initial_start_tokens = self._tokenize(prompt)
        start_tokens = [_ for _ in initial_start_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= max_tokens:
            pad_len = self.maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.maxlen]
                sample_index = self.maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self._sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = self._detokenize(initial_start_tokens + tokens_generated)
        return txt
