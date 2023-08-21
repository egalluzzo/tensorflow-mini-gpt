import tensorflow as tf
from tensorflow import keras
from generator import TextGenerator

class TextGenerationCallback(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        maxlen: Integer, the number of input tokens
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
            self,
            model: tf.keras.Model,
            maxlen: int,
            max_tokens: int,
            index_to_word: dict[int, str],
            start_prompt: str,
            top_k: int = 10,
            print_every: int = 1
    ):
        self.generator = TextGenerator(
            model=model,
            maxlen=maxlen,
            index_to_word=index_to_word,
            top_k=top_k
        )
        self.max_tokens = max_tokens
        self.start_prompt = start_prompt
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        
        txt = self.generator.generate_text(
            prompt=self.start_prompt,
            max_tokens=self.max_tokens
        )
        print(f"generated text:\n{txt}\n")
