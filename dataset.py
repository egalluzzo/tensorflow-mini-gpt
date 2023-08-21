import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os
import random
import string

def create_dataset_and_vocab(
        directories: list[str],
        vocab_size: int,
        maxlen: int,
        batch_size = 128):

    # The dataset contains each review in a separate text file
    # The text files are present in four different folders
    # Create a list all files
    filenames = []
    for dir in directories:
        for f in os.listdir(dir):
            filenames.append(os.path.join(dir, f))

    print(f"{len(filenames)} files")

    # Create a dataset from text files
    random.shuffle(filenames)
    text_ds = tf.data.TextLineDataset(filenames)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)

    # Create a vectorization layer and adapt it to the text
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size - 1,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices

    def prepare_lm_inputs_labels(text):
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y
    
    text_ds = text_ds.map(prepare_lm_inputs_labels)
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

    return (text_ds, vocab)


def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    stripped_html = tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1 ")
    return tf.strings.regex_replace(stripped_html, f"\\s+", " ")
