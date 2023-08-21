# Implement a mini-GPT.  This is mostly copied from
# https://keras.io/examples/generative/text_generation_with_miniature_gpt/
#
# Changes include:
# - More transformer blocks
# - Using Gelu instead of Relu (doesn't really make much difference)
# - Saving at checkpoints

import os
from model import create_model
from dataset import create_dataset_and_vocab
from text_callback import TextGenerationCallback
from save_callback import SaveCallback

vocab_size = 30000     # Only consider the top 30k words
maxlen = 80            # Max sequence size
start_prompt = "this movie is"
checkpoints_dir="./checkpoints"
text_dirs = [
    "data/aclImdb/train/pos",
    "data/aclImdb/train/neg",
    "data/aclImdb/test/pos",
    "data/aclImdb/test/neg",
]
batch_save_interval = 20
vocab_file = f"{checkpoints_dir}/vocab.txt"

# vocab_size = 40000     # Only consider the top 40k words
# maxlen = 120           # Max sequence size
# start_prompt = "he said"
# text_dirs = [
#     "data/gutenberg/data/text",
# ]
# batch_save_interval = 200
# vocab_file = "checkpoints/book_vocab.txt"

model = create_model(
    vocab_size = vocab_size,
    maxlen = maxlen,
)

(text_ds, vocab) = create_dataset_and_vocab(
    directories=text_dirs,
    vocab_size=vocab_size,
    maxlen=maxlen
)

os.makedirs(checkpoints_dir, exist_ok=True)

with open(vocab_file, 'w') as fp:
    fp.write('\n'.join(vocab))

# Tokenize starting prompt
num_tokens_generated = 40
text_gen_callback = TextGenerationCallback(
    model=model,
    maxlen=maxlen,
    max_tokens=num_tokens_generated,
    index_to_word=vocab,
    start_prompt=start_prompt)
save_callback = SaveCallback(
    model=model,
    checkpoints_dir=checkpoints_dir,
    batch_save_interval=20)

model.fit(
    text_ds,
    verbose=2,
    epochs=50,
    callbacks=[save_callback, text_gen_callback])
