import tensorflow as tf
import argparse
from generator import TextGenerator

# Make sure this agrees with train.py
maxlen = 80 # Max sequence size

def load_vocab(vocab_file):
    return [line[:-1] for line in vocab_file]

parser = argparse.ArgumentParser(description='Run the generative model.')
parser.add_argument('prompt', nargs='+',
                    help='the starting prompt for the generative model')
parser.add_argument('--checkpoint', metavar="checkpointDir", nargs=1,
                    required=True, help='checkpoint directory to load')
parser.add_argument('--vocab', metavar="vocabFile", nargs=1,
                    type=argparse.FileType('r'), required=True,
                    help='vocab file to load')

args = parser.parse_args()
prompt = " ".join(args.prompt).lower()

# Load our saved model.
model = tf.keras.models.load_model(args.checkpoint[0])
vocab = load_vocab(args.vocab[0])

# Generate some text!
generator = TextGenerator(model=model, maxlen=maxlen, index_to_word=vocab)
text = generator.generate_text(prompt=prompt, max_tokens=maxlen)
print(f"Generated text: {text}")
