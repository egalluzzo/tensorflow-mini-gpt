# TensorFlow Mini GPT

This project is heavily based on [this tutorial from Keras](https://keras.io/examples/generative/text_generation_with_miniature_gpt/).  It implements a small GPT model with a customizable number of Transformer blocks and attention heads per block.  It uses word embedding, rather than token embedding.  The positional encoding layer consists of learned weights, rather than being sinusoidal.  Differences from the above tutorial are:

* More transformer blocks
* Using Gelu instead of Relu (doesn't really make much difference)
* Saving at checkpoints

## Training

To train this model, you will need some text files with one utterance per line.  For testing, I downloaded a set of IMDB movie reviews into the `data` directory as follows:

```shell
mkdir data
cd data
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xf aclImdb_v1.tar.gz
```

Then you can create a Python virtual environment and run `train.py` as follows.  Note that this will likely take several hours.  You may need to replace the Mac-specific `tensorflow-metal` in `requirements.txt` with the equivalent GPU acceleration for your platform.

```shell
python -m venv ./.venv/
pip install -r requirements.txt
python train.py
```

## Running

After training the model, you can generate text as follows:

```shell
python --checkpoint=./checkpoints/ckpt-epoch-50 --vocab=./checkpoints/vocab.txt "this movie is"
```

(or whatever prompt you want to start with).
