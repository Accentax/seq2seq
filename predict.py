#!/usr/bin/env python
from allennlp.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq


import itertools

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.dataset_readers.copynet_seq2seq import CopyNetDatasetReader

from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq


from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import Seq2SeqPredictor
from allennlp.training.trainer import Trainer

from allennlp.models.archival import load_archive
from allennlp.predictors import Seq2SeqPredictor

from allennlp.models.archival import load_archive, archive_model
serialization_dir="./run_1"
archive_model(serialization_dir=serialization_dir,
                      archive_path=serialization_dir + "/new_path.tar.gz")
archive = load_archive('./run_1/new_path.tar.gz')
predictor = Seq2SeqPredictor.from_archive(archive, 'seq2seq')
validation_file="small_test.tsv"
reader = CopyNetDatasetReader(
    target_namespace="target_tokens",
    source_tokenizer=WordTokenizer(),
    target_tokenizer=WordTokenizer(),
    source_token_indexers={'tokens': SingleIdTokenIndexer()},
    #target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')}
    )
validation_dataset = reader.read(validation_file)


for instance in itertools.islice(validation_dataset, 10):
    print('SOURCE:', instance.fields['source_tokens'].tokens)
    print('GOLD:', instance.fields['target_tokens'].tokens)
    print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])

