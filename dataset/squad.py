import json
import os

import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.python.platform import gfile

from dataset.rc_dataset import RCDataset, default_tokenizer, process_tokens
from utils.log import logger


class SQuAD(RCDataset):
    def __init__(self, args):
        super(SQuAD, self).__init__(args)
        self.w_len = 10

    def next_batch_feed_dict_by_dataset(self, dataset, _slice, samples):
        if self.args.use_char_embedding:
            data = {
                "documents_bt:0": dataset[0][_slice],
                "questions_bt:0": dataset[1][_slice],
                "documents_bt_char:0": dataset[2][_slice],
                "questions_bt_char:0": dataset[3][_slice],
                "answer_start:0": dataset[4][_slice],
                "answer_end:0": dataset[5][_slice]
            }
        else:
            data = {
                "documents_bt:0": dataset[0][_slice],
                "questions_bt:0": dataset[1][_slice],
                # TODO: substitute with real data
                # "documents_btk:0": np.zeros([samples, self.d_len, self.w_len]),
                # "questions_btk:0": np.zeros([samples, self.q_len, self.w_len]),
                "answer_start:0": dataset[2][_slice],
                "answer_end:0": dataset[3][_slice]
            }
        return data, samples

    def preprocess_input_sequences(self, data):
        if not self.args.use_char_embedding:
            documents, questions, answer_spans = data
        else:
            documents, questions, documents_char, questions_char, answer_spans = data
            documents_char_ok = pad_sequences(documents_char, maxlen = self.d_len, dtype = "int32", padding = "post",
                                              truncating = "post")
            questions_char_ok = pad_sequences(questions_char, maxlen = self.q_len, dtype = "int32", padding = "post",
                                              truncating = "post")

        documents_ok = pad_sequences(documents, maxlen = self.d_len, dtype = "int32", padding = "post", truncating = "post")
        questions_ok = pad_sequences(questions, maxlen = self.q_len, dtype = "int32", padding = "post", truncating = "post")

        # FIXME: here can not use the array ,because the postiton is counted under character. not words
        answer_start = [np.array([int(i == answer_span[0]) for i in range(self.d_len)]) for answer_span in answer_spans]
        answer_end = [np.array([int(i == answer_span[1]) for i in range(self.d_len)]) for answer_span in answer_spans]
        if self.args.use_char_embedding:
            return documents_ok, questions_ok, documents_char_ok, questions_char_ok, np.asarray(answer_start), np.asarray(answer_end)
        else:
            return documents_ok, questions_ok, np.asarray(answer_start), np.asarray(answer_end)

    def prepare_data(self, data_dir, train_file, valid_file, max_vocab_num, output_dir = ""):
        """
        build word vocabulary and character vocabulary.
        """
        if not gfile.Exists(os.path.join(data_dir, output_dir)):
            os.mkdir(os.path.join(data_dir, output_dir))
        os_train_file = os.path.join(data_dir, train_file)
        os_valid_file = os.path.join(data_dir, valid_file)
        vocab_file = os.path.join(data_dir, output_dir, "vocab.%d" % max_vocab_num)
        char_vocab_file = os.path.join(data_dir, output_dir, "char_vocab")

        vocab_data_file = os.path.join(data_dir, output_dir, "data.txt")

        def save_data(d_data, q_data):
            """
            save all data to a file and use it build vocabulary.
            """
            with open(vocab_data_file, mode = "w", encoding = "utf-8") as f:
                f.write("\t".join(d_data) + "\n")
                f.write("\t".join(q_data) + "\n")

        if not gfile.Exists(vocab_data_file):
            d, q, _ = self.read_squad_data(os_train_file)
            v_d, v_q, _ = self.read_squad_data(os_valid_file)
            save_data(d, q)
            save_data(v_d, v_q)
        if not gfile.Exists(vocab_file):
            logger("Start create vocabulary.")
            word_counter = self.gen_vocab(vocab_data_file, max_count = self.args.max_count)
            self.save_vocab(word_counter, vocab_file, max_vocab_num)
        if not gfile.Exists(char_vocab_file):
            logger("Start create character vocabulary.")
            char_counter = self.gen_char_vocab(vocab_data_file)
            self.save_char_vocab(char_counter, char_vocab_file, max_vocab_num = 70)

        return os_train_file, os_valid_file, vocab_file, char_vocab_file

    def read_squad_data(self, file):
        """
        read squad data file in string form
        :return tuple of (documents, questions, answer_spans)
        """
        logger("Reading SQuAD data.")

        def extract(sample_data):
            document = sample_data["context"]
            for qas in sample_data["qas"]:
                question = qas["question"]
                for ans in qas["answers"]:
                    answer_len = len(ans["text"])
                    answer_span = [ans["answer_start"], ans["answer_start"] + answer_len]
                    assert (ans["text"] == document[ans["answer_start"]:(ans["answer_start"] + answer_len)])
                    assert (ans["answer_start"] != answer_span[1] != 0)
                    documents.append(document)
                    questions.append(question)
                    answer_spans.append(answer_span)

        documents, questions, answer_spans = [], [], []
        f = json.load(open(file, encoding = "utf-8"))
        data_list, version = f["data"], f["version"]
        logger("SQuAD version: {}".format(version))
        [extract(sample) for data in data_list for sample in data["paragraphs"]]

        if self.args.debug:
            documents, questions, answer_spans = documents[:1000], questions[:1000], answer_spans[:1000]

        return documents, questions, answer_spans

    def squad_data_to_idx(self, vocab_file, char_vocab_file = None, *args):
        """
        convert string list to index list form.         
        """
        logger("Convert string data to index.")
        word_dict = self.load_vocab(vocab_file)
        if self.args.use_char_embedding:
            char_dict = self.load_vocab(self.char_vocab_file)
        res_data = []
        for idx, i in enumerate(args):
            tmp = [self.sentence_to_token_ids(document, word_dict) for document in i]
            res_data.append(tmp.copy())
            if self.args.use_char_embedding:
                tmp_c = [self.words_to_char_ids(document, char_dict) for document in i]
                res_data.append(tmp_c.copy())
        logger("Convert string2index done.")
        return res_data

    def token_idx_map(self, context, answer_span):
        logger("Convert answer to position in the context.")
        answer_se = []
        for i in range(len(context)):
            answer_tokens = process_tokens(default_tokenizer(context[i][answer_span[i][0]: answer_span[i][1]]))
            con = process_tokens(default_tokenizer(context[i][:answer_span[i][0]]))
            a_start_idx = len(con)
            a_end_idx = len(con) + len(answer_tokens)
            answer_se.append([a_start_idx, a_end_idx])
        return answer_se

    # noinspection PyAttributeOutsideInit
    def get_data_stream(self):
        # prepare data

        os_train_file, os_valid_file, self.vocab_file, self.char_vocab_file = self.prepare_data(self.args.data_root,
                                                                                                self.args.train_file,
                                                                                                self.args.valid_file,
                                                                                                self.args.max_vocab_num,
                                                                                                self.args.tmp_dir)
        # read data
        documents, questions, answer_spans = self.read_squad_data(os_train_file)
        v_documents, v_questions, v_answer_spans = self.read_squad_data(os_valid_file)

        use_char = self.args.use_char_embedding and self.char_vocab_file
        if use_char:
            documents_ids, documents_char_ids, questions_ids, questions_char_ids, v_documents_ids, v_documents_char_ids, v_questions_ids, v_questions_char_ids = self.squad_data_to_idx(
                self.vocab_file, self.char_vocab_file, documents,
                questions,
                v_documents, v_questions)
        else:
            if not self.char_vocab_file and self.args.use_char_embedding:
                # Warning(
                #     'No char vocabulary file has been found, char embedding will be ignored. set the value of self.char_vocab_file to enable the char embedding in squad.py.')
                raise Exception(
                    "No char vocabulary file has been found,. set the value of self.char_vocab_file to enable the char embedding in squad.py.")
            documents_ids, questions_ids, v_documents_ids, v_questions_ids = self.squad_data_to_idx(self.vocab_file, self.char_vocab_file,
                                                                                                    documents, questions,
                                                                                                    v_documents, v_questions)
        v_answer_spans = self.token_idx_map(v_documents, v_answer_spans)
        answer_spans = self.token_idx_map(documents, answer_spans)

        # SQuAD cannot access the test data
        # first 9/10 train data     ->     train data
        # last  1/10 train data     ->     valid data
        # valid data                ->     test data
        train_num = len(documents) * 9 // 10
        self.train_data = (documents_ids[:train_num], questions_ids[:train_num], answer_spans[:train_num]) if not use_char \
            else (documents_ids[:train_num], questions_ids[:train_num], documents_char_ids[:train_num], questions_char_ids[:train_num],
                  answer_spans[:train_num])
        self.valid_data = (documents_ids[train_num:], questions_ids[train_num:], answer_spans[train_num:]) if not use_char else (
            documents_ids[train_num:], questions_ids[train_num:], documents_char_ids[train_num:], questions_char_ids[train_num:],
            answer_spans[train_num:])
        self.test_data = (v_documents_ids, v_questions_ids, v_answer_spans) if not use_char else (
            v_documents_ids, v_questions_ids, v_documents_char_ids, v_questions_char_ids, v_answer_spans)

        def get_max_length(d_bt):
            lens = [len(i) for i in d_bt]
            return max(lens)

        # data statistics
        self.d_len = get_max_length(self.train_data[0])
        self.q_len = get_max_length(self.train_data[1])

        self.d_char_len = max([get_max_length(i) for i in self.train_data[2]]) if use_char else None
        self.q_char_len = max([get_max_length(i) for i in self.train_data[3]]) if use_char else None

        self.train_sample_num = len(self.train_data[0])
        self.valid_sample_num = len(self.valid_data[0])
        self.test_sample_num = len(self.test_data[0])
        self.train_idx = np.random.permutation(self.train_sample_num // self.args.batch_size)
        print(self.d_char_len)
        print(self.q_char_len)
        return self.d_len, self.q_len, self.train_sample_num, self.valid_sample_num, self.test_sample_num, self.d_char_len, self.q_char_len
