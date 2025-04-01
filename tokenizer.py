import re
import urllib.request

UNK_TOKEN="<|unk|>"
EOT_TOKEN="<|eot|>"
EOS_TOKEN="<|eos|>"


def get_sample_book():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    return load_sample_book()

def load_sample_book():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_text = raw_text + EOT_TOKEN
    return raw_text

def split_text(text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed

def build_vocabulary(text):
    preprocessed = split_text(text)
    preprocessed = sorted(set(preprocessed))
    preprocessed.append(UNK_TOKEN)
    preprocessed.append(EOS_TOKEN)
    preprocessed.append(EOT_TOKEN)
    vocab = dict()
    for i, word in enumerate(preprocessed):
        vocab[word] = i
    return vocab

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }

    def encode(self, text):
        preprocessed = split_text(text)
        unk_id = self.str_to_int[UNK_TOKEN]
        ids = [self.str_to_int.get(item, unk_id) for item in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text
    
    def __len__(self):
        return len(self.str_to_int)

if __name__=="__main__":
    raw_text = load_sample_book()
    vocab = build_vocabulary(raw_text)
    tokenizer = SimpleTokenizerV2(vocab)

    TEST_SENTENCE = "I like to I corner I --"
    ids = tokenizer.encode(TEST_SENTENCE)
    print("Ids: ", ids)
    decoded_sentence = tokenizer.decode(ids)
    print("Decoded sentence: ", decoded_sentence)