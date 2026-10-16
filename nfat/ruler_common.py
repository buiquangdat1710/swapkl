"""Phần dùng chung cho RULER, KHÔNG import code LoLCATs — chạy được cả ở env nfat (tf 4.43)
lẫn env swaa (tf 4.57). nfat/eval_ruler.py và nfat/run_ruler_vllm.py đều import từ đây."""

# tên trong bộ sinh ruler_niah.py <-> tên task chính thức của RULER
RULER_NAME = {'S-NIAH-PK': 'niah_single_1', 'S-NIAH-N': 'niah_single_2',
              'S-NIAH-W': 'niah_single_3', 'MK-1': 'niah_multikey_1',
              'MK-2': 'niah_multikey_2', 'MK-3': 'niah_multikey_3',
              'MV': 'niah_multivalue', 'MQ': 'niah_multiquery', 'VT': 'vt',
              'CWE': 'cwe', 'FWE': 'fwe'}


class BaseProcessor:
    """Lớp đệm cho mô hình base: 'chat template' = BOS + nội dung, không vai trò."""
    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return (self.tokenizer.bos_token or '') + messages[-1]['content']


class ChatProcessor:
    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)
