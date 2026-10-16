"""
Dựng prompt truy xuất "kim trong đống rơm" theo thang độ khó của RULER.

Passkey hiện dùng trong dự án tương đương biến thể DỄ NHẤT (S-NIAH-1): đống rơm là vài câu
lặp lại, kim là một số 5 chữ số. Chỗ mọi phương pháp trạng thái-cố-định thực sự sụp là các
biến thể khó hơn, nơi arXiv:2608.28444 báo SWA chỉ đạt 23,0 còn LoLCATs 5,8 so với 99,8 của
attention đầy đủ. Không có thước đo này thì không đo được tiến bộ của bất kỳ hướng nào.

Năm biến thể, xếp theo độ khó:

  s1  đống rơm lặp lại      + giá trị số 7 chữ số      (≈ passkey cũ)
  s2  đống rơm văn bản thật + giá trị số 7 chữ số      (nhiễu ngữ nghĩa)
  s3  đống rơm văn bản thật + giá trị UUID 32 ký tự    (chuỗi dài, mọi token phải đúng)
  mk  nhiều kim khác NHÃN, hỏi đúng một nhãn           (phải phân biệt, không chỉ tìm thấy)
  mv  một nhãn, nhiều GIÁ TRỊ, phải lấy đủ             (nhiều lần truy xuất trong một lượt)

Chấm điểm theo RULER: sinh văn bản rồi kiểm giá trị đúng có xuất hiện trong đầu ra không.
Với `mv` là tỉ lệ số giá trị lấy được. Cách chấm này rộng lượng hơn exact-match và là cách
các bài báo về RULER dùng, nên số liệu so sánh được với họ.
"""
import random
import string
from typing import List, Tuple

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")

PREAMBLE = ("Some special magic information is hidden within the following text. "
            "Make sure to memorize it. I will quiz you about it afterwards.\n")

TAIL = "\nWhat is the special magic {label}? The special magic {label} is:"


def _uuid(rng: random.Random) -> str:
    h = ''.join(rng.choice('0123456789abcdef') for _ in range(32))
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'


def _number(rng: random.Random, n: int = 7) -> str:
    return str(rng.randint(10 ** (n - 1), 10 ** n - 1))


def _word(rng: random.Random) -> str:
    return ''.join(rng.choice(string.ascii_lowercase) for _ in range(8))


def make_value(rng, kind: str) -> str:
    return {'number': _number, 'uuid': _uuid, 'word': _word}[kind](rng)


def needle_line(label: str, value: str) -> str:
    return f'The special magic {label} is: {value}.'


def build(tokenizer, variant: str, target_len: int, depth: float, rng: random.Random,
          haystack_words: List[str] = None, n_extra: int = 3
          ) -> Tuple[List[int], List[str], str]:
    """
    Trả về (prompt_ids, gold_values, câu hỏi đã dùng).

    `target_len` là số token mong muốn của prompt; hàm không bao giờ vượt quá.
    `depth` trong [0,1] là vị trí tương đối của kim trong đống rơm.
    `haystack_words`: kho từ lấy từ văn bản thật; None thì dùng câu lặp lại.
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    natural = variant in ('s2', 's3', 'mk', 'mv') and haystack_words
    val_kind = 'uuid' if variant == 's3' else 'number'

    if variant == 'mv':
        label = _word(rng)
        golds = [make_value(rng, val_kind) for _ in range(n_extra + 1)]
        needles = [needle_line(label, v) for v in golds]
    elif variant == 'mk':
        labels = [_word(rng) for _ in range(n_extra + 1)]
        values = [make_value(rng, val_kind) for _ in labels]
        pick = rng.randrange(len(labels))
        label, golds = labels[pick], [values[pick]]
        needles = [needle_line(l, v) for l, v in zip(labels, values)]
    else:
        label = _word(rng)
        golds = [make_value(rng, val_kind)]
        needles = [needle_line(label, golds[0])]

    head = enc(PREAMBLE)
    tail = enc(TAIL.format(label=label))
    needle_ids = [enc(' ' + n) for n in needles]
    budget = target_len - len(bos) - len(head) - len(tail) - sum(len(n) for n in needle_ids)
    if budget < 0:
        raise ValueError(f'target_len {target_len} quá nhỏ cho biến thể {variant}')

    if natural:
        # ghép từ thật cho tới khi đủ ngân sách; ước lượng rồi cắt cho chính xác
        approx = max(1, budget // 2)
        text = ' '.join(rng.choice(haystack_words) for _ in range(approx * 2))
        body = enc(text)[:budget]
    else:
        g = enc(FILLER)
        body = (g * (budget // len(g) + 1))[:budget]

    # rải các kim vào thân, kim CHÍNH (hoặc kim đầu) đặt ở đúng `depth`
    k = len(needle_ids)
    if k == 1:
        cuts = [int(round(depth * len(body)))]
    else:
        # trải đều quanh depth để mọi kim đều nằm trong đống rơm
        span = len(body)
        base = int(round(depth * span))
        step = max(1, span // (k + 1))
        cuts = sorted(min(span, max(0, base + (i - k // 2) * step)) for i in range(k))
    ids = bos + head
    prev = 0
    for c, nid in zip(cuts, needle_ids):
        ids += body[prev:c] + nid
        prev = c
    ids += body[prev:] + tail
    return ids, golds, label


def score(text: str, golds: List[str]) -> float:
    """Tỉ lệ giá trị đúng xuất hiện trong đầu ra (cách chấm của RULER)."""
    t = text.lower()
    return sum(g.lower() in t for g in golds) / len(golds)


def load_haystack_words(path: str, max_docs: int = 2000) -> List[str]:
    """
    Đọc văn bản thật từ một thư mục (.txt / .jsonl / .parquet / .arrow) làm đống rơm.
    Trả về danh sách từ. Không có đường dẫn thì trả về None và bên gọi dùng câu lặp lại.
    """
    import glob
    import json
    import os
    if not path or not os.path.exists(path):
        return None
    texts = []
    for f in sorted(glob.glob(os.path.join(path, '**', '*'), recursive=True))[:200]:
        if len(texts) >= max_docs:
            break
        try:
            if f.endswith('.txt'):
                texts.append(open(f, encoding='utf-8', errors='ignore').read())
            elif f.endswith('.jsonl'):
                for i, line in enumerate(open(f, encoding='utf-8', errors='ignore')):
                    if i >= max_docs:
                        break
                    texts.append(json.loads(line).get('text', ''))
            elif f.endswith(('.arrow', '.parquet')):
                import pyarrow as pa
                import pyarrow.parquet as pq
                tbl = (pq.read_table(f) if f.endswith('.parquet')
                       else pa.ipc.open_stream(pa.memory_map(f, 'rb')).read_all())
                if 'text' in tbl.column_names:
                    texts += tbl.column('text').to_pylist()[:max_docs]
        except Exception:
            continue
    words = ' '.join(t for t in texts if t).split()
    return words if len(words) > 10_000 else None
