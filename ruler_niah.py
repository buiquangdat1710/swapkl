"""
ruler_niah.py — RULER data generator, doi chieu truc tiep voi ma nguon
NVIDIA/RULER (scripts/data/synthetic/niah.py, constants.py).

== BAN v2: SUA LOI DATA NGHIEM TRONG ==
Ban v1 dung 160 danh tu hardcode lam key. RULER goc dung key ghep
tinh tu-danh tu (wonderwords): 912 adj x 6782 noun = 6.18 TRIEU key.

Hau qua tren MK-2/MK-3 (haystack toan la needle): voi 160 key va ~2200
needle nhieu o 32K, key muc tieu bi trung trung binh 13 lan voi 13 gia tri
KHAC NHAU -> cau hoi VO NGHIEM. Do duoc: 47% mau mo ho o 2K, 100% o 32K.
Duong cong "suy giam theo do dai" thuc chat la duong cong ti le trung key.

Ban nay:
  1. Dung dung key space cua RULER (wonderwords, co fallback).
  2. THEM bao dam tat dinh: needle nhieu khong bao gio trung key muc tieu
     (RULER dua vao xac suat ~0.16%; o day ep ve 0).
  3. Template/answer-prefix lay nguyen van constants.py, ke ca phep bien doi
     so it (Some->A, are->is, numbers->number) khi chi co 1 dap an.
  4. Haystack noise/needle noi bang "\n" dung nhu RULER.
  5. Pool needle cua train/test khac nhau (tranh nhiem cheo).
  6. VT co one-shot demonstration nhu Table 8 cua paper.
"""

import os
import re
import json
import uuid
import random
import string
import hashlib
from typing import List, Dict, Any, Union, Tuple

DATA_VERSION = "v2"          # doi -> cache eval set cu tu dong bi bo qua

NOISE_HAYSTACK = ("The grass is green. The sky is blue. The sun is yellow. "
                  "Here we go. There and back again.")

NEEDLE_TMPL = "One of the special magic {tnv} for {key} is: {value}."

NIAH_TEMPLATE = (
    "Some special magic {tnv} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {tnv} afterwards.\n"
    "{context}\n"
    "What are all the special magic {tnv} for {query} mentioned in the provided text?"
)
NIAH_ANSWER_PREFIX = (
    " The special magic {tnv} for {query} mentioned in the provided text are"
)

VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the "
    "following text.\n\n{context}\n"
    "Question: Find all variables that are assigned the value {query} in the "
    "text above."
)
VT_ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)

CWE_TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more often "
    "than others. Memorize the ones that appear most often.\n{context}\n"
    "Question: What are the {num_cw} most common words in the above list?"
)
CWE_ANSWER_PREFIX = " Answer: The top {num_cw} words that appear most often in the list are:"

FWE_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. "
    "Find the three most frequently appeared coded words. {context}\n"
    "Question: Do not provide any explanation. Please ignore the dots '....'. "
    "What are the three most frequently appeared words in the above coded text?"
)
FWE_ANSWER_PREFIX = (
    " Answer: According to the coded text above, the three most frequently "
    "appeared words are:"
)

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "S-NIAH-PK": dict(haystack="repeat", k="words", v="numbers", nk=1, nv=1, nq=1),
    "S-NIAH-N":  dict(haystack="essay",  k="words", v="numbers", nk=1, nv=1, nq=1),
    "S-NIAH-W":  dict(haystack="essay",  k="words", v="uuids",   nk=1, nv=1, nq=1),
    "MK-1":      dict(haystack="essay",  k="words", v="numbers", nk=4, nv=1, nq=1),
    "MK-2":      dict(haystack="needle", k="words", v="numbers", nk=1, nv=1, nq=1),
    "MK-3":      dict(haystack="needle", k="uuids", v="uuids",   nk=1, nv=1, nq=1),
    "MV":        dict(haystack="essay",  k="words", v="numbers", nk=1, nv=4, nq=1),
    "MQ":        dict(haystack="essay",  k="words", v="numbers", nk=4, nv=1, nq=4),
    "VT":        dict(haystack="repeat", k="vt",    v="vt",      nk=1, nv=1, nq=1),
    # --- bien the KHO hon: tang so item phai nho cung luc ---
    "MV-8":      dict(haystack="essay",  k="words", v="numbers", nk=1, nv=8, nq=1),
    "MQ-8":      dict(haystack="essay",  k="words", v="numbers", nk=8, nv=1, nq=8),
    # --- AGGREGATION: phai tich luy thong ke tren TOAN BO context.
    #     Day la nhom task ma bo nho hoi quy co loi the ly thuyet so voi
    #     attention, va KHONG the sua bang cach hoc dinh dang.
    "CWE":       dict(haystack="agg", k="cwe", v="cwe", nk=1, nv=10, nq=1),
    "FWE":       dict(haystack="agg", k="fwe", v="fwe", nk=1, nv=3,  nq=1),
}

_TOK_PER_ANS = {"numbers": 8, "uuids": 32, "vt": 8, "cwe": 8, "fwe": 8}


def max_new_tokens(task: str) -> int:
    """Du cho cau tra loi DAI NHAT cua task. UUID ~25-30 token, khong phai 4."""
    if task == "VT":
        return 64
    if task == "CWE":
        return 120        # RULER: tokens_to_generate
    if task == "FWE":
        return 50
    c = TASK_CONFIG[task]
    return int(_TOK_PER_ANS[c["v"]] * c["nv"] * c["nq"] * 1.5) + 24


_FALLBACK_ADJS = """abundant academic acidic adorable adventurous aggressive agreeable alert
aloof amused ancient angry animated annoyed anxious arrogant ashamed astonishing attractive
average awful bewildered bitter bizarre blushing boiling bored brainy brave breezy brief
bright broad bumpy calm careful cautious charming cheerful chilly chubby clever cloudy
clumsy colossal combative comfortable concerned condemned confused cooperative courageous
crazy creepy crowded cruel curious cynical damp dangerous dapper dazzling delicious
delightful depressed determined dirty disgusted distinct dizzy doubtful drab dull eager
elated elegant embarrassed enchanting energetic enormous enthusiastic envious exuberant
faithful famous fancy fantastic fierce filthy fine foolish fragile frail frantic friendly
frightened funny gentle gifted gigantic glamorous gleaming glorious gorgeous graceful
grieving grumpy handsome happy harsh healthy helpful helpless hilarious hollow homely
horrible hungry hurt hushed husky icy ideal immense impartial innocent inquisitive
itchy jealous jittery jolly joyous juicy kind lazy lively lonely loud lovely lucky
magnificent massive melodic melted merry mighty misty modern motionless muddy mushy
mysterious narrow nervous nutritious obedient obnoxious odd old orange outrageous
panicky perfect petite plain pleasant polished powerful precious prickly proud puzzled
quaint quiet rapid relieved repulsive resonant ripe roasted robust rotten rough round
sassy scary scattered scrawny selfish shaggy shallow sharp shiny shivering shrill silky
silly skinny sleepy slimy smiling smooth soggy solid sparkling spicy splendid spotless
stale steady steep sticky stormy strange striped stunning sturdy sweet swift talented
tame tart tasty tender tense terrible thankful thoughtful thundering tidy tired tough
tranquil troubled uneven unique untidy upset uptight vast victorious vivacious voiceless
wandering warm weary wicked wide witty wobbly wonderful wooden worried yummy zany
zealous zesty""".split()

_FALLBACK_NOUNS = """abbey airport anchor antelope apple arrow attic avenue badge balloon
bamboo barrel basket beacon bicycle blanket bottle boulder bracket bridge bucket butter
cabinet cactus camera candle canyon carpet castle cavern ceiling cellar chapel cherry
chimney cinema clover compass copper cottage crayon crystal cushion dagger daisy diamond
dolphin donkey dragon drawer eagle ember engine envelope fabric falcon feather ferry
fiddle flute forest fossil fountain garden gazelle glacier granite harbor hammer harvest
helmet hillside hornet island jacket jasmine jungle kettle kitten ladder lantern lattice
lemon lighthouse lobster lumber magnet mailbox mantle marble meadow mirror monsoon mosaic
mountain muffin nectar needle nickel notebook oyster paddle palace pancake parrot pebble
pelican pepper piano pillar pigeon pocket prairie pumpkin quarry quilt rabbit rainbow
ribbon river rocket saddle sapphire satchel scarf shadow shelter shovel silver sparrow
spindle sponge squirrel stadium station statue stencil summit sunset swallow tavern
teapot temple thunder timber tornado tractor trumpet tunnel turtle umbrella valley
velvet village violin walnut wardrobe waterfall whistle willow window wisdom yogurt
zebra""".split()

_WORDS_CACHE: List[str] = []


def get_words() -> List[str]:
    """Key space cua RULER: tich Descartes adjective x noun."""
    global _WORDS_CACHE
    if _WORDS_CACHE:
        return _WORDS_CACHE
    try:
        import wonderwords
        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        src = "wonderwords (giong RULER)"
    except Exception:
        nouns, adjs = _FALLBACK_NOUNS, _FALLBACK_ADJS
        src = "fallback nhung san - nen `pip install wonderwords` de khop RULER"
    _WORDS_CACHE = sorted({f"{a}-{n}" for a in adjs for n in nouns})
    print(f"[ruler] key space = {len(_WORDS_CACHE):,} ({src})")
    return _WORDS_CACHE


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _rand_key(kind: str, rng: random.Random) -> str:
    if kind == "words":
        return rng.choice(get_words())
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _rand_val(kind: str, rng: random.Random) -> str:
    if kind == "numbers":
        return str(rng.randint(1000000, 9999999))
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class RulerNIAH:
    def __init__(self, processor, essay_sents: List[str],
                 max_essay_sents: int = 80000, pool_seed: int = 7):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pool_seed = pool_seed
        self.essay = essay_sents[:max_essay_sents]
        self.essay_lens = self._lens(self.essay)
        self.noise = [NOISE_HAYSTACK]           # RULER lap CA KHOI, khong tach cau
        self.noise_lens = self._lens(self.noise)
        self._needle_pool: Dict[str, Tuple[List[str], List[int], List[str]]] = {}
        empty = processor.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)
        self.chat_overhead = len(self.tok(empty, add_special_tokens=False).input_ids)

    def _lens(self, sents, batch=2000):
        out = []
        for i in range(0, len(sents), batch):
            out.extend(len(e) for e in
                       self.tok(sents[i:i + batch], add_special_tokens=False).input_ids)
        return out

    def _needle_haystack(self, k_kind, v_kind, n=12000):
        ck = f"{k_kind}|{v_kind}"
        if ck not in self._needle_pool:
            r = random.Random(self.pool_seed)
            keys = [_rand_key(k_kind, r) for _ in range(n)]
            sents = [NEEDLE_TMPL.format(tnv=v_kind, key=k, value=_rand_val(v_kind, r))
                     for k in keys]
            self._needle_pool[ck] = (sents, self._lens(sents), keys)
        return self._needle_pool[ck]

    def _pool(self, cfg):
        if cfg["haystack"] == "essay":
            return self.essay, self.essay_lens, None
        if cfg["haystack"] == "needle":
            return self._needle_haystack(cfg["k"], cfg["v"])
        return self.noise, self.noise_lens, None

    def _fill(self, pool, lens, keys, budget, rng, banned):
        """`banned`: key muc tieu -> filler mang key nay bi BO QUA, dam bao
        cau hoi luon co dap an DUY NHAT (cho nay ban v1 hong)."""
        n_pool = len(pool)
        i = rng.randrange(n_pool) if n_pool > 1 else 0
        chosen, total, guard = [], 0, 0
        while total < budget and guard < budget + 3000:
            j = i % n_pool
            i += 1
            guard += 1
            if keys is not None and keys[j] in banned:
                continue
            if total + lens[j] + 1 > budget:
                break
            chosen.append(pool[j])
            total += lens[j] + 1
        return chosen

    def _sample_niah(self, task, target_len, depth, rng):
        cfg = TASK_CONFIG[task]
        tnv = cfg["v"]

        keys, seen = [], set()
        while len(keys) < cfg["nk"]:
            k = _rand_key(cfg["k"], rng)
            if k not in seen:
                seen.add(k)
                keys.append(k)

        values = [[_rand_val(cfg["v"], rng) for _ in range(cfg["nv"])] for _ in keys]
        flat = [(k, v) for k, vs in zip(keys, values) for v in vs]
        needles = [NEEDLE_TMPL.format(tnv=tnv, key=k, value=v) for k, v in flat]

        q_idx = rng.sample(range(cfg["nk"]), cfg["nq"])
        queries = [keys[i] for i in q_idx]
        answers = [v for i in q_idx for v in values[i]]
        query = (", ".join(queries[:-1]) + ", and " + queries[-1]
                 if len(queries) > 1 else queries[0])

        tmpl, prefix = NIAH_TEMPLATE, NIAH_ANSWER_PREFIX
        if cfg["nq"] * cfg["nv"] == 1:
            for a, b in (("Some", "A"), ("are all", "is"), ("are", "is")):
                tmpl = tmpl.replace(a, b)
                prefix = prefix.replace(a, b)
            tnv_disp = tnv[:-1]
        else:
            tnv_disp = tnv

        skeleton = (tmpl.format(tnv=tnv_disp, context="", query=query)
                    + prefix.format(tnv=tnv_disp, query=query))
        overhead = (len(self.tok(skeleton, add_special_tokens=False).input_ids)
                    + sum(len(self.tok(n, add_special_tokens=False).input_ids)
                          for n in needles)
                    + self.chat_overhead + 8)
        budget = max(0, target_len - overhead)

        pool, lens, pkeys = self._pool(cfg)
        chosen = self._fill(pool, lens, pkeys, budget, rng, set(keys))

        qset = set(queries)
        gold_idx = [i for i, (k, _) in enumerate(flat) if k in qset]
        dist_idx = [i for i in range(len(needles)) if i not in set(gold_idx)]

        base = int(round(len(chosen) * float(depth) / 100.0))
        if len(gold_idx) > 1:
            span = max(1, len(chosen) // len(gold_idx))
            for n_i, i in enumerate(gold_idx):
                p = (base + n_i * span) % (len(chosen) + 1)
                chosen.insert(p, needles[i])
        else:
            chosen.insert(max(0, min(len(chosen), base)), needles[gold_idx[0]])
        for i in dist_idx:
            chosen.insert(rng.randrange(len(chosen) + 1), needles[i])

        sep = " " if cfg["haystack"] == "essay" else "\n"
        body = (tmpl.format(tnv=tnv_disp, context=sep.join(chosen), query=query)
                + prefix.format(tnv=tnv_disp, query=query))
        return body, answers

    _VT_DEMO = (
        "Memorize and track the chain(s) of variable assignment hidden in the "
        "following text.\n\nThe grass is green. VAR AAA = 12345 The sky is blue. "
        "VAR BBB = AAA The sun is yellow. VAR CCC = BBB Here we go. "
        "There and back again.\n"
        "Question: Find all variables that are assigned the value 12345 in the "
        "text above. Answer: According to the chain(s) of variable assignment in "
        "the text above, 3 variables are assigned the value 12345, they are: "
        "AAA BBB CCC\n\n"
    )

    def _sample_vt(self, target_len, depth, rng, num_hops=4):
        def vname():
            return "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        value = str(rng.randint(10000, 99999))
        names, seen = [], set()
        while len(names) < num_hops + 1:
            n = vname()
            if n not in seen:
                seen.add(n)
                names.append(n)
        lines = [f"VAR {names[0]} = {value}"]
        lines += [f"VAR {b} = {a}" for a, b in zip(names, names[1:])]

        skeleton = (self._VT_DEMO + VT_TEMPLATE.format(context="", query=value)
                    + VT_ANSWER_PREFIX.format(num_v=len(names), query=value))
        overhead = (len(self.tok(skeleton, add_special_tokens=False).input_ids)
                    + sum(len(self.tok(l, add_special_tokens=False).input_ids)
                          for l in lines) + self.chat_overhead + 8)
        chosen = self._fill(self.noise, self.noise_lens, None,
                            max(0, target_len - overhead), rng, set())
        base = int(round(len(chosen) * float(depth) / 100.0))
        step = max(1, len(chosen) // (len(lines) + 1))
        for off, l in enumerate(lines):
            chosen.insert(max(0, min(len(chosen), base + off * step)), l)
        body = (self._VT_DEMO + VT_TEMPLATE.format(context="\n".join(chosen), query=value)
                + VT_ANSWER_PREFIX.format(num_v=len(names), query=value))
        return body, names


    # ------------------------------------------------------- AGGREGATION
    _CWE_DEMO_N = 40      # RULER: few-shot dung 40 tu, freq 10/3

    def _cwe_example(self, num_words, f_common, f_uncommon, num_cw, rng):
        """RULER get_example(): num_cw tu 'pho bien' lap f_common lan,
        phan con lai lap f_uncommon lan, tron deu, danh so 1. w 2. w ..."""
        pool = get_words()
        sel = rng.sample(pool, min(num_words, len(pool)))
        common, uncommon = sel[:num_cw], sel[num_cw:]
        wl = common * int(f_common) + uncommon * int(f_uncommon)
        rng.shuffle(wl)
        ctx = " ".join(f"{i+1}. {w}" for i, w in enumerate(wl))
        return ctx, common

    def _sample_cwe(self, target_len, rng, num_cw=10, f_common=30, f_uncommon=3):
        # RULER: duoi 4096 token thi 10 tu pho bien x30 lan da vuot ngan sach,
        # nen dung tan suat thap hon (6/1) va demo nho hon (20 tu, 3/1).
        if target_len < 4096:
            f_common, f_uncommon = 6, 1
            demo_n, demo_fc, demo_fu = 20, 3, 1
        else:
            demo_n, demo_fc, demo_fu = self._CWE_DEMO_N, 10, 3
        demo_ctx, demo_ans = self._cwe_example(demo_n, demo_fc, demo_fu, num_cw, rng)
        demo = (CWE_TEMPLATE.format(context=demo_ctx, num_cw=num_cw)
                + CWE_ANSWER_PREFIX.format(num_cw=num_cw) + " "
                + " ".join(f"{i+1}. {w}" for i, w in enumerate(demo_ans)) + "\n")
        fixed = len(self.tok(demo + CWE_TEMPLATE.format(context="", num_cw=num_cw)
                             + CWE_ANSWER_PREFIX.format(num_cw=num_cw),
                             add_special_tokens=False).input_ids) + self.chat_overhead + 8
        budget = max(1, target_len - fixed)

        # Uoc luong so token / muc "N. adj-noun" thay vi tim nhi phan
        # (tim nhi phan tokenize lai context 32K 18 lan -> CWE cham 3x so voi MV).
        if not hasattr(self, f"_cwe_tpe{f_common}"):
            probe, _ = self._cwe_example(120, f_common, f_uncommon, num_cw, rng)
            n_items = (f_common - f_uncommon) * num_cw + f_uncommon * 120
            setattr(self, f"_cwe_tpe{f_common}",
                    len(self.tok(probe, add_special_tokens=False).input_ids) / n_items)
        tpe = getattr(self, f"_cwe_tpe{f_common}")

        def n_words_for(n_items):
            return max(num_cw + 1, (n_items - (f_common - f_uncommon) * num_cw) // f_uncommon)

        nw = n_words_for(int(budget / tpe))
        best = None
        for _ in range(4):                      # <=4 lan hieu chinh, khong phai 18
            ctx, ans = self._cwe_example(nw, f_common, f_uncommon, num_cw, rng)
            n = len(self.tok(ctx, add_special_tokens=False).input_ids)
            if n <= budget:
                best = (ctx, ans)
                if n >= budget * 0.97:
                    break
                nw = int(nw * min(1.6, budget / max(1, n)))
            else:
                nw = int(nw * 0.93 * budget / n)
            nw = max(num_cw + 1, nw)
        if best is None:
            best = self._cwe_example(num_cw + 1, f_common, f_uncommon, num_cw, rng)
        ctx, ans = best
        body = (demo + CWE_TEMPLATE.format(context=ctx, num_cw=num_cw)
                + CWE_ANSWER_PREFIX.format(num_cw=num_cw))
        return body, ans

    def _sample_fwe(self, target_len, rng, alpha=2.0, vocab_size=2000, wordlen=6):
        """RULER: tan suat tu thu k ~ k^-alpha (Zipf/Zeta). Tra 3 tu dau bang."""
        if not hasattr(self, "_fwe_vocab"):
            v = sorted({"".join(rng.choices(string.ascii_lowercase, k=wordlen))
                        for _ in range(vocab_size * 3)})[:vocab_size]
            rng.shuffle(v)
            v[0] = "..."                        # RULER: hang dau coi la nhieu
            self._fwe_vocab = v
            self._fwe_zeta = sum(k ** -alpha for k in range(1, len(v) + 1))
            self._fwe_tpw = len(self.tok(" ".join(v[:200]),
                                         add_special_tokens=False).input_ids) / 200
        vocab, zeta, tpw = self._fwe_vocab, self._fwe_zeta, self._fwe_tpw
        w_alpha = [(i + 1) ** -alpha / zeta for i in range(len(vocab))]

        def build(num_words):
            words = []
            for w, a in zip(vocab, w_alpha):
                words.extend([w] * int(num_words * a))
            rng.shuffle(words)
            return " ".join(words), vocab[1:4]

        fixed = len(self.tok(FWE_TEMPLATE.format(context="") + FWE_ANSWER_PREFIX,
                             add_special_tokens=False).input_ids) + self.chat_overhead + 8
        budget = max(1, target_len - fixed)
        nw, best = int(budget / max(tpw, 0.1)), None
        for _ in range(4):
            ctx, ans = build(nw)
            n = len(self.tok(ctx, add_special_tokens=False).input_ids)
            if n <= budget:
                best = (ctx, ans)
                if n >= budget * 0.97:
                    break
                nw = int(nw * min(1.6, budget / max(1, n)))
            else:
                nw = int(nw * 0.93 * budget / n)
            nw = max(100, nw)
        ctx, ans = best if best else build(100)
        return FWE_TEMPLATE.format(context=ctx) + FWE_ANSWER_PREFIX, ans

    def sample(self, task, target_len, depth, rng) -> Dict[str, Any]:
        if task not in TASK_CONFIG:
            raise ValueError(f"task khong hop le: {task}")
        if task == "VT":
            body, answers = self._sample_vt(target_len, depth, rng)
        elif task == "CWE":
            body, answers = self._sample_cwe(target_len, rng)
        elif task == "FWE":
            body, answers = self._sample_fwe(target_len, rng)
        else:
            body, answers = self._sample_niah(task, target_len, depth, rng)
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False,
            add_generation_prompt=True)
        return {"task": task, "prompt": prompt, "answers": answers,
                "answer": ", ".join(answers), "depth": float(depth),
                "n_tokens": len(self.tok(prompt, add_special_tokens=False).input_ids),
                "target_len": target_len}


def target_text(sample: Dict[str, Any]) -> str:
    """
    Chuoi dung lam nhan khi train — phai noi LIEN MACH voi answer-prefix.

    NIAH prefix ket thuc bang "... is" / "... are"  -> can THEM dau cach.
    VT   prefix ket thuc bang "... they are: "      -> KHONG them, tranh cach doi.
    Nhieu dap an noi bang dau cach, dung nhu vi du cua RULER ("Answer: 12345 54321").
    """
    if sample["task"] == "CWE":
        return " " + " ".join(f"{i+1}. {w}"
                              for i, w in enumerate(sample["answers"]))
    body = " ".join(sample["answers"])
    return body if sample["task"] == "VT" else " " + body


def load_essay_sentences(split: str = "train", train_frac: float = 0.8) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("sgoel9/paul_graham_essays", split="train")
    sents = split_sentences(" ".join(ds["text"]))
    cut = int(len(sents) * train_frac)
    return sents[:cut] if split == "train" else sents[cut:]


DEPTH_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def eval_set_path(cache_dir, task, length, n, seed):
    return os.path.join(cache_dir,
                        f"eval_{DATA_VERSION}_{task}_{length}_{n}_s{seed}.jsonl")


def build_eval_set(gen, task, length, n_samples, seed, cache_dir="./ruler_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    path = eval_set_path(cache_dir, task, length, n_samples, seed)
    if os.path.exists(path):
        return path
    tag = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed * 1000003 + length + tag % 100000)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for i in range(n_samples):
            f.write(json.dumps(gen.sample(task, length,
                                          DEPTH_GRID[i % len(DEPTH_GRID)], rng)) + "\n")
    os.replace(tmp, path)
    return path


def load_eval_set(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def score(prediction: str, answers: Union[str, List[str]]) -> float:
    """Recall [0,1], string-match nhu RULER. Dap an ngan (<=4 ky tu, vd ten
    bien VT) khop theo ranh gioi tu de khong trung ngau nhien."""
    if isinstance(answers, str):
        answers = [answers]
    p = prediction.lower()
    hit = 0
    for a in answers:
        a = a.lower().strip()
        if not a:
            continue
        if len(a) <= 4:
            hit += bool(re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", p))
        else:
            hit += a in p
    return hit / max(1, len(answers))


def summarize_by_depth(records):
    agg = {}
    for r in records:
        agg.setdefault(r["depth"], []).append(float(r["correct"]))
    return {d: sum(v) / len(v) for d, v in sorted(agg.items())}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - hw), min(1.0, c + hw)


def _count_ambiguous(sample) -> bool:
    """True neu mot key duoc hoi co nhieu gia tri hon so luong dap an ky vong."""
    t = sample["task"]
    if t in ("VT", "CWE", "FWE"):
        return False
    nv = TASK_CONFIG[t]["nv"]
    m = re.search(r"magic \w+ for (.+?) mentioned in the provided text\?", sample["prompt"])
    if not m:
        return True
    for kk in m.group(1).replace(", and ", ", ").split(", "):
        vals = set(re.findall(rf"magic \w+ for {re.escape(kk.strip())} is: (\S+?)\.",
                              sample["prompt"]))
        if len(vals) != nv:
            return True
    return False


def validate(gen, tasks=None, lengths=(2048, 8192, 32768), n=40, rng=None) -> bool:
    """Kiem tra tinh hop le cua data. BAT BUOC 0% mau mo ho."""
    rng = rng or random.Random(0)
    tasks = tasks or list(TASK_CONFIG)
    print(f"{'task':10s} {'L':>7} {'n_tok':>7} {'%mo ho':>8} {'#dap an':>8}")
    bad = 0
    for t in tasks:
        for L in lengths:
            amb, s = 0, None
            for _ in range(n):
                s = gen.sample(t, L, rng.choice(DEPTH_GRID), rng)
                amb += _count_ambiguous(s)
            pct = 100 * amb / n
            bad += pct > 0
            print(f"{t:10s} {L:7d} {s['n_tokens']:7d} {pct:7.0f}% "
                  f"{len(s['answers']):8d}{'  <-- HONG' if pct else ''}")
    print("\nOK: moi task deu co dap an duy nhat." if not bad
          else f"\nCO {bad} o HONG - khong dung duoc.")
    return bad == 0