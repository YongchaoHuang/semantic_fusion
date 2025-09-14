# -*- coding: utf-8 -*-
# yongchao.huang@abdn.ac.uk
# begin.

# simple corpus.
"""

import math, random, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# 0) Utilities / Repro
# -------------------------------
SEED = 111
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pad_to_len(x, L, pad=0):
    return x + [pad] * (L - len(x))

# -------------------------------
# 1) Synthetic corpus generator
# -------------------------------
SUBJECTS = ["Alice", "Bob", "Carol", "Dave", "Eve"]
OBJECTS  = ["task", "paper", "model", "project", "meal"]
VERBS    = ["finishes", "reviews", "trains", "starts", "cooks"]
ADJ_POS  = ["good", "great", "excellent", "pleasant", "wonderful"]
ADJ_NEG  = ["bad", "poor", "terrible", "unpleasant", "awful"]
INTENS   = ["slightly", "moderately", "very", "extremely"]
PUNCT    = [".", "!", "?"]
COMMAS   = [","]
CONJ     = ["and", "but"]

SUBJ2PRON = {"Alice":"she","Bob":"he","Carol":"she","Dave":"he","Eve":"she"}
DEFAULT_PRON = "they"

def _rand_adj():
    is_pos = random.random() < 0.5
    return random.choice(ADJ_POS if is_pos else ADJ_NEG), is_pos

def _clause_tokens(subj=None, force_pron=False):
    if subj is None:
        subj = random.choice(SUBJECTS)
    head = SUBJ2PRON.get(subj, DEFAULT_PRON) if force_pron else subj
    verb = random.choice(VERBS)
    obj  = random.choice(OBJECTS)
    intens = random.choices(INTENS, weights=[2,2,3,2], k=1)[0]
    adj, _ = _rand_adj()
    toks = [head, verb, "the", obj, ",", intens, adj]
    toks = [t if (t in SUBJECTS or t in PUNCT or t in COMMAS) else t.lower() for t in toks]
    return toks

def make_corpus(n_train=8000, n_val=1200, max_len=28,
                adj_pos_train=None, adj_neg_train=None,
                adj_pos_val=None,   adj_neg_val=None):
    adj_pos_train = adj_pos_train if adj_pos_train is not None else ADJ_POS
    adj_neg_train = adj_neg_train if adj_neg_train is not None else ADJ_NEG
    adj_pos_val   = adj_pos_val   if adj_pos_val   is not None else ADJ_POS
    adj_neg_val   = adj_neg_val   if adj_neg_val   is not None else ADJ_NEG

    def sample_with_lists(use_pos, use_neg, two_clause_p=0.6):
        subj = random.choice(SUBJECTS)
        c1 = _clause_tokens(subj=subj, force_pron=False)
        if c1[-1] in ADJ_POS + ADJ_NEG:
            c1[-1] = random.choice(use_pos if random.random() < 0.5 else use_neg)
        if random.random() < two_clause_p:
            conj = random.choice(CONJ)
            c2 = _clause_tokens(subj=subj, force_pron=True)
            if c2[-1] in ADJ_POS + ADJ_NEG:
                c2[-1] = random.choice(use_pos if random.random() < 0.5 else use_neg)
            end = random.choices(PUNCT, weights=[8,3,1], k=1)[0]
            sent = c1 + [conj] + c2 + [end]
        else:
            end = random.choices(PUNCT, weights=[8,3,1], k=1)[0]
            sent = c1 + [end]
        return [t if (t in SUBJECTS or t in PUNCT or t in COMMAS) else t.lower() for t in sent]

    train = [sample_with_lists(adj_pos_train, adj_neg_train) for _ in range(n_train)]
    val   = [sample_with_lists(adj_pos_val,   adj_neg_val)   for _ in range(n_val)]
    train = [s[:max_len] for s in train]
    val   = [s[:max_len] for s in val]
    return train, val

# -------------------------------
# 2) Vocab + tokenization
# -------------------------------
SPECIALS = ["<pad>", "<bos>", "<eos>"]
def build_vocab(sents: List[List[str]]):
    vocab = {tok for s in sents for tok in s}
    stoi = {sp:i for i,sp in enumerate(SPECIALS)}
    for tok in sorted(vocab):
        if tok not in stoi: stoi[tok] = len(stoi)
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def encode_sentence(tokens: List[str], stoi: Dict[str,int], max_len: int):
    ids = [stoi["<bos>"]] + [stoi[t] for t in tokens] + [stoi["<eos>"]]
    ids = ids[:max_len]
    ids = pad_to_len(ids, max_len, stoi["<pad>"])
    return ids, len(ids)

# -------------------------------
# 3) Fuzzy semantic features
# -------------------------------
@dataclass
class FeatureBank:
    names: List[str]

def power_membership(x: float, c: float, tau: float=0.5):
    return float(pow(0.9, abs(x - c) / max(tau, 1e-6)))

def tri_memberships(x: float, centers: List[float], tau: float=0.5):
    return [power_membership(x, c, tau) for c in centers]

NOUNS = set(OBJECTS + ["research", "dataset", "result", "paper", "report", "code", "tool"])
VERB_SET = set(v.lower() for v in VERBS)
ADJ_SET_POS = set(ADJ_POS)
ADJ_SET_NEG = set(ADJ_NEG)
INTENS_STRENGTH = {"slightly": 0.2, "moderately": 0.5, "very": 0.8, "extremely": 1.0}
PRONOUNS = {"he","she","they"}

def sentiment_score(token: str):
    if token in ADJ_SET_POS: return 1.0
    if token in ADJ_SET_NEG: return -1.0
    return 0.0

def strength_signal(token: str, punct: str):
    base = INTENS_STRENGTH.get(token, 0.0)
    if punct == "!": base = min(1.0, base + 0.2)
    return base

def build_feature_bank() -> FeatureBank:
    names = [
        "is_noun","is_verb","is_adj",
        "is_subject","is_object","is_head",
        "is_bos","is_eos","is_comma","is_question",
        "pos_low","pos_med","pos_high",
        "neg_low","neg_med","neg_high",
        "str_low","str_med","str_high",
        "coref_subject",
        "is_capitalized","is_pronoun"
    ]
    return FeatureBank(names)

def compute_semantics(tokens: List[str], max_len: int, fb: FeatureBank, stoi: Dict[str,int]) -> np.ndarray:
    T = min(len(tokens)+2, max_len)
    F = len(fb.names)
    M = np.zeros((max_len, F), dtype=np.float32)

    subj_index = 1 if T > 1 else 0
    head_index = 2 if T > 2 else 1
    obj_index  = 4 if T > 4 else min(T-2, 4)
    end_tok = tokens[-1] if len(tokens)>0 else "."
    end_punct = end_tok if end_tok in PUNCT else "."

    aligned = ["<bos>"] + tokens[:max_len-2] + ["<eos>"]
    for t in range(T):
        tok = aligned[t]
        is_bos = (t==0)
        is_eos = (t==T-1)
        is_comma = (tok == ",")
        is_question = (tok == "?")
        is_cap = (len(tok)>0 and tok[0].isupper() and tok not in PUNCT and tok not in COMMAS)
        is_pron = tok in PRONOUNS

        is_noun = float(tok in NOUNS or (tok.isalpha() and tok.lower() not in VERB_SET and tok not in ADJ_SET_POS and tok not in ADJ_SET_NEG and tok not in SUBJECTS and tok not in PRONOUNS))
        is_verb = float(tok.lower() in VERB_SET)
        is_adj  = float(tok in ADJ_SET_POS or tok in ADJ_SET_NEG)

        is_subj = float(t == subj_index)
        is_head = float(t == head_index)
        is_obj  = float(t == obj_index)

        s = sentiment_score(tok)
        pos_mag = max(0.0, s)
        neg_mag = max(0.0, -s)
        pos_low, pos_med, pos_high = tri_memberships(pos_mag, [0.2, 0.6, 1.0], tau=0.35)
        neg_low, neg_med, neg_high = tri_memberships(neg_mag, [0.2, 0.6, 1.0], tau=0.35)

        intens_tok = aligned[t-1].lower() if t-1 >= 0 else ""
        strength = strength_signal(intens_tok, end_punct)
        str_low, str_med, str_high = tri_memberships(strength, [0.2, 0.6, 1.0], tau=0.35)

        coref_subject = float(tok in {"he","she","they"} and any(p in aligned[:t] for p in SUBJECTS))

        feat = [
            is_noun, is_verb, is_adj, is_subj, is_obj, is_head,
            float(is_bos), float(is_eos), float(is_comma), float(is_question),
            pos_low, pos_med, pos_high, neg_low, neg_med, neg_high,
            str_low, str_med, str_high, coref_subject,
            float(is_cap), float(is_pron),
        ]
        M[t,:] = np.array(feat, dtype=np.float32)

    return M

# -------------------------------
# 4) Dataset
# -------------------------------
class LMDataset(torch.utils.data.Dataset):
    def __init__(self, sents, stoi, fb: FeatureBank, max_len=28):
        self.sents = sents
        self.stoi = stoi
        self.fb = fb
        self.max_len = max_len
    def __len__(self): return len(self.sents)
    def __getitem__(self, idx):
        toks = self.sents[idx]
        ids, _ = encode_sentence(toks, self.stoi, self.max_len)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        pad_id = self.stoi["<pad>"]
        mask = (x != pad_id).float()
        M = compute_semantics(toks, self.max_len, self.fb, self.stoi)
        M = torch.from_numpy(M[:-1, :])
        return x, y, mask, M

# -------------------------------
# 5) Models
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=512)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight  # weight tying
    def forward(self, x, attn_mask=None):
        h = self.tok(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=attn_mask)
        logits = self.lm_head(h)
        return logits, h

class SemanticFusionLM(nn.Module):
    def __init__(self, vocab_size, feat_dim, d_model=128, nhead=4, nlayers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.sem_proj = nn.Linear(feat_dim, d_model)
        self.gate = nn.Linear(d_model + feat_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=512)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight  # weight tying
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, max(64, feat_dim*2)),
            nn.ReLU(),
            nn.Linear(max(64, feat_dim*2), feat_dim),
        )
    def forward(self, x, sem, attn_mask=None):
        e = self.tok(x)
        s = self.sem_proj(sem)
        g = torch.sigmoid(self.gate(torch.cat([e, sem], dim=-1)))
        h0 = e + s + g * s
        h0 = self.pos(h0)
        h = self.enc(h0, src_key_padding_mask=attn_mask)
        logits = self.lm_head(h)
        sem_pred = torch.sigmoid(self.aux_head(h))
        return logits, sem_pred, h

# -------------------------------
# 6) Loss
# -------------------------------
LS_EPS = 0.02       # slightly lower label smoothing (favor PPL)
UNI_LAM = 0.01      # slightly lower uniformizer (favor PPL)
UNI_ON  = True      # toggle: set False to ablate uniformizer

def lm_step(logits, y, mask, label_smoothing=0.0):
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        y.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing
    )
    loss = (loss * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)
    return loss

def bce_step(pred, target, mask):
    loss = F.binary_cross_entropy(pred, target, reduction="none").mean(-1)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss

def adj_uniform_kl_loss(logits, y, mask, groups, lam=UNI_LAM):
    if not UNI_ON or lam <= 0:
        return torch.tensor(0.0, device=logits.device)
    B, L, V = logits.size()
    device = logits.device
    y = y.view(-1)
    mask = (mask.view(-1) > 0.5)

    pos_ids = groups.pos_adj.to(device)
    neg_ids = groups.neg_adj.to(device)

    def kl_to_uniform(sel_mask, cls_ids):
        if sel_mask.sum() == 0:
            return None
        rows = torch.nonzero(sel_mask, as_tuple=False).squeeze(-1)
        g = logits.view(-1, V)[rows][:, cls_ids]  # [N, K]
        p = torch.softmax(g, dim=-1)
        K = g.size(1)
        kl = (p * (torch.log(p + 1e-9) - math.log(1.0 / K))).sum(dim=-1)
        return kl.mean()

    losses = []
    lp = kl_to_uniform(mask & torch.isin(y, pos_ids), pos_ids)
    ln = kl_to_uniform(mask & torch.isin(y, neg_ids), neg_ids)
    if lp is not None: losses.append(lp)
    if ln is not None: losses.append(ln)
    if not losses:
        return torch.tensor(0.0, device=device)
    return lam * sum(losses) / len(losses)

# -------------------------------
# 7) Decoding helpers
# -------------------------------
def detokenize(ids, itos):
    toks = [itos[i] for i in ids if itos[i] not in {"<bos>", "<eos>", "<pad>"}]
    out = []
    for t in toks:
        if t in {".", ",", "!", "?"} and out:
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return " ".join(out)

def top_p_top_k_sample(logits, temperature=1.0, top_p=0.9, top_k: int = 0):
    """Softmax -> optional top-k clamp -> nucleus -> sample."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)

    # top-k clamp
    if top_k and top_k > 0 and top_k < probs.numel():
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        mask = torch.full_like(probs, 0.0)
        mask[topk_idx] = topk_vals
        probs = mask
        s = probs.sum()
        if s > 0:
            probs = probs / s

    # nucleus on current probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= top_p
    if sorted_probs.numel() > 0:
        keep[0] = True
    filtered = sorted_probs[keep]
    filtered_idx = sorted_idx[keep]
    s = filtered.sum()
    if s > 0:
        filtered = filtered / s
    choice = torch.multinomial(filtered, 1).item()
    return int(filtered_idx[choice].item())

def sample_group_mixture(next_logits: torch.Tensor,
                         group_ids: List[int],
                         temperature: float = 1.3,
                         top_p: float = 0.95,
                         alpha_uniform: float = 0.85):
    """
    Sample inside group_ids from q = (1-alpha)*softmax(l/T) + alpha*Uniform(group),
    and apply nucleus on q (NOT on p) to avoid clipping low-prob holdouts.
    """
    if not group_ids:
        return int(torch.argmax(next_logits).item())
    idx = torch.tensor(group_ids, device=next_logits.device, dtype=torch.long)
    g = next_logits[idx] / max(temperature, 1e-8)
    p = torch.softmax(g, dim=-1)
    u = torch.full_like(p, 1.0 / p.numel())
    q = (1 - alpha_uniform) * p + alpha_uniform * u  # mix first

    sp, si = torch.sort(q, descending=True)          # nucleus on q
    cdf = torch.cumsum(sp, dim=-1)
    keep = cdf <= top_p
    keep[0] = True
    sp = sp[keep]; si = si[keep]
    sp = sp / sp.sum()
    choice = torch.multinomial(sp, 1).item()
    return int(idx[si[choice]].item())

def simple_repetition_penalty_k(logits: torch.Tensor, prev_ids: List[int],
                                k: int = 3, penalty: float = 2.5, itos=None):
    """Apply a repetition penalty to the last-k generated IDs."""
    l = logits.clone()
    if not prev_ids or penalty <= 1.0:
        return l
    last = prev_ids[-k:] if len(prev_ids) >= k else prev_ids
    for tid in last:
        l[tid] -= math.log(max(penalty, 1.0001))
    if itos is not None and prev_ids:
        last_tok = itos[prev_ids[-1]]
        if last_tok in {".", "!", "?", ","}:
            l[prev_ids[-1]] -= math.log(1.8)
    return l

def build_semantics_for_prefix(prefix_ids, itos, fb, max_len):
    L = min(len(prefix_ids), max_len)
    toks = [itos[i] for i in prefix_ids if itos[i] not in {"<bos>", "<pad>"}]
    M_np = compute_semantics(toks, max_len, fb, None)
    M = torch.from_numpy(M_np[:L, :]).float()
    if L == 0:
        M = torch.zeros(1, len(fb.names), dtype=torch.float32)
    return M

def apply_control(M, fb, control, mode="last"):
    if not control: return M
    name_to_idx = {n:i for i,n in enumerate(fb.names)}
    M = M.clone()
    idxs = [M.size(0)-1] if mode == "last" else list(range(M.size(0)))
    for k,v in control.items():
        if k in name_to_idx:
            j = name_to_idx[k]
            for i in idxs:
                M[i, j] = float(v)
    return M

# -------------------------------
# 7b) Vocab groups + steering (for 1-clause generator)
# -------------------------------
@dataclass
class VocabGroups:
    subjects: torch.Tensor
    verbs:    torch.Tensor
    the_id:   int
    objects:  torch.Tensor
    comma_id: int
    intens:   torch.Tensor
    pos_adj:  torch.Tensor
    neg_adj:  torch.Tensor
    exclam:   int
    qmark:    int
    period:   int

def build_vocab_groups(stoi: Dict[str,int]) -> VocabGroups:
    def ids(xs): return torch.tensor([stoi[x] for x in xs if x in stoi], dtype=torch.long, device=DEVICE)
    subjects = ids(SUBJECTS)
    verbs    = ids(VERBS)
    the_id   = stoi.get("the", -1)
    objects  = ids(OBJECTS)
    comma_id = stoi.get(",", -1)
    intens   = ids(INTENS)
    pos_adj  = ids(ADJ_POS)
    neg_adj  = ids(ADJ_NEG)
    exclam   = stoi.get("!", -1)
    qmark    = stoi.get("?", -1)
    period   = stoi.get(".", -1)
    return VocabGroups(subjects, verbs, the_id, objects, comma_id, intens, pos_adj, neg_adj, exclam, qmark, period)

def apply_logit_steer(logits: torch.Tensor, control: Dict[str,float], groups: VocabGroups, state: int):
    """State-aware steering: INTENS=6, ADJ=7, PUNCT=8"""
    l = logits.clone()
    pos = control.get("pos_high", 0.0) - control.get("neg_high", 0.0)
    neg = control.get("neg_high", 0.0) - control.get("pos_high", 0.0)
    strength = max(control.get("str_high", 0.0), control.get("str_med", 0.0))
    is_question = control.get("is_question", 0.0)

    if state == 6 and groups.intens.numel() > 0 and strength > 0:
        l[groups.intens] += 1.2 * (0.5 + strength)

    if state == 7:
        if groups.pos_adj.numel() > 0 and pos > 0:
            l[groups.pos_adj] += 6.0 * pos
            l[groups.neg_adj] -= 3.0 * pos
        if groups.neg_adj.numel() > 0 and neg > 0:
            l[groups.neg_adj] += 6.0 * neg
            l[groups.pos_adj] -= 3.0 * neg

    if state == 8:
        if groups.exclam >= 0:
            l[groups.exclam] += 2.8 * max(0.0, strength) * max(0.0, pos)
        if groups.qmark >= 0:
            l[groups.qmark] += 2.8 * is_question

    return l

# -------------------------------
# 7c) Finite-State Grammar (1-clause generation)
# -------------------------------
@dataclass
class Grammar:
    start_state: int = 1
    end_state:   int = 9
# states: 1 SUBJ -> 2 VERB -> 3 'the' -> 4 OBJ -> 5 ',' -> 6 INTENS -> 7 ADJ -> 8 PUNCT -> 9 END

def allowed_ids_for_state(state: int, groups: VocabGroups) -> List[int]:
    if state == 1:   return groups.subjects.tolist()
    if state == 2:   return groups.verbs.tolist()
    if state == 3:   return [groups.the_id] if groups.the_id >= 0 else []
    if state == 4:   return groups.objects.tolist()
    if state == 5:   return [groups.comma_id] if groups.comma_id >= 0 else []
    if state == 6:   return groups.intens.tolist()
    if state == 7:   return (groups.pos_adj.tolist() + groups.neg_adj.tolist())
    if state == 8:   return [i for i in [groups.period, groups.exclam, groups.qmark] if i >= 0]
    return []

def next_state(state: int) -> int:
    return min(state + 1, 9)

# -------------------------------
# 7d) Prompt support (helpers)
# -------------------------------
PROMPT_SPLIT_RE = re.compile(r"[A-Za-z]+|[.,!?]")

def tokenize_prompt(text: str) -> List[str]:
    """Split prompt into tokens consistent with this toy vocab."""
    toks = PROMPT_SPLIT_RE.findall(text)
    out = []
    for t in toks:
        if t in SUBJECTS or t in PUNCT or t in COMMAS:
            out.append(t)                 # keep case for names; keep punctuation/comma
        else:
            out.append(t.lower())         # everything else lower-cased
    return out

def infer_state_from_prefix(prefix_toks: List[str], stoi: Dict[str,int], itos: Dict[int,str], groups: "VocabGroups") -> int:
    """Validate prefix against the FSG and return the next decoding state."""
    grammar = Grammar()
    state = grammar.start_state
    for t in prefix_toks:
        if state == grammar.end_state:
            break
        if t not in stoi:
            raise ValueError(f"Prompt token '{t}' not in vocabulary.")
        allow = set(allowed_ids_for_state(state, groups))
        tid = stoi[t]
        if tid in allow:
            state = next_state(state)
        else:
            expected = [itos[i] for i in allow]
            raise ValueError(f"Prompt token '{t}' invalid at grammar state {state}. "
                             f"Expected one of {expected}.")
    return state

# -------------------------------
# 7e) Baseline *fair* generator (grammar + last-3 repetition penalty) with prompt
# -------------------------------
_VOCAB_GROUPS: VocabGroups = None  # set in run_experiment()

@torch.no_grad()
def generate_baseline_fair(model, stoi, itos, max_len=28, temperature=0.8, top_p=0.9, top_k=20,
                           prompt: Optional[str] = None):
    model.eval()
    groups = _VOCAB_GROUPS
    grammar = Grammar()
    bos, eos, pad = stoi["<bos>"], stoi["<eos>"], stoi["<pad>"]
    ids = [bos]
    state = grammar.start_state

    # --- Prompt prefix handling ---
    if prompt:
        prefix_toks = tokenize_prompt(prompt)
        state = infer_state_from_prefix(prefix_toks, stoi, itos, groups)
        ids += [stoi[t] for t in prefix_toks]

    for _ in range(max_len - 1):
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        attn_mask = (x == pad)
        logits, _ = model(x, attn_mask=attn_mask)
        next_logits = logits[0, -1, :]

        # grammar mask
        allow = allowed_ids_for_state(state, groups)
        mask = torch.full_like(next_logits, float("-inf"))
        if allow:
            mask[torch.tensor(allow, device=next_logits.device)] = 0.0
        next_logits = next_logits + mask

        # last-3 repetition guard (stronger)
        next_logits = simple_repetition_penalty_k(next_logits, ids, k=3, penalty=2.5, itos=itos)
        nid = top_p_top_k_sample(next_logits, temperature, top_p, top_k=top_k)
        ids.append(nid)

        state = next_state(state)
        if state == grammar.end_state:
            break
    ids.append(eos)
    return detokenize(ids, itos)

# -------------------------------
# 7f) Fusion generator (with class mixture sampling) with prompt
# -------------------------------
@torch.no_grad()
def generate_fusion(model, stoi, itos, fb, max_len=28,
                    temperature=0.8, top_p=0.9, control=None,
                    prompt: Optional[str] = None):
    model.eval()
    groups = _VOCAB_GROUPS
    grammar = Grammar()
    bos, eos, pad = stoi["<bos>"], stoi["<eos>"], stoi["<pad>"]

    ids = [bos]
    state = grammar.start_state

    # Prompt prefix
    if prompt:
        prefix_toks = tokenize_prompt(prompt)
        state = infer_state_from_prefix(prefix_toks, stoi, itos, groups)
        ids += [stoi[t] for t in prefix_toks]

    ctrl = control or {}
    pos = ctrl.get("pos_high", 0.0) - ctrl.get("neg_high", 0.0)
    neg = ctrl.get("neg_high", 0.0) - ctrl.get("pos_high", 0.0)

    for _ in range(max_len - 1):
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        L = x.size(1)
        M = build_semantics_for_prefix(ids, itos, fb, max_len)
        if M.size(0) > L: M = M[:L,:]
        elif M.size(0) < L:
            M = torch.cat([M, torch.zeros(L - M.size(0), M.size(1))], dim=0)
        M = M.to(DEVICE).unsqueeze(0)

        attn_mask = (x == pad)
        logits, sem_pred, _ = model(x, M, attn_mask=attn_mask)
        next_logits = logits[0, -1, :]

        allow = allowed_ids_for_state(state, groups)

        # HARD class restriction under strong control
        if state == 7 and allow:
            if pos > 0.6:
                allow = list(set(allow).intersection(set(groups.pos_adj.tolist())))
            elif neg > 0.6:
                allow = list(set(allow).intersection(set(groups.neg_adj.tolist())))

        # HARD punctuation under strong control
        if state == 8 and allow:
            if ctrl.get("is_question", 0.0) > 0.6 and groups.qmark in allow:
                nid = groups.qmark
                ids.append(nid); state = next_state(state)
                if state == grammar.end_state: break
                continue
            if (pos > 0.6) and (ctrl.get("str_high", 0.0) > 0.6) and groups.exclam in allow:
                nid = groups.exclam
                ids.append(nid); state = next_state(state)
                if state == grammar.end_state: break
                continue

        # Class-mixture sampling at ADJ under strong control
        strong = (pos > 0.6) or (neg > 0.6)
        if state == 7 and allow and strong:
            if pos > 0.6:
                nid = sample_group_mixture(next_logits, allow, temperature=1.5, top_p=1.0, alpha_uniform=0.97)
            else:
                nid = sample_group_mixture(next_logits, allow, temperature=1.3, top_p=0.95, alpha_uniform=0.85)
            ids.append(nid)
            state = next_state(state)
            if state == grammar.end_state: break
            continue

        # Otherwise: grammar mask + steering + moderate repetition guard
        mask = torch.full_like(next_logits, float("-inf"))
        if allow:
            mask[torch.tensor(allow, device=next_logits.device)] = 0.0
        next_logits = next_logits + mask

        next_logits = apply_logit_steer(next_logits, ctrl, groups, state)
        next_logits = simple_repetition_penalty_k(next_logits, ids, k=3, penalty=1.5, itos=itos)

        nid = top_p_top_k_sample(next_logits, temperature, top_p, top_k=0)
        ids.append(nid)

        state = next_state(state)
        if state == grammar.end_state:
            break

    ids.append(eos)
    return detokenize(ids, itos)

# -------------------------------
# 8) Schedulers
# -------------------------------
def build_warmup_cosine_scheduler(optimizer, num_warmup, num_steps):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / max(1, num_warmup)
        progress = float(step - num_warmup) / max(1, num_steps - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------------------
# 9) Run experiment
# -------------------------------
def run_experiment():
    # Hyperparams
    max_len = 28
    batch_size = 64
    epochs = 6
    d_model = 128
    nhead = 4
    nlayers = 4
    dim_ff = 256
    aux_lambda = 0.5
    base_lr = 3e-4

    # OOD adjective split
    rng = random.Random(SEED)
    pos_all = ADJ_POS[:]; neg_all = ADJ_NEG[:]
    rng.shuffle(pos_all); rng.shuffle(neg_all)
    split_pos = len(pos_all)//2
    split_neg = len(neg_all)//2
    ADJ_POS_TRAIN = pos_all[:split_pos]; ADJ_POS_HOLD = pos_all[split_pos:]
    ADJ_NEG_TRAIN = neg_all[:split_neg]; ADJ_NEG_HOLD = neg_all[split_neg:]

    # Data (train uses only TRAIN subsets; val uses FULL sets)
    train_sents, val_sents = make_corpus(
        n_train=8000, n_val=1200, max_len=max_len-2,
        adj_pos_train=ADJ_POS_TRAIN, adj_neg_train=ADJ_NEG_TRAIN,
        adj_pos_val=ADJ_POS,        adj_neg_val=ADJ_NEG
    )
    stoi, itos = build_vocab(train_sents + val_sents)
    fb = build_feature_bank()
    feat_dim = len(fb.names)
    pad_id = stoi["<pad>"]

    # hold-out token ids for seen-only PPL
    holdout_ids = set([stoi[w] for w in (ADJ_POS_HOLD + ADJ_NEG_HOLD) if w in stoi])

    # Build vocab groups for losses/generation
    global _VOCAB_GROUPS
    _VOCAB_GROUPS = build_vocab_groups(stoi)

    train_ds = LMDataset(train_sents, stoi, fb, max_len=max_len)
    val_ds   = LMDataset(val_sents,   stoi, fb, max_len=max_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # -------- Baseline --------
    base = TinyTransformerLM(len(stoi), d_model, nhead, nlayers, dim_ff).to(DEVICE)
    opt_b = torch.optim.AdamW(base.parameters(), lr=base_lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(0.10 * total_steps))
    sch_b = build_warmup_cosine_scheduler(opt_b, warmup_steps, total_steps)
    global_step_b = 0

    def eval_perplexity(model, loader):
        model.eval()
        total_loss, total_tok = 0.0, 0.0
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,y,mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                attn_mask = (x==pad_id)
                logits,_ = model(x, attn_mask=attn_mask)
                loss = lm_step(logits, y, mask, label_smoothing=0.0)  # eval w/o LS
                total_loss += loss.item() * mask.sum().item()
                total_tok  += mask.sum().item()
        ppl = math.exp(total_loss / max(total_tok,1.0))
        return ppl

    def eval_perplexity_seen_only(model, loader, holdout_ids: set):
        model.eval()
        total_loss, total_tok = 0.0, 0.0
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,y,mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                attn_mask = (x==pad_id)
                logits,_ = model(x, attn_mask=attn_mask)
                vocab = logits.size(-1)
                ce = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none")
                m = mask.reshape(-1)
                y_cpu = y.reshape(-1).detach().cpu().numpy()
                keep = torch.tensor([int(int(t) not in holdout_ids) for t in y_cpu], device=ce.device, dtype=ce.dtype)
                w = m * keep
                if w.sum() > 0:
                    total_loss += (ce * w).sum().item()
                    total_tok  += w.sum().item()
        if total_tok == 0:
            return float("nan")
        return math.exp(total_loss / total_tok)

    print(f"Uniformizer ON? {UNI_ON}  (UNI_LAM={UNI_LAM}, LS_EPS={LS_EPS})")
    print("Training baseline...")
    for ep in range(1, epochs+1):
        base.train()
        for x,y,mask,M in train_loader:
            x,y,mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            attn_mask = (x==pad_id)
            logits,_ = base(x, attn_mask=attn_mask)
            # LS + optional uniformizer
            loss = lm_step(logits, y, mask, label_smoothing=LS_EPS) \
                   + adj_uniform_kl_loss(logits, y, mask, _VOCAB_GROUPS, lam=UNI_LAM)
            opt_b.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            opt_b.step(); sch_b.step()
            global_step_b += 1
        ppl = eval_perplexity(base, val_loader)
        print(f"[Baseline] epoch {ep:02d}  val PPL: {ppl:.3f}")

    base_ppl = eval_perplexity(base, val_loader)
    base_seen_ppl = eval_perplexity_seen_only(base, val_loader, holdout_ids)

    # -------- Semantic Fusion + Aux --------
    fusion = SemanticFusionLM(len(stoi), feat_dim, d_model, nhead, nlayers, dim_ff).to(DEVICE)
    opt_f = torch.optim.AdamW(fusion.parameters(), lr=base_lr, weight_decay=0.01)
    sch_f = build_warmup_cosine_scheduler(opt_f, warmup_steps, total_steps)
    global_step_f = 0

    def eval_perplexity_fusion(model, loader):
        model.eval()
        total_loss, total_tok = 0.0, 0.0
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,y,mask,M = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), M.to(DEVICE)
                attn_mask = (x==pad_id)
                logits,sem_pred,_ = model(x, M, attn_mask=attn_mask)
                loss = lm_step(logits, y, mask, label_smoothing=0.0)
                total_loss += loss.item() * mask.sum().item()
                total_tok  += mask.sum().item()
        ppl = math.exp(total_loss / max(total_tok,1.0))
        return ppl

    def eval_perplexity_fusion_seen_only(model, loader, holdout_ids: set):
        model.eval()
        total_loss, total_tok = 0.0, 0.0
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,y,mask,M = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), M.to(DEVICE)
                attn_mask = (x==pad_id)
                logits,sem_pred,_ = model(x, M, attn_mask=attn_mask)
                vocab = logits.size(-1)
                ce = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none")
                m = mask.reshape(-1)
                y_cpu = y.reshape(-1).detach().cpu().numpy()
                keep = torch.tensor([int(int(t) not in holdout_ids) for t in y_cpu], device=ce.device, dtype=ce.dtype)
                w = m * keep
                if w.sum() > 0:
                    total_loss += (ce * w).sum().item()
                    total_tok  += w.sum().item()
        if total_tok == 0:
            return float("nan")
        return math.exp(total_loss / total_tok)

    print("\nTraining semantic fusion + auxiliary...")
    for ep in range(1, epochs+1):
        fusion.train()
        for x,y,mask,M in train_loader:
            x,y,mask,M = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), M.to(DEVICE)
            attn_mask = (x==pad_id)
            logits,sem_pred,_ = fusion(x, M, attn_mask=attn_mask)
            loss_lm = lm_step(logits, y, mask, label_smoothing=LS_EPS)
            loss_aux = bce_step(sem_pred, M, mask)
            loss_uni = adj_uniform_kl_loss(logits, y, mask, _VOCAB_GROUPS, lam=UNI_LAM)
            loss = loss_lm + aux_lambda * loss_aux + loss_uni
            opt_f.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            opt_f.step(); sch_f.step()
            global_step_f += 1
        ppl = eval_perplexity_fusion(fusion, val_loader)
        print(f"[Fusion+Aux] epoch {ep:02d}  val PPL: {ppl:.3f}")

    fusion_ppl = eval_perplexity_fusion(fusion, val_loader)
    fusion_seen_ppl = eval_perplexity_fusion_seen_only(fusion, val_loader, holdout_ids)

    def sem_mse(model, loader):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,mask,M = x.to(DEVICE), mask.to(DEVICE), M.to(DEVICE)
                attn_mask = (x==pad_id)
                logits,sem_pred,_ = model(x, M, attn_mask=attn_mask)
                err = ((sem_pred - M)**2).mean(dim=-1)
                tot += (err * mask).sum().item()
                cnt += mask.sum().item()
        return tot / max(cnt,1.0)

    mse = sem_mse(fusion, val_loader)

    print("\n======== RESULTS ========")
    print(f"Baseline PPL        : {base_ppl:.3f}")
    print(f"Fusion+Aux PPL      : {fusion_ppl:.3f}")
    print(f"(Seen-only) Baseline PPL : {base_seen_ppl:.3f}")
    print(f"(Seen-only) Fusion  PPL  : {fusion_seen_ppl:.3f}")
    print(f"Semantic pred MSE ↓ : {mse:.4f}  (lower is better)")
    print("=========================")

    # Per-token CE (baseline + fusion)
    def per_token_loss_base(model, loader, focus: List[str], stoi):
        model.eval()
        tok_ids = [stoi[t] for t in focus if t in stoi]
        total = {t: [0.0, 0.0] for t in tok_ids}
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,y,mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                attn_mask = (x == stoi["<pad>"])
                logits,_ = model(x, attn_mask=attn_mask)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                ce = ce.reshape(x.size(0), x.size(1))
                for tid in tok_ids:
                    hit = (y == tid).float()
                    total[tid][0] += (ce * hit).sum().item()
                    total[tid][1] += hit.sum().item()
        return {focus[i]: (total[tok_ids[i]][0] / max(total[tok_ids[i]][1],1.0)) for i in range(len(tok_ids))}

    def per_token_loss_fusion(model, loader, focus: List[str], stoi):
        model.eval()
        tok_ids = [stoi[t] for t in focus if t in stoi]
        total = {t: [0.0, 0.0] for t in tok_ids}
        with torch.no_grad():
            for x,y,mask,M in loader:
                x,y,mask,M = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), M.to(DEVICE)
                attn_mask = (x == stoi["<pad>"])
                logits,_,_ = model(x, M, attn_mask=attn_mask)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                ce = ce.reshape(x.size(0), x.size(1))
                for tid in tok_ids:
                    hit = (y == tid).float()
                    total[tid][0] += (ce * hit).sum().item()
                    total[tid][1] += hit.sum().item()
        return {focus[i]: (total[tok_ids[i]][0] / max(total[tok_ids[i]][1],1.0)) for i in range(len(tok_ids))}

    focus = ["good","great","terrible","slightly","very","!","?",","]
    print("Baseline focus CE:", per_token_loss_base(base, val_loader, focus, stoi))
    print("Fusion   focus CE:", per_token_loss_fusion(fusion, val_loader, focus, stoi))

    # ---- Generations (1-clause for clarity) ----
    print("\n--- Baseline generations (FAIR: grammar + last-3 penalty) ---")
    for _ in range(3):
        print("•", generate_baseline_fair(base, stoi, itos, max_len=max_len, temperature=0.7, top_p=0.9, top_k=20))

    print("\n--- Fusion (neutral) generations ---")
    for _ in range(3):
        print("•", generate_fusion(fusion, stoi, itos, fb, max_len=max_len, temperature=0.7, top_p=0.9, control=None))

    print("\n--- Fusion (controlled: positive & strong) ---")
    pos_strong = {"pos_high": 0.95, "str_high": 0.9}
    for _ in range(3):
        print("•", generate_fusion(fusion, stoi, itos, fb, max_len=max_len,
                                   temperature=0.7, top_p=0.9, control=pos_strong))

    print("\n--- Fusion (controlled: negative & question) ---")
    neg_question = {"neg_high": 0.95, "is_question": 1.0, "str_med": 0.6}
    for _ in range(3):
        print("•", generate_fusion(fusion, stoi, itos, fb, max_len=max_len,
                                   temperature=0.8, top_p=0.9, control=neg_question))

    # ---- Prompted ----
    print("\n--- Prompted ---")
    prompt = "Carol starts the model,"  # must follow the grammar: SUBJ VERB 'the' OBJ ','
    print("Prompt:", prompt)
    print("Baseline (fair):", generate_baseline_fair(
        base, stoi, itos, max_len=max_len, temperature=0.7, top_p=0.9, top_k=20,
        prompt=prompt))
    print("Fusion (no control)        :", generate_fusion(
        fusion, stoi, itos, fb, max_len=max_len, temperature=0.7, top_p=0.9,
        control=None, prompt=prompt))
    print("Fusion (controlled: positive & strong)        :", generate_fusion(
        fusion, stoi, itos, fb, max_len=max_len, temperature=0.7, top_p=0.9,
        control=pos_strong, prompt=prompt))
    print("Fusion (controlled: negative & question)        :", generate_fusion(
        fusion, stoi, itos, fb, max_len=max_len, temperature=0.7, top_p=0.9,
        control=neg_question, prompt=prompt))

    # ---- Control success sanity check ----
    def last_punct(s: str):
        return s.strip()[-1] if s.strip() and s.strip()[-1] in ".!?" else ""
    def last_adj(s: str):
        m = re.search(r",\s+(?:[a-z]+\s+)?([a-z]+)[.!?]$", s.strip())
        return m.group(1) if m else ""

    def control_success(fusion, stoi, itos, fb, N=200):
        pos_ctrl = {"pos_high":0.95, "str_high":0.9}
        neg_ctrl = {"neg_high":0.95, "is_question":1.0, "str_med":0.6}
        counts = {"pos_adj":0, "pos_exc":0, "neg_adj":0, "neg_q":0}
        pos_set = {w.lower() for w in ADJ_POS}
        neg_set = {w.lower() for w in ADJ_NEG}
        for _ in range(N):
            s1 = generate_fusion(fusion, stoi, itos, fb, control=pos_ctrl, temperature=0.7, top_p=0.9)
            a1, p1 = last_adj(s1), last_punct(s1)
            counts["pos_adj"] += int(a1 in pos_set)
            counts["pos_exc"] += int(p1 == "!")
            s2 = generate_fusion(fusion, stoi, itos, fb, control=neg_ctrl, temperature=0.8, top_p=0.9)
            a2, p2 = last_adj(s2), last_punct(s2)
            counts["neg_adj"] += int(a2 in neg_set)
            counts["neg_q"]   += int(p2 == "?")
        for k in counts: counts[k] /= N
        print("\nControl success rates over", N, "samples:")
        print("  positive: adj=", counts["pos_adj"], "  !=", counts["pos_exc"])
        print("  negative: adj=", counts["neg_adj"], "  ?=", counts["neg_q"])

    control_success(fusion, stoi, itos, fb, N=200)

    # ---- Confusion table ----
    def control_confusion(fusion, stoi, itos, fb, N=200):
        pos_ctrl = {"pos_high":0.95, "str_high":0.9}
        neg_ctrl = {"neg_high":0.95, "str_med":0.6}
        pos_set = {w.lower() for w in ADJ_POS}
        neg_set = {w.lower() for w in ADJ_NEG}
        mat = np.zeros((2,3), dtype=int)
        def cls(word):
            if word in pos_set: return 0
            if word in neg_set: return 1
            return 2
        for _ in range(N):
            s1 = generate_fusion(fusion, stoi, itos, fb, control=pos_ctrl, temperature=0.7, top_p=0.9)
            mat[0, cls(last_adj(s1))] += 1
            s2 = generate_fusion(fusion, stoi, itos, fb, control=neg_ctrl, temperature=0.8, top_p=0.9)
            mat[1, cls(last_adj(s2))] += 1
        def rowfmt(r):
            tot = mat[r].sum()
            return f"{mat[r,0]:4d} ({mat[r,0]/max(tot,1):.2f}) | {mat[r,1]:4d} ({mat[r,1]/max(tot,1):.2f}) | {mat[r,2]:4d} ({mat[r,2]/max(tot,1):.2f})"
        print("\nConfusion table (rows=intended, cols=realized):")
        print("               POS         NEG        OTHER")
        print(f"Intended POS : {rowfmt(0)}")
        print(f"Intended NEG : {rowfmt(1)}")

    control_confusion(fusion, stoi, itos, fb, N=200)

    # ---- OOD test (held-out adjectives) ----
    def ood_test(fusion, stoi, itos, fb, adj_pos_hold, adj_neg_hold, N=200):
        pos_ctrl = {"pos_high":0.95, "str_high":0.9}
        neg_ctrl = {"neg_high":0.95, "str_med":0.6}
        pos_hold = {w.lower() for w in adj_pos_hold}
        neg_hold = {w.lower() for w in adj_neg_hold}
        pos_hit = 0; pos_tot = 0
        neg_hit = 0; neg_tot = 0
        for _ in range(N):
            s1 = generate_fusion(fusion, stoi, itos, fb, control=pos_ctrl, temperature=0.7, top_p=0.9)
            pos_hit += int(last_adj(s1) in pos_hold); pos_tot += 1
            s2 = generate_fusion(fusion, stoi, itos, fb, control=neg_ctrl, temperature=0.8, top_p=0.9)
            neg_hit += int(last_adj(s2) in neg_hold); neg_tot += 1
        print("\nOOD control test (held-out adjectives):")
        print(f"  POS control -> holdout adj rate: {pos_hit}/{pos_tot} = {pos_hit/max(pos_tot,1):.2f}")
        print(f"  NEG control -> holdout adj rate: {neg_hit}/{neg_tot} = {neg_hit/max(neg_tot,1):.2f}")
        print("  (Higher is better; shows the controller can target words unseen in training.)")

    print("\nHeld-out adjectives (POS):", ADJ_POS_HOLD)
    print("Held-out adjectives (NEG):", ADJ_NEG_HOLD)
    ood_test(fusion, stoi, itos, fb, ADJ_POS_HOLD, ADJ_NEG_HOLD, N=200)

if __name__ == "__main__":
    run_experiment()

"""# end."""
