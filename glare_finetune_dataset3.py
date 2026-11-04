# %% [markdown]
# GLARE-faithful fine-tuning (NTv3) para o Dataset3 (Train/ e Test/ com um .fa por classe)
# Reimplementa: model.py (backbone + pooling + GLAREHead dual-head) e train.py
# (loss ponderada SF+Order, class weights sqrt_inv, AdamW 2 grupos, cosine+warmup, bf16,
#  grad-accum, clip 1.0, janelas deslizantes na validacao/teste, early stopping por SF-F1 ponderado).
# Fora do escopo: DDP, TemperatureScaler, MahalanobisOOD (modulo ood.py nao fornecido).

# %%
import os, re, math, csv, time, json, random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig, get_cosine_schedule_with_warmup

# ----------------------------- CONFIG (valores do default.yaml / train.py) -----------------------------
CFG = dict(
    dataset="Dataset3",
    data_root="./Data",
    base_model="InstaDeepAI/NTv3_8M_pre",   # 650M não cabe em 8 GB com ajuste completo
    max_length=512,                # paper: 512 na comparação de backbones; 4096 no modelo final
    use_lora=False,                # paper: backbone totalmente ajustado
    lora_r=16, lora_alpha=32, lora_dropout=0.1, lora_target_modules=["linear", "mha_output"],  # ignorados se use_lora=False
    pooling="mean",                # paper: mean pooling mascarado
    hidden_dim=512, head_dropout=0.1,
    batch_size=8, grad_accum=8,    # batch efetivo 64 (paper: 16 x 4); se faltar memória: 4 x 16
    max_epochs=10, patience=8,     # paper: até 10 épocas, paciência 8
    lambda_sf=1.0, lambda_order=0.0,   # paper: só loss de superfamília
    label_smoothing=0.05, class_weight_scheme="sqrt_inv",
    backbone_lr=5e-5, head_lr=3e-4, weight_decay=0.01, warmup_fraction=0.05,
    val_fraction=0.1, seed=42,
    sliding_overlap=0.5, rc_augment=True,
    num_workers=4, out_dir="runs/glare_dataset3",
)

# Superfamilia (nome do arquivo normalizado) -> ordem. Ajuste se seus nomes de arquivo forem outros.
SF_TO_ORDER = {
    "BELPAO": "LTR", "PAO": "LTR", "COPIA": "LTR", "ERV": "LTR", "GYPSY": "LTR",
    "L1": "LINE", "LINE1": "LINE", "SINE": "SINE",
    "MARINER": "TIR", "TC1MARINER": "TIR", "HAT": "TIR", "MUTATOR": "TIR", "MULE": "TIR",
    "PIF": "TIR", "PIFHARBINGER": "TIR", "CACTA": "TIR",
    "NONTE": "NONTE", "RANDOM": "NONTE",
}
RC_MAP = str.maketrans("ACGTacgt", "TGCAtgca")
EXTS = (".fa", ".fasta", ".fna")

def norm_label(name):
    return re.sub(r"[^A-Z0-9]", "", name.upper())

# ----------------------------- DADOS -----------------------------
def read_fasta(path):
    seqs, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seqs.append("".join(cur))
                cur = []
            elif cur is not None:
                cur.append(line)
    if cur is not None:
        seqs.append("".join(cur))
    return seqs

def clean(seq):
    return re.sub(r"[^ACGTN]", "N", seq.upper().replace("U", "T"))

def load_split(folder):
    recs = []
    for f in sorted(x for x in os.listdir(folder) if x.lower().endswith(EXTS)):
        sf = os.path.splitext(f)[0]
        key = norm_label(sf)
        if key not in SF_TO_ORDER:
            raise KeyError(f"Classe '{sf}' ({key}) sem ordem em SF_TO_ORDER")
        for s in read_fasta(os.path.join(folder, f)):
            recs.append({"sequence": clean(s), "superfamily": sf, "order": SF_TO_ORDER[key]})
    return recs

def sequence_windows(sequence, max_length, overlap=0.5):
    """Identico a data.sequence_windows: stride=max(1,int(L*(1-overlap))), mantem janela final parcial."""
    if len(sequence) <= max_length:
        return [sequence]
    stride = max(1, int(max_length * (1.0 - overlap)))
    windows, start = [], 0
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        windows.append(sequence[start:end])
        if end >= len(sequence):
            break
        start += stride
    return windows

def encode(tokenizer, seq, max_length, pad_id):
    enc = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True,
                    max_length=max_length, pad_to_multiple_of=128)
    ids = enc["input_ids"].squeeze(0)
    mask = enc["attention_mask"].squeeze(0) if "attention_mask" in enc else (ids != pad_id).long()
    return ids, mask

class TEDataset(Dataset):
    """Treino: recorte aleatorio se L>max_length + reverse-complement (p=0.5) opcional."""
    def __init__(self, records, tokenizer, max_length, o2i, s2i, pad_id, augment=False, rc_augment=False):
        self.r, self.tok, self.L = records, tokenizer, max_length
        self.o2i, self.s2i, self.pad, self.aug, self.rc = o2i, s2i, pad_id, augment, rc_augment
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        rec = self.r[i]; seq = rec["sequence"]
        if self.aug and len(seq) > self.L:
            st = random.randint(0, len(seq) - self.L); seq = seq[st:st + self.L]
        if self.rc and random.random() < 0.5:
            seq = seq.translate(RC_MAP)[::-1]
        ids, mask = encode(self.tok, seq, self.L, self.pad)
        return {"input_ids": ids, "attention_mask": mask,
                "order_label": torch.tensor(self.o2i[rec["order"]]),
                "superfamily_label": torch.tensor(self.s2i[rec["superfamily"]])}

class SlidingWindowDataset(Dataset):
    """Validacao/teste: todas as janelas de cada sequencia, com seq_idx para agregar logits."""
    def __init__(self, records, tokenizer, max_length, o2i, s2i, pad_id, overlap=0.5):
        self.tok, self.L, self.pad = tokenizer, max_length, pad_id
        self.w = []
        for k, rec in enumerate(records):
            for win in sequence_windows(rec["sequence"], max_length, overlap):
                self.w.append((k, win, o2i[rec["order"]], s2i[rec["superfamily"]]))
        self.n_sequences = len(records)
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        k, win, o, s = self.w[i]
        ids, mask = encode(self.tok, win, self.L, self.pad)
        return {"input_ids": ids, "attention_mask": mask, "order_label": torch.tensor(o),
                "superfamily_label": torch.tensor(s), "seq_idx": torch.tensor(k)}

def compute_class_weights(labels, num_classes, scheme="sqrt_inv"):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    present = counts > 0
    w = np.zeros(num_classes)
    if scheme == "none":
        w[present] = 1.0
    else:
        w[present] = (1.0 / counts[present]) if scheme == "inv" else (1.0 / np.sqrt(counts[present]))
        w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)

# ----------------------------- MODELO (model.py) -----------------------------
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__(); self.attn = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, h, mask):
        s = self.attn(h).squeeze(-1).masked_fill(mask == 0, -1e9)
        return (h * torch.softmax(s, dim=-1).unsqueeze(-1)).sum(dim=1)

class GLAREHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_orders, num_superfamilies, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout))
        self.order_head = nn.Linear(hidden_dim // 2, num_orders)
        self.superfamily_head = nn.Linear(hidden_dim // 2, num_superfamilies)
    def forward(self, x):
        h = self.trunk(x); return self.order_head(h), self.superfamily_head(h)

def load_backbone(name):
    try:
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        return AutoModel.from_pretrained(name, config=cfg, trust_remote_code=True)
    except Exception as e:
        print(f"AutoModel falhou ({e!r}); usando AutoModelForMaskedLM")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        return AutoModelForMaskedLM.from_pretrained(name, config=cfg, trust_remote_code=True)

class GLAREModel(nn.Module):
    def __init__(self, cfg, num_orders, num_superfamilies, tokenizer):
        super().__init__()
        backbone = load_backbone(cfg["base_model"])
        self.use_lora = cfg["use_lora"]
        if self.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            backbone = get_peft_model(backbone, LoraConfig(
                r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["lora_target_modules"], bias="none", task_type=TaskType.FEATURE_EXTRACTION))
        else:
            for p in backbone.parameters(): p.requires_grad = True
        self.backbone = backbone
        self.pooling = cfg["pooling"]
        hs = self._probe_hidden(tokenizer)
        if self.pooling == "attention": self.pool = AttentionPool(hs)
        self.head = GLAREHead(hs, cfg["hidden_dim"], num_orders, num_superfamilies, cfg["head_dropout"])
    def _hidden(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        if getattr(out, "hidden_states", None) is not None: return out.hidden_states[-1]
        if getattr(out, "last_hidden_state", None) is not None: return out.last_hidden_state
        return out[0]
    @torch.no_grad()
    def _probe_hidden(self, tok):
        ids, mask = encode(tok, "ACGT" * 32, 128, tok.pad_token_id or 0)
        was = self.backbone.training; self.backbone.eval()
        h = self._hidden(ids.unsqueeze(0), mask.unsqueeze(0)); self.backbone.train(was)
        return h.shape[-1]
    def forward(self, ids, mask=None):
        if mask is None: mask = torch.ones_like(ids)
        h = self._hidden(ids, mask)
        if self.pooling == "attention": pooled = self.pool(h, mask)
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float())

# ----------------------------- TREINO (train.py) -----------------------------
def train_epoch(model, loader, opt, sched, o_crit, s_crit, device, lam_o, lam_s, accum):
    model.train(); tot = to = ts = 0.0; nb = 0; n = len(loader)
    opt.zero_grad(set_to_none=True)
    for i, b in enumerate(loader):
        ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
        ol, sl = b["order_label"].to(device), b["superfamily_label"].to(device)
        last = ((i + 1) % accum == 0) or (i + 1 == n)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o_log, s_log = model(ids, mask)
            oloss, sloss = o_crit(o_log.float(), ol), s_crit(s_log.float(), sl)
            loss = lam_s * sloss + lam_o * oloss
        (loss / accum).backward()
        tot += loss.item(); to += oloss.item(); ts += sloss.item(); nb += 1
        if last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    d = max(nb, 1); return {"train_loss": tot / d, "order_loss": to / d, "sf_loss": ts / d}

@torch.no_grad()
def evaluate(model, loader, n_seq, o_crit, s_crit, device, lam_o, lam_s, sf_names, order_names, sf_to_order):
    model.eval(); tot = 0.0; nb = 0
    S = torch.zeros(n_seq, len(sf_names), device=device); C = torch.zeros(n_seq, device=device)
    Y = torch.full((n_seq,), -1, dtype=torch.long, device=device)
    for b in loader:
        ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
        ol, sl, k = b["order_label"].to(device), b["superfamily_label"].to(device), b["seq_idx"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o_log, s_log = model(ids, mask)
        tot += (lam_s * s_crit(s_log.float(), sl) + lam_o * o_crit(o_log.float(), ol)).item(); nb += 1
        S.index_add_(0, k, s_log.float()); C.index_add_(0, k, torch.ones(k.size(0), device=device)); Y[k] = sl
    pred = (S / C.clamp(min=1).unsqueeze(1)).argmax(-1).cpu().numpy(); true = Y.cpu().numpy()
    sf_f1 = f1_score(true, pred, average="weighted", zero_division=0)
    po = [sf_to_order[sf_names[i]] for i in pred]; to_ = [sf_to_order[sf_names[i]] for i in true]
    return {"val_loss": tot / max(nb, 1), "sf_f1": float(sf_f1),
            "order_f1": float(f1_score(to_, po, average="weighted", zero_division=0))}, true, pred

def make_loader(ds, cfg, shuffle=False, drop_last=False):
    nw = cfg["num_workers"]
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=nw, pin_memory=True,
                      drop_last=drop_last, persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)

def main(cfg=CFG):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(cfg["out_dir"], exist_ok=True)
    tr_dir = os.path.join(cfg["data_root"], cfg["dataset"], "Train"); te_dir = os.path.join(cfg["data_root"], cfg["dataset"], "Test")
    trainval, test = load_split(tr_dir), load_split(te_dir)
    sf_names = sorted({r["superfamily"] for r in trainval}); order_names = sorted({SF_TO_ORDER[norm_label(s)] for s in sf_names})
    s2i = {s: i for i, s in enumerate(sf_names)}; o2i = {o: i for i, o in enumerate(order_names)}
    sf_to_order = {s: SF_TO_ORDER[norm_label(s)] for s in sf_names}
    miss = {r["superfamily"] for r in test} - set(sf_names)
    assert not miss, f"Classes no Test ausentes no Train: {miss}"
    y = [s2i[r["superfamily"]] for r in trainval]
    tr_idx, va_idx = train_test_split(np.arange(len(trainval)), test_size=cfg["val_fraction"], stratify=y, random_state=cfg["seed"])
    train_recs = [trainval[i] for i in tr_idx]; val_recs = [trainval[i] for i in va_idx]
    print(f"train={len(train_recs)} val={len(val_recs)} test={len(test)} | SF={len(sf_names)} {sf_names} | orders={order_names}")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    L = cfg["max_length"]
    train_ds = TEDataset(train_recs, tok, L, o2i, s2i, pad, augment=True, rc_augment=cfg["rc_augment"])
    val_ds = SlidingWindowDataset(val_recs, tok, L, o2i, s2i, pad, cfg["sliding_overlap"])
    test_ds = SlidingWindowDataset(test, tok, L, o2i, s2i, pad, cfg["sliding_overlap"])
    train_loader = make_loader(train_ds, cfg, shuffle=True, drop_last=True)
    val_loader, test_loader = make_loader(val_ds, cfg), make_loader(test_ds, cfg)

    model = GLAREModel(cfg, len(order_names), len(sf_names), tok).to(device)
    ow = compute_class_weights([o2i[r["order"]] for r in train_recs], len(order_names), cfg["class_weight_scheme"]).to(device)
    sw = compute_class_weights([s2i[r["superfamily"]] for r in train_recs], len(sf_names), cfg["class_weight_scheme"]).to(device)
    o_crit = nn.CrossEntropyLoss(weight=ow, label_smoothing=cfg["label_smoothing"])
    s_crit = nn.CrossEntropyLoss(weight=sw, label_smoothing=cfg["label_smoothing"])

    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    hd = list(model.head.parameters()) + (list(model.pool.parameters()) if cfg["pooling"] == "attention" else [])
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["backbone_lr"]}, {"params": hd, "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    total_steps = math.ceil(len(train_loader) / cfg["grad_accum"]) * cfg["max_epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * cfg["warmup_fraction"]), total_steps)

    best, bad, rows, ckpt = -1.0, 0, [], os.path.join(cfg["out_dir"], "best_model.pt")
    for ep in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        tm = train_epoch(model, train_loader, opt, sched, o_crit, s_crit, device, cfg["lambda_order"], cfg["lambda_sf"], cfg["grad_accum"])
        vm, _, _ = evaluate(model, val_loader, val_ds.n_sequences, o_crit, s_crit, device, cfg["lambda_order"], cfg["lambda_sf"], sf_names, order_names, sf_to_order)
        print(f"Epoch {ep:02d}/{cfg['max_epochs']} train_loss={tm['train_loss']:.4f} val_loss={vm['val_loss']:.4f} "
              f"order_f1={vm['order_f1']:.4f} sf_f1={vm['sf_f1']:.4f} {time.time()-t0:.0f}s")
        rows.append({"epoch": ep, **tm, **vm})
        if vm["sf_f1"] > best:
            best, bad = vm["sf_f1"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_sf_f1": best, "config": cfg,
                        "sf_names": sf_names, "order_names": order_names}, ckpt)
        else:
            bad += 1
            if bad >= cfg["patience"]: print("Early stopping."); break
    with open(os.path.join(cfg["out_dir"], "train_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"]); model.to(device)
    tm_, true, pred = evaluate(model, test_loader, test_ds.n_sequences, o_crit, s_crit, device, cfg["lambda_order"], cfg["lambda_sf"], sf_names, order_names, sf_to_order)
    print(f"TEST sf_f1={tm_['sf_f1']:.4f} order_f1={tm_['order_f1']:.4f}")
    print(classification_report(true, pred, labels=list(range(len(sf_names))), target_names=sf_names, zero_division=0))
    cm = confusion_matrix(true, pred, labels=list(range(len(sf_names))))
    np.save(os.path.join(cfg["out_dir"], "test_confusion_matrix.npy"), cm)
    json.dump({"sf_names": sf_names, "order_names": order_names, "test": tm_}, open(os.path.join(cfg["out_dir"], "summary.json"), "w"), indent=2)
    return model, true, pred, sf_names

# %%
if __name__ == "__main__":
    main()
