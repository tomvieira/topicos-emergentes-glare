# %%
# Experimento: NTv3 (modelo base, congelado) + classificador leve
# Entrada: arquivos .fa por classe em  <DATA_ROOT>/<dataset>/Train/  e  .../Test/
#          (um arquivo por classe, ex.: TIR.fa, LTR.fa, Non-TE.fa)
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

# %%
# =============================================
# CONFIGURAÇÃO
# =============================================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

dataset = 'Dataset5'
DATA_ROOT = '../Data/fasta'                      # ajuste para sua estrutura
TRAIN_DIR = os.path.join(DATA_ROOT, dataset, 'Train')
TEST_DIR = os.path.join(DATA_ROOT, dataset, 'Test')

MODEL_ID = "InstaDeepAI/NTv3_8M_pre"             # versões: 8M, 100M, 650M (sufixo _pre)
PESOS_ALEATORIOS = False                         # True = controle sem pré-treino
CAMADA = -1                                      # índice do hidden_state a usar

WIN = 4096                                       # tamanho da janela (nt)
STRIDE = 2048                                    # passo entre janelas (50% de sobreposição)
MAX_WINDOWS = 8                                  # máx. de janelas por sequência (controla custo)
BATCH_WIN = 16                                   # janelas por lote na extração

USAR_PESO_DE_CLASSE = False                      # True se o seu dataset for desbalanceado
ORDEM_EXIBICAO = None                            # ex.: ["LTR", "LINE", "CLASS II", "Non-TE"] ou None

CACHE_DIR = '../Data/embeddings_ntv3'
os.makedirs(CACHE_DIR, exist_ok=True)
TAG = MODEL_ID.split('/')[-1] + ('_rand' if PESOS_ALEATORIOS else '') + f'_L{CAMADA}'

# %%
# =============================================
# 1. LEITURA DOS FASTA (um arquivo por classe)
# =============================================
EXTS = ('.fa', '.fasta', '.fna')


def ler_fasta(path):
    ids, seqs = [], []
    cur_id, cur = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append(''.join(cur))
                cur_id, cur = line[1:].split()[0], []
            else:
                cur.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append(''.join(cur))
    return ids, seqs


def limpar(seq):
    seq = seq.upper().replace('U', 'T')
    return re.sub(r'[^ACGTN]', 'N', seq)


def carregar_split(pasta, classes=None):
    arquivos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(EXTS))
    nomes = [os.path.splitext(f)[0] for f in arquivos]
    if classes is None:
        classes = nomes
    else:
        extras = set(nomes) - set(classes)
        if extras:
            raise ValueError(f"Classes no {pasta} ausentes no treino: {extras}")
    ids, seqs, labels = [], [], []
    for f, nome in zip(arquivos, nomes):
        i, s = ler_fasta(os.path.join(pasta, f))
        ids += i
        seqs += [limpar(x) for x in s]
        labels += [classes.index(nome)] * len(s)
    return ids, seqs, np.array(labels, dtype=np.int64), classes


ids_tr, seqs_tr, y_tr_all, classes = carregar_split(TRAIN_DIR)
ids_te, seqs_te, y_te, _ = carregar_split(TEST_DIR, classes)
num_classes = len(classes)

print("Classes:", classes)
print("Treino:", {c: int((y_tr_all == i).sum()) for i, c in enumerate(classes)})
print("Teste :", {c: int((y_te == i).sum()) for i, c in enumerate(classes)})

# %%
# =============================================
# 2. NTv3: janelas + embeddings (modelo congelado)
# =============================================
def carregar_modelo():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if PESOS_ALEATORIOS:
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    return tok, model.to(device).eval()


def janelas(seq):
    if len(seq) <= WIN:
        return [seq]
    starts = list(range(0, len(seq) - WIN + 1, STRIDE))
    if starts[-1] + WIN < len(seq):
        starts.append(len(seq) - WIN)
    if len(starts) > MAX_WINDOWS:
        idx = np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in idx]
    return [seq[s:s + WIN] for s in starts]


@torch.no_grad()
def extrair_embeddings(seqs, tok, model):
    itens = [(i, w) for i, s in enumerate(seqs) for w in janelas(s)]
    ordem = sorted(range(len(itens)), key=lambda k: len(itens[k][1]))  # reduz padding
    soma, cont = None, np.zeros(len(seqs))
    t0 = time.time()
    for b in range(0, len(ordem), BATCH_WIN):
        idx = ordem[b:b + BATCH_WIN]
        enc = tok([itens[k][1] for k in idx], add_special_tokens=False, padding=True,
                  pad_to_multiple_of=128, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items() if k in ('input_ids', 'attention_mask')}
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[CAMADA]
        mask = enc['attention_mask'].unsqueeze(-1).float()
        if mask.shape[1] != h.shape[1]:   # camadas intermediárias podem ter resolução menor
            mask = nn.functional.adaptive_max_pool1d(mask.transpose(1, 2), h.shape[1]).transpose(1, 2)
        vec = ((h.float() * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        if soma is None:
            soma = np.zeros((len(seqs), vec.shape[1]), dtype=np.float32)
        for j, k in enumerate(idx):
            soma[itens[k][0]] += vec[j]
            cont[itens[k][0]] += 1
        if (b // BATCH_WIN) % 50 == 0:
            print(f"  {b}/{len(ordem)} janelas ({time.time() - t0:.0f}s)")
    return soma / cont[:, None]


def obter_embeddings(nome_split, seqs):
    path = os.path.join(CACHE_DIR, f"{TAG}_{dataset}_{nome_split}.npy")
    if os.path.exists(path):
        print(f"Carregando cache: {path}")
        return np.load(path)
    tok, model = carregar_modelo()
    emb = extrair_embeddings(seqs, tok, model)
    np.save(path, emb)
    del model
    torch.cuda.empty_cache()
    return emb


# %%
# Teste rápido (rode antes do passo completo): confere formato e tempo
tok_t, model_t = carregar_modelo()
t0 = time.time()
e = extrair_embeddings(seqs_tr[:4], tok_t, model_t)
print("Formato dos embeddings:", e.shape, f"| {time.time() - t0:.1f}s para 4 sequências")
del model_t
torch.cuda.empty_cache()

# %%
X_tr_all = obter_embeddings('train', seqs_tr)
X_te = obter_embeddings('test', seqs_te)
print("Treino:", X_tr_all.shape, "| Teste:", X_te.shape)

# %%
# =============================================
# 3. Treino / validação (80/20 estratificado, como no seu script)
# =============================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr_all, y_tr_all, test_size=0.2, stratify=y_tr_all, random_state=42)

scaler = StandardScaler().fit(X_tr)
X_tr, X_val, X_te_s = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te)


class EmbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_loader = DataLoader(EmbDataset(X_tr, y_tr), batch_size=32, shuffle=True)
val_loader = DataLoader(EmbDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(EmbDataset(X_te_s, y_te), batch_size=32)

# %%
# =============================================
# 4. Classificador leve sobre os embeddings
# =============================================
class MLPHead(nn.Module):
    def __init__(self, dim, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        return self.net(x)


model = MLPHead(X_tr.shape[1], 256, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

if USAR_PESO_DE_CLASSE:
    cont = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    pesos = torch.tensor(cont.sum() / (num_classes * np.maximum(cont, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=pesos)
else:
    criterion = nn.CrossEntropyLoss()


def acc_batch(out, y):
    return (out.argmax(1) == y).float().mean().item()


num_epochs, patience = 100, 20
best_val_loss, sem_melhora = float('inf'), 0
best_model_path = f"ntv3_head_{TAG}_{dataset}.pt"
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    tl = ta = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tl += loss.item()
        ta += acc_batch(out, y)
    tl /= len(train_loader)
    ta /= len(train_loader)

    model.eval()
    vl = va = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            vl += criterion(out, y).item()
            va += acc_batch(out, y)
    vl /= len(val_loader)
    va /= len(val_loader)

    train_losses.append(tl); val_losses.append(vl)
    train_accs.append(ta); val_accs.append(va)
    print(f"Epoch {epoch+1}: Train Loss = {tl:.4f}, Train Acc = {ta:.4f}, "
          f"Val Loss = {vl:.4f}, Val Acc = {va:.4f}")

    if vl < best_val_loss:
        best_val_loss, sem_melhora = vl, 0
        torch.save(model.state_dict(), best_model_path)
    else:
        sem_melhora += 1
        if sem_melhora >= patience:
            print(f"Early stopping após {epoch+1} épocas")
            break

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc'); plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curves')
plt.tight_layout()
plt.show()

# %%
# =============================================
# 5. Avaliação no teste
# =============================================
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        all_preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())
all_preds, all_labels = np.array(all_preds), np.array(all_labels)

# Ordem de exibição (mesma para matriz e métricas)
if ORDEM_EXIBICAO is not None:
    ordem_idx = [classes.index(c) for c in ORDEM_EXIBICAO]
else:
    ordem_idx = list(range(num_classes))
nomes_exib = [classes[i] for i in ordem_idx]

cm = confusion_matrix(all_labels, all_preds, labels=ordem_idx)
print(cm)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_xlabel('Predicted label', fontsize=18)
ax.set_ylabel('True label', fontsize=18)
ax.set_xticks(np.arange(len(nomes_exib)))
ax.set_yticks(np.arange(len(nomes_exib)))
ax.set_xticklabels(nomes_exib, fontsize=14)
ax.set_yticklabels(nomes_exib, fontsize=14)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for t in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, t, f"{int(cm[t, j])}", ha="center", va="center",
                color="white" if t == j else "black", fontsize=18)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.show()

# %%
metrics = classification_report(all_labels, all_preds, labels=ordem_idx,
                                target_names=nomes_exib, output_dict=True, zero_division=0)
metrics_df = pd.DataFrame(metrics).transpose()


def accuracy_by_class(cm):
    accs = []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        total = TP + TN + FP + FN
        accs.append((TP + TN) / total if total > 0 else 0.0)
    return accs


def specificity(cm):
    out = []
    for i in range(cm.shape[0]):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        out.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return out


def format_by_class(vals, names):
    return [f'{names[i]}: {v:.3f}' for i, v in enumerate(vals)]


acc_by_class = accuracy_by_class(cm)
spec_by_class = specificity(cm)
print("Acurácia por classe:", format_by_class(acc_by_class, nomes_exib))
print(f"Acurácia Macro: {np.mean(acc_by_class):.3f}")
print("Especificidade por classe:", format_by_class(spec_by_class, nomes_exib))
print(f"Especificidade Macro: {np.mean(spec_by_class):.3f}")

df_new = metrics_df.loc[nomes_exib].copy()
df_new['specificity'] = spec_by_class
df_new['acc'] = acc_by_class
df_new = df_new.reindex(columns=['acc', 'precision', 'recall', 'f1-score', 'specificity', 'support'])
df_new = pd.concat([df_new, pd.DataFrame([df_new.mean(numeric_only=True)], index=['Mean'])])
print(df_new)

print(f"\nF1 macro   : {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"F1 ponderado: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
df_new.to_csv(f"resultados_{TAG}_{dataset}.csv")

# %%
# =============================================
# 6. Controle: regressão logística nos mesmos embeddings
# =============================================
lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
pred_lr = lr.predict(X_te_s)
print(f"[Reg. logística] F1 macro: {f1_score(y_te, pred_lr, average='macro'):.4f} | "
      f"F1 ponderado: {f1_score(y_te, pred_lr, average='weighted'):.4f}")