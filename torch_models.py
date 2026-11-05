"""PyTorch implementations of all spatio-temporal models (same designs as models.py).

Tensor layout: (batch, time, node, channel). Every model maps a batch dict to
{"pred": (B, H, N) normalised forecasts, optional "logits": (B, H, N, 3), optional gates}.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def T(x):
    return torch.as_tensor(x, dtype=torch.float32)


class Base(nn.Module):
    def __init__(self, cfg, G, N, Tin, H, Fin, n_slots):
        super().__init__()
        self.cfg, self.N, self.Tin, self.H, self.Fin, self.n_slots = cfg, N, Tin, H, Fin, n_slots
        for k in ["adj_sym", "geo_rw", "geo_rwT", "corr_rw", "prof_rw", "raw_rw"]:
            if k in G:
                self.register_buffer(k, T(G[k]), persistent=False)
        self.register_buffer("cheb", torch.stack([T(c) for c in G["cheb"]]), persistent=False)


# ============================================================================ LSTM
class LSTM(Base):
    def __init__(self, *a):
        super().__init__(*a)
        dh = self.cfg.get("hidden", 64)
        self.rnn = nn.LSTM(self.Fin, dh, num_layers=self.cfg.get("layers", 1), batch_first=True)
        self.out = nn.Linear(dh, self.H)

    def forward(self, b):
        x = b["x"]
        B, Tn, N, Fd = x.shape
        h, _ = self.rnn(x.permute(0, 2, 1, 3).reshape(B * N, Tn, Fd))
        return {"pred": self.out(h[:, -1]).reshape(B, N, self.H).transpose(1, 2)}


# ============================================================================ T-GCN
class TGCN(Base):
    def __init__(self, *a):
        super().__init__(*a)
        dh = self.cfg.get("hidden", 64)
        self.g = nn.Linear(self.Fin + dh, 2 * dh)
        self.c = nn.Linear(self.Fin + dh, dh)
        self.out = nn.Linear(dh, self.H)
        self.dh = dh

    def forward(self, b):
        x = b["x"]
        B, Tn, N, _ = x.shape
        A = self.adj_sym
        h = x.new_zeros(B, N, self.dh)
        for t in range(Tn):
            g = torch.sigmoid(self.g(A @ torch.cat([x[:, t], h], -1)))
            r, u = g.chunk(2, -1)
            c = torch.tanh(self.c(A @ torch.cat([x[:, t], r * h], -1)))
            h = u * h + (1 - u) * c
        return {"pred": self.out(h).transpose(1, 2)}


# ============================================================================ STGCN
class TConvGLU(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv2d(cin, 2 * cout, (k, 1))

    def forward(self, x):                       # x: (B,T,N,C)
        y = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        a, g = y.chunk(2, -1)
        return a * torch.sigmoid(g)


class STGCN(Base):
    def __init__(self, *a):
        super().__init__(*a)
        c = self.cfg.get("channels", [64, 16, 64])
        K = self.cheb.shape[0]
        self.blocks = nn.ModuleList()
        cin, Tn = self.Fin, self.Tin
        for _ in range(2):
            self.blocks.append(nn.ModuleDict({
                "t1": TConvGLU(cin, c[0], 3), "t2": TConvGLU(c[1], c[2], 3), "ln": nn.LayerNorm(c[2])}))
            self.blocks[-1].W = nn.Parameter(torch.empty(K, c[0], c[1]))
            self.blocks[-1].bias = nn.Parameter(torch.zeros(c[1]))
            nn.init.xavier_uniform_(self.blocks[-1].W)
            cin, Tn = c[2], Tn - 4
        self.tout = TConvGLU(cin, cin, Tn)
        self.ln = nn.LayerNorm(cin)
        self.fc = nn.Linear(cin, cin)
        self.out = nn.Linear(cin, self.H)
        self.drop = nn.Dropout(self.cfg.get("dropout", 0.1))

    def forward(self, b):
        x = b["x"]
        for blk in self.blocks:
            h = blk["t1"](x)
            g = sum(torch.matmul(self.cheb[k] @ h, blk.W[k]) for k in range(self.cheb.shape[0]))
            x = self.drop(blk["ln"](blk["t2"](F.relu(g + blk.bias))))
        h = self.ln(self.tout(x))[:, 0]
        return {"pred": self.out(F.relu(self.fc(h))).transpose(1, 2)}


# ============================================================================ Graph WaveNet
class GWNet(Base):
    def __init__(self, *a):
        super().__init__(*a)
        c = self.cfg
        R, S, E = c.get("res", 32), c.get("skip", 256), c.get("end", 512)
        self.dil = c.get("dilations", [1, 2, 1, 2, 1, 2, 1, 2])
        self.order, nsup = 2, 3
        self.start = nn.Linear(self.Fin, R)
        self.e1 = nn.Parameter(torch.randn(self.N, 10) * 0.1)
        self.e2 = nn.Parameter(torch.randn(10, self.N) * 0.1)
        self.f = nn.ModuleList([nn.Conv2d(R, R, (2, 1), dilation=(d, 1)) for d in self.dil])
        self.g = nn.ModuleList([nn.Conv2d(R, R, (2, 1), dilation=(d, 1)) for d in self.dil])
        self.skip = nn.ModuleList([nn.Linear(R, S) for _ in self.dil])
        self.gc = nn.ModuleList([nn.Linear(R * (1 + nsup * self.order), R) for _ in self.dil])
        self.bn = nn.ModuleList([nn.LayerNorm(R) for _ in self.dil])
        self.end1, self.end2 = nn.Linear(S, E), nn.Linear(E, self.H)
        self.drop = nn.Dropout(c.get("dropout", 0.3))

    def forward(self, b):
        x = b["x"]
        rf = 1 + sum(self.dil)
        if x.shape[1] < rf:
            x = F.pad(x, (0, 0, 0, 0, rf - x.shape[1], 0))
        h = self.start(x)
        Aadp = torch.softmax(F.relu(self.e1 @ self.e2), 1)
        sups = [self.geo_rw, self.geo_rwT, Aadp]
        skip = 0
        for i, d in enumerate(self.dil):
            res = h
            hc = h.permute(0, 3, 1, 2)
            h = (torch.tanh(self.f[i](hc)) * torch.sigmoid(self.g[i](hc))).permute(0, 2, 3, 1)
            skip = skip + self.skip[i](h[:, -1:])
            outs = [h]
            for A in sups:
                z = h
                for _ in range(self.order):
                    z = A @ z
                    outs.append(z)
            h = self.drop(self.gc[i](torch.cat(outs, -1)))
            h = self.bn[i](h + res[:, -h.shape[1]:])
        y = self.end2(F.relu(self.end1(F.relu(skip))))[:, 0]
        return {"pred": y.transpose(1, 2)}


# ============================================================================ AGCRN
class AGCRN(Base):
    def __init__(self, *a):
        super().__init__(*a)
        dh, de, K = self.cfg.get("hidden", 64), self.cfg.get("embed", 10), 2
        self.layers_n = self.cfg.get("layers", 2)
        self.E = nn.Parameter(torch.randn(self.N, de))
        self.cells = nn.ModuleList()
        cin = self.Fin
        for _ in range(self.layers_n):
            cell = nn.ParameterDict({
                "Wg": nn.Parameter(torch.empty(de, K, cin + dh, 2 * dh)), "bg": nn.Parameter(torch.zeros(de, 2 * dh)),
                "Wc": nn.Parameter(torch.empty(de, K, cin + dh, dh)), "bc": nn.Parameter(torch.zeros(de, dh))})
            nn.init.xavier_uniform_(cell["Wg"])
            nn.init.xavier_uniform_(cell["Wc"])
            self.cells.append(cell)
            cin = dh
        self.dh = dh
        self.out = nn.Conv2d(1, self.H, (1, dh))

    @staticmethod
    def gconv(x, A, Wn, bn):
        xg = torch.stack([x, A @ x], 2)                                 # (B,N,K,C)
        return torch.einsum("bnki,nkio->bno", xg, Wn) + bn

    def forward(self, b):
        x = b["x"]
        B, Tn, N, _ = x.shape
        A = torch.softmax(F.relu(self.E @ self.E.T), 1)
        seq = x
        for cell in self.cells:
            Wg = torch.einsum("nd,dkio->nkio", self.E, cell["Wg"])
            Wc = torch.einsum("nd,dkio->nkio", self.E, cell["Wc"])
            bg, bc = self.E @ cell["bg"], self.E @ cell["bc"]
            h = x.new_zeros(B, N, self.dh)
            outs = []
            for t in range(Tn):
                zr = torch.sigmoid(self.gconv(torch.cat([seq[:, t], h], -1), A, Wg, bg))
                z, r = zr.chunk(2, -1)
                c = torch.tanh(self.gconv(torch.cat([seq[:, t], z * h], -1), A, Wc, bc))
                h = r * h + (1 - r) * c
                outs.append(h)
            seq = torch.stack(outs, 1)
        y = self.out(h.unsqueeze(1))[..., 0]                                # (B,H,N)
        return {"pred": y}


# ============================================================================ STID
class STID(Base):
    def __init__(self, *a):
        super().__init__(*a)
        d, L = self.cfg.get("hidden", 32), self.cfg.get("layers", 3)
        D = 4 * d
        self.ts = nn.Linear(self.Tin * self.Fin, d)
        self.node = nn.Parameter(torch.randn(self.N, d) * 0.1)
        self.tod = nn.Parameter(torch.randn(self.n_slots, d) * 0.1)
        self.dow = nn.Parameter(torch.randn(7, d) * 0.1)
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Dropout(self.cfg.get("dropout", 0.15)),
                                                nn.Linear(D, D)) for _ in range(L)])
        self.out = nn.Linear(D, self.H)

    def forward(self, b):
        x = b["x"]
        B, Tn, N, Fd = x.shape
        h = self.ts(x.permute(0, 2, 1, 3).reshape(B, N, Tn * Fd))
        tod = self.tod[b["tod"][:, -1]][:, None].expand(B, N, -1)
        dow = self.dow[b["dow"][:, -1]][:, None].expand(B, N, -1)
        h = torch.cat([h, self.node[None].expand(B, N, -1), tod, dow], -1)
        for blk in self.enc:
            h = h + blk(h)
        return {"pred": self.out(h).transpose(1, 2)}


# ============================================================================ iTransformer
class ITransformer(Base):
    def __init__(self, *a):
        super().__init__(*a)
        d, L = self.cfg.get("d", 128), self.cfg.get("layers", 2)
        self.emb = nn.Linear(2 * self.Tin, d)      # variate token = [load window, day-lag window]
        self.cov = nn.Linear(self.Tin, d)
        layer = nn.TransformerEncoderLayer(d, self.cfg.get("heads", 8), 2 * d, dropout=0.1, batch_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.H)

    def forward(self, b):
        x = b["x"]
        N = x.shape[2]
        tok = self.emb(torch.cat([x[..., 0], x[..., 2]], 1).transpose(1, 2))
        cov = self.cov(b["cal"].transpose(1, 2))
        h = self.enc(torch.cat([tok, cov], 1))
        return {"pred": self.out(self.ln(h[:, :N])).transpose(1, 2)}


# ============================================================================ STAEformer
class STAEformer(Base):
    def __init__(self, *a):
        super().__init__(*a)
        c = self.cfg
        di, dt, dd, da = c.get("d_in", 24), c.get("d_tod", 24), c.get("d_dow", 24), c.get("d_adp", 80)
        d = di + dt + dd + da
        heads = c.get("heads", 4)
        self.inp = nn.Linear(self.Fin, di)
        self.tod = nn.Parameter(torch.randn(self.n_slots, dt) * 0.1)
        self.dow = nn.Parameter(torch.randn(7, dd) * 0.1)
        self.adp = nn.Parameter(torch.empty(self.Tin, self.N, da))
        nn.init.xavier_uniform_(self.adp)

        def enc(L):
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, c.get("ff", 256), dropout=0.1,
                                                                    batch_first=True), L)
        self.tl = enc(c.get("t_layers", 3))
        self.sl = enc(c.get("s_layers", 3))
        self.out = nn.Linear(self.Tin * d, self.H)

    def forward(self, b):
        x = b["x"]
        B, Tn, N, _ = x.shape
        h = torch.cat([self.inp(x), self.tod[b["tod"]][:, :, None].expand(B, Tn, N, -1),
                       self.dow[b["dow"]][:, :, None].expand(B, Tn, N, -1), self.adp[None].expand(B, -1, -1, -1)], -1)
        d = h.shape[-1]
        h = self.tl(h.transpose(1, 2).reshape(B * N, Tn, d)).reshape(B, N, Tn, d).transpose(1, 2)
        h = self.sl(h.reshape(B * Tn, N, d)).reshape(B, Tn, N, d)
        return {"pred": self.out(h.transpose(1, 2).reshape(B, N, Tn * d)).transpose(1, 2)}


# ============================================================================ CRAG-Net
class CRAGNet(Base):
    """Correlation- and Ramp-Aware Graph Network (see paper, Sec. IV)."""

    def __init__(self, *a):
        super().__init__(*a)
        c = self.cfg
        R, S, E, de = c.get("res", 32), c.get("skip", 64), c.get("end", 128), c.get("node_emb", 16)
        self.dil = c.get("dilations", [1, 2, 4, 4])
        self.graphs = c.get("graphs", ["geo", "corr", "adp"])
        self.use_exo = c.get("exo", True)
        self.delta = c.get("delta", True)
        self.prior = c.get("prior", "both")
        self.aux = c.get("aux", False)
        G = len(self.graphs)
        fin = self.Fin if self.use_exo else 3
        self.start = nn.Linear(fin, R)
        self.node = nn.Parameter(torch.randn(self.N, de) * 0.1)
        self.node_in = nn.Linear(de, R, bias=False)
        self.tod = nn.Parameter(torch.randn(self.n_slots, R) * 0.1)
        self.dow = nn.Parameter(torch.randn(7, R) * 0.1)
        self.e2 = nn.Parameter(torch.randn(de, self.N) * 0.1)
        self.fuse = nn.Linear(de, max(G, 1) * len(self.dil))
        self.f = nn.ModuleList([nn.Conv2d(R, R, (2, 1), dilation=(d, 1)) for d in self.dil])
        self.g = nn.ModuleList([nn.Conv2d(R, R, (2, 1), dilation=(d, 1)) for d in self.dil])
        self.skip = nn.ModuleList([nn.Linear(R, S) for _ in self.dil])
        self.selfw = nn.ModuleList([nn.Linear(R, R) for _ in self.dil])
        self.gw = nn.ParameterList([nn.Parameter(torch.empty(max(G, 1), R, R)) for _ in self.dil])
        for w in self.gw:
            nn.init.xavier_uniform_(w)
        self.ln = nn.ModuleList([nn.LayerNorm(R) for _ in self.dil])
        self.end1, self.end2 = nn.Linear(S + de, E), nn.Linear(E, self.H)
        if self.aux:
            self.cls = nn.Linear(E, self.H * 3)
        if self.prior:
            self.gate = nn.Linear(E, 2 * self.H)
            with torch.no_grad():
                self.gate.weight.mul_(0.1)
                self.gate.bias.zero_()
        self.drop = nn.Dropout(c.get("dropout", 0.2))

    def supports(self):
        out = []
        for g in self.graphs:
            if g == "geo":
                out.append(self.geo_rw)
            elif g == "corr":
                out.append(self.corr_rw)
            elif g == "prof":
                out.append(self.prof_rw)
            elif g == "raw":
                out.append(self.raw_rw)
            elif g == "adj":
                out.append(self.adj_sym)
            elif g == "adp":
                out.append(torch.softmax(F.relu(self.node @ self.e2), 1))
            elif g == "none":
                out.append(None)          # "no message" option inside the node-adaptive softmax
        return out

    def fusion_weights(self):
        G, L = len(self.graphs), len(self.dil)
        return torch.softmax(self.fuse(self.node).reshape(self.N, L, max(G, 1)), -1)

    def forward(self, b):
        x = b["x"] if self.use_exo else b["x"][..., :3]
        B, Tn, N, _ = x.shape
        h = self.start(x) + self.node_in(self.node) + self.tod[b["tod"]][:, :, None]
        if self.use_exo:
            h = h + self.dow[b["dow"]][:, :, None]
        sups = self.supports()
        alpha = self.fusion_weights()
        skip = 0
        for li, d in enumerate(self.dil):
            res = h
            hc = h.permute(0, 3, 1, 2)
            h = (torch.tanh(self.f[li](hc)) * torch.sigmoid(self.g[li](hc))).permute(0, 2, 3, 1)
            skip = skip + self.skip[li](h[:, -1])
            z = self.selfw[li](h)
            for gi, A in enumerate(sups):
                if A is not None:
                    z = z + alpha[:, li, gi, None] * ((A @ h) @ self.gw[li][gi])
            h = self.ln[li](self.drop(z) + res[:, -z.shape[1]:])
        hid = F.relu(self.end1(torch.cat([F.relu(skip), self.node[None].expand(B, -1, -1)], -1)))
        out = self.end2(hid).transpose(1, 2)                                        # (B,H,N)
        res = {}
        if self.prior:
            gl = self.gate(hid).reshape(B, N, 2, self.H).permute(0, 3, 2, 1)        # (B,H,2,N)
            if self.prior in ("daily", "both"):
                gd = 1 + torch.tanh(gl[:, :, 0])
                out = out + gd * (b["lag_fut"] - b["lag_last"][:, None])
                res["gate_daily"] = gd
            if self.prior in ("profile", "both"):
                gp = torch.tanh(gl[:, :, 1]) if self.prior == "both" else 1 + torch.tanh(gl[:, :, 1])
                out = out + gp * (b["prof_fut"] - b["prof_last"][:, None])
                res["gate_prof"] = gp
        res["pred"] = b["last"][:, None] + out if self.delta else out
        if self.aux:
            res["logits"] = self.cls(hid).reshape(B, N, self.H, 3).transpose(1, 2)
        return res


REGISTRY = {"LSTM": LSTM, "T-GCN": TGCN, "STGCN": STGCN, "GWNet": GWNet, "AGCRN": AGCRN, "STID": STID,
            "iTransformer": ITransformer, "STAEformer": STAEformer, "CRAG-Net": CRAGNet}


# ============================================================================ CRAG-Net (MLP backbone)
class CRAGNetMLP(CRAGNet):
    """CRAG-Net with an identity-embedding MLP encoder (STID-style) instead of the dilated TCN.

    Per zone, the whole input window is flattened and encoded together with zone, time-of-day and
    day-of-week embeddings; residual MLP blocks alternate with node-adaptive multi-graph mixing; the head
    is the same ramp-decomposed head with gated seasonal ramp priors.
    """

    def __init__(self, cfg, G, N, Tin, H, Fin, n_slots):
        Base.__init__(self, cfg, G, N, Tin, H, Fin, n_slots)
        c = self.cfg
        d, L, de = c.get("hidden", 32), c.get("layers", 3), c.get("node_emb", 32)
        D = 4 * d
        self.graphs = c.get("graphs", ["geo", "corr", "adp"])
        self.use_exo = c.get("exo", True)
        self.delta = c.get("delta", True)
        self.prior = c.get("prior", "both")
        self.aux = c.get("aux", False)
        self.L = L
        Gn = len(self.graphs)
        fin = self.Fin if self.use_exo else 3
        self.ts = nn.Linear(self.Tin * fin, d)
        self.node = nn.Parameter(torch.randn(self.N, de) * 0.1)
        self.node_proj = nn.Linear(de, d)
        self.tod = nn.Parameter(torch.randn(self.n_slots, d) * 0.1)
        self.dow = nn.Parameter(torch.randn(7, d) * 0.1)
        self.e2 = nn.Parameter(torch.randn(de, self.N) * 0.1)
        self.fuse = nn.Linear(de, max(Gn, 1) * L)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Dropout(c.get("dropout", 0.15)),
                                                nn.Linear(D, D)) for _ in range(L)])
        self.gw = nn.ParameterList([nn.Parameter(torch.zeros(max(Gn, 1), D, D)) for _ in range(L)])
        for w in self.gw:
            nn.init.xavier_uniform_(w, gain=0.1)
        self.ln = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        E = D
        self.end2 = nn.Linear(E, self.H)
        if self.aux:
            self.cls = nn.Linear(E, self.H * 3)
        if self.prior:
            self.gate = nn.Linear(E, 2 * self.H)
            with torch.no_grad():
                self.gate.weight.mul_(0.1)
                self.gate.bias.zero_()
        self.dil = list(range(L))      # for fusion_weights() shape
        if "none" in self.graphs:
            with torch.no_grad():
                b_ = self.fuse.bias.view(L, Gn)
                b_[:, self.graphs.index("none")] = self.cfg.get("none_bias", 2.0)

    def forward(self, b):
        x = b["x"] if self.use_exo else b["x"][..., :3]
        B, Tn, N, Fd = x.shape
        h = self.ts(x.permute(0, 2, 1, 3).reshape(B, N, Tn * Fd))
        tod = self.tod[b["tod"][:, -1]][:, None].expand(B, N, -1)
        dow = self.dow[b["dow"][:, -1]][:, None].expand(B, N, -1)
        h = torch.cat([h, self.node_proj(self.node)[None].expand(B, -1, -1), tod, dow], -1)    # (B,N,D)
        sups = self.supports()
        alpha = self.fusion_weights()
        for li in range(self.L):
            h = h + self.mlp[li](h)
            if any(A is not None for A in sups):
                z = 0
                for gi, A in enumerate(sups):
                    if A is not None:
                        z = z + alpha[:, li, gi, None] * ((A @ h) @ self.gw[li][gi])
                h = self.ln[li](h + z) if self.cfg.get("graph_ln", True) else h + z
        hid = h
        out = self.end2(hid).transpose(1, 2)
        res = {}
        if self.prior:
            gl = self.gate(hid).reshape(B, N, 2, self.H).permute(0, 3, 2, 1)
            if self.prior in ("daily", "both"):
                # zero-initialised gate (default for the MLP backbone): the prior is used only where it helps
                gd = torch.tanh(gl[:, :, 0]) if self.cfg.get("gate_init0", True) else 1 + torch.tanh(gl[:, :, 0])
                out = out + gd * (b["lag_fut"] - b["lag_last"][:, None])
                res["gate_daily"] = gd
            if self.prior in ("profile", "both"):
                gp = torch.tanh(gl[:, :, 1]) if self.prior == "both" else 1 + torch.tanh(gl[:, :, 1])
                out = out + gp * (b["prof_fut"] - b["prof_last"][:, None])
                res["gate_prof"] = gp
        res["pred"] = b["last"][:, None] + out if self.delta else out
        if self.aux:
            res["logits"] = self.cls(hid).reshape(B, N, self.H, 3).transpose(1, 2)
        return res


REGISTRY["CRAG-MLP"] = CRAGNetMLP
