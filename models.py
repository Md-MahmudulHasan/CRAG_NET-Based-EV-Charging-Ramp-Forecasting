"""Spatio-temporal forecasting models (baselines + proposed), written with autograd.numpy.

Every model maps a batch
    x   : (B, T, N, F) input features (F[0] = normalised load, F[1] = first difference, rest exogenous)
    tod : (B, T) time-of-day slot,   dow : (B, T) day-of-week
to a dict with
    pred   : (B, H, N) normalised load forecasts
    logits : optional (B, H, N, 3) ramp-event logits (down / none / up)

Baselines follow the original architectures, scaled to a CPU budget (see paper, Sec. V-A):
LSTM, T-GCN (Zhao et al. 2019), STGCN (Yu et al. 2018), Graph WaveNet (Wu et al. 2019),
AGCRN (Bai et al. 2020), STID (Shao et al. 2022), iTransformer (Liu et al. 2024),
STAEformer (Liu et al. 2023).  Proposed: CRAG-Net (correlation- and ramp-aware graph network).
"""
from __future__ import annotations

from nn import (np, onp, F32, linear_init, linear, ln_init, layernorm, mlp_init, mlp, relu, sigmoid, tanh,
                gelu, softmax, dropout, tconv_init, tconv, gru_init, gru_cell, lstm_init, lstm_cell,
                mha_init, mha, glorot)


class Base:
    name = "base"
    ramp_head = False

    def __init__(self, cfg, G, N, T, H, F, n_slots, rng):
        self.cfg, self.G, self.N, self.T, self.H, self.F, self.n_slots = cfg, G, N, T, H, F, n_slots
        self.rng = rng
        self.params = self.init(rng)

    def init(self, rng):
        raise NotImplementedError

    def __call__(self, p, b, train=False, rng=None):
        raise NotImplementedError


# ============================================================================ LSTM (node-wise, shared)
class LSTM(Base):
    name = "LSTM"

    def init(self, rng):
        dh = self.cfg.get("hidden", 64)
        return {"cell": lstm_init(rng, self.F, dh), "out": linear_init(rng, dh, self.H)}

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        B, T, N, _ = x.shape
        dh = p["cell"]["Wh"].shape[0]
        h = onp.zeros((B, N, dh), F32)
        c = onp.zeros((B, N, dh), F32)
        for t in range(T):
            h, c = lstm_cell(p["cell"], x[:, t], (h, c))
        y = linear(p["out"], h)                       # (B,N,H)
        return {"pred": np.swapaxes(y, 1, 2)}


# ============================================================================ T-GCN
class TGCN(Base):
    name = "T-GCN"

    def init(self, rng):
        dh = self.cfg.get("hidden", 64)
        return {"g": linear_init(rng, self.F + dh, 2 * dh), "c": linear_init(rng, self.F + dh, dh),
                "out": linear_init(rng, dh, self.H)}

    def __call__(self, p, b, train=False, rng=None):
        A = self.G["adj_sym"]
        x = b["x"]
        B, T, N, _ = x.shape
        dh = p["c"]["W"].shape[1]
        h = onp.zeros((B, N, dh), F32)
        for t in range(T):
            xh = np.concatenate([x[:, t], h], -1)
            g = sigmoid(linear(p["g"], np.matmul(A, xh)))
            r, u = g[..., :dh], g[..., dh:]
            c = tanh(linear(p["c"], np.matmul(A, np.concatenate([x[:, t], r * h], -1))))
            h = u * h + (1 - u) * c
        return {"pred": np.swapaxes(linear(p["out"], h), 1, 2)}


# ============================================================================ STGCN
class STGCN(Base):
    name = "STGCN"

    def init(self, rng):
        c = self.cfg.get("channels", [32, 16, 32])
        K = len(self.G["cheb"])
        blocks, cin, T = [], self.F, self.T
        for _ in range(2):
            blocks.append({"t1": tconv_init(rng, cin, 2 * c[0], 3),
                           "cheb": {"W": glorot(rng, (K, c[0], c[1])), "b": onp.zeros(c[1], F32)},
                           "t2": tconv_init(rng, c[1], 2 * c[2], 3), "ln": ln_init(c[2])})
            cin, T = c[2], T - 4
        return {"blocks": blocks, "tout": tconv_init(rng, cin, 2 * cin, T), "ln": ln_init(cin),
                "fc": linear_init(rng, cin, cin), "out": linear_init(rng, cin, self.H)}

    @staticmethod
    def glu(p, x):
        y = tconv(p, x)
        c = y.shape[-1] // 2
        return y[..., :c] * sigmoid(y[..., c:])

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        Ts = self.G["cheb"]
        for blk in p["blocks"]:
            h = self.glu(blk["t1"], x)
            g = 0.0
            for k, Tk in enumerate(Ts):
                g = g + np.matmul(np.matmul(Tk, h), blk["cheb"]["W"][k])
            g = relu(g + blk["cheb"]["b"])
            x = layernorm(blk["ln"], self.glu(blk["t2"], g))
            x = dropout(x, self.cfg.get("dropout", 0.1), rng, train)
        h = layernorm(p["ln"], self.glu(p["tout"], x))[:, 0]   # (B,N,C)
        y = linear(p["out"], relu(linear(p["fc"], h)))
        return {"pred": np.swapaxes(y, 1, 2)}


# ============================================================================ Graph WaveNet
class GWNet(Base):
    name = "GWNet"

    def init(self, rng):
        cfg = self.cfg
        R, S, E = cfg.get("res", 32), cfg.get("skip", 64), cfg.get("end", 128)
        self.dil = cfg.get("dilations", [1, 2, 1, 2, 1, 2, 1, 2])
        self.order = 2
        nsup = 3
        p = {"start": linear_init(rng, self.F, R),
             "e1": (rng.standard_normal((self.N, 10)) * 0.1).astype(F32),
             "e2": (rng.standard_normal((10, self.N)) * 0.1).astype(F32),
             "layers": [], "end1": linear_init(rng, S, E), "end2": linear_init(rng, E, self.H)}
        for _ in self.dil:
            p["layers"].append({"f": tconv_init(rng, R, R, 2), "g": tconv_init(rng, R, R, 2),
                                "skip": linear_init(rng, R, S),
                                "gc": linear_init(rng, R * (1 + nsup * self.order), R),
                                "ln": ln_init(R)})
        return p

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        rf = 1 + sum(self.dil)
        if x.shape[1] < rf:
            pad = onp.zeros((x.shape[0], rf - x.shape[1]) + x.shape[2:], F32)
            x = np.concatenate([pad, x], 1)
        h = linear(p["start"], x)
        Aadp = softmax(relu(np.matmul(p["e1"], p["e2"])), axis=1)
        sups = [self.G["geo_rw"], self.G["geo_rwT"], Aadp]
        skip = 0.0
        for d, L in zip(self.dil, p["layers"]):
            res = h
            f = tanh(tconv(L["f"], h, d))
            g = sigmoid(tconv(L["g"], h, d))
            h = f * g
            skip = skip + linear(L["skip"], h[:, -1:])
            outs = [h]
            for A in sups:
                z = h
                for _ in range(self.order):
                    z = np.matmul(A, z)
                    outs.append(z)
            h = linear(L["gc"], np.concatenate(outs, -1))
            h = dropout(h, self.cfg.get("dropout", 0.3), rng, train)
            h = layernorm(L["ln"], h + res[:, -h.shape[1]:])
        y = linear(p["end2"], relu(linear(p["end1"], relu(skip))))[:, 0]   # (B,N,H)
        return {"pred": np.swapaxes(y, 1, 2)}


# ============================================================================ AGCRN
class AGCRN(Base):
    name = "AGCRN"

    def init(self, rng):
        dh, de, K = self.cfg.get("hidden", 64), self.cfg.get("embed", 10), 2
        cin = self.F + dh
        return {"E": (rng.standard_normal((self.N, de))).astype(F32),
                "Wg": glorot(rng, (de, K, cin, 2 * dh)) * 0.5, "bg": onp.zeros((de, 2 * dh), F32),
                "Wc": glorot(rng, (de, K, cin, dh)) * 0.5, "bc": onp.zeros((de, dh), F32),
                "out": linear_init(rng, dh, self.H)}

    @staticmethod
    def avwgcn(x, A, Wn, bn):
        """Node-adaptive graph conv (K=2: I, A). x: (B,N,C); Wn: (K,N,C,O) node-specific weights."""
        y = 0.0
        for k, xk in enumerate((x, np.matmul(A, x))):
            y = y + np.swapaxes(np.matmul(np.swapaxes(xk, 0, 1), Wn[k]), 0, 1)   # batched over nodes
        return y + bn

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        B, T, N, _ = x.shape
        dh = p["Wc"].shape[-1]
        E = p["E"]
        A = softmax(relu(np.matmul(E, E.T)), axis=1)
        Wg = np.einsum("nd,dkio->knio", E, p["Wg"])      # computed once per forward (shared over time)
        Wc = np.einsum("nd,dkio->knio", E, p["Wc"])
        bg, bc = np.matmul(E, p["bg"]), np.matmul(E, p["bc"])
        h = onp.zeros((B, N, dh), F32)
        for t in range(T):
            xh = np.concatenate([x[:, t], h], -1)
            zr = sigmoid(self.avwgcn(xh, A, Wg, bg))
            z, r = zr[..., :dh], zr[..., dh:]
            c = tanh(self.avwgcn(np.concatenate([x[:, t], z * h], -1), A, Wc, bc))
            h = r * h + (1 - r) * c
        return {"pred": np.swapaxes(linear(p["out"], h), 1, 2)}


# ============================================================================ STID
class STID(Base):
    name = "STID"

    def init(self, rng):
        d, L = self.cfg.get("hidden", 32), self.cfg.get("layers", 3)
        D = 4 * d
        return {"ts": linear_init(rng, self.T * self.F, d),
                "node": (rng.standard_normal((self.N, d)) * 0.1).astype(F32),
                "tod": (rng.standard_normal((self.n_slots, d)) * 0.1).astype(F32),
                "dow": (rng.standard_normal((7, d)) * 0.1).astype(F32),
                "enc": [mlp_init(rng, [D, D, D]) for _ in range(L)],
                "out": linear_init(rng, D, self.H)}

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        B, T, N, F = x.shape
        xs = np.reshape(np.transpose(x, (0, 2, 1, 3)), (B, N, T * F))
        h = linear(p["ts"], xs)
        node = np.broadcast_to(p["node"][None], (B, N, p["node"].shape[1]))
        tod = np.broadcast_to(p["tod"][b["tod"][:, -1]][:, None, :], (B, N, p["tod"].shape[1]))
        dow = np.broadcast_to(p["dow"][b["dow"][:, -1]][:, None, :], (B, N, p["dow"].shape[1]))
        h = np.concatenate([h, node, tod, dow], -1)
        for blk in p["enc"]:
            h = h + mlp(blk, dropout(h, self.cfg.get("dropout", 0.15), rng, train))
        return {"pred": np.swapaxes(linear(p["out"], h), 1, 2)}


# ============================================================================ iTransformer
class ITransformer(Base):
    name = "iTransformer"

    def init(self, rng):
        d, L = self.cfg.get("d", 64), self.cfg.get("layers", 2)
        return {"emb": linear_init(rng, self.T, d), "cov": linear_init(rng, self.T, d),
                "layers": [{"att": mha_init(rng, d), "ln1": ln_init(d), "ff": mlp_init(rng, [d, 2 * d, d]),
                            "ln2": ln_init(d)} for _ in range(L)],
                "ln": ln_init(d), "out": linear_init(rng, d, self.H)}

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        B, T, N, F = x.shape
        tok = linear(p["emb"], np.transpose(x[..., 0], (0, 2, 1)))          # (B,N,d)
        cov = linear(p["cov"], np.transpose(b["cal"], (0, 2, 1)))           # (B,4,d) covariate tokens
        h = np.concatenate([tok, cov], 1)
        for L in p["layers"]:
            h = layernorm(L["ln1"], h + dropout(mha(L["att"], h, 4), 0.1, rng, train))
            h = layernorm(L["ln2"], h + dropout(mlp(L["ff"], h, gelu), 0.1, rng, train))
        y = linear(p["out"], layernorm(p["ln"], h[:, :N]))
        return {"pred": np.swapaxes(y, 1, 2)}


# ============================================================================ STAEformer
class STAEformer(Base):
    name = "STAEformer"

    def init(self, rng):
        c = self.cfg
        di, dt, dd, da = c.get("d_in", 16), c.get("d_tod", 16), c.get("d_dow", 16), c.get("d_adp", 32)
        d = di + dt + dd + da
        Lt, Ls = c.get("t_layers", 2), c.get("s_layers", 2)

        def layer():
            return {"att": mha_init(rng, d), "ln1": ln_init(d), "ff": mlp_init(rng, [d, 2 * d, d]), "ln2": ln_init(d)}
        return {"inp": linear_init(rng, self.F, di),
                "tod": (rng.standard_normal((self.n_slots, dt)) * 0.1).astype(F32),
                "dow": (rng.standard_normal((7, dd)) * 0.1).astype(F32),
                "adp": glorot(rng, (self.T, self.N, da)),
                "tl": [layer() for _ in range(Lt)], "sl": [layer() for _ in range(Ls)],
                "out": linear_init(rng, self.T * d, self.H)}

    def block(self, L, h, train, rng):
        h = layernorm(L["ln1"], h + dropout(mha(L["att"], h, self.cfg.get("heads", 2)), 0.1, rng, train))
        return layernorm(L["ln2"], h + dropout(mlp(L["ff"], h, relu), 0.1, rng, train))

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"]
        B, T, N, F = x.shape
        e = [linear(p["inp"], x),
             np.broadcast_to(p["tod"][b["tod"]][:, :, None, :], (B, T, N, p["tod"].shape[1])),
             np.broadcast_to(p["dow"][b["dow"]][:, :, None, :], (B, T, N, p["dow"].shape[1])),
             np.broadcast_to(p["adp"][None], (B, T, N, p["adp"].shape[2]))]
        h = np.concatenate(e, -1)                                   # (B,T,N,d)
        h = np.swapaxes(h, 1, 2)                                    # (B,N,T,d) temporal attention over T
        for L in p["tl"]:
            h = self.block(L, h, train, rng)
        h = np.swapaxes(h, 1, 2)                                    # (B,T,N,d) spatial attention over N
        for L in p["sl"]:
            h = self.block(L, h, train, rng)
        h = np.reshape(np.swapaxes(h, 1, 2), (B, N, -1))
        return {"pred": np.swapaxes(linear(p["out"], h), 1, 2)}


# ============================================================================ Proposed: CRAG-Net
class CRAGNet(Base):
    """Correlation- and Ramp-Aware Graph Network.

    * multi-graph spatial mixing over {geographic kernel, training-set ramp-correlation, adaptive}
      graphs with node-adaptive fusion weights (softmax over graphs, conditioned on node embeddings);
    * gated dilated causal TCN backbone (receptive field = input window);
    * calendar/node identity embeddings + exogenous weather;
    * ramp parameterisation: the head predicts the cumulative ramp Delta_h = y_{t+h} - y_t and the
      level is recovered as y_t + Delta_h; an auxiliary head classifies ramp events (down/none/up).
    """
    name = "CRAG-Net"
    ramp_head = True

    def init(self, rng):
        c = self.cfg
        R, S, E, de = c.get("res", 32), c.get("skip", 64), c.get("end", 128), c.get("node_emb", 16)
        self.dil = c.get("dilations", [1, 2, 4, 4])
        self.graphs = c.get("graphs", ["geo", "corr", "adp"])
        self.use_exo = c.get("exo", True)
        self.delta = c.get("delta", True)
        G = len(self.graphs)
        fin = self.F if self.use_exo else 3
        p = {"start": linear_init(rng, fin, R),
             "node": (rng.standard_normal((self.N, de)) * 0.1).astype(F32),
             "node_in": linear_init(rng, de, R, bias=False),
             "tod": (rng.standard_normal((self.n_slots, R)) * 0.1).astype(F32),
             "dow": (rng.standard_normal((7, R)) * 0.1).astype(F32),
             "e2": (rng.standard_normal((de, self.N)) * 0.1).astype(F32),
             "fuse": linear_init(rng, de, max(G, 1) * len(self.dil)),
             "layers": [], "end1": linear_init(rng, S + de, E), "end2": linear_init(rng, E, self.H)}
        if self.ramp_head_on():
            p["cls"] = linear_init(rng, E, self.H * 3)
        self.prior = c.get("prior", "both")        # "daily" | "profile" | "both" | None
        if self.prior:
            p["gate"] = linear_init(rng, E, 2 * self.H)
            p["gate"]["W"] *= 0.1
        for _ in self.dil:
            p["layers"].append({"f": tconv_init(rng, R, R, 2), "g": tconv_init(rng, R, R, 2),
                                "skip": linear_init(rng, R, S),
                                "gw": glorot(rng, (max(G, 1), R, R)), "self": linear_init(rng, R, R),
                                "ln": ln_init(R)})
        return p

    def ramp_head_on(self):
        return self.cfg.get("aux", True)

    def __call__(self, p, b, train=False, rng=None):
        x = b["x"] if self.use_exo else b["x"][..., :3]
        B, T, N, _ = x.shape
        h = linear(p["start"], x) + linear(p["node_in"], p["node"])
        h = h + p["tod"][b["tod"]][:, :, None, :]
        if self.use_exo:
            h = h + p["dow"][b["dow"]][:, :, None, :]
        sups = []
        for gname in self.graphs:
            if gname == "geo":
                sups.append(self.G["geo_rw"])
            elif gname == "corr":
                sups.append(self.G["corr_rw"])
            elif gname == "adj":
                sups.append(self.G["adj_sym"])
            elif gname == "raw":
                sups.append(self.G["raw_rw"])
            elif gname == "prof":
                sups.append(self.G["prof_rw"])
            elif gname == "adp":
                sups.append(softmax(relu(np.matmul(p["node"], p["e2"])), axis=1))
        G = len(sups)
        nl = len(self.dil)
        if G > 0:
            alpha = softmax(np.reshape(linear(p["fuse"], p["node"]), (N, nl, G)), axis=-1)  # node-adaptive fusion
        skip = 0.0
        for li, (d, L) in enumerate(zip(self.dil, p["layers"])):
            res = h
            h = tanh(tconv(L["f"], h, d)) * sigmoid(tconv(L["g"], h, d))
            skip = skip + linear(L["skip"], h[:, -1])
            z = linear(L["self"], h)
            for gi, A in enumerate(sups):
                z = z + alpha[:, li, gi][:, None] * np.matmul(np.matmul(A, h), L["gw"][gi])
            z = dropout(z, self.cfg.get("dropout", 0.2), rng, train)
            h = layernorm(L["ln"], z + res[:, -z.shape[1]:])
        node = np.broadcast_to(p["node"][None], (B, N, p["node"].shape[1]))
        hid = relu(linear(p["end1"], np.concatenate([relu(skip), node], -1)))           # (B,N,E)
        out = linear(p["end2"], hid)                                                     # (B,N,H)
        out = np.swapaxes(out, 1, 2)                                                     # (B,H,N)
        if self.prior:
            # seasonal ramp priors, modulated by node/state-dependent gates:
            #   daily  : yesterday's ramp over the same horizon  y(t+h-S) - y(t-S)
            #   profile: training-set calendar profile ramp       P(t+h) - P(t)   (TOU / workday structure)
            gl = np.swapaxes(np.reshape(linear(p["gate"], hid), (B, N, 2, self.H)), 1, 3)   # (B,H,2,N)
            gates = {}
            if self.prior in ("daily", "both"):
                gates["gate_daily"] = 1.0 + tanh(gl[:, :, 0])
                out = out + gates["gate_daily"] * (b["lag_fut"] - b["lag_last"][:, None, :])
            if self.prior in ("profile", "both"):
                gates["gate_prof"] = tanh(gl[:, :, 1]) if self.prior == "both" else 1.0 + tanh(gl[:, :, 1])
                out = out + gates["gate_prof"] * (b["prof_fut"] - b["prof_last"][:, None, :])
        pred = b["last"][:, None, :] + out if self.delta else out
        res = {"pred": pred}
        if self.prior and not train:
            res.update({k: onp.asarray(v) for k, v in gates.items()})
        if self.ramp_head_on():
            lg = np.reshape(linear(p["cls"], hid), (B, N, self.H, 3))
            res["logits"] = np.swapaxes(lg, 1, 2)                                        # (B,H,N,3)
        return res


REGISTRY = {"LSTM": LSTM, "T-GCN": TGCN, "STGCN": STGCN, "GWNet": GWNet, "AGCRN": AGCRN,
            "STID": STID, "iTransformer": ITransformer, "STAEformer": STAEformer, "CRAG-Net": CRAGNet}
