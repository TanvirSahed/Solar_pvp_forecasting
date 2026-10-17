"""Download the reference papers for the SET Lab model-selection report into ./ref.

Usage:  python tools/fetch_refs.py [--out ref] [--only R06,R12]

Strategy per paper (first hit wins):
  1. arXiv PDF (direct, or an arXiv preprint found via Semantic Scholar)
  2. Publisher open-access PDF (Nature/Sci. Rep., MDPI)
  3. Semantic Scholar `openAccessPdf`
Paywalled papers (most Elsevier / IEEE TSTE) are logged as MANUAL with their DOI link.
A status table is written to ref/README.md.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8"}

# (id, short name, citation, link) -- from SET_Lab_Dl_Model_Selection.pdf, "Complete References"
REFS = [
    ("R01", "SolarBench", "Nie et al. (2026). SolarBench: A global solar energy nowcasting benchmark", "arXiv:2609.06187"),
    ("R02", "TSFM_PowerSystem_Assessment", "Za'ter & Hodge (2026). Empirical Assessment of Time-Series Foundation Models for Power System Forecasting Applications", "arXiv:2604.22077"),
    ("R03", "FETS_Benchmark", "Obermeier et al. (2026). FETS Benchmark: Foundation Models Enable Scalable and Generalizable Energy Time Series Forecasting", "arXiv:2604.22328"),
    ("R04", "TSFM_PhysicsSynthetic_ColdStart_PV", "Longarini et al. (2026). Time series Foundation Models based on Physics-Informed Synthetic Histories for Cold-Start PV Forecasting", "arXiv:2606.07457"),
    ("R05", "SolarSeer", "Bai et al. (2025). SolarSeer: Ultrafast and accurate 24-hour solar irradiance forecasts", "arXiv:2508.03590"),
    ("R06", "HDCF-Mamba", "Shi et al. (2026). HDCF-Mamba: Bridging Global Dependencies and Local Dynamics for Multi-Scale PV Forecasting. Energies 19(5), 1315", "doi:10.3390/en19051315"),
    ("R07", "PatchTST_ACI_PV", "Suresh (2025). Benchmarking Transformer Variants for Hour-Ahead PV Forecasting: PatchTST with Adaptive Conformal Inference. Energies 18(18), 5000", "doi:10.3390/en18185000"),
    ("R08", "MTSA-QRN_CQR", "Wang et al. (2026). Probabilistic PV Power Forecasting ... Multi-Scale Temporal-Spatial Attention and CQR. Sustainability 18(2), 739", "doi:10.3390/su18020739"),
    ("R09", "WeatherPatternExperts_Drift", "Wang et al. (2026). Adapting to drift: Weather-pattern experts for short-term PV power forecasting. Adv. Eng. Informatics 74, 104781", "doi:10.1016/j.aei.2026.104781"),
    ("R10", "DualLayerDrift_Retraining", "Cao et al. (2026). Dual-Layer Drift Detection and Adaptive Retraining for Ultra-Short-Term PV Forecasting. Energies 19(18), 4391", "doi:10.3390/en19184391"),
    ("R11", "PV-MM-Diffusion", "Huang et al. (2026). PV-MM-diffusion: multi-modal diffusion model for ultra-short-term probabilistic PV forecasting. Applied Energy 413, 127816", "doi:10.1016/j.apenergy.2026.127816"),
    ("R12", "Chronos-2", "Ansari et al. (2025). Chronos-2: From Univariate to Universal Forecasting", "arXiv:2510.15821"),
    ("R13", "TiRex-2", "Podest et al. (2026). TiRex-2: Generalizing TiRex to Multivariate Data and Streaming", "arXiv:2607.01204"),
    ("R14", "TimesFM", "Das et al. (2024). A Decoder-Only Foundation Model for Time-Series Forecasting", "arXiv:2310.10688"),
    ("R15", "Moirai-2", "Liu et al. (2026). Moirai 2.0: When Less Is More for Time Series Forecasting", "arXiv:2511.11698"),
    ("R16", "RFKAN", "Rong et al. (2025). Recurrent Fourier-Kolmogorov Arnold Networks for PV power forecasting. Scientific Reports", "doi:10.1038/s41598-025-88959-5"),
    ("R17", "QuantumML_PV", "Sagingalieva et al. (2025). Photovoltaic power forecasting using quantum machine learning. Solar Energy 302, 114016", "doi:10.1016/j.solener.2025.114016"),
    ("R18", "DigitalTwin_Transformer_PV", "Zhao (2024). A novel digital-twin approach based on transformer for PV power prediction. Scientific Reports", "doi:10.1038/s41598-024-76711-4"),
    ("R19", "PowerMamba", "Menati et al. (2025). PowerMamba: A Deep State Space Model and Comprehensive Benchmark for Time Series Prediction in Electric Power Systems", "arXiv:2412.06112"),
    ("R20", "SSM_Transformer_RNN_GridBenchmark", "Hong & Lee (2026). Benchmarking State Space Models, Transformers, and Recurrent Networks for US Grid Forecasting", "arXiv:2602.21415"),
    ("R21", "DEST-GNN", "Yang et al. (2025). DEST-GNN: double-explored spatio-temporal GNN for multi-site intra-hour PV forecasting. Applied Energy 378, 124744", "doi:10.1016/j.apenergy.2024.124744"),
    ("R22", "MSF-GCN", "Sima et al. (2025). Multi-scale fused GCN for multi-site PV power forecasting. Energy Conv. Manag. 333, 119773", "doi:10.1016/j.enconman.2025.119773"),
    ("R23", "FourierST-GNN", "Su et al. (2026). Spatio-temporal GNN with Fourier features for multi-site PV forecasting. EPSR 251, 112171", "doi:10.1016/j.epsr.2025.112171"),
    ("R24", "RL_Ensemble_Renewable", "Wu et al. (2025). A reinforcement learning-based ensemble forecasting framework for renewable energy forecasting. Renewable Energy 244, 122692", "doi:10.1016/j.renene.2025.122692"),
    ("R25", "PPO_DynamicModelSelection_PV", "Wang et al. (2026). Context-aware dynamic model selection framework based on PPO for day-ahead PV forecasting. Scientific Reports", "doi:10.1038/s41598-026-69955-9"),
    ("R26", "GreedyCompensatory_SDQN_PV", "Cong et al. (2026). Greedy compensatory agent based day-ahead PV forecasting with simplified DQN. Renewable Energy 256, 124616", "doi:10.1016/j.renene.2025.124616"),
    ("R27", "GraphCast", "Lam et al. (2023). GraphCast: Learning skillful medium-range global weather forecasting", "arXiv:2212.12794"),
    ("R28", "GenCast", "Price et al. (2024). Probabilistic weather forecasting with machine learning. Nature", "doi:10.1038/s41586-024-08252-9"),
    ("R29", "Aurora", "Bodnar et al. (2025). A foundation model for the Earth system. Nature", "doi:10.1038/s41586-025-09005-y"),
    ("R30", "FourCastNet", "Pathak et al. (2022). FourCastNet: Adaptive Fourier Neural Operators weather model", "arXiv:2202.11214"),
    ("R31", "Federated_PV", "Michalakopoulos et al. (2025). Comparative analysis of federated learning strategies for short-term PV forecasting. Energy Reports 13", "doi:10.1016/j.egyr.2025.03.043"),
    ("R32", "CEEMDAN-CNN-BiLSTM-SDQN", "Jia et al. (2026). A CEEMDAN-CNN-BiLSTM-SDQN Framework for PV Power Forecasting. Energies 19(7), 1649", "doi:10.3390/en19071649"),
    ("R33", "Copula_RL_ProbForecast", "Zhao et al. (2026). Probability Forecasting of Daily PV Output Based on Dynamic Copula and RL. Electronics 15(7), 1387", "doi:10.3390/electronics15071387"),
    ("R34", "RL_EMS_TransferLearning", "Xu et al. (2025). Enhancing RL-Based Energy Management Through Transfer Learning With Load and PV Forecasting. IEEE Access", "doi:10.1109/ACCESS.2025.3548990"),
    ("R35", "Robust_PV_Adversarial_DRL", "Liu et al. (2025). Robust PV Forecasting Against Multi-Modal Adversarial Attack via DRL. IEEE TSTE 16(4)", "doi:10.1109/TSTE.2025.3548089"),
]

# Architecture papers we re-implement from (not in the 35, but needed to build the models faithfully)
FOUNDATIONAL = [
    ("F01", "PatchTST", "Nie et al. (2023). A Time Series is Worth 64 Words (PatchTST)", "arXiv:2211.14730"),
    ("F02", "iTransformer", "Liu et al. (2024). iTransformer: Inverted Transformers Are Effective for TS Forecasting", "arXiv:2310.06625"),
    ("F03", "TimeXer", "Wang et al. (2024). TimeXer: Empowering Transformers for TS Forecasting with Exogenous Variables", "arXiv:2402.19072"),
    ("F04", "Mamba", "Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "arXiv:2312.00752"),
    ("F05", "Mamba-2_SSD", "Dao & Gu (2024). Transformers are SSMs (Mamba-2)", "arXiv:2405.21060"),
    ("F06", "S-Mamba", "Wang et al. (2024). Is Mamba Effective for Time Series Forecasting?", "arXiv:2403.11144"),
    ("F07", "TiRex", "Auer et al. (2025). TiRex: Zero-Shot Forecasting Across Long and Short Horizons", "arXiv:2505.23719"),
    ("F08", "Chronos", "Ansari et al. (2024). Chronos: Learning the Language of Time Series", "arXiv:2403.07815"),
    ("F09", "GCN", "Kipf & Welling (2017). Semi-Supervised Classification with GCNs", "arXiv:1609.02907"),
    ("F10", "GAT", "Velickovic et al. (2018). Graph Attention Networks", "arXiv:1710.10903"),
    ("F11", "GATv2", "Brody et al. (2022). How Attentive are Graph Attention Networks?", "arXiv:2105.14491"),
    ("F12", "GraphWaveNet", "Wu et al. (2019). Graph WaveNet (adaptive adjacency)", "arXiv:1906.00121"),
    ("F13", "MTGNN", "Wu et al. (2020). Connecting the Dots: Multivariate TS Forecasting with GNNs", "arXiv:2005.11650"),
    ("F14", "DANN", "Ganin et al. (2016). Domain-Adversarial Training of Neural Networks", "arXiv:1505.07818"),
    ("F15", "CQR", "Romano et al. (2019). Conformalized Quantile Regression", "arXiv:1905.03222"),
    ("F16", "ACI", "Gibbs & Candes (2021). Adaptive Conformal Inference Under Distribution Shift", "arXiv:2106.00170"),
    ("F17", "KAN", "Liu et al. (2024). KAN: Kolmogorov-Arnold Networks", "arXiv:2404.19756"),
    ("F18", "TCN", "Bai et al. (2018). An Empirical Evaluation of Generic Conv. and Recurrent Networks (TCN)", "arXiv:1803.01271"),
    ("F19", "DLinear", "Zeng et al. (2023). Are Transformers Effective for Time Series Forecasting?", "arXiv:2205.13504"),
    ("F20", "TFT", "Lim et al. (2021). Temporal Fusion Transformers", "arXiv:1912.09363"),
    ("F21", "TSLib_Survey", "Wang et al. (2024). Deep Time Series Models: A Comprehensive Survey and Benchmark (TSLib)", "arXiv:2407.13278"),
]


def get(url: str, timeout: int = 60) -> tuple[bytes, str, str]:
    """HTTP GET. Prefers system curl (handles Nature's cookie redirect); falls back to urllib."""
    if shutil.which("curl"):
        with tempfile.TemporaryDirectory() as td:
            body, jar = Path(td) / "b", Path(td) / "c"
            r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), "-A", UA["User-Agent"],
                                "-H", "Accept: " + UA["Accept"], "-c", str(jar), "-b", str(jar),
                                "-o", str(body), "-w", "%{http_code} %{content_type} %{url_effective}", url],
                               capture_output=True, text=True)
            code, _, rest = r.stdout.partition(" ")
            ctype, _, final = rest.partition(" ")
            if not code.startswith("2"):
                raise OSError(f"HTTP {code or r.returncode}")
            return body.read_bytes(), ctype, final
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", ""), r.geturl()


MDPI_JOURNAL = {"en": "energies", "su": "sustainability", "electronics": "electronics"}


def mdpi_cdn(doi: str) -> str | None:
    """10.3390/en19051315 -> https://mdpi-res.com/d_attachment/energies/energies-19-01315/article_deploy/energies-19-01315.pdf"""
    m = re.fullmatch(r"10\.3390/([a-z]+)(\d{2})(\d{2})(\d{4})", doi)
    if not m or m.group(1) not in MDPI_JOURNAL:
        return None
    j, vol, art = MDPI_JOURNAL[m.group(1)], m.group(2), f"{int(m.group(4)):05d}"
    stem = f"{j}-{vol}-{art}"
    return f"https://mdpi-res.com/d_attachment/{j}/{stem}/article_deploy/{stem}.pdf"


def elsevier_pii(doi: str) -> str | None:
    try:
        b, _, _ = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", 30)
        for ln in json.loads(b)["message"].get("link", []):
            m = re.search(r"PII:(S[0-9X]+)", ln.get("URL", ""))
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def is_pdf(b: bytes) -> bool:
    return b[:5] == b"%PDF-"


def s2_lookup(doi: str) -> dict:
    url = ("https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(doi)
           + "?fields=title,externalIds,openAccessPdf")
    try:
        b, _, _ = get(url, 30)
        return json.loads(b)
    except Exception:
        return {}


def candidates(link: str) -> list[tuple[str, str]]:
    kind, ident = link.split(":", 1)
    if kind == "arXiv":
        return [("arxiv", f"https://arxiv.org/pdf/{ident}")]
    doi = ident
    out: list[tuple[str, str]] = []
    if doi.startswith("10.1038/"):  # Nature / Scientific Reports
        out.append(("nature", f"https://www.nature.com/articles/{doi.split('/', 1)[1]}.pdf"))
    if doi.startswith("10.3390/"):  # MDPI: www.mdpi.com blocks scripts (403); its CDN does not
        cdn = mdpi_cdn(doi)
        if cdn:
            out.append(("mdpi-cdn", cdn))
    if doi.startswith("10.1016/"):  # Elsevier: only works for open-access (hybrid OA) articles
        pii = elsevier_pii(doi)
        if pii:
            out.append(("sciencedirect-oa", f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true"))
    if doi.startswith("10.1109/ACCESS"):  # IEEE Access is gold OA: doi -> /document/<arnumber>
        try:
            _, _, landing = get(f"https://doi.org/{doi}", 30)
            m = re.search(r"/document/(\d+)", landing)
            if m:
                out.append(("ieee-oa", f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={m.group(1)}"))
        except Exception:
            pass
    meta = s2_lookup(doi)
    arx = (meta.get("externalIds") or {}).get("ArXiv")
    if arx:
        out.append(("arxiv-preprint", f"https://arxiv.org/pdf/{arx}"))
    oa = (meta.get("openAccessPdf") or {}).get("url")
    if oa:
        out.append(("s2-oa", oa))
    return out


def fetch(entries, outdir: Path) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid, short, cite, link in entries:
        dest = outdir / f"{rid}_{short}.pdf"
        kind, ident = link.split(":", 1)
        url = f"https://arxiv.org/abs/{ident}" if kind == "arXiv" else f"https://doi.org/{ident}"
        row = dict(id=rid, cite=cite, url=url, file="", status="MANUAL", source="")
        if dest.exists() and is_pdf(dest.read_bytes()[:5]):
            row.update(file=dest.name, status="OK", source="cached")
            rows.append(row); print(f"[cached] {rid}"); continue
        for src, cand in candidates(link):
            try:
                b, ctype, _ = get(cand)
                if is_pdf(b):
                    dest.write_bytes(b)
                    row.update(file=dest.name, status="OK", source=src)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"   {rid} {src} failed: {type(e).__name__}: {e}")
            time.sleep(1.0)
        print(f"[{row['status']:6s}] {rid} {short} ({row['source']})")
        rows.append(row)
        time.sleep(1.5)  # be polite to arXiv / publishers
    return rows


def write_readme(path: Path, main: list[dict], found: list[dict]) -> None:
    def table(rows, sub=""):
        lines = ["| ID | Paper | Link | Local PDF |", "|---|---|---|---|"]
        for r in rows:
            f = f"[{r['file']}]({sub}{r['file']})" if r["status"] == "OK" else "**not downloaded (paywalled - get via MIST library)**"
            lines.append(f"| {r['id']} | {r['cite']} | <{r['url']}> | {f} |")
        return "\n".join(lines)

    ok = sum(r["status"] == "OK" for r in main)
    path.write_text(
        "# Reference papers\n\n"
        "Generated by `python tools/fetch_refs.py`. Re-run any time; already-downloaded files are skipped.\n\n"
        f"## R01-R35: papers cited in `Resources/SET_Lab_Dl_Model_Selection.pdf` ({ok}/{len(main)} downloaded)\n\n"
        + table(main) +
        "\n\n## Foundational architecture papers (needed to re-implement the models)\n\n"
        + table(found, "foundational/") + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref")
    ap.add_argument("--only", default="", help="comma-separated IDs, e.g. R06,F01")
    a = ap.parse_args()
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    sel = lambda L: [e for e in L if not only or e[0] in only]  # noqa: E731
    out = Path(a.out)
    main_rows = fetch(sel(REFS), out)
    found_rows = fetch(sel(FOUNDATIONAL), out / "foundational")
    if not only:
        write_readme(out / "README.md", main_rows, found_rows)
        (out / "manifest.json").write_text(json.dumps(main_rows + found_rows, indent=1), encoding="utf-8")
