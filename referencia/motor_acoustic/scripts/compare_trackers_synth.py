"""Compara ridge harmônico+Viterbi, argmax por quadro e pYIN em sinal sintético.
Fundamental propositalmente fraca (caso de motor). Só para calibrar, não é resultado do artigo."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from motoracoustic.config import PipelineConfig
from motoracoustic.synth import motor_tone, add_noise
from motoracoustic.preprocess import preprocess
from motoracoustic.ridge import track_rotation, track_rotation_framewise
from motoracoustic.baselines import pyin_track
from motoracoustic.evaluate import f0_error_metrics
SR=16000; cfg=PipelineConfig()
f0=np.linspace(35,55,SR*8)
print(f"{'SNR':>5} | {'harm+Viterbi':>14} | {'argmax':>14} | {'pYIN':>14}   (fração dentro de 2% / erro de oitava)")
for snr in (20,6,0,-6):
    v=[];a=[];p=[]
    for seed in range(3):
        x=preprocess(add_noise(motor_tone(f0,SR,rng=seed),snr,rng=seed+50),SR,cfg)
        r=track_rotation(x,SR,cfg); ref=np.interp(r["t"],np.arange(len(f0))/SR,f0)
        v.append(f0_error_metrics(r["f0"],ref)); a.append(f0_error_metrics(track_rotation_framewise(x,SR,cfg),ref))
        pf,_,_=pyin_track(x,SR,cfg); tp=np.arange(len(pf))*cfg.stft_hop/SR
        p.append(f0_error_metrics(pf,np.interp(tp,np.arange(len(f0))/SR,f0)))
    fmt=lambda L:f"{np.mean([m['within_2pct'] for m in L]):.2f} / {np.mean([m['octave_err'] for m in L]):.2f}"
    print(f"{snr:>5} | {fmt(v):>14} | {fmt(a):>14} | {fmt(p):>14}")
