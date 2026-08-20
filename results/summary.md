# Summary: what the data actually shows

## Q1 -- Does Ps increase through Grover amplification?
- Qubit (N=8): ideal Ps = 0.9453 vs classical baseline 1/N = 0.1250 -> amplification confirmed.
- Qutrit (N=9): ideal Ps = 0.9836 vs classical baseline 1/N = 0.1111 -> amplification confirmed.

## Q2 -- Does the qutrit provide a larger search space per physical carrier?
- At the matched register sizes used throughout (qubit m=3, qutrit m=2): N_qutrit / N_qubit = 1.125 (9 vs 8) using ONE FEWER physical qudit.

## Q3 -- Does the larger Hilbert space remain advantageous after decoherence?
- Ideal-case gap (qutrit - qubit): +0.0383
- Noisy-case gap at T1=50.0us, Tphi=20.0us: +0.0348
- No sign change detected across the matched T1=Tphi sweep (10-160us): qutrit ahead at every point tested in this range.