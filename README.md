# FarmLab: Moringa Ultrasonic Pop Classification System
---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/title-slide.png" alt="MUP1">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/roadmap.png" alt="MUP2">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/research-question-and-hypothesis.png" alt="MUP3">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/experimental-variables-and-design-summary.png" alt="MUP4">

---

```shell
Capture (384 kHz WAV)
-> Band 20-100 kHz + RMS thresholding
-> Spectral features (peak, centroid, bw, rms, dur)
-> Train/test split (grouped by file)
-> Models: LogisticRegression | RandomForest | SVM
-> Metrics: PR/ROC, confusion matrix, demo predictions
```

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/code-review.png" alt="MUP5"> 


```python
# 1) Discover files
wavs = scan("data/*.wav")

# 2) Detect events
events = []
for f in wavs:
    x, sr = load_wav(f)                # 384000 expected
    x = bandlimit(x, sr, 20000, 100000)
    for seg in rms_detector(x, sr, thresh_db=6, min_ms=2, max_ms=50):
        feats = spectral_features(x, sr, seg)  # peak_hz, centroid_hz, bw_hz, rms, dur_ms
        feats["label"] = lookup_label(f, seg)  # from a small seed label file
        feats["file"]  = f
        events.append(feats)

# 3) Train/val/test split by basename
train_df, val_df, test_df = grouped_split(events, group="file")

# 4) Train a baseline model
Xtr, ytr = to_matrix(train_df), train_df["label"]
clf = RandomForestClassifier().fit(Xtr, ytr)

# 5) Evaluate and export demo artifacts
evaluate_and_plot(clf, val_df, test_df)
save_model(clf, "model.joblib")

```

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/ultrasonic-pipeline-overview.png" alt="MUP6">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/data-handling.png" alt="MUP7">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/data-sample.png" alt="MUP8">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/discussion.png" alt="MUP9">

---

<img src="https://github.com/Automated-Aeroponics-Innovations/FarmLab-Product-Development/blob/RES-moringa/conclusion.png" alt="MUP10">

---

## Current Tool Stack >> Hardware
* Prusa Core One
* Bambu Lab H2D
* FormLabs Form 4
* Makera Carvera Air
* 3DPotter PotterBot
* Coming Soon: SnapMaker U1

## Current Tool Stack >> Software
* FreeCAD
* Blender
* Preform
* PrusaSlicer
* Bambu Slicer

---

# Materials Data Sheet

Project scope: 3 identical stations built around the VIVOSUN VGrow Smart Box, each with a Dodotronic Ultramic, a Raspberry Pi 5, and local NVMe storage. Camera monitoring via VIVOSUN GrowCam. Data path for all stations: `/home/griermarkov/nvme0n1/data/rec`.

---

## Bill of Materials — Per Station

| Item                                                                   | Qty | Key Specs and Notes                                                                                                                               | Source                                 |
| ---------------------------------------------------------------------- | --: | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| VIVOSUN VGrow Smart Box + DWC kit                                      |   1 | Smart hydroponic grow box with DWC module. App monitoring, built-in sensors (water level and temperature), 60-day harvest marketing claim.        |                                        |
| VIVOSUN GrowCam Smart Camera                                           |   1 | 2K 4MP, 2.4 GHz Wi-Fi, microSD storage, 2-way audio, night vision, time-lapse, IP66. Use for visual context and time alignment with audio.        |                                        |
| Dodotronic Ultramic 384K EVO (USB)                                     |   1 | USB ultrasound mic. Typical sampling up to 384 kHz (1 ch). Frequency response into ultrasonic band; vendor user guide covers setup and operation. | ([thegrowguys.com.au][1])              |
| Raspberry Pi 5 (16 GB)                                                 |   1 | Quad-core A76, PCIe 2.0 x1 FFC for NVMe HAT, dual 4K HDMI, USB 3.0. Adequate for on-device feature extraction and SVM/RF inference.               | ([Samsung Semiconductor Downloads][2]) |
| GeekPi N04 NVMe M.2 HAT for Pi 5                                       |   1 | NVMe to PCIe adapter for Pi 5 (keys into PCIe FFC). Accepts M-key NVMe SSDs (2280, 2260, 2242, 2230).                                             | ([Raspberry Pi][3])                    |
| Samsung PRO Plus microSDXC 1 TB (OS)                                   |   1 | UHS-I, V30, A2. Use for Raspberry Pi OS 64-bit boot.                                                                                              | ([Amazon][4])                          |
| Samsung 990 PRO 1 TB NVMe (Data)                                       |   1 | PCIe 4.0 x4 drive used at PCIe 2.0 x1 via HAT. High endurance and throughput margin for continuous WAV ingest.                                    |                                        |
| USB-C 5V power supply, 27 W or higher                                  |   1 | Official Pi 5 PSU recommended (USB-C PD). Budget headroom for spikes.                                                                             | ([Samsung Semiconductor Downloads][2]) |
| Short shielded USB cable, USB-A to USB-B or USB-C (per Ultramic model) |   1 | Keep length short to reduce EMI in ultrasonic band.                                                                                               | Vendor docs                            |
| Microbe Life Hydroponics Photosynthesis Plus, 32 oz                    |   1 | Nutrient/biological additive for hydroponics. Follow product dosing and safety.                                                                   | ([52Pi Store][5])                      |

Notes:

* If the Ultramic model you have uses USB-A, use a high-quality shielded USB-A to the Pi’s USB 3.0 port. If it is a USB-C variant, match accordingly. Core usage guidance is in the Dodotronic user guide. ([thegrowguys.com.au][1])
* The VGrow bundle page confirms the DWC kit integration and built-in sensors useful for contextual metadata. 

---

## Totals — 3 Stations

* VGrow Smart Box + DWC: 3
* VIVOSUN GrowCam Smart Camera: 3
* Dodotronic Ultramic 384K EVO: 3
* Raspberry Pi 5 16 GB: 3
* GeekPi N04 NVMe HAT: 3
* Samsung PRO Plus microSD 1 TB: 3
* Samsung 990 PRO 1 TB NVMe: 3
* USB-C 27 W PSU: 3
* Shielded USB mic cables: 3
* Microbe Life Photosynthesis Plus 32 oz: 1 to 3 (usage dependent)

---

## Compatibility and Integration Checks

* Pi 5 PCIe storage: The Pi 5 exposes PCIe 2.0 x1 via FFC. The GeekPi N04 NVMe HAT adapts this for standard M-key NVMe SSDs (2230-2280). Use the 990 PRO 1 TB for data capture. ([Samsung Semiconductor Downloads][2])
* Ultramic capture rate: Plan 384 kHz, mono, 16-bit WAV in 10 s chunks. This mirrors typical bat/ultrasound capture practice and aligns with the Ultramic guide. ([thegrowguys.com.au][1])
* Hydroponic context: The VGrow kit includes water level and temperature sensing available in the app. Record these as metadata alongside audio. 
* Camera: The GrowCam is 2.4 GHz Wi-Fi only. Place the Pi and camera on the same LAN for time alignment. Consider microSD for on-camera storage and export.

---

## Power and Thermal Budget (Per Station, conservative)

* Pi 5 under load: 6 to 12 W typical. Add cooling if continuous 24x7. ([Samsung Semiconductor Downloads][2])
* NVMe SSD: 2 to 5 W typical write. Ensure airflow around the NVMe HAT. ([Raspberry Pi][3])
* Ultramic: USB-powered, low draw (mA range), but keep cable short and away from pump/fan wiring to reduce EMI. ([thegrowguys.com.au][1])

---

## Storage Budget

* WAV capture at 384 kHz, 16-bit, mono: approx 0.73 MB/s. A 1 TB NVMe holds roughly 380 to 420 hours of continuous audio per station depending on overhead and indexing.
* Use directory: `/home/griermarkov/nvme0n1/data/rec` with 10 s chunking and rotation as defined in the ops plan.

---

## Consumables and Maintenances

* Nutrients and water changes per VGrow app guidance. Keep dosing logs in your README experiment journal. 
* Periodically clean reservoir and tubing to minimize pump noise injection into the ultrasonic band.

---

## Optional Accessories

* Small USB fan or heatsink kit for Pi 5 case to avoid thermal throttling during long captures. ([Samsung Semiconductor Downloads][2])
* Acoustic isolation pads to decouple the Ultramic from the Smart Box chassis.

---

## Methodology Alignment

* Feature extraction and modeling will follow deep scattering, MFCC, and basic features feeding SVM/RBF and CNN baselines, consistent with prior plant bioacoustics work. Our scattering and CNN use cases are motivated by the documented accuracy improvements over MFCC and basic features in controlled studies.  
