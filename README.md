# Face Studio — RetinaFace + ArcFace

A local browser webcam app with multi-person recognition, face boxes, detection confidence, cosine similarity, and guided registration at five angles.

## Run

Requires Python 3.10+ and a webcam. On Apple Silicon, use native ARM64 Python.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py
python rebuild_profiles.py  # converts existing saved registration photos, if any
python app.py
```

Open **http://127.0.0.1:5000** and allow camera access. If the port is occupied, use `FACE_STUDIO_PORT=5001 python app.py` and open **http://127.0.0.1:5001**.

1. Enable the camera. The browser requests 4K if available; actual resolution depends on the camera and browser. Camera selection uses the browser's default device.
2. Open **Register face**, enter a name, and begin registration.
3. Face forward, then turn gently left, right, up, and down. The mirrored preview behaves like a mirror; left and right refer to your own shoulders.
4. Hold each accepted pose while eight samples are collected: 40 total.
5. Switch to **Recognition**. Each usable face is compared with all registered people. Matches show names; uncertain faces show **Unknown**.

Poor lighting, blur, small or clipped faces, low detector confidence, extra people, or the wrong pose pause enrollment. Canceling preserves existing people. Each completed enrollment creates an independent numbered folder.

## Models and inference

Browser webcam → JPEG frame (longest side ≤1280px) → Flask → **RetinaFace ResNet34** detection and five landmarks → **112×112 similarity-aligned face crop** → **ArcFace ResNet50 trained on WebFace600K** → normalized **512-dimensional embedding** → cosine comparison → browser overlays.

RetinaFace uses the R34 ONNX checkpoint from [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch), with 640×640 aspect-preserving resize and padding, BGR mean subtraction, anchor decoding and non-maximum suppression. Boxes and landmarks are mapped back to the submitted frame. The recognition checkpoint is `w600k_r50.onnx` from [InsightFace's official buffalo_l release](https://github.com/deepinsight/insightface/releases/tag/v0.7); only that recognition file is extracted. Detection uses RetinaFace, not buffalo_l's SCRFD detector.

Recognition follows the checkpoint's RGB `(pixel − 127.5) / 127.5` preprocessing and five-point alignment. Enrollment samples are individually normalized, averaged, and normalized again to create a reference per registration. Live faces pass size (initially at least 100px per side), visibility, lighting, confidence, coarse yaw, and blur checks before contributing to identity comparison. Calibration captures technically usable crops from 50px per side so stricter selection rules can be evaluated offline. Sharpness is measured on a standardized 128×128 grayscale crop, with an initial minimum Laplacian variance of 20. This avoids applying a fixed edge threshold at different image scales. The warning reports the measured score and required minimum. Set `FACE_STUDIO_BLUR_THRESHOLD` to adjust this heuristic after checking live frames; it is not a calibrated image-quality model. Enrollment keeps its existing quality gates; live settings can be calibrated and may reject distant or moving faces; they need validation at the facility.

Model files total approximately **263 MB**. The downloader pins release URLs, expected sizes, and SHA256 hashes in `models/manifest.json`. First setup downloads the larger buffalo_l archive to extract ArcFace. Downloads are verified before replacement; inference needs no internet. SHA256 values were recorded from the selected upstream release artifacts.

ONNX Runtime uses CPU inference with four intra-operation threads by default. To experiment with CoreML acceleration:

```sh
FACE_STUDIO_PROVIDER=CoreMLExecutionProvider python app.py
```

CoreML is optional, requires a compatible ONNX Runtime build, and can fall back to CPU for unsupported operations. Benchmark both performance and similarity scores before adopting it. This project does not require CUDA, PyTorch, or the InsightFace Python package.

## Threshold and validation

The default cosine threshold is **0.60**, an uncalibrated starting point. SFace's threshold does not transfer automatically to ArcFace. Override it with:

```sh
ARCFACE_MATCH_THRESHOLD=0.65 python app.py
```

Raise it to reject more candidate matches; lower it to accept more, with greater risk of wrong identities. Calibrate using separate genuine and impostor captures from the entrance, then evaluate on further held-out recordings. Cosine similarity ranges from −1 to 1; it is not identity probability. Detection confidence is a separate score.

Registration collects examples for comparison; it does not retrain the recognition model. Pose guidance uses coarse landmark ratios calibrated against the initial front-facing samples, not exact head angles. Use modest turns, not full profiles.

Live recognition now uses Supervision's ByteTrack to associate face boxes, keeps up to 24 recent samples per track, and selects up to 8 quality-ranked views within a 3-second window. The average embedding is weighted by sharpness, size, and a coarse yaw cue. By default, confirmation requires 3 agreeing individual matches, an aggregate similarity of 0.60, and a 0.04 margin over the second candidate. A sudden incompatible embedding clears the history. Evidence expires after 1 second without a new usable sample; gaps longer than 1.5 seconds or frame-size changes reset tracking. These rules reduce flicker but do not guarantee that tracks remain correct when people cross or occlude one another.

Confirmed sightings are recorded once per track in `data/entries.jsonl`; the Recognition tab shows recent sightings. Calibration and registration do not produce sightings. These are face-sighting logs, not directional doorway-crossing events: fragmented tracks or restarting the camera can create duplicates. There is no liveness protection.

## Calibration tab

1. Start the camera and select **Calibration**.
2. Select the registered person who will walk through, or **Unregistered person**. Record one labeled person at a time; multiple visible faces are skipped rather than assigned ambiguous ground truth.
3. Choose **Tuning**, click **Record passage**, walk naturally, and finish. Record at least 2 registered passages and 1 unregistered passage (preferably many more people and repetitions). A passage lasts 2–30 seconds; up to 150 measurements are saved.
4. Click **Compare settings**. The app evaluates 144 combinations of sharpness, face size, coarse yaw, similarity threshold, candidate margin, and required agreeing views. It minimizes total identity errors and misses, preferring fewer wrong identities/false accepts, fewer duplicate tracks, and stricter settings when totals tie.
5. Choose **Validation** and record new passages, again including at least 2 registered and 1 unregistered. Compare again to evaluate the frozen candidate on these fresh tests. Validation never selects the candidate settings. Clear the tests before starting a new tuning round after validation.
6. Review current versus candidate counts, including wrong identities, missed people, accepted unregistered people, track duplicates, passages without face measurements, and time to first correct confirmation. Click **Apply reviewed settings** only when satisfied. Applying is blocked if validation has wrong identities/false accepts or never correctly recognizes registered entrants. Some misses can remain and are shown for review.

The automatic comparison reuses the same quality-weighted decision function as live recognition. Calibration saves 512-dimensional embeddings, timestamps, tracker IDs, and quality metrics in owner-only files under `data/calibration/`, not raw videos or live images. Even crops rejected by the normal blur threshold are measured during calibration, provided they meet basic technical requirements. Empty passages count as missed registered entrants; unregistered passages without usable samples need extra review because their unknown result does not establish recognition discrimination.

The gallery is fingerprinted. Changes to enrolled identities/embeddings invalidate existing tests. Candidate reports are in memory; after a server restart, run Compare again before validation/applying. Applied settings persist in `data/recognition_settings.json`. Environment match/blur values set initial defaults; saved reviewed settings take precedence. Clear calibration tests deletes recordings and the report, preserving registrations, logs, and applied settings.

This first workflow is a single-person pilot. It does not calibrate group crossings, lighting automatically, detector parameters, or ByteTrack parameters. Test crowded entrance behavior separately, and expand testing with gallery size. A few successful passages do not establish a production false-match rate.

## Existing registrations and storage

Every person folder stores 40 JPEG face crops, `metadata.json`, and **`profile_arcface.npz`**. The ArcFace archive contains a checkpoint identifier, 512-dimensional sample embeddings, and an averaged reference. Archives from other models are never compared with ArcFace.

`python rebuild_profiles.py` creates separate ArcFace archives from each folder's 40 saved photos. It requires exactly one detected face in every image and writes a completed profile atomically. Failures are reported per folder; re-register those people. Original JPEGs, metadata, and old SFace `profile.npz` files are preserved. Running the script again skips already converted profiles. Folders without compatible ArcFace archives appear as invalid in the people list until converted or registered again. The legacy global `data/profile.npz` is ignored.

Registration folders are the sole source of identities. Names in metadata take precedence. File changes and deletions are checked on every request; unchanged embeddings are cached. Editing JPEGs does not update an existing ArcFace archive automatically. Archives identify this specific checkpoint; changing recognition weights requires rebuilding enrollment embeddings and using a new model identifier.

The **Recognition** tab lists people with previews, names, sample counts, and folders. **Rename** changes the display name. **Remove** deletes the selected folder, its images, and profiles after confirmation. Invalid folders can also be removed. Registering the same person twice creates two independent entries.

Crops include a 20% margin and preserve the captured pose before alignment. Incomplete registrations remain in memory and are discarded when canceled. Registration files have owner-only permissions and are excluded from Git. Live frames are not recorded. The server binds to loopback and rejects other hostnames and cross-origin requests; keep it local.

## Checks and licensing

```sh
python -m unittest discover -s tests -v
```

Checks cover registration, deletion, multi-person matching, threshold and quality gates, model output decoding, alignment, RGB preprocessing, and blank-frame inference when weights are available.

RetinaFace implementation: [upstream source and MIT license](https://github.com/yakhyo/retinaface-pytorch). ArcFace checkpoint: [InsightFace model licensing](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md). InsightFace's supplied pretrained weights are for non-commercial research; obtain appropriate licensing before commercial deployment. Their code license does not grant commercial rights to the weights.
# pnp_thesis
