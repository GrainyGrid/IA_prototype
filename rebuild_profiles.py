"""Build separate ArcFace profiles from saved enrollment photos; preserve SFace files."""
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

import app as service
from face_models import EMBEDDING_SIZE, MODEL_ID, PROFILE_NAME


def rebuild_folder(engine, folder):
    target = folder / PROFILE_NAME
    if target.is_symlink():
        raise ValueError("Linked profiles are not supported.")
    if target.exists():
        with np.load(target, allow_pickle=False) as profile:
            if str(profile['model_id'].item()) == MODEL_ID and profile['features'].shape == (service.TOTAL, EMBEDDING_SIZE):
                service.reference_embedding(profile['features'])
                return 'already converted'
        raise ValueError('An incompatible ArcFace archive exists. Preserve it elsewhere before rebuilding.')
    metadata_path = folder / 'metadata.json'
    if metadata_path.is_symlink():
        raise ValueError('Linked metadata is not supported.')
    metadata = json.loads(metadata_path.read_text())
    names = metadata['images']
    name = metadata['name']
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        raise ValueError('Invalid person name.')
    if not isinstance(names, list) or len(names) != service.TOTAL or len(set(names)) != service.TOTAL:
        raise ValueError('Expected 40 distinct enrollment images.')
    features = []
    for filename in names:
        if not isinstance(filename, str) or Path(filename).name != filename or '\\' in filename:
            raise ValueError('Invalid image path.')
        path = folder / filename
        if path.is_symlink():
            raise ValueError('Linked images are not supported.')
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Unreadable image: {filename}')
        _, faces = engine.detector.detect(frame)
        if faces is None or len(faces) != 1:
            raise ValueError(f'{filename}: expected exactly one detected face; register again.')
        features.append(engine.feature(frame, faces[0]))
    features = np.stack(features)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.arcface-', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            np.savez_compressed(output, name=name.strip(), features=features, model_id=MODEL_ID,
                                reference_embedding=service.reference_embedding(features),
                                registration_directory=f'registration/{folder.name}')
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 'converted 40 photos'


def main():
    engine = service.FaceEngine()
    engine.initialize()
    failures = 0
    folders = sorted(service.REGISTRATIONS.iterdir()) if service.REGISTRATIONS.exists() else []
    for folder in folders:
        if not folder.is_dir() or folder.is_symlink() or folder.name.startswith('.'):
            continue
        try:
            print(f'{folder.name}: {rebuild_folder(engine, folder)}', flush=True)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            failures += 1
            print(f'{folder.name}: FAILED — {error}', flush=True)
    return failures


if __name__ == '__main__':
    raise SystemExit(bool(main()))
