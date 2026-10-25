const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), capture = document.createElement('canvas');
let stream = null, running = false, generation = 0, registration = false, controller = null;
let lastTime = 0, lastCount = 0;
let peopleSignature = '';
const stages = ['front', 'left', 'right', 'up', 'down'];
const prompts = { front: 'Look straight at the camera.', left: 'Turn slightly toward your left shoulder.', right: 'Turn slightly toward your right shoulder.', up: 'Lift your chin slightly.', down: 'Lower your chin slightly.' };

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type");

  let result;
  if (contentType && contentType.includes("application/json")) {
    result = await response.json();
  } else {
    const text = await response.text();
    // This will print the actual HTML error snippet or route failure in your UI message box
    throw new Error(`Server error (${response.status}): Check terminal logs.`);
  }

  if (!response.ok) throw new Error(result.error || 'Request failed.');
  return result;
}
function message(text) { $('message').textContent = text; }
function mode(register) {
  if (window.calibrationActive) { message('Finish or discard the calibration recording before switching tabs.'); return; }
  const calibrate = register === 'calibration';
  register = register === true;
  $('tab-calibration').setAttribute('aria-selected', String(calibrate));
  $('calibration-panel').hidden = !calibrate;
  $('tab-live').setAttribute('aria-selected', String(!register && !calibrate));
  $('tab-register').setAttribute('aria-selected', String(register));
  $('live-panel').hidden = register || calibrate;
  $('register-panel').hidden = !register;
}
function renderPeople(status) {
  const people = [...status.people, ...status.invalid_profiles];
  const availableTags = status.ble_available_tags || ['alice', 'bob', 'charlie'];
  const signature = JSON.stringify({ people, availableTags });
  if (signature === peopleSignature) return;
  peopleSignature = signature;
  $('people-list').replaceChildren();
  for (const person of people) {
    const row = document.createElement('div'); row.className = 'person-row';
    const header = document.createElement('div'); header.className = 'person-header';
    const image = document.createElement('img'); image.className = 'person-photo'; image.alt = `${person.name} registration preview`;
    image.src = `/api/people/${encodeURIComponent(person.id)}/thumbnail`;
    image.onerror = () => { image.hidden = true; };
    const details = document.createElement('div'); details.className = 'person-details';
    const name = document.createElement('strong'); name.textContent = person.name;
    const folder = document.createElement('small'); folder.textContent = person.directory;
    const tagText = person.ble_tag_id ? ` · BLE: TAG-${person.ble_tag_id}` : ' · Face only';
    const info = document.createElement('small'); info.textContent = person.error || `${person.sample_count} samples${tagText}`;
    if (person.error) info.className = 'danger';
    details.append(name, folder, info); header.append(image, details); row.append(header);

    // BLE Tag Pairing Dropdown per registered person
    if (!person.error) {
      const bleRow = document.createElement('div');
      bleRow.style.marginTop = '8px';
      const bleSelect = document.createElement('select');
      bleSelect.style.marginBottom = '4px';
      bleSelect.style.padding = '6px 8px';
      bleSelect.style.fontSize = '11px';
      bleSelect.setAttribute('aria-label', `BLE tag for ${person.name}`);
      bleSelect.add(new Option('2FA: None (Face only)', ''));
      for (const tag of availableTags) {
        bleSelect.add(new Option(`2FA: TAG-${tag}`, tag));
      }
      bleSelect.value = person.ble_tag_id || '';
      bleSelect.onchange = async () => {
        bleSelect.disabled = true;
        const chosen = bleSelect.value || null;
        try {
          const updated = await api(`/api/people/${encodeURIComponent(person.id)}/ble`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ble_tag_id: chosen }),
          });
          update(updated);
          message(chosen ? `Paired TAG-${chosen} to ${person.name}.` : `Removed BLE tag from ${person.name}.`);
        } catch (error) {
          message(error.message);
        } finally {
          bleSelect.disabled = false;
        }
      };
      bleRow.append(bleSelect);
      row.append(bleRow);
    }

    const actions = document.createElement('div'); actions.className = 'person-actions';
    const rename = document.createElement('button'); rename.className = 'text-button'; rename.textContent = 'Rename';
    const remove = document.createElement('button'); remove.className = 'text-button danger'; remove.textContent = 'Remove';
    rename.setAttribute('aria-label', `Rename ${person.name} (${person.id})`);
    remove.setAttribute('aria-label', `Remove ${person.name} (${person.id})`);
    const edit = document.createElement('form'); edit.className = 'person-edit'; edit.hidden = true;
    const input = document.createElement('input'); input.value = person.name; input.maxLength = 60; input.required = true;
    input.setAttribute('aria-label', `New name for ${person.id}`);
    const save = document.createElement('button'); save.className = 'text-button'; save.textContent = 'Save'; save.type = 'submit';
    const cancelEdit = document.createElement('button'); cancelEdit.className = 'text-button'; cancelEdit.textContent = 'Cancel'; cancelEdit.type = 'button';
    cancelEdit.onclick = () => { edit.hidden = true; actions.hidden = false; };
    rename.onclick = () => { edit.hidden = false; actions.hidden = true; input.focus(); };
    edit.onsubmit = async event => {
      event.preventDefault(); save.disabled = true;
      try {
        update(await api(`/api/people/${encodeURIComponent(person.id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: input.value }) }));
        edit.hidden = true; actions.hidden = false; message('Person name updated.');
      } catch (error) { message(error.message); }
      finally { save.disabled = false; }
    };
    remove.onclick = async () => {
      if (!confirm(`Permanently remove ${person.name} (${person.id})? This deletes the registration folder, its images, and embeddings.`)) return;
      remove.disabled = true;
      try { update(await api(`/api/people/${encodeURIComponent(person.id)}`, { method: 'DELETE' })); message('Registration removed from recognition.'); }
      catch (error) { message(error.message); remove.disabled = false; }
    };
    edit.append(input, save, cancelEdit); actions.append(rename, remove); row.append(actions, edit);
    $('people-list').append(row);
  }
}
function update(status) {
  window.faceStudioStatus = status;
  window.calibrationActive = Boolean(status.calibration?.active);
  document.dispatchEvent(new CustomEvent('face-studio-status', { detail: status }));
  $('match-threshold').textContent = status.threshold.toFixed(3);
  registration = status.registering;
  $('profile-name').textContent = status.person_count ? `${status.person_count} registered ${status.person_count === 1 ? 'person' : 'people'}` : 'No people registered';
  $('profile-description').textContent = 'Every completed registration folder is used for live recognition. Manage names, BLE tags, and registrations below.';
  $('profile-status').textContent = status.registered ? 'Ready to recognize' : 'Not registered';
  if ($('ble-status')) {
    $('ble-status').textContent = !status.ble_enabled ? 'Disabled (BLE_DISABLED=1)' : (status.ble_last_scan ? 'Scanning active' : 'Starting scanner…');
  }
  if ($('ble-nearby')) {
    const visible = status.ble_visible_tags || [];
    $('ble-nearby').textContent = visible.length
      ? visible.map(t => `TAG-${t.tag} (${t.rssi} dBm)`).join(', ')
      : 'None in range';
  }

  if ($('ble-tag') && $('ble-detected-hint')) {
    const bleSelect = $('ble-tag');
    const visible = status.ble_visible_tags || [];
    const currentVal = bleSelect.value;

    // Build fresh options list: Default "None" + all currently broadcasting tags
    const options = [{ value: '', text: 'No BLE tag (Face only)' }];
    visible.forEach(t => {
      options.push({ value: t.tag, text: `TAG-${t.tag} (Signal: ${t.rssi} dBm)` });
    });

    // If a user selected a tag that is temporarily out of range, keep it visible in the dropdown
    if (currentVal && !options.some(o => o.value === currentVal)) {
      options.push({ value: currentVal, text: `TAG-${currentVal} (Previously selected)` });
    }

    // Only update the dropdown DOM if the list of available tags changed (prevents resetting user clicks)
    const existingValues = Array.from(bleSelect.options).map(o => o.value).join(',');
    const newValues = options.map(o => o.value).join(',');
    if (existingValues !== newValues) {
      bleSelect.replaceChildren();
      options.forEach(opt => {
        const el = new Option(opt.text, opt.value);
        bleSelect.add(el);
      });
      bleSelect.value = currentVal;
    }

    // Update hint text
    if (visible.length > 0) {
      $('ble-detected-hint').textContent = `${visible.length} tag(s) found nearby. Click to select your device.`;
    } else {
      $('ble-detected-hint').textContent = 'No ESP32 tags detected nearby. Power on your tag to see it appear.';
    }
  }

  $('stored-samples').textContent = status.people.reduce((total, person) => total + person.sample_count, 0);
  renderPeople(status);
  const count = status.registering ? status.count : (status.completed || lastCount === status.total ? status.total : status.count);
  lastCount = count;
  $('progress').value = count;
  $('progress-text').textContent = `${count} / ${status.total}`;
  $('progress-title').textContent = count === status.total ? 'Registration complete' : 'Registration progress';
  $('name').disabled = registration;
  if ($('ble-tag')) $('ble-tag').disabled = registration;
  $('begin').hidden = registration; $('begin').disabled = !stream;
  $('cancel').hidden = !registration;
  $('mode-label').textContent = registration ? 'REGISTRATION MODE' : status.registered ? 'RECOGNITION MODE' : 'DETECTION MODE';
  document.querySelectorAll('[data-stage]').forEach((element, index) => {
    const n = Math.max(0, Math.min(8, count - index * 8));
    element.querySelector('small').textContent = `${n} / 8`;
    element.classList.toggle('done', n === 8);
    element.classList.toggle('active', registration && status.stage === element.dataset.stage);
  });
  if (status.completed) $('instruction').textContent = `All five angles captured. Saved to ${status.registration_directory}. Switch to Recognition to see your live match.`;
  else if (registration) $('instruction').textContent = `${prompts[status.stage]} ${status.message || ''}`;
  if (status.warning) message(status.warning);
}
function draw(result) {
  overlay.width = result.width; overlay.height = result.height;
  const context = overlay.getContext('2d');
  context.clearRect(0, 0, overlay.width, overlay.height);
  const renderScale = Math.min(overlay.clientWidth / result.width, overlay.clientHeight / result.height) || 1;
  const unit = 1 / renderScale;
  const padding = 8 * unit, lineHeight = 21 * unit;
  const maxTextWidth = Math.max(1, result.width - 2 * padding);
  context.font = `${16 * unit}px sans-serif`;
  context.textBaseline = 'top';
  const wrapText = text => {
    const lines = [];
    let line = '';
    for (const character of text) {
      if (line && context.measureText(line + character).width > maxTextWidth) {
        lines.push(line.trimEnd()); line = '';
      }
      line += character;
    }
    if (line) lines.push(line.trimEnd());
    return lines;
  };
  const peopleById = Object.fromEntries((result.people || []).map(p => [p.id, p]));
  for (const face of result.faces) {
    const [x, y, w, h] = face.box;
    const left = result.width - x - w;
    const personRecord = face.person_id ? peopleById[face.person_id] : null;
    const assignedTag = personRecord?.ble_tag_id;

    let color = '#efd597'; // Default yellow (unknown / unconfirmed)
    let authLine = null;
    if (face.matched) {
      if (assignedTag) {
        if (face.ble_verified) {
          color = '#abe195'; // Green: Face + BLE Tag verified
          authLine = `QUALIFIED: ${face.label} (TAG-${assignedTag} OK)`;
        } else {
          color = '#f29b9b'; // Red: Face matched, but required BLE tag missing
          authLine = `DENIED: face ok, TAG-${assignedTag} missing`;
        }
      } else {
        color = '#abe195'; // Face-only user matched
        authLine = `QUALIFIED: ${face.label} (Face only)`;
      }
    }

    context.strokeStyle = color; context.lineWidth = 2 * unit;
    context.strokeRect(left, y, w, h);
    context.fillStyle = color;
    for (const [lx, ly] of face.landmarks) { context.beginPath(); context.arc(result.width - lx, ly, 2 * unit, 0, Math.PI * 2); context.fill(); }
    const similarity = face.similarity == null ? '' : ` · similarity ${face.similarity.toFixed(3)}`;
    const lines = [
      ...wrapText(face.label + (face.track_id != null ? ` · track ${face.track_id} · ${face.sample_count} views` : '')),
      ...(authLine ? wrapText(authLine) : []),
      ...wrapText(`Face confidence ${(face.confidence * 100).toFixed(1)}%${similarity}`),
      ...(face.quality_warning ? wrapText(face.quality_warning) : []),
    ];
    const textWidth = Math.min(result.width, Math.max(...lines.map(line => context.measureText(line).width)) + 2 * padding);
    const textHeight = lines.length * lineHeight + 2 * padding;
    const tx = Math.max(0, Math.min(left, result.width - textWidth));
    const preferredY = y >= textHeight + 4 * unit ? y - textHeight - 4 * unit : y + h + 4 * unit;
    const ty = Math.max(0, Math.min(preferredY, result.height - textHeight));
    context.fillStyle = '#17281ef2'; context.fillRect(tx, ty, textWidth, textHeight);
    context.fillStyle = color;
    lines.forEach((line, index) => context.fillText(line, tx + padding, ty + padding + index * lineHeight));
  }
  $('face-count').textContent = result.faces.length;
  const primary = [...result.faces].sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3])[0];
  $('confidence').textContent = primary ? `${(primary.confidence * 100).toFixed(1)}%` : '—';
  $('similarity').textContent = primary?.similarity != null ? primary.similarity.toFixed(3) : '—';
  const now = performance.now();
  if (lastTime) {
    const currentFps = 1000 / (now - lastTime);
    window.fpsSmoothed = window.fpsSmoothed ? window.fpsSmoothed * 0.7 + currentFps * 0.3 : currentFps;
    $('fps').textContent = `${Math.round(window.fpsSmoothed)} FPS`;
  } else {
    $('fps').textContent = '— FPS';
  }
  lastTime = now;
  message(result.message); update(result);
}
async function loop(token) {
  if (!running || token !== generation) return;
  try {
    if (video.readyState >= 2) {
      const scale = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight));
      const width = Math.round(video.videoWidth * scale), height = Math.round(video.videoHeight * scale);
      capture.width = width; capture.height = height;
      capture.getContext('2d').drawImage(video, 0, 0, width, height);
      const blob = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', 0.90));
      if (!running || token !== generation) return;
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 15000);
      let result;
      try { result = await api('/api/frame', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob, signal: controller.signal }); }
      finally { clearTimeout(timeout); }
      if (running && token === generation) draw(result);
    }
  } catch (error) {
    if (token !== generation) return;
    message(error.name === 'AbortError' ? 'Processing timed out. Stop and restart the camera.' : error.message);
    running = false;
    return;
  }
  if (running && token === generation) setTimeout(() => loop(token), 0);
}
async function startCamera() {
  if (stream) return;
  $('start').disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs localhost or HTTPS. Open http://127.0.0.1:5000.');
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 }, facingMode: 'user' }, audio: false });
    video.srcObject = stream; await video.play();
    update(await api('/api/video/reset', { method: 'POST' }));
    stream.getVideoTracks()[0].addEventListener('ended', () => stopCamera());
    $('empty').hidden = true; $('caption').hidden = false;
    $('live-dot').classList.add('on'); $('stop').disabled = false; $('begin').disabled = false;
    $('resolution').textContent = `${video.videoWidth} × ${video.videoHeight}`;
    running = true; lastTime = 0; window.fpsSmoothed = 0; loop(++generation);
  } catch (error) {
    await stopCamera();
    message(error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser, then try again.' : error.name === 'NotFoundError' ? 'No camera found. Connect a webcam and try again.' : error.message);
  } finally { $('start').disabled = false; }
}
async function stopCamera() {
  running = false; ++generation; controller?.abort();
  stream?.getTracks().forEach(track => track.stop()); stream = null; video.srcObject = null;
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
  $('empty').hidden = false; $('caption').hidden = true; $('live-dot').classList.remove('on'); $('stop').disabled = true; $('begin').disabled = true; $('resolution').textContent = 'CAMERA OFF';
  for (const id of ['face-count', 'confidence', 'similarity']) $(id).textContent = '—';
  try { if (registration) update(await api('/api/cancel', { method: 'POST' })); update(await api('/api/video/reset', { method: 'POST' })); }
  catch (error) { message(error.message); return; }
  message('Camera stopped.');
}
$('start').onclick = startCamera; $('stop').onclick = stopCamera;
$('tab-live').onclick = () => mode(false); $('tab-register').onclick = $('go-register').onclick = () => mode(true); $('begin').onclick = async () => {
  if (!stream || !running) { message('Start or restart the camera before registering.'); return; }
  if (!$('name').value.trim()) { $('name').focus(); message('Enter your name first.'); return; } $('begin').disabled = true;
  const bleTag = $('ble-tag') ? ($('ble-tag').value || null) : null;
  try {
    update(await api('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: $('name').value, ble_tag_id: bleTag }),
    }));
    $('instruction').textContent = prompts.front;
  }
  catch (error) { message(error.message); $('begin').disabled = false; }
};
$('cancel').onclick = async () => { try { update(await api('/api/cancel', { method: 'POST' })); $('instruction').textContent = 'Registration canceled. Existing people are unchanged.'; } catch (error) { message(error.message); } };
$('refresh-people').onclick = async () => { try { update(await api('/api/status')); message('People list refreshed.'); } catch (error) { message(error.message); } };
setInterval(() => { if (!running) api('/api/status').then(update).catch(() => { }); }, 2000);
window.addEventListener('pagehide', () => { running = false; controller?.abort(); stream?.getTracks().forEach(track => track.stop()); });
api('/api/status').then(update).catch(error => message(error.message));