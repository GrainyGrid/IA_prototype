(() => {
  let busy = false, personSignature = '', active = false;
  const buttonIds = ['calibration-start', 'calibration-stop', 'calibration-cancel', 'calibration-analyze', 'calibration-apply', 'calibration-clear'];
  function render(status) {
    const state = status.calibration;
    if (!state) return;
    active = state.active;
    const signature = JSON.stringify(status.people.map(p => [p.id,p.name]));
    if (signature !== personSignature) {
      const selected = $('calibration-person').value;
      $('calibration-person').replaceChildren(new Option('Unregistered person', ''));
      status.people.forEach(p => $('calibration-person').add(new Option(`${p.name} (${p.id})`, p.id)));
      if (status.people.some(p => p.id === selected)) $('calibration-person').value = selected;
      personSignature = signature;
    }
    const report = state.report;
    $('calibration-person').disabled = active || busy;
    $('calibration-split').disabled = active || busy;
    $('calibration-start').hidden = active;
    $('calibration-stop').hidden = !active;
    $('calibration-cancel').hidden = !active;
    $('calibration-start').disabled = busy || !running || status.registering || !status.people.length;
    $('calibration-stop').disabled = busy;
    $('calibration-cancel').disabled = busy;
    $('calibration-analyze').disabled = active || busy;
    $('calibration-clear').disabled = busy;
    $('calibration-apply').disabled = busy || active || !report?.can_apply || report.applied || report.validation.wrong > 0 || report.validation.false_accepts > 0 || (report?.can_apply && report.validation.missed >= report.validation.known);
    $('calibration-progress').textContent = active ?
      `Recording: ${state.elapsed}s · ${state.frames} face measurements · ${state.skipped} skipped frames. Walk naturally, then finish. Limit: 30 seconds.` :
      `${state.takes.length} saved passages. ${report?.applied ? 'Reviewed settings are applied.' : report ? 'Record fresh validation passages, then compare again.' : 'Collect tuning passages, then compare settings.'}`;
    $('calibration-takes').replaceChildren();
    for (const take of state.takes) {
      const row = document.createElement('div');
      const person = status.people.find(p => p.id === take.expected);
      row.textContent = `${take.split === 'tuning' ? 'Tuning' : 'Validation'} · ${person?.name || (take.expected ? take.expected : 'Unregistered')} · ${take.samples} samples`;
      $('calibration-takes').append(row);
    }
    const target = $('calibration-report'); target.replaceChildren();
    if (!report) return;
    const heading = document.createElement('h3'); heading.textContent = `Recommended settings (${report.candidate_count} combinations compared)`;
    const settings = document.createElement('div'); settings.className = 'settings';
    const s = report.settings;
    settings.textContent = `Sharpness ≥ ${s.sharpness}; face size ≥ ${s.min_size}px; pose cue ≤ ${s.max_yaw}; similarity ≥ ${s.threshold}; candidate separation ≥ ${s.margin}; at least ${s.min_samples} agreeing views.`;
    target.append(heading, settings);
    for (const split of ['tuning','validation']) {
      const title = document.createElement('h3'); title.textContent = split === 'tuning' ? 'Tuning results' : 'Fresh validation results';
      target.append(title);
      const candidate = report[split], baseline = report[`baseline_${split}`];
      const table = document.createElement('table');
      for (const values of [['Per passage','Current','Candidate'], ['Correct',baseline.correct,candidate.correct], ['No face samples',baseline.no_samples,candidate.no_samples],
        ['Wrong identity',baseline.wrong,candidate.wrong], ['Missed registered',baseline.missed,candidate.missed],
        ['Unregistered accepted',baseline.false_accepts,candidate.false_accepts], ['Duplicate tracks',baseline.duplicate_tracks,candidate.duplicate_tracks],
        ['Average delay',baseline.mean_delay == null ? '—' : `${baseline.mean_delay}s`,candidate.mean_delay == null ? '—' : `${candidate.mean_delay}s`]]) {
        const row = document.createElement('tr');
        values.forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }); table.append(row);
      }
      target.append(table);
      const count = document.createElement('p'); count.textContent = `${candidate.known} registered and ${candidate.unregistered} unregistered passages. Wrong and missed counts can overlap.`; target.append(count);
    }
  }
  async function action(name, body = {}) {
    if (busy) return;
    busy = true; buttonIds.forEach(id => $(id).disabled = true);
    try {
      const status = await api(`/api/calibration/${name}`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      update(status);
      message(name === 'apply' ? 'Reviewed settings applied and saved.' : name === 'start' ? 'Walk one labeled person through the camera view.' : 'Calibration updated.');
    } catch (error) { message(error.message); }
    finally { busy = false; if (window.faceStudioStatus) render(window.faceStudioStatus); }
  }
  $('tab-calibration').onclick = () => mode('calibration');
  $('calibration-start').onclick = () => action('start', {person_id:$('calibration-person').value || null, split:$('calibration-split').value});
  $('calibration-stop').onclick = () => action('stop');
  $('calibration-cancel').onclick = () => action('cancel');
  $('calibration-analyze').onclick = () => action('analyze');
  $('calibration-apply').onclick = () => action('apply');
  $('calibration-clear').onclick = () => {
    if (confirm('Delete saved calibration tests and their measurements? Registered people, entry logs, and applied settings are kept.')) action('clear');
  };
  document.addEventListener('face-studio-status', event => render(event.detail));
  if (window.faceStudioStatus) render(window.faceStudioStatus);
  async function entries() {
    try {
      const result = await api('/api/entries');
      const target = $('entry-list'); target.replaceChildren();
      if (!result.entries.length) target.textContent = 'No confirmed sightings yet.';
      result.entries.slice(-10).reverse().forEach(entry => {
        const row = document.createElement('div');
        row.textContent = `${entry.name} · ${new Date(entry.timestamp).toLocaleTimeString()}`; target.append(row);
      });
    } catch (error) { $('entry-list').textContent = error.message; }
  }
  entries(); setInterval(entries, 3000);
})();
