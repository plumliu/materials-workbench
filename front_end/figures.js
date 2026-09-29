const $ = id => document.getElementById(id);
let manual = '', items = [], current = null, baseline = '', loaded = false;
let loading = false, saving = false, refreshing = 0, zoom = 1;
const starting = new Set();
const bridge = () => $('wpd').contentWindow?.workbench;
const busy = item => item?.processing?.status === 'running';
function dirty() { return loaded && !loading && !busy(current) && bridge()?.snapshot() !== baseline; }
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw Error((await response.json()).error || response.statusText);
  return response;
}
function query(item = current, extra = {}, name = manual) {
  return new URLSearchParams({manual: name, id: item.id, ...extra});
}
function message(text) { $('editor-message').textContent = text; }
function failure(item) {
  if (!['failed', 'interrupted', 'needs_resolution'].includes(item?.processing?.status)) return '';
  const details = item.processing.issues?.map(issue => issue.message).join('\n') || item.processing.message || item.processing.error || '处理已中断';
  return `智能识别未完成：${details}。可继续手工标注，或重新识别。`;
}
function state() {
  const running = busy(current), changed = dirty();
  $('state').textContent = running ? '智能识别中' : changed ? '有未保存修改' :
    ({unreviewed:'未确认', draft:'草稿', verified:'已确认'}[current?.review_status] || '');
  $('state').className = 'state ' + (!changed ? current?.review_status : '');
  $('save').disabled = $('verify').disabled = !loaded || running || loading || saving;
  $('recognize').disabled = !current?.can_annotate || running || loading || saving;
  $('recognition-cover').hidden = !running;
  $('wpd').inert = running || loading || !loaded;
  $('wpd').tabIndex = running || loading || !loaded ? -1 : 0;
  $('wpd').parentElement.setAttribute('aria-busy', String(running || loading));
  if (running && document.activeElement === $('wpd')) $('wpd').blur();
}
function list() {
  const search = $('search').value.toLowerCase(), filter = $('filter').value;
  $('counts').textContent = `${items.length} 张 · ${items.filter(i => busy(i)).length} 张识别中`;
  $('list').replaceChildren();
  for (const item of items) {
    if (!(item.id + ' ' + item.caption).toLowerCase().includes(search) ||
        filter === 'ready' && (!item.can_annotate || busy(item)) ||
        filter === 'verified' && item.review_status !== 'verified' ||
        filter === 'pending' && (item.ready || busy(item)) ||
        filter === 'running' && !busy(item)) continue;
    const button = document.createElement('button');
    button.className = 'item' + (current?.id === item.id ? ' selected' : '');
    const title = document.createElement('strong');
    title.textContent = item.id; title.title = item.caption;
    const detail = document.createElement('small');
    detail.textContent = `页 ${item.page} · ${busy(item) ? '智能识别中' : !item.can_annotate ? '来源缺失' :
      ({verified:'已确认', draft:'草稿', unreviewed:'可手工标注'}[item.review_status] || '可手工标注')}`;
    button.append(title, detail);
    button.onclick = () => open(item).catch(e => message(e.message));
    $('list').append(button);
  }
}
async function refresh() {
  if (!manual) return;
  const name = manual, ticket = ++refreshing;
  const fresh = await (await api('/api/figures?' + new URLSearchParams({manual: name}))).json();
  if (name !== manual || ticket !== refreshing) return;
  items = fresh;
  const selected = items.find(i => i.id === current?.id);
  if (selected && !loading && !saving && !starting.has(name + "/" + selected.id)) {
    const wasBusy = busy(current);
    if (busy(selected)) {
      current = selected; loaded = false; baseline = '';
    } else if (wasBusy) {
      await open(selected, true);
    } else {
      // Keep the loaded version until save/reopen, so another window cannot be overwritten.
      current.processing = selected.processing;
    }
  }
  list(); state();
}
async function choose(name) {
  if (loading || saving) { $('manual').value = manual; return; }
  if (dirty() && !confirm('当前修改尚未保存，确定切换手册吗？')) { $('manual').value = manual; return; }
  manual = name; current = null; loaded = false; baseline = ''; items = [];
  $('placeholder').hidden = false; state();
  await refresh();
  if (items.length) await open(items[0]);
}
async function waitForBridge() {
  if (bridge()?.ready) return;
  await new Promise((resolve, reject) => {
    const start = Date.now();
    const timer = setInterval(() => {
      if (bridge()?.ready) { clearInterval(timer); resolve(); }
      else if (Date.now() - start > 15000) { clearInterval(timer); reject(Error('WPD 未加载，请刷新后重试')); }
    }, 100);
  });
}
async function open(item, afterRecognition = false) {
  if (loading || saving) return;
  if (!afterRecognition && dirty() && !confirm('当前修改尚未保存，确定切换 Figure 吗？')) return;
  current = item; baseline = ''; loaded = false; loading = true;
  list(); state(); message(failure(item));
  $('title').textContent = item.id + (item.caption ? ' · ' + item.caption.replace(/^Figure\s+[\d.]+\s*/, '') : '');
  $('placeholder').hidden = false;
  $('placeholder').textContent = item.can_annotate ? '正在载入 WPD 项目…' : '本图来源 PDF 缺失，请检查 intake 记录。';
  $('source-label').textContent = '原PDF来源：页' + item.page;
  $('source').src = '/api/source-image?' + query(item, {page: item.page});
  $('pdf').href = '/api/figure-pdf?' + query(item);
  zoom = 1; $('source').style.width = '100%';
  try {
    if (!busy(item) && item.can_annotate) {
      await waitForBridge();
      const route = item.ready ? '/api/figure-tar?' : '/api/figure-pdf?';
      const blob = await (await api(route + query(item))).blob();
      baseline = item.ready ? await bridge().load(blob) : await bridge().loadPDF(blob, item.id + '.pdf');
      loaded = true; $('placeholder').hidden = true;
    }
  } finally { loading = false; state(); }
}
async function save(status) {
  if (!loaded || busy(current) || loading || saving) return;
  saving = true; state(); message("正在保存…");
  const item = current;
  try {
    const snapshot = bridge().snapshot(), blob = await bridge().archive(snapshot);
    current = await (await api('/api/figure-save?' + query(item, {version: JSON.stringify(item.version), status}), {
      method: 'POST', headers: {'Content-Type':'application/x-tar'}, body: blob,
    })).json();
    baseline = snapshot;
    message(status === 'verified' ? '已确认并保存。' : '草稿已保存。');
  } catch (error) { message(error.message); }
  finally { saving = false; state(); await refresh().catch(e => message(e.message)); }
}
$('recognize').onclick = () => {
  $('recognition-target').textContent = current.id;
  $('recognition-dialog').showModal();
};
$('recognition-cancel').onclick = () => $('recognition-dialog').close();
$('recognition-confirm').onclick = async () => {
  const item = current, name = manual, previousBaseline = baseline, previousLoaded = loaded;
  $('recognition-dialog').close();
  const key = name + '/' + item.id;
  starting.add(key);
  current = {...item, processing: {status:'running'}}; loaded = false; state(); message('');
  try {
    const started = await (await api('/api/figure-recognize?' + query(item, {}, name), {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({confirm_reset:true, version:item.version}),
    })).json();
    if (manual === name && current?.id === item.id) { current = started; baseline = ''; state(); }
    await refresh();
  } catch (error) {
    if (manual === name && current?.id === item.id) {
      current = item; baseline = previousBaseline; loaded = previousLoaded; state(); message(error.message);
    }
  } finally { starting.delete(key); }
};
function resize(id,host,min,max){let start=0,width=0;const bar=$(id);const set=v=>{host.style.gridTemplateColumns=Math.min(max(),Math.max(min,v))+'px 8px minmax(0,1fr)';};bar.onpointerdown=e=>{if(e.button!==0)return;start=e.clientX;width=host.children[0].getBoundingClientRect().width;bar.setPointerCapture(e.pointerId);document.body.classList.add('resizing');};bar.onpointermove=e=>{if(bar.hasPointerCapture(e.pointerId))set(width+e.clientX-start);};bar.onpointerup=bar.onpointercancel=e=>{if(bar.hasPointerCapture(e.pointerId))bar.releasePointerCapture(e.pointerId);document.body.classList.remove('resizing');};bar.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();set(host.children[0].getBoundingClientRect().width+(e.key==='ArrowLeft'?-20:20));}};}

resize('sidebar-divider', $('shell'), 150, () => 500);
resize('source-divider', $('panes'), 180, () => Math.max(180, $('panes').clientWidth - 500));
$('toggle-source').onclick = () => {
  const hidden = $('panes').classList.toggle('collapsed');
  $('panes').style.gridTemplateColumns = '';
  $('toggle-source').textContent = hidden ? '显示原 PDF' : '收起原 PDF';
};
$('zoom-out').onclick = () => { zoom = Math.max(.4, zoom - .2); $('source').style.width = zoom * 100 + '%'; };
$('zoom-in').onclick = () => { zoom = Math.min(4, zoom + .2); $('source').style.width = zoom * 100 + '%'; };
$('save').onclick = () => save('draft'); $('verify').onclick = () => save('verified');
$('search').oninput = $('filter').onchange = list;
$('manual').onchange = e => choose(e.target.value).catch(e => message(e.message));
window.onbeforeunload = e => { if (dirty() || saving) { e.preventDefault(); e.returnValue = ''; } };
(async () => {
  const all = await (await api('/api/overview')).json();
  const names = all.filter(i => i.intake).map(i => i.name);
  $('manual').replaceChildren(...names.map(n => new Option(n, n)));
  const selected = new URLSearchParams(location.search).get('manual');
  if (names.length) {
    $('manual').value = names.includes(selected) ? selected : names[0];
    await choose($('manual').value);
  }
})().catch(e => message(e.message));
setInterval(() => { if (!loading && !saving) { state(); refresh().catch(e => message(e.message)); } }, 3000);
