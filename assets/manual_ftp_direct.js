(function () {
  function $(id) { return document.getElementById(id); }
  function val(id) { const el = $(id); return el ? (el.value || '').trim() : ''; }
  function setVal(id, v) { const el = $(id); if (el && v) { el.value = v; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); } }
  function status(msg, ok) {
    const el = $('domestic-manual-ftp-inline-status');
    if (el) {
      el.textContent = msg;
      el.style.color = ok === false ? '#b91c1c' : '#0f5132';
      el.style.fontWeight = '700';
    }
    const progress = $('domestic-ftp-progress-text');
    if (progress && /启动|开始|连接|校验/.test(msg)) progress.textContent = '下载状态：' + msg;
  }
  function parseFtpText(text) {
    text = (text || '').replace(/复制|Copy|COPY/g, ' ').replace(/[\t\r]+/g, ' ');
    const compact = text.replace(/\s+/g, ' ').trim();
    const hosts = [];
    const seen = new Set();
    const hostRe = /(?:ftp:\/\/)?(ftp[0-9A-Za-z.-]*\.tpdc\.ac\.cn)/ig;
    let m;
    while ((m = hostRe.exec(compact)) !== null) {
      const h = m[1].replace(/[，,。;；:：\/]+$/g, '');
      if (h && !seen.has(h.toLowerCase())) { hosts.push(h); seen.add(h.toLowerCase()); }
    }
    const port = (compact.match(/(?:端\s*口|port)\s*[:：]?\s*(\d{2,5})/i) || compact.match(/\b(6201|21)\b/i) || [,''])[1];
    const username = (compact.match(/(?:用\s*户\s*名|用户|账号|user(?:name)?)\s*[:：]?\s*([A-Za-z0-9_@.\-]+)/i) || compact.match(/\b(download_[0-9A-Za-z_\-]+)\b/i) || [,''])[1];
    const password = (compact.match(/(?:密\s*码|password|pass)\s*[:：]?\s*([A-Za-z0-9_@.\-]+)/i) || [,''])[1];
    return {hosts, port, username, password};
  }
  function applyBulkText() {
    const text = val('domestic-ftp-bulk-text');
    if (!text) return;
    const p = parseFtpText(text);
    if (p.hosts[0]) setVal('domestic-ftp-host-input', p.hosts[0]);
    if (p.hosts[1]) setVal('domestic-ftp-backup-host-input', p.hosts[1]);
    if (p.port) setVal('domestic-ftp-port-input', p.port);
    if (p.username) setVal('domestic-ftp-username-input', p.username);
    if (p.password) setVal('domestic-ftp-password-input', p.password);
    if (p.hosts.length || p.username || p.password) status('已从整段文本解析 FTP 信息，请核对后点击开始下载。', true);
  }
  document.addEventListener('input', function (ev) {
    if (ev.target && ev.target.id === 'domestic-ftp-bulk-text') applyBulkText();
  }, true);
  document.addEventListener('paste', function (ev) {
    if (ev.target && ev.target.id === 'domestic-ftp-bulk-text') setTimeout(applyBulkText, 80);
  }, true);
  document.addEventListener('click', async function (ev) {
    const target = ev.target && ev.target.closest ? ev.target.closest('#domestic-manual-ftp-download-btn') : null;
    if (!target) return;
    // Use the direct backend route and stop Dash's delegated listener from creating duplicate processes.
    ev.preventDefault();
    ev.stopPropagation();
    if (ev.stopImmediatePropagation) ev.stopImmediatePropagation();
    try {
      applyBulkText();
      const state = $('domestic-download-action-state');
      const manifest = state && state.dataset ? (state.dataset.manifestPath || '') : '';
      const payload = {
        manifest_path: manifest,
        host: val('domestic-ftp-host-input'),
        backup_host: val('domestic-ftp-backup-host-input'),
        port: val('domestic-ftp-port-input'),
        username: val('domestic-ftp-username-input'),
        password: val('domestic-ftp-password-input'),
        save_dir: val('domestic-ftp-save-dir-input')
      };
      if (!payload.host || !payload.port || !payload.username || !payload.password) {
        status('缺少 FTP 参数：请填写主机、端口、用户名、密码，或先粘贴整段 FTP 账号。', false);
        return;
      }
      target.disabled = true;
      const oldText = target.textContent;
      target.textContent = '正在启动FTP下载...';
      status('已点击开始 FTP 下载：正在请求后端启动 Python 下载进程...', true);
      const res = await fetch('/__manual_ftp_download', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        status('FTP 下载启动失败：' + (data.error || res.statusText || '未知错误'), false);
      } else {
        status('✅ FTP 下载已启动。PID：' + data.pid + '；主机：' + data.primary_host + '；保存目录：' + data.download_dir, true);
      }
      target.disabled = false;
      target.textContent = oldText || '开始 FTP 下载';
    } catch (e) {
      status('FTP 下载启动异常：' + e, false);
      try { target.disabled = false; target.textContent = '开始 FTP 下载'; } catch (_) {}
    }
  }, true);
})();
