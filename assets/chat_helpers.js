window.dash_clientside = Object.assign({}, window.dash_clientside, {
  chatHelpers: {
    noop: function(x) { return x; }
  }
});

(function () {
  const clientKey = 'som_agent_client_id';
  let hbTimer = null;
  let chatPinnedToBottom = true;
  let lastChatScrollHeight = 0;

  function clientId() {
    let cid = window.sessionStorage.getItem(clientKey);
    if (!cid) {
      cid = 'client-' + Math.random().toString(36).slice(2) + '-' + Date.now();
      window.sessionStorage.setItem(clientKey, cid);
    }
    return cid;
  }

  function currentSessionId() {
    const meta = document.getElementById('session-meta');
    if (!meta) return null;
    return meta.dataset.sessionId || null;
  }

  function postJson(url, body, useBeacon) {
    const payload = JSON.stringify(body || {});
    if (useBeacon && navigator.sendBeacon) {
      try {
        navigator.sendBeacon(url, new Blob([payload], { type: 'application/json' }));
        return;
      } catch (e) {}
    }
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
      keepalive: true
    }).catch(function () {});
  }

  function sendHeartbeat() {
    postJson('/__heartbeat', { client_id: clientId(), session_id: currentSessionId(), ts: Date.now() }, false);
  }

  function announcePageClose() {
    // 不再使用前端显式 shutdown_intent 立即终止任务。
    // 页面真正关闭后，heartbeat 自然中断，由后端 heartbeat_timeout 统一回收。
  }

  function bindChatEnter() {
    const textarea = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    if (!textarea || !sendBtn || textarea.dataset.boundEnter === '1') return;
    textarea.dataset.boundEnter = '1';
    textarea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
      }
    });
  }

  function chatBox() {
    return document.getElementById('chat-history');
  }

  function isNearBottom(box) {
    if (!box) return true;
    return (box.scrollHeight - box.scrollTop - box.clientHeight) < 48;
  }

  function bindChatScroll() {
    const box = chatBox();
    if (!box || box.dataset.boundScroll === '1') return;
    box.dataset.boundScroll = '1';
    box.addEventListener('scroll', function () {
      chatPinnedToBottom = isNearBottom(box);
      lastChatScrollHeight = box.scrollHeight;
    });
  }

  function scrollChatToBottom(force) {
    const box = chatBox();
    if (!box) return;
    if (force || chatPinnedToBottom) {
      box.scrollTop = box.scrollHeight;
      chatPinnedToBottom = true;
    }
    lastChatScrollHeight = box.scrollHeight;
  }

  function maybeAutoScroll(force) {
    const box = chatBox();
    if (!box) return;
    bindChatScroll();
    const grew = box.scrollHeight > (lastChatScrollHeight + 2);
    if (force || (grew && chatPinnedToBottom)) {
      scrollChatToBottom(true);
      return;
    }
    lastChatScrollHeight = box.scrollHeight;
  }

  function editorInput() {
    return document.getElementById('layer-edit-event');
  }

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function showSelectedState(layer, selected) {
    const handles = layer.querySelectorAll('.resize-handle');
    if (selected) {
      layer.style.border = '1px solid rgba(37,99,235,0.95)';
      layer.style.boxShadow = '0 0 0 2px rgba(37,99,235,0.18)';
      handles.forEach(function (h) { h.style.opacity = '1'; });
    } else {
      layer.style.border = '1px solid transparent';
      layer.style.boxShadow = 'none';
      handles.forEach(function (h) { h.style.opacity = '0'; });
    }
  }

  function selectOverlayItem(layer) {
    document.querySelectorAll('.overlay-item').forEach(function (el) {
      el.classList.remove('overlay-selected');
      showSelectedState(el, false);
    });
    if (!layer) return;
    layer.classList.add('overlay-selected');
    showSelectedState(layer, true);
  }

  function emitLayerState(layer) {
    const sink = editorInput();
    if (!layer || !sink) return;
    const payload = {
      target: layer.dataset.target,
      layer: layer.dataset.layer,
      left_pct: clamp(parseFloat(layer.style.left || '0') / 100, 0, 1),
      top_pct: clamp(parseFloat(layer.style.top || '0') / 100, 0, 1),
      width_pct: clamp(parseFloat(layer.style.width || '0') / 100, 0.04, 1),
      height_pct: clamp(parseFloat(layer.style.height || '0') / 100, 0.04, 1)
    };
    sink.value = JSON.stringify(payload);
    sink.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function bindOverlayItem(layer) {
    if (!layer || layer.dataset.editorBound === '1') return;
    layer.dataset.editorBound = '1';

    let mode = null;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;
    let startWidth = 0;
    let startHeight = 0;
    let resizeDir = '';

    function canvasMetrics() {
      const canvas = layer.closest('.map-canvas');
      return {
        canvas: canvas,
        w: (canvas && canvas.clientWidth) || 1,
        h: (canvas && canvas.clientHeight) || 1
      };
    }

    function onPointerMove(e) {
      if (!mode) return;
      const m = canvasMetrics();
      if (!m.canvas) return;
      const dx = (e.clientX - startX) / m.w * 100;
      const dy = (e.clientY - startY) / m.h * 100;
      let left = startLeft;
      let top = startTop;
      let width = startWidth;
      let height = startHeight;
      const minW = 6;
      const minH = 6;

      if (mode === 'drag') {
        left = clamp(startLeft + dx, 0, 100 - startWidth);
        top = clamp(startTop + dy, 0, 100 - startHeight);
      } else if (mode === 'resize') {
        if (resizeDir.indexOf('e') >= 0) width = clamp(startWidth + dx, minW, 100 - startLeft);
        if (resizeDir.indexOf('s') >= 0) height = clamp(startHeight + dy, minH, 100 - startTop);
        if (resizeDir.indexOf('w') >= 0) {
          left = clamp(startLeft + dx, 0, startLeft + startWidth - minW);
          width = clamp(startWidth - (left - startLeft), minW, 100 - left);
        }
        if (resizeDir.indexOf('n') >= 0) {
          top = clamp(startTop + dy, 0, startTop + startHeight - minH);
          height = clamp(startHeight - (top - startTop), minH, 100 - top);
        }
      }

      left = clamp(left, 0, 100 - width);
      top = clamp(top, 0, 100 - height);
      layer.style.left = left + '%';
      layer.style.top = top + '%';
      layer.style.width = width + '%';
      layer.style.height = height + '%';
      e.preventDefault();
    }

    function endInteraction() {
      if (!mode) return;
      mode = null;
      emitLayerState(layer);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', endInteraction);
      window.removeEventListener('pointercancel', endInteraction);
    }

    layer.addEventListener('pointerdown', function (e) {
      const handle = e.target.closest('.resize-handle');
      const m = canvasMetrics();
      if (!m.canvas || e.button !== 0) return;
      selectOverlayItem(layer);
      startX = e.clientX;
      startY = e.clientY;
      startLeft = parseFloat(layer.style.left || '0');
      startTop = parseFloat(layer.style.top || '0');
      startWidth = parseFloat(layer.style.width || '10');
      startHeight = parseFloat(layer.style.height || '10');

      if (handle) {
        mode = 'resize';
        resizeDir = handle.dataset.dir || 'se';
      } else {
        mode = 'drag';
        resizeDir = '';
      }

      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', endInteraction);
      window.addEventListener('pointercancel', endInteraction);
      e.preventDefault();
    });

    layer.addEventListener('mousedown', function () {
      selectOverlayItem(layer);
    });
  }

  function bindLayerEditors() {
    document.querySelectorAll('.overlay-item').forEach(bindOverlayItem);
    document.querySelectorAll('.map-canvas').forEach(function (canvas) {
      if (canvas.dataset.boundDeselect === '1') return;
      canvas.dataset.boundDeselect = '1';
      canvas.addEventListener('mousedown', function (e) {
        if (!e.target.closest('.overlay-item')) selectOverlayItem(null);
      });
    });
  }

  function installObserver() {
    const appBody = document.getElementById('app-body');
    if (!appBody || appBody.dataset.chatObserver === '1') return;
    appBody.dataset.chatObserver = '1';
    const observer = new MutationObserver(function () {
      bindChatEnter();
      bindLayerEditors();
      maybeAutoScroll(false);
    });
    observer.observe(appBody, { childList: true, subtree: true });
    bindChatEnter();
    bindLayerEditors();
    maybeAutoScroll(true);
  }

  function startHeartbeatLoop() {
    if (hbTimer) return;
    sendHeartbeat();
    hbTimer = window.setInterval(function () {
      sendHeartbeat();
      bindChatEnter();
      bindLayerEditors();
      bindChatScroll();
    }, 10000);
  }

  document.addEventListener('DOMContentLoaded', function () {
    installObserver();
    startHeartbeatLoop();
  });
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) sendHeartbeat();
  });
})();
