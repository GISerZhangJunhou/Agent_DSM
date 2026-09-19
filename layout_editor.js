
(function(){
  const DEFAULT_STAGE_H = 900;
  const DEFAULT_MAP_FRAME = { x: 40, y: 56, width: 860, height: 640 };
  const DEFAULT_CENTER = [30.67, 104.06];
  const DEFAULT_ZOOM = 9;

  const editor = {
    payload: null,
    key: null,
    map: null,
    resultOverlay: null,
    covariateOverlays: [],
    activeRaster: null,
    rasterCache: null,
    rasterCacheLoading: false,
    rasterCachePromise: null,
    rasterCacheRequestId: 0,
    lastHoverPixelKey: '',
    sampleLayer: null,
    baseLayers: [],
    mapFrame: { ...DEFAULT_MAP_FRAME },
    items: [],
    selectedId: null,
    pendingPalette: null,
    lastServerPalette: null,
    drag: null,
    uiBound: false,
    activeLayoutKey: null,
    lastBasemapAutoKey: null,
    lastServerActiveLayoutKey: null,
    styleCatalog: { palettes: [], north_arrows: [] },
    overlaySignature: null,
    sampleSignature: null,
    lastReadoutHtml: null,
    overlayImageCache: new Map(),
    preloadedLayoutSig: null,
  };


  function syncStageHeight(){
    const stage = stageNode();
    if(!stage) return;
    // Use the viewport instead of center-shell measured height. During Dash
    // rerenders center-shell can report a short transitional height, which makes
    // the map shrink while the card below stays empty.
    const vh = Math.max(720, Math.floor((window.innerHeight || 900) * 0.76));
    const h = Math.min(860, Math.max(700, vh));
    stage.style.height = `${h}px`;
    stage.style.minHeight = `${h}px`;
  }

  function stageSize(){
    const n = stageNode();
    const w = Math.max(520, Math.floor(n?.clientWidth || 0));
    const h = Math.max(640, Math.floor(n?.clientHeight || DEFAULT_STAGE_H));
    return { w, h };
  }
  function responsiveDefaultMapFrame(){
    const { w, h } = stageSize();
    const pad = 16;
    const width = Math.max(360, w - pad * 2);
    const height = Math.max(280, h - pad * 2);
    return { x: pad, y: pad, width, height };
  }
  function normalizeMapFrame(box){
    const { w, h } = stageSize();
    const fallback = responsiveDefaultMapFrame();
    const width = Math.max(240, Math.min(Number(box?.width || fallback.width), w - 4));
    const height = Math.max(180, Math.min(Number(box?.height || fallback.height), h - 4));
    const x = Math.max(0, Math.min(Number(box?.x ?? fallback.x), Math.max(0, w - width)));
    const y = Math.max(0, Math.min(Number(box?.y ?? fallback.y), Math.max(0, h - height)));
    return { x, y, width, height };
  }

  function qs(sel, root){ return (root || document).querySelector(sel); }
  function qsa(sel, root){ return Array.from((root || document).querySelectorAll(sel)); }
  function clamp(v, mn, mx){ return Math.max(mn, Math.min(mx, v)); }
  function stageNode(){ return qs('#layout-stage'); }
  function payloadNode(){ return qs('#layout-stage-payload'); }
  function mapFrameNode(){ return qs('#map-frame-item'); }
  function overlayNode(){ return qs('#overlay-layer'); }
  function mapNode(){ return qs('#leaflet-map'); }
  function selectedBox(){ return qs('#selected-id'); }
  function sessionId(){ return qs('#session-meta')?.dataset.sessionId || 'default'; }
  function parsePayload(){
    const n = payloadNode();
    if(!n) return null;
    try { return JSON.parse(n.textContent || '{}'); } catch(e){ return null; }
  }
  function activeLayoutPersistKey(){ return `som-active-layout:${sessionId()}`; }
  function paletteScopeKey(){
    const p = editor.payload || parsePayload() || {};
    return `${sessionId()}:${p.active_layout_key || editor.activeLayoutKey || 'default'}:${p.result_key || 'empty'}`;
  }
  function palettePersistKey(){ return `som-palette:${paletteScopeKey()}`; }
  function paletteTifKey(){ return `som-palette-tif:${paletteScopeKey()}`; }
  function styleBundlePersistKey(){ return `som-style-bundle:${paletteScopeKey()}`; }

  function mergePersistedStyleBundle(payload){
    if(!payload || !payload.result || !payload.result.tif_path) return payload;
    try{
      const scope = `${sessionId()}:${payload.active_layout_key || editor.activeLayoutKey || 'default'}:${payload.result_key || 'empty'}`;
      const raw = sessionStorage.getItem(`som-style-bundle:${scope}`);
      if(!raw) return payload;
      const styled = JSON.parse(raw);
      if(!styled || styled.tif_path !== payload.result.tif_path) return payload;
      const base = payload.result;
      payload.result = {
        ...base,
        ...styled,
        kind: base.kind,
        sample_points: base.sample_points,
        sample_point_count: base.sample_point_count,
        legend_title: base.legend_title || styled.legend_title || '值',
      };
    }catch(e){}
    return payload;
  }

  function getLayouts(payload){ return Array.isArray(payload?.layouts) ? payload.layouts : []; }
  function getLayoutByKey(payload, key){ return getLayouts(payload).find(x => x.key === key) || null; }

  function setPayloadFromLayout(layout){
    if(!editor.payload) return;
    if(!layout){
      editor.payload.result = null;
      editor.payload.result_key = 'empty';
      return;
    }
    editor.payload.active_layout_key = layout.key;
    editor.payload.title = layout.title || editor.payload.title;
    editor.payload.result = layout.result || null;
    editor.payload.result_key = layout.result_key || 'empty';
    mergePersistedStyleBundle(editor.payload);
  }

  function persistKey(payload){
    const rk = payload?.result_key || 'empty';
    const lk = editor.activeLayoutKey || payload?.active_layout_key || 'empty';
    return `som-layout-v236:${sessionId()}:${lk}:${rk}`;
  }
  function saveState(){
    if(!editor.payload) return;
    const dump = {
      mapFrame: editor.mapFrame,
      items: editor.items,
      mapView: editor.map ? { center: editor.map.getCenter(), zoom: editor.map.getZoom() } : null,
    };
    try { sessionStorage.setItem(persistKey(editor.payload), JSON.stringify(dump)); } catch(e) {}
  }
  function loadState(payload){
    try {
      const raw = sessionStorage.getItem(persistKey(payload));
      return raw ? JSON.parse(raw) : null;
    } catch(e) { return null; }
  }


  function preferredActiveLayoutKey(payload){
    let stored = null;
    try { stored = sessionStorage.getItem(activeLayoutPersistKey()); } catch(e) {}
    const layouts = getLayouts(payload);
    const valid = new Set(layouts.map(x => x.key));
    // User-selected layer is stored client-side. A server-side active key is
    // written to sessionStorage only when a new result/task changes the active
    // layout, so normal polling will not force the map back to an old layer.
    if(stored && valid.has(stored)) return stored;
    if(payload?.active_layout_key && valid.has(payload.active_layout_key)) return payload.active_layout_key;
    return layouts.length ? layouts[0].key : null;
  }

  function markActiveTab(){
    qsa('.layout-tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.layoutKey === editor.activeLayoutKey));
    const sel = qs('#layout-select');
    if(sel && editor.activeLayoutKey) sel.value = editor.activeLayoutKey;
  }

  function updateRightPanelForActiveLayout(){
    const resultBox = qs('#result-kind-box');
    const result = editor.payload?.result;
    if(resultBox){
      const kind = result?.kind || '暂无';
      resultBox.textContent = kind === 'som_map' ? '有机质制图布局' : kind === 'gcp_map' ? '不确定性布局' : kind === 'uploaded_raster' ? `上传栅格：${result?.source_name || '未命名图层'}` : kind === 'uploaded_raster_stack' ? '上传协变量叠加预览' : '暂无';
    }
  }

  function switchLayout(key){
    if(!editor.payload) return;
    const layout = getLayoutByKey(editor.payload, key);
    if(!layout) return;
    editor.activeLayoutKey = key;
    try { sessionStorage.setItem(activeLayoutPersistKey(), key); } catch(e) {}
    setPayloadFromLayout(layout);
    editor.pendingPalette = null;
    let desiredPalette = resultPaletteForLayout(layout);
    try {
      const stored = sessionStorage.getItem(palettePersistKey());
      const storedTif = sessionStorage.getItem(paletteTifKey());
      const tif = resultTifForLayout(layout);
      if(stored && (!storedTif || !tif || storedTif === tif)) desiredPalette = stored;
    } catch(e) {}
    buildState(editor.payload);
    renderMapFrame();
    renderItems();
    // V236: do not destroy/recreate the Leaflet map on every layer switch.
    // The old initMap(true) path reloaded tiles, samples and overlay images, so
    // switching dozens of uploaded covariates felt slow and sometimes appeared
    // stuck.  The server has already published all preview overlay URLs; here we
    // only replace the active image overlay and keep the map instance alive.
    if(!editor.map){
      initMap(true);
    } else {
      if(desiredPalette && desiredPalette !== qs('#result-ramp-select')?.value) setPaletteValue(desiredPalette, {});
      applyResultOverlay(editor.payload?.result || null);
      autoSetBasemapForPayload(editor.payload, {force:true});
      applyBasemapVisibility();
      try { editor.map.invalidateSize(); } catch(e) {}
    }
    preloadAllLayoutOverlays(editor.payload);
    markActiveTab();
    updateRightPanelForActiveLayout();
    applyInteractionModeClass();
  }

  function resultPaletteForLayout(layout){
    return layout?.result?.palette || '';
  }
  function resultTifForLayout(layout){
    return layout?.result?.tif_path || '';
  }


  function placeholderSvg(text){
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="160"><rect width="100%" height="100%" rx="16" fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#334155" font-size="22" font-family="Microsoft YaHei,Arial">${String(text||'图片')}</text></svg>`;
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  }

  function renderStyleCatalog(){
    const paletteGrid = qs('#palette-grid');
    const northGrid = qs('#north-arrow-grid');
    if(paletteGrid && Array.isArray(editor.styleCatalog?.palettes) && editor.styleCatalog.palettes.length){
      paletteGrid.innerHTML = '';
      editor.styleCatalog.palettes.forEach(opt => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'palette-swatch-btn';
        btn.dataset.palette = opt.value;
        btn.dataset.colors = JSON.stringify(opt.colors || []);
        btn.title = opt.label || opt.value;
        btn.innerHTML = `<span class="palette-swatch-bar" style="background:linear-gradient(90deg, ${(opt.colors || []).join(', ')})"></span><span class="palette-swatch-label">${String(opt.label || opt.value || '').split('｜').pop()}</span>`;
        paletteGrid.appendChild(btn);
      });
      const sel = qs('#result-ramp-select');
      if(sel){
        const current = editor.pendingPalette || sel.value || editor.payload?.result?.palette || '';
        sel.innerHTML = '';
        editor.styleCatalog.palettes.forEach(opt => {
          const o = document.createElement('option');
          o.value = opt.value;
          o.textContent = opt.label || opt.value;
          if(opt.value === current) o.selected = true;
          sel.appendChild(o);
        });
      }
    }
    if(northGrid){
      const builtins = [
        {key:'arcgis', label:'ArcGIS', src:'/assets/north_arrow_arcgis.svg'},
        {key:'qgis', label:'QGIS', src:'/assets/north_arrow_qgis.svg'},
        {key:'compass', label:'罗盘', src:'/assets/north_arrow_compass.svg'},
        {key:'star', label:'星芒', src:'/assets/north_arrow_star.svg'},
        {key:'triangle', label:'三角', src:'/assets/north_arrow_triangle.svg'},
        {key:'survey', label:'测绘', src:'/assets/north_arrow_survey.svg'},
        {key:'minimal', label:'极简', src:'/assets/north_arrow_minimal.svg'},
        {key:'classic', label:'经典', src:'/assets/north_arrow_classic.svg'},
      ];
      const dynamic = Array.isArray(editor.styleCatalog?.north_arrows) ? editor.styleCatalog.north_arrows.slice(0, 12) : [];
      const seen = new Set();
      const arrows = builtins.concat(dynamic).filter(x => { if(!x || !x.key || seen.has(x.key)) return false; seen.add(x.key); return true; });
      northGrid.innerHTML = '';
      arrows.forEach(opt => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'preset-btn';
        btn.dataset.addType = 'north_arrow';
        btn.dataset.preset = opt.key;
        btn.innerHTML = `<img src="${opt.src}" class="preset-icon"><span>${opt.label || opt.key}</span>`;
        northGrid.appendChild(btn);
      });
      updateStyleSelectionHighlights();
    }
  }

  async function refreshStyleCatalog(){
    try{
      const resp = await fetch('/__style_catalog', {cache:'no-store'});
      const data = await resp.json();
      if(data && data.ok){
        editor.styleCatalog = { palettes: data.palettes || [], north_arrows: data.north_arrows || [] };
        renderStyleCatalog();
        const desired = editor.pendingPalette || qs('#result-ramp-select')?.value || editor.payload?.result?.palette || '';
        if(desired) setPaletteValue(desired, {});
      }
    }catch(e){}
  }

  function currentPaletteColors(){
    const activeName = editor.pendingPalette || editor.payload?.result?.palette || qs('#result-ramp-select')?.value || '';
    const dynamic = (editor.styleCatalog?.palettes || []).find(x => x.value === activeName);
    if(dynamic && Array.isArray(dynamic.colors) && dynamic.colors.length) return dynamic.colors.slice();
    return (editor.payload?.result?.palette_colors || ['#f7fcf5','#a1d99b','#238b45']).slice();
  }
  function gradientCss(colors){ return `linear-gradient(to top, ${colors.join(',')})`; }
  function svgDataUri(svg){ return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg); }
  function northArrowSrc(preset){
    const dynamic = (editor.styleCatalog?.north_arrows || []).find(x => x.key === preset);
    if(dynamic && dynamic.src) return dynamic.src;
    const builtin = {
      arcgis: '/assets/north_arrow_arcgis.svg',
      qgis: '/assets/north_arrow_qgis.svg',
      minimal: '/assets/north_arrow_minimal.svg',
      classic: '/assets/north_arrow_classic.svg',
      compass: '/assets/north_arrow_compass.svg',
      star: '/assets/north_arrow_star.svg',
      triangle: '/assets/north_arrow_triangle.svg',
      survey: '/assets/north_arrow_survey.svg',
    };
    return builtin[preset] || builtin.arcgis;
  }


  function isFormalDeliverablePayload(payload){
    const k = payload?.result?.kind || '';
    return k === 'som_map' || k === 'gcp_map';
  }

  function defaultItems(payload){
    const prefs = payload?.layout_preferences || {};
    const elems = prefs.layout_elements || {};
    const isFormal = isFormalDeliverablePayload(payload);
    // Upload/covariate preview is a working map, not a deliverable layout.
    // Formal cartographic components are added only after SOM mapping or
    // GCP+AOA analysis has produced a final result layer.
    if(!isFormal) return [];
    const visible = (prefKey, elemKey) => {
      if(Object.prototype.hasOwnProperty.call(elems, elemKey)) return elems[elemKey] !== false;
      if(Object.prototype.hasOwnProperty.call(prefs, prefKey)) return prefs[prefKey] !== false;
      return true;
    };
    const st = stageSize();
    const frame = editor.mapFrame || responsiveDefaultMapFrame();
    const frameX = Number(frame.x || 0);
    const frameY = Number(frame.y || 0);
    const frameW = Math.max(360, Math.min(Number(frame.width || st.w || 980), Number(st.w || 980)));
    const frameH = Math.max(300, Math.min(Number(frame.height || st.h || 660), Number(st.h || 660)));
    const pad = Math.max(18, Math.min(32, Math.round(frameW * 0.028)));
    // V239：正式制图部件的坐标必须相对“地图框”计算，再换算成 stage 绝对坐标。
    // 之前按整个 stage 计算，地图框在页面中有偏移时，标题/图例/指北针会跑到地图框外。
    const titleW = Math.min(frameW - 280, Math.max(360, Math.round(frameW * 0.56)));
    const titleX = frameX + Math.round((frameW - titleW) / 2);
    const titleY = frameY + 22;
    const northSize = Math.min(72, Math.max(58, Math.round(frameW * 0.062)));
    const northX = frameX + frameW - pad - northSize;
    const northY = frameY + 26;
    const legendW = Math.min(145, Math.max(118, Math.round(frameW * 0.13)));
    const legendH = Math.min(310, Math.max(230, Math.round(frameH * 0.34)));
    const legendX = frameX + frameW - pad - legendW;
    const legendY = frameY + Math.min(frameH - legendH - 46, Math.max(138, (northY - frameY) + northSize + 34));
    const scaleY = frameY + Math.max(260, frameH - 92);
    const items = [];
    const titleText = prefs.title_text || payload?.title || '专题图';
    if(visible('show_title', 'title')){
      items.push({ id:`text_auto_title`, type:'text', text:titleText, x:titleX, y:titleY, width:titleW, height:60, zIndex:80, fill:'transparent', stroke:'transparent', textColor:'#17345a', fontSize:25, opacity:1, preset:'plain', textAlign:'center' });
    }
    if(visible('show_north', 'north_arrow')){
      items.push({ id:`north_auto`, type:'north_arrow', x:northX, y:northY, width:northSize, height:northSize, zIndex:81, fill:'transparent', stroke:'#17345a', textColor:'#17345a', fontSize:22, opacity:1, preset:'arcgis' });
    }
    if(visible('show_scale', 'scale_bar')){
      items.push({ id:`scale_auto`, type:'scale_bar', x:pad, y:scaleY, width:250, height:58, zIndex:82, fill:'transparent', stroke:'#17345a', textColor:'#17345a', fontSize:20, opacity:1, preset:'line-ticks' });
    }
    if(visible('show_legend', 'legend')){
      items.push({ id:`legend_auto`, type:'legend', x:legendX, y:legendY, width:legendW, height:legendH, zIndex:83, fill:'transparent', stroke:'transparent', textColor:'#17345a', fontSize:20, opacity:1, preset:'transparent', legendTitle:(prefs.legend_title || payload?.result?.legend_title || '值') });
    }
    return items;
  }

  function legendHtml(item){
    const r = editor.payload?.result || {};
    const colors = currentPaletteColors();
    return `
      <div class="legend-title">${escapeHtml(item.legendTitle || r.legend_title || '值')}</div>
      <div class="legend-inner">
        <div class="legend-ramp" style="background:${gradientCss(colors)}"></div>
        <div class="legend-labels">
          <div>${r.zmax != null ? Number(r.zmax).toFixed(3) : '高'}</div>
          <div>${r.zmin != null ? Number(r.zmin).toFixed(3) : '低'}</div>
        </div>
      </div>`;
  }
  function scaleHtml(){
    return `<div class="scale-content"><div class="scale-bar-inner-wrap"><div class="scale-bar-inner"></div></div><div class="scale-labels"><span class="scale-label-left">0</span><span class="scale-label-right">100 km</span></div></div>`;
  }
  function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }


  function appendAssistantBubbleOnce(key, text){
    if(!text) return;
    const dedupKey = `som-chat-ui-once:${sessionId()}:${key}`;
    try { if(sessionStorage.getItem(dedupKey) === '1') return; sessionStorage.setItem(dedupKey, '1'); } catch(e) {}
    const box = document.getElementById('chat-history');
    if(box){
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.justifyContent = 'flex-start';
      row.style.marginBottom = '10px';
      const bubble = document.createElement('div');
      bubble.textContent = text;
      bubble.style.backgroundColor = '#f5f7fb';
      bubble.style.padding = '12px 14px';
      bubble.style.borderRadius = '16px';
      bubble.style.maxWidth = '88%';
      bubble.style.whiteSpace = 'pre-wrap';
      bubble.style.lineHeight = '1.6';
      bubble.style.fontSize = '15px';
      bubble.style.color = '#111827';
      bubble.style.boxShadow = '0 1px 3px rgba(0,0,0,0.06)';
      row.appendChild(bubble);
      box.appendChild(row);
      try { box.scrollTop = box.scrollHeight; } catch(e) {}
    }
    // 持久化到后端活动会话，避免刷新后完成提示丢失；前端立即追加，避免等待 Dash store 回写。
    try {
      fetch('/__client_chat_ack', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId(), key, text})}).catch(()=>{});
    } catch(e) {}
  }

  function notifyLayerVisible(bundle, reason){
    if(!bundle || !bundle.tif_path) return;
    const title = editor.payload?.title || bundle.source_name || bundle.legend_title || '地图图层';
    const sig = overlayBundleSignature(bundle);
    const msg = reason === 'style' ? `地图样式已更新完成：${title}` : `地图图层已加载完成：${title}`;
    appendAssistantBubbleOnce(`${reason || 'layer'}:${sig}`, msg);
    const pendingKey = reason === 'style' ? 'pending_style_loaded_messages' : 'pending_layer_loaded_messages';
    const pending = editor.payload && Array.isArray(editor.payload[pendingKey]) ? editor.payload[pendingKey] : [];
    pending.forEach((item, idx) => {
      const text = (item && typeof item === 'object') ? item.text : item;
      if(!text) return;
      const stamp = (item && typeof item === 'object') ? (item.created_at || '') : '';
      appendAssistantBubbleOnce(`deferred:${pendingKey}:${sig}:${idx}:${stamp}`, String(text));
    });
  }

  function createHandles(el){
    ['nw','ne','sw','se'].forEach(dir => {
      const h = document.createElement('span');
      h.className = `resize-handle handle-${dir}`;
      h.dataset.dir = dir;
      el.appendChild(h);
    });
  }

  function applyClasses(el, item){
    el.className = 'layout-item';
    el.dataset.id = item.id;
    el.dataset.type = item.type;
    if(item.locked) el.classList.add('locked');
    if(item.type === 'text'){
      el.classList.add('text-item', `title-style-${['banner','plain','shadow','outline'].includes(item.preset) ? item.preset : 'banner'}`);
    } else if(item.type === 'legend') {
      el.classList.add('legend-panel', `legend-style-${item.preset || 'transparent'}`);
    } else if(item.type === 'scale_bar') {
      el.classList.add('scale-item', `scale-style-${item.preset || 'line-ticks'}`);
    } else if(item.type === 'north_arrow' || item.type === 'image') {
      el.classList.add('image-item');
    } else if(['rect','circle','line','arrow'].includes(item.type)) {
      el.classList.add('shape-item');
      if(item.type === 'rect') el.classList.add('rect-shape');
      if(item.type === 'circle') el.classList.add('circle-shape');
      if(item.type === 'line') el.classList.add('line-shape');
      if(item.type === 'arrow') el.classList.add('arrow-shape');
    }
  }
  function applyBox(el, item){
    el.style.left = `${item.x}px`;
    el.style.top = `${item.y}px`;
    el.style.width = `${item.width}px`;
    el.style.height = `${item.height}px`;
    el.style.zIndex = String(item.zIndex || 10);
    el.style.opacity = String(item.opacity ?? 1);
  }
  function niceDistance(meters){
    if(meters <= 0) return 100;
    const p = Math.pow(10, Math.floor(Math.log10(meters)));
    const n = meters / p;
    const nice = n >= 5 ? 5 : n >= 2 ? 2 : 1;
    return nice * p;
  }
  function updateSingleScaleBar(el){
    if(!editor.map || !el) return;
    const content = el.querySelector('.scale-content');
    const inner = el.querySelector('.scale-bar-inner');
    const wrap = el.querySelector('.scale-bar-inner-wrap');
    const labels = el.querySelector('.scale-labels');
    const labelRight = el.querySelector('.scale-label-right');
    if(!inner || !labelRight || !wrap || !labels || !content) return;
    const lat = editor.map.getCenter().lat;
    const zoom = editor.map.getZoom();
    const mpp = 156543.03392804097 * Math.cos(lat*Math.PI/180) / Math.pow(2, zoom);
    const maxPx = Math.max(100, el.clientWidth - 20);
    const rawMeters = maxPx * mpp * 0.48;
    const nice = niceDistance(rawMeters);
    const px = clamp(nice / mpp, 70, maxPx);
    content.style.width = `${px}px`;
    wrap.style.width = `${px}px`;
    inner.style.width = `${px}px`;
    labels.style.width = `${px}px`;
    labelRight.textContent = nice >= 1000 ? `${(nice/1000).toFixed(nice % 1000 === 0 ? 0 : 1)} km` : `${Math.round(nice)} m`;
  }
  function updateScaleBars(){ qsa('.layout-item[data-type="scale_bar"]', overlayNode()).forEach(updateSingleScaleBar); }

  function applyVisual(el, item){
    if(item.type === 'text') {
      el.textContent = item.text || '新建文字';
      el.style.color = item.textColor || '#17345a';
      el.style.fontSize = `${item.fontSize || 24}px`;
      el.style.background = item.fill || 'transparent';
      el.style.borderColor = item.stroke || '#17345a';
      el.style.padding = '8px 12px';
      el.style.borderRadius = '10px';
      el.style.fontWeight = '800';
      el.style.textAlign = item.textAlign || 'left';
    } else if(item.type === 'north_arrow') {
      el.innerHTML = `<img src="${northArrowSrc(item.preset)}" draggable="false">`;
    } else if(item.type === 'legend') {
      el.innerHTML = legendHtml(item);
      el.style.color = item.textColor || '#17345a';
    } else if(item.type === 'scale_bar') {
      el.innerHTML = scaleHtml();
      updateSingleScaleBar(el);
    } else if(item.type === 'image') {
      el.innerHTML = `<img src="${item.src || placeholderSvg('图片')}" draggable="false">`;
    } else if(item.type === 'rect' || item.type === 'circle') {
      el.style.borderColor = item.stroke || '#17345a';
      el.style.background = item.fill || 'rgba(255,255,255,0.25)';
    } else if(item.type === 'line') {
      el.style.borderTopColor = item.stroke || '#17345a';
    } else if(item.type === 'arrow') {
      el.style.setProperty('--shape-color', item.stroke || '#17345a');
      el.style.color = item.stroke || '#17345a';
    }
  }

  function clampAutoItemToMapFrame(item){
    if(!item || !isFormalDeliverablePayload(editor.payload)) return item;
    const id = String(item.id || '');
    if(!(id.startsWith('text_auto_') || id === 'north_auto' || id === 'scale_auto' || id === 'legend_auto')) return item;
    const f = editor.mapFrame || responsiveDefaultMapFrame();
    const margin = 8;
    const w = Math.max(20, Number(item.width || 80));
    const h = Math.max(20, Number(item.height || 40));
    item.x = clamp(Number(item.x || 0), Number(f.x || 0) + margin, Number(f.x || 0) + Number(f.width || 0) - w - margin);
    item.y = clamp(Number(item.y || 0), Number(f.y || 0) + margin, Number(f.y || 0) + Number(f.height || 0) - h - margin);
    return item;
  }

  function renderItems(){
    const root = overlayNode();
    if(!root) return;
    root.innerHTML = '';
    editor.items.sort((a,b)=>(a.zIndex||0)-(b.zIndex||0)).forEach(item => {
      clampAutoItemToMapFrame(item);
      const el = document.createElement('div');
      applyClasses(el, item);
      applyBox(el, item);
      applyVisual(el, item);
      createHandles(el);
      root.appendChild(el);
    });
    selectById(editor.selectedId || null);
    updateScaleBars();
    renderLayerList();
    saveState();
  }

  function renderMapFrame(){
    const box = mapFrameNode();
    if(!box) return;
    editor.mapFrame = normalizeMapFrame(editor.mapFrame);
    box.style.left = `${editor.mapFrame.x}px`;
    box.style.top = `${editor.mapFrame.y}px`;
    box.style.width = `${editor.mapFrame.width}px`;
    box.style.height = `${editor.mapFrame.height}px`;
    box.style.display = 'block';
    box.style.position = 'absolute';
    ensureMapInteractionSwitch();
    applyInteractionModeClass();
    if(editor.map) setTimeout(() => editor.map.invalidateSize(), 30);
    saveState();
  }

  function selectById(id){
    if(id === 'mapframe' && editor.interactionMode !== 'layout') id = null;
    editor.selectedId = id;
    qsa('.layout-item', overlayNode()).forEach(el => el.classList.toggle('selected', el.dataset.id === id));
    const mf = mapFrameNode();
    if(mf) mf.classList.toggle('selected', id === 'mapframe' && editor.interactionMode === 'layout');
    if(selectedBox()) selectedBox().textContent = id || '无';
    updateStyleSelectionHighlights();
    renderLayerList();
    applyInteractionModeClass();
  }
  function itemById(id){ return editor.items.find(x => x.id === id); }
  function itemLabel(item){
    if(!item) return '';
    const labels = {text:'文字', image:'图片', rect:'矩形', circle:'圆形', line:'线', arrow:'箭头', north_arrow:'指北针', legend:'图例', scale_bar:'比例尺'};
    if(item.id === 'mapframe') return '地图框';
    return `${labels[item.type] || item.type} · ${item.text || item.legendTitle || item.id}`;
  }
  function renderLayerList(){
    const host = qs('#layout-layer-list');
    if(!host) return;
    const layers = [{id:'mapframe', type:'mapframe', zIndex:0}].concat(editor.items.slice().sort((a,b)=>(b.zIndex||0)-(a.zIndex||0)));
    host.innerHTML = '';
    layers.forEach(item => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'layer-row';
      if(editor.selectedId === item.id) row.classList.add('active');
      if(item.locked) row.classList.add('locked');
      row.dataset.layerId = item.id;
      row.innerHTML = `<span>${escapeHtml(itemLabel(item))}</span><small>${item.locked ? '锁定' : ''}</small>`;
      host.appendChild(row);
    });
  }
  function normalizeZ(){
    editor.items.sort((a,b)=>(a.zIndex||0)-(b.zIndex||0)).forEach((item, idx) => { item.zIndex = 20 + idx; });
  }
  function moveSelectedLayer(direction){
    const item = itemById(editor.selectedId);
    if(!item) return;
    normalizeZ();
    if(direction === 'up') item.zIndex += 2;
    if(direction === 'down') item.zIndex -= 2;
    if(direction === 'top') item.zIndex = 999;
    if(direction === 'bottom') item.zIndex = 1;
    normalizeZ();
    renderItems();
    selectById(item.id);
    saveState();
  }
  function duplicateSelected(){
    const item = itemById(editor.selectedId);
    if(!item) return;
    const cp = JSON.parse(JSON.stringify(item));
    cp.id = `${item.type}_${Date.now()}_${Math.floor(Math.random()*9999)}`;
    cp.x = clamp((cp.x||0)+24, 0, stageSize().w - (cp.width||120));
    cp.y = clamp((cp.y||0)+24, 0, stageSize().h - (cp.height||80));
    cp.zIndex = 80 + editor.items.length;
    editor.items.push(cp);
    renderItems();
    selectById(cp.id);
  }
  function alignSelected(axis){
    if(editor.selectedId === 'mapframe'){
      const st = stageSize();
      if(axis === 'h') editor.mapFrame.x = Math.round((st.w - editor.mapFrame.width)/2);
      if(axis === 'v') editor.mapFrame.y = Math.round((st.h - editor.mapFrame.height)/2);
      renderMapFrame();
      return;
    }
    const item = itemById(editor.selectedId);
    if(!item) return;
    const st = stageSize();
    if(axis === 'h') item.x = Math.round((st.w - item.width)/2);
    if(axis === 'v') item.y = Math.round((st.h - item.height)/2);
    renderItems(); selectById(item.id); saveState();
  }
  function toggleLockSelected(){
    const item = itemById(editor.selectedId);
    if(!item) return;
    item.locked = !item.locked;
    renderItems(); selectById(item.id); saveState();
  }
  function templateKey(){ return `som-layout-template:${sessionId()}`; }
  function saveTemplate(){
    try { sessionStorage.setItem(templateKey(), JSON.stringify({mapFrame: editor.mapFrame, items: editor.items})); alert('当前布局模板已保存。'); } catch(e){ alert('模板保存失败。'); }
  }
  function loadTemplate(){
    try {
      const raw = sessionStorage.getItem(templateKey());
      if(!raw){ alert('当前还没有保存过模板。'); return; }
      const t = JSON.parse(raw);
      if(t.mapFrame) editor.mapFrame = normalizeMapFrame(t.mapFrame);
      if(Array.isArray(t.items)) editor.items = t.items;
      renderMapFrame(); renderItems(); saveState();
    } catch(e){ alert('模板读取失败。'); }
  }
  function hideContextMenu(){ const m=qs('#layout-context-menu'); if(m) m.classList.add('hidden'); }
  function showContextMenu(x,y){
    const m=qs('#layout-context-menu'); if(!m) return;
    const st=stageNode()?.getBoundingClientRect();
    m.style.left = `${Math.max(8, x-(st?.left||0))}px`;
    m.style.top = `${Math.max(8, y-(st?.top||0))}px`;
    m.classList.remove('hidden');
  }

  function clearMap(){
    if(editor.hoverKeepAliveTimer){ try { clearInterval(editor.hoverKeepAliveTimer); } catch(e) {} }
    if(editor.map){ try { editor.map.remove(); } catch(e) {} }
    editor.map = null; editor.resultOverlay = null; editor.covariateOverlays = []; editor.activeRaster = null; editor.rasterCache = null; editor.rasterCacheLoading = false; editor.rasterCachePromise = null; editor.rasterCacheRequestId = 0; editor.lastHoverPixelKey = ''; editor.lastHoverLatLng = null; editor.lastHoverEvent = null; editor.hoverInsideMap = false; editor.hoverKeepAliveTimer = null; editor.sampleLayer = null; editor.baseLayers = []; editor.pixelReadout = null; editor.pixelProbe = null; editor.hoverInstalled = false; editor.overlaySignature = null; editor.sampleSignature = null; editor.lastReadoutHtml = null;
  }

  function samplePointsToggleNode(){ return qs('#sample-points-toggle-input input[type=checkbox]'); }
  function areSamplePointsEnabled(){
    const pref = editor.payload?.map?.show_samples;
    if(pref === true) return true;
    if(pref === false) return false;
    return !!samplePointsToggleNode()?.checked;
  }
  function uploadedSamplePoints(){
    const pts = editor.payload?.uploaded_samples?.points;
    return Array.isArray(pts) ? pts : [];
  }
  function currentSamplePoints(bundle){
    if(!areSamplePointsEnabled()) return [];
    const fromBundle = bundle && Array.isArray(bundle.sample_points) ? bundle.sample_points : [];
    const fromUpload = uploadedSamplePoints();
    // During upload/covariate preview, prefer the original uploaded sample table
    // so points stay visible over raster previews.  Final SOM/GCP layouts set
    // map.show_samples=false and therefore return no points here.
    return fromUpload.length ? fromUpload : fromBundle;
  }

  function applyInteractionModeClass(){
    const st = stageNode();
    if(!st) return;
    const mode = editor.interactionMode === 'layout' ? 'layout' : 'query';
    st.classList.toggle('map-query-mode', mode === 'query');
    st.classList.toggle('map-layout-mode', mode === 'layout');
    const mf = mapFrameNode();
    if(mf && mode === 'query') mf.classList.remove('selected');
    qsa('.map-mode-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.mapMode === mode));
  }

  function setMapInteractionMode(mode){
    editor.interactionMode = mode === 'layout' ? 'layout' : 'query';
    if(editor.interactionMode === 'query') editor.selectedId = null;
    applyInteractionModeClass();
    if(editor.map){
      try { editor.map.dragging.enable(); editor.map.scrollWheelZoom.enable(); editor.map.doubleClickZoom.enable(); } catch(e) {}
      setTimeout(() => editor.map && editor.map.invalidateSize(), 30);
    }
  }

  function ensureMapInteractionSwitch(){
    // V104：像元查询是实时悬停能力，不再作为一个需要用户切换的
    // “浏览/查值”模式出现。布局编辑、缩放和平移可以并存；用户只要
    // 把鼠标移到结果图上，就应看到像元值。保留函数空实现，避免旧
    // 事件绑定报错。
    return null;
  }

  function ensurePixelProbe(){
    const host = mapNode();
    if(!host) return null;
    if(editor.pixelProbe && editor.pixelProbe.nodeType === 1) return editor.pixelProbe;
    let probe = host.querySelector('.raster-hover-probe');
    if(!probe){
      probe = document.createElement('div');
      probe.className = 'raster-hover-probe';
      probe.innerHTML = '<span class="raster-hover-line raster-hover-line-h"></span><span class="raster-hover-line raster-hover-line-v"></span><span class="raster-hover-cell"></span>';
      host.appendChild(probe);
    }
    editor.pixelProbe = probe;
    return probe;
  }

  function movePixelProbeFromEvent(ev){
    const probe = ensurePixelProbe();
    const host = mapNode();
    if(!probe || !host || !ev) return;
    const rect = host.getBoundingClientRect();
    const relX = ev.clientX - rect.left;
    const relY = ev.clientY - rect.top;
    probe.style.left = `${relX}px`;
    probe.style.top = `${relY}px`;
    probe.classList.add('active');
  }

  function hidePixelProbe(){
    const probe = editor.pixelProbe || qs('.raster-hover-probe', mapNode());
    if(probe) probe.classList.remove('active');
    const readout = editor.pixelReadout || qs('.map-pixel-readout', mapNode());
    if(readout) readout.classList.remove('active');
  }

  function movePixelReadoutFromEvent(ev){
    movePixelProbeFromEvent(ev);
    const el = ensurePixelReadout();
    const host = mapNode();
    if(!el || !host || !ev) return;
    const rect = host.getBoundingClientRect();
    const relX = ev.clientX - rect.left;
    const relY = ev.clientY - rect.top;
    const w = el.offsetWidth || 260;
    const h = el.offsetHeight || 54;
    let left = relX + 16;
    let top = relY + 16;
    if(left + w + 12 > rect.width) left = Math.max(12, relX - w - 16);
    if(top + h + 12 > rect.height) top = Math.max(12, relY - h - 16);
    el.style.left = `${Math.max(12, left)}px`;
    el.style.top = `${Math.max(12, top)}px`;
    el.style.bottom = 'auto';
    el.classList.add('active');
  }

  function resultValueLabel(result){
    const kind = result?.kind || '';
    if(kind === 'som_map') return { label:'土壤有机质', unit:'g/kg', layer: result?.source_name || '土壤有机质预测图' };
    if(kind === 'gcp_map') return { label: result?.legend_title || '不确定性', unit:'g/kg', layer: result?.source_name || '不确定性图层' };
    if(kind === 'uploaded_raster') return { label:'栅格值', unit:'', layer: result?.source_name || result?.legend_title || '上传栅格图层' };
    return { label: result?.legend_title || '栅格值', unit:'', layer: result?.source_name || '' };
  }
  function ensurePixelReadout(){
    // V102：不要再用 Leaflet control。旧样式里有
    // `.map-frame-item .leaflet-control-container { display:none !important; }`，
    // 会把所有 Leaflet 控件整体隐藏，导致像元值读数永远不可见。
    // 这里改为直接把读数框挂到 #leaflet-map 内部，独立于 control-container。
    if(!editor.map || !window.L) return null;
    ensureMapInteractionSwitch();
    const host = mapNode();
    if(!host) return null;
    if(editor.pixelReadout && editor.pixelReadout.nodeType === 1) return editor.pixelReadout;
    let el = host.querySelector('.map-pixel-readout');
    if(!el){
      el = document.createElement('div');
      el.className = 'map-pixel-readout';
      el.innerHTML = '正在准备像元查询…';
      host.appendChild(el);
      try {
        window.L.DomEvent.disableClickPropagation(el);
        window.L.DomEvent.disableScrollPropagation(el);
      } catch(e) {}
    }
    editor.pixelReadout = el;
    return el;
  }
  function setPixelReadout(html, ev){
    const el = ensurePixelReadout();
    if(!el) return;
    const next = html || '正在准备像元查询…';
    // V174：读数框不再在每个轮询/每个mousemove上反复重写 DOM。
    // 反复 innerHTML 会触发布局重排，叠加 500ms 轮询就表现为“一闪一闪”。
    if(editor.lastReadoutHtml !== next){
      el.innerHTML = next;
      editor.lastReadoutHtml = next;
    }
    el.classList.add('active');
    // 只移动位置，不清空内容；鼠标仍在地图内时值应常驻显示。
    if(ev) movePixelReadoutFromEvent(ev);
  }

  function base64ToFloat32Array(b64){
    try{
      const bin = atob(b64 || '');
      const len = bin.length;
      const bytes = new Uint8Array(len);
      for(let i=0; i<len; i++) bytes[i] = bin.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }catch(e){
      console.warn('[raster-cache] base64 decode failed', e);
      return null;
    }
  }

  function rasterCacheMatches(result){
    return !!(editor.rasterCache && result && editor.rasterCache.tif_path === result.tif_path && editor.rasterCache.values);
  }

  async function preloadRasterCache(result){
    if(!result || !result.tif_path) return null;
    if(rasterCacheMatches(result)) return editor.rasterCache;
    if(editor.rasterCacheLoading && editor.rasterCachePromise) return editor.rasterCachePromise;
    const requestId = ++editor.rasterCacheRequestId;
    editor.rasterCacheLoading = true;
    editor.rasterCache = { tif_path: result.tif_path, loading: true };
    editor.rasterCachePromise = fetch('/__raster_cache', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({tif_path: result.tif_path, admin_region: result.admin_region || ''})
    }).then(async resp => {
      let data = null;
      try { data = await resp.json(); } catch(e) { data = {ok:false, error:`HTTP ${resp.status}`}; }
      if(requestId !== editor.rasterCacheRequestId) return null;
      if(!resp.ok || !data.ok) throw new Error(data?.error || (`HTTP ${resp.status}`));
      const values = base64ToFloat32Array(data.values_b64);
      if(!values || values.length !== Number(data.width) * Number(data.height)){
        throw new Error('像元缓存长度异常');
      }
      const cache = {
        tif_path: result.tif_path,
        width: Number(data.width),
        height: Number(data.height),
        bounds: data.bounds,
        values,
        source_width: Number(data.source_width || data.width),
        source_height: Number(data.source_height || data.height),
        downsampled: !!data.downsampled,
        unit: data.unit || '',
        nodata_meta: data.nodata_meta || null,
        loaded_at: Date.now()
      };
      editor.rasterCache = cache;
      editor.rasterCacheLoading = false;
      // V107: If the mouse is already stopped on the map while the cache finishes
      // loading, repaint the sticky readout immediately. Do not wait for a new
      // mousemove event; otherwise the user sees "正在载入" until they move.
      try {
        if(editor.hoverInsideMap && editor.lastHoverLatLng && typeof editor.__renderStickyRasterHover === 'function'){
          requestAnimationFrame(() => editor.__renderStickyRasterHover(editor.lastHoverLatLng, editor.lastHoverEvent));
        }
      } catch(e) {}
      return cache;
    }).catch(err => {
      if(requestId === editor.rasterCacheRequestId){
        editor.rasterCache = {tif_path: result.tif_path, error: err?.message || String(err)};
        editor.rasterCacheLoading = false;
      }
      console.warn('[raster-cache] load failed', err);
      return null;
    });
    return editor.rasterCachePromise;
  }

  function localSampleRaster(latlng, result){
    const cache = editor.rasterCache;
    if(!cache || cache.loading) return {status:'loading'};
    if(cache.error) return {status:'error', error: cache.error};
    if(!cache.values || !cache.bounds || !result || cache.tif_path !== result.tif_path) return {status:'missing'};
    const b = cache.bounds;
    const south = Number(b?.[0]?.[0]);
    const west = Number(b?.[0]?.[1]);
    const north = Number(b?.[1]?.[0]);
    const east = Number(b?.[1]?.[1]);
    if(!Number.isFinite(south) || !Number.isFinite(west) || !Number.isFinite(north) || !Number.isFinite(east) || east === west || north === south){
      return {status:'error', error:'像元缓存空间范围无效'};
    }
    const lng = Number(latlng.lng), lat = Number(latlng.lat);
    if(lng < west || lng > east || lat < south || lat > north) return {status:'nodata'};
    const col = Math.floor((lng - west) / (east - west) * cache.width);
    const row = Math.floor((north - lat) / (north - south) * cache.height);
    if(row < 0 || col < 0 || row >= cache.height || col >= cache.width) return {status:'nodata'};
    const idx = row * cache.width + col;
    const v = cache.values[idx];
    if(!Number.isFinite(v)) return {status:'nodata', row, col};
    return {status:'ok', value:v, row, col, width:cache.width, height:cache.height};
  }
  function activeRasterResult(){
    // Always use the layer that is actually active in the UI. This keeps the
    // hover readout synchronized with the upload-recognition dropdown and the
    // layout tab/dropdown.
    if(editor.activeRaster && editor.activeRaster.tif_path) return editor.activeRaster;
    const p = editor.payload || parsePayload() || {};
    const activeKey = editor.activeLayoutKey || p.active_layout_key;
    const layout = activeKey ? getLayoutByKey(p, activeKey) : null;
    if(layout && layout.result && layout.result.tif_path) return layout.result;
    if(p.result && p.result.tif_path) return p.result;
    const layouts = getLayouts(p);
    const first = layouts.find(x => x?.result?.tif_path);
    return first ? first.result : null;
  }
  function installRasterHover(){
    if(!editor.map || editor.hoverInstalled) return;
    editor.hoverInstalled = true;
    ensurePixelReadout();
    ensurePixelProbe();
    const host = mapNode();
    let raf = 0;
    let pending = null;

    function pointIsInsideMap(ev){
      const h = mapNode();
      if(!h || !ev) return false;
      const rect = h.getBoundingClientRect();
      return ev.clientX >= rect.left && ev.clientX <= rect.right && ev.clientY >= rect.top && ev.clientY <= rect.bottom;
    }

    function renderRealtimeValue(latlng, ev){
      if(!latlng || !Number.isFinite(latlng.lat) || !Number.isFinite(latlng.lng)) return;
      editor.hoverInsideMap = true;
      editor.lastHoverLatLng = latlng;
      if(ev) editor.lastHoverEvent = ev;
      movePixelReadoutFromEvent(ev || editor.lastHoverEvent);
      const result = activeRasterResult();
      if(!result || !result.tif_path){
        setPixelReadout('当前没有可查询的 GeoTIFF 结果图层', ev || editor.lastHoverEvent);
        return;
      }
      if(!rasterCacheMatches(result)){
        // V160: do not loop forever on "正在载入前端缓存" when cache loading
        // failed for an uploaded GeoTIFF. Uploaded covariate layers and final
        // result layers use the same cache path; if the cache has an error, show
        // the actual error so the user knows whether it is CRS/path/file access.
        const meta = resultValueLabel(result);
        if(editor.rasterCache && editor.rasterCache.tif_path === result.tif_path && editor.rasterCache.error){
          setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>经度 ${latlng.lng.toFixed(5)}，纬度 ${latlng.lat.toFixed(5)}<br/>像元缓存错误：${editor.rasterCache.error}`, ev || editor.lastHoverEvent);
          return;
        }
        if(editor.rasterCache && editor.rasterCache.tif_path === result.tif_path && editor.rasterCache.loading){
          setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>经度 ${latlng.lng.toFixed(5)}，纬度 ${latlng.lat.toFixed(5)}<br/>正在载入前端像元缓存…`, ev || editor.lastHoverEvent);
          return;
        }
        preloadRasterCache(result);
        setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>经度 ${latlng.lng.toFixed(5)}，纬度 ${latlng.lat.toFixed(5)}<br/>正在载入前端像元缓存…`, ev || editor.lastHoverEvent);
        return;
      }
      const sampled = localSampleRaster(latlng, result);
      const meta = resultValueLabel(result);
      if(sampled.status === 'loading' || sampled.status === 'missing'){
        setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>经度 ${latlng.lng.toFixed(5)}，纬度 ${latlng.lat.toFixed(5)}<br/>正在载入前端像元缓存…`, ev || editor.lastHoverEvent);
        return;
      }
      if(sampled.status === 'error'){
        setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>像元缓存错误：${sampled.error || '未知错误'}`, ev || editor.lastHoverEvent);
        return;
      }
      if(sampled.status === 'nodata'){
        setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>经度 ${latlng.lng.toFixed(5)}，纬度 ${latlng.lat.toFixed(5)}<br/>该位置为 NODATA`, ev || editor.lastHoverEvent);
        return;
      }
      const v = Number(sampled.value);
      const txt = Number.isFinite(v) ? v.toFixed(3) : String(sampled.value);
      const unit = meta.unit ? ` ${meta.unit}` : '';
      const cacheNote = editor.rasterCache?.downsampled ? '<span class="pixel-rc">查询缓存为轻量化栅格</span>' : '';
      const rc = `<span class="pixel-rc">行 ${sampled.row}，列 ${sampled.col}</span>`;
      setPixelReadout(`图层：${meta.layer || '当前栅格'}<br/>经度 ${latlng.lng.toFixed(5)}，纬度 ${latlng.lat.toFixed(5)}<br/>${meta.label}：<b>${txt}${unit}</b>${rc}${cacheNote}`, ev || editor.lastHoverEvent);
    }

    // Expose the renderer to the raster-cache loader. When cache loading finishes
    // and the cursor has not moved, V107 repaints the last cursor position instead
    // of waiting for another mousemove event.
    editor.__renderStickyRasterHover = renderRealtimeValue;

    function scheduleLocalRender(latlng, ev){
      editor.hoverInsideMap = true;
      editor.lastHoverLatLng = latlng;
      if(ev) editor.lastHoverEvent = ev;
      pending = {latlng, ev};
      if(raf) return;
      raf = requestAnimationFrame(function(){
        raf = 0;
        const p = pending;
        pending = null;
        if(p) renderRealtimeValue(p.latlng, p.ev);
      });
    }

    function latlngFromPointer(ev){
      if(!editor.map || !host || !window.L) return null;
      const rect = host.getBoundingClientRect();
      if(ev.clientX < rect.left || ev.clientX > rect.right || ev.clientY < rect.top || ev.clientY > rect.bottom) return null;
      const pt = window.L.point(ev.clientX - rect.left, ev.clientY - rect.top);
      return editor.map.containerPointToLatLng(pt);
    }

    function hideOnlyWhenTrulyOutside(ev){
      // Leaflet mouseout can fire when the cursor moves between internal map panes,
      // image overlays, tiles, controls, or the transparent hover readout. That was
      // the reason the readout disappeared while the cursor was still on the map.
      // V107 hides only when the pointer is genuinely outside #leaflet-map.
      if(ev && pointIsInsideMap(ev)) return;
      editor.hoverInsideMap = false;
      hidePixelProbe();
    }

    editor.map.on('mousemove', function(ev){ scheduleLocalRender(ev.latlng, ev.originalEvent); });
    editor.map.on('mouseout', function(ev){
      const oe = ev && ev.originalEvent;
      setTimeout(() => hideOnlyWhenTrulyOutside(oe), 30);
    });

    if(host && !host.__somRasterHoverDomInstalled){
      host.__somRasterHoverDomInstalled = true;
      host.classList.add('realtime-raster-query');
      host.addEventListener('mouseenter', function(ev){
        const latlng = latlngFromPointer(ev);
        if(latlng) scheduleLocalRender(latlng, ev);
      }, {passive:true});
      host.addEventListener('mousemove', function(ev){
        const latlng = latlngFromPointer(ev);
        if(latlng) scheduleLocalRender(latlng, ev);
      }, {passive:true});
      host.addEventListener('mouseleave', function(ev){
        setTimeout(() => hideOnlyWhenTrulyOutside(ev), 30);
      }, {passive:true});
    }

    if(!document.__somRasterHoverDocumentInstalled){
      document.__somRasterHoverDocumentInstalled = true;
      document.addEventListener('pointermove', function(ev){
        const host2 = mapNode();
        if(!editor.map || !host2 || !window.L) return;
        const rect = host2.getBoundingClientRect();
        if(ev.clientX < rect.left || ev.clientX > rect.right || ev.clientY < rect.top || ev.clientY > rect.bottom){
          if(editor.hoverInsideMap) hideOnlyWhenTrulyOutside(ev);
          return;
        }
        const pt = window.L.point(ev.clientX - rect.left, ev.clientY - rect.top);
        const latlng = editor.map.containerPointToLatLng(pt);
        scheduleLocalRender(latlng, ev);
      }, true);
      document.addEventListener('pointerleave', function(ev){ hideOnlyWhenTrulyOutside(ev); }, true);
    }

    // V107: sticky keep-alive. If the pointer is stationary, no mousemove event is
    // emitted. This interval does not query the backend; it only keeps the existing
    // readout/probe active and, when needed, repaints the last known lat/lng from
    // the in-memory raster cache. This is what makes the value stay visible until
    // the cursor actually leaves the map.
    if(!editor.hoverKeepAliveTimer){
      editor.hoverKeepAliveTimer = setInterval(function(){
        if(!editor.hoverInsideMap || !editor.lastHoverLatLng) return;
        const el = editor.pixelReadout || qs('.map-pixel-readout', mapNode());
        const probe = editor.pixelProbe || qs('.raster-hover-probe', mapNode());
        if(el) el.classList.add('active');
        if(probe) probe.classList.add('active');
        const result = activeRasterResult();
        if(result && result.tif_path && rasterCacheMatches(result)){
          renderRealtimeValue(editor.lastHoverLatLng, editor.lastHoverEvent);
        }
      }, 350);
    }
  }
  function boundsFromSamplePoints(points){
    if(!Array.isArray(points) || !points.length) return null;
    let south = Infinity, west = Infinity, north = -Infinity, east = -Infinity;
    points.forEach(p => {
      const lat = Number(p?.lat), lon = Number(p?.lon);
      if(!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      south = Math.min(south, lat); north = Math.max(north, lat);
      west = Math.min(west, lon); east = Math.max(east, lon);
    });
    if(!Number.isFinite(south) || !Number.isFinite(west) || !Number.isFinite(north) || !Number.isFinite(east)) return null;
    if(Math.abs(north - south) < 1e-6){ north += 0.01; south -= 0.01; }
    if(Math.abs(east - west) < 1e-6){ east += 0.01; west -= 0.01; }
    return [[south, west], [north, east]];
  }

  function applySamplePoints(points){
    if(!editor.map || !window.L) return;
    if(editor.sampleLayer){ try { editor.map.removeLayer(editor.sampleLayer); } catch(e){} editor.sampleLayer = null; }
    if(!Array.isArray(points) || !points.length) return;
    const group = window.L.layerGroup();
    const maxN = Math.min(points.length, 1500);
    for(let i=0; i<maxN; i++){
      const p = points[i] || {};
      const lat = Number(p.lat), lon = Number(p.lon);
      if(!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      const somText = (p.som !== undefined && p.som !== null) ? `<br/>SOM：${p.som}` : '';
      const marker = window.L.circleMarker([lat, lon], {
        radius: 1.8,
        color: '#ffffff',
        weight: 0.6,
        fillColor: '#2563eb',
        fillOpacity: 0.62,
        opacity: 0.72,
        interactive: true,
        pane: 'markerPane'
      }).bindPopup(`样点<br/>经度：${lon.toFixed(5)}<br/>纬度：${lat.toFixed(5)}${somText}`);
      group.addLayer(marker);
    }
    editor.sampleLayer = group.addTo(editor.map);
  }

  function clearCovariateOverlays(){
    if(Array.isArray(editor.covariateOverlays)){
      editor.covariateOverlays.forEach(layer => { try { if(editor.map) editor.map.removeLayer(layer); } catch(e){} });
    }
    editor.covariateOverlays = [];
  }

  function stableStringifyBounds(bounds){
    try { return JSON.stringify(bounds || null); } catch(e) { return String(bounds || ''); }
  }
  function overlayBundleSignature(bundle){
    if(!bundle) return 'empty';
    if(Array.isArray(bundle.overlay_stack) && bundle.overlay_stack.length){
      return 'stack::' + bundle.overlay_stack.map(layer => {
        if(!layer) return 'null';
        return [layer.tif_path || '', layer.overlay_url || '', stableStringifyBounds(layer.bounds), layer.opacity ?? layer.stack_opacity ?? ''].join('|');
      }).join('::');
    }
    if(bundle.overlay_url && bundle.bounds){
      return ['single', bundle.tif_path || '', bundle.overlay_url || '', stableStringifyBounds(bundle.bounds), bundle.opacity ?? '', bundle.source_name || '', bundle.kind || ''].join('|');
    }
    return 'empty';
  }


  function collectPreloadableOverlayUrls(payload){
    const urls = [];
    const add = (bundle) => {
      if(!bundle) return;
      if(Array.isArray(bundle.overlay_stack)){
        bundle.overlay_stack.forEach(add);
        return;
      }
      if(bundle.overlay_url) urls.push(String(bundle.overlay_url));
    };
    getLayouts(payload).forEach(layout => add(layout?.result));
    add(payload?.result);
    return Array.from(new Set(urls));
  }

  function preloadImageUrl(url){
    if(!url) return;
    if(editor.overlayImageCache.has(url)) return;
    const img = new Image();
    editor.overlayImageCache.set(url, img);
    img.decoding = 'async';
    img.loading = 'eager';
    img.src = url;
  }

  function preloadAllLayoutOverlays(payload){
    if(!payload) return;
    const urls = collectPreloadableOverlayUrls(payload);
    const sig = urls.join('|');
    if(sig && sig === editor.preloadedLayoutSig) return;
    editor.preloadedLayoutSig = sig;
    // Browser image cache warm-up: all uploaded raster preview PNGs are fetched
    // once after upload/preview publication, so later dropdown switching reuses
    // cached images instead of waiting for network/disk reads.
    urls.forEach(preloadImageUrl);
  }
  function sampleLayerSignature(bundle){
    const enabled = areSamplePointsEnabled();
    if(!enabled) return 'off';
    const pts = currentSamplePoints(bundle);
    if(!Array.isArray(pts) || !pts.length) return 'on:0';
    const first = pts[0] || {};
    const last = pts[pts.length - 1] || {};
    return ['on', pts.length, first.lon, first.lat, first.som, last.lon, last.lat, last.som].join('|');
  }
  function syncSampleLayer(bundle, force){
    const sig = sampleLayerSignature(bundle);
    if(!force && editor.sampleSignature === sig) return;
    editor.sampleSignature = sig;
    if(areSamplePointsEnabled()) applySamplePoints(currentSamplePoints(bundle));
    else if(editor.sampleLayer){ try { editor.map.removeLayer(editor.sampleLayer); } catch(e){} editor.sampleLayer = null; }
  }

  function applyResultOverlay(bundle){
    if(!editor.map) return;
    const nextSig = overlayBundleSignature(bundle);
    const sameOverlay = editor.overlaySignature === nextSig;

    // V174：Dash/前端每500ms会轮询一次 payload。若图层没有变，不能反复
    // remove/add imageOverlay，也不能重置 rasterCache；否则悬浮值框会回到
    // “正在载入前端像元缓存…”，同时地图层会闪烁。
    if(sameOverlay){
      if(bundle && bundle.overlay_url && bundle.bounds){
        editor.activeRaster = bundle;
        if(!rasterCacheMatches(bundle) && !(editor.rasterCache && editor.rasterCache.tif_path === bundle.tif_path && editor.rasterCache.loading)){
          preloadRasterCache(bundle);
        }
      }
      syncSampleLayer(bundle, false);
      return;
    }

    editor.overlaySignature = nextSig;
    clearCovariateOverlays();
    if(editor.resultOverlay){ try { editor.map.removeLayer(editor.resultOverlay); } catch(e){} editor.resultOverlay = null; }
    if(bundle && Array.isArray(bundle.overlay_stack) && bundle.overlay_stack.length){
      editor.activeRaster = null;
      editor.lastHoverPixelKey = '';
      editor.rasterCache = null;
      editor.rasterCacheLoading = false;
      editor.rasterCachePromise = null;
      bundle.overlay_stack.forEach((layer, idx) => {
        if(!layer || !layer.overlay_url || !layer.bounds) return;
        const opacity = Math.max(0.12, Math.min(0.55, Number(layer.stack_opacity ?? layer.opacity ?? 0.42)));
        try {
          const img = window.L.imageOverlay(layer.overlay_url, layer.bounds, { opacity, interactive:false });
          img.addTo(editor.map);
          editor.covariateOverlays.push(img);
        } catch(e) {}
      });
      // 不隐藏悬浮框：若鼠标仍在地图内，保持最后一次读数，避免闪一下消失。
    } else if(bundle && bundle.overlay_url && bundle.bounds){
      const cacheStillValid = editor.rasterCache && editor.rasterCache.tif_path === bundle.tif_path && (editor.rasterCache.values || editor.rasterCache.loading || editor.rasterCache.error);
      editor.activeRaster = bundle;
      editor.lastHoverPixelKey = '';
      if(!cacheStillValid){
        editor.rasterCache = null;
        editor.rasterCacheLoading = false;
        editor.rasterCachePromise = null;
      }
      editor.resultOverlay = window.L.imageOverlay(bundle.overlay_url, bundle.bounds, { opacity: bundle.opacity ?? 0.55, interactive:false });
      try { editor.resultOverlay.once('load', function(){ notifyLayerVisible(bundle, 'layer'); }); } catch(e) {}
      editor.resultOverlay.addTo(editor.map);
      if(!cacheStillValid) preloadRasterCache(bundle);
    } else {
      editor.activeRaster = null;
      editor.rasterCache = null;
      editor.rasterCacheLoading = false;
      editor.rasterCachePromise = null;
      if(!editor.hoverInsideMap) hidePixelProbe();
    }
    syncSampleLayer(bundle, true);
    renderItems();
    saveState();
  }

  function showMapFallback(message){
    const mapEl = mapNode();
    if(!mapEl) return;
    mapEl.innerHTML = `<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#475569;font-size:14px;background:#f8fafc;">${message}</div>`;
  }

  function basemapToggleNode(){ return qs('#basemap-toggle-input input[type=checkbox]'); }
  function hasResultOverlay(payload){
    const r = payload?.result || null;
    const hasRaster = !!(r && ((r.overlay_url && r.bounds) || (Array.isArray(r.overlay_stack) && r.overlay_stack.length)));
    const hasSamples = !!(payload?.uploaded_samples && Array.isArray(payload.uploaded_samples.points) && payload.uploaded_samples.points.length);
    const hasUpload = !!(payload?.map?.has_uploaded_data);
    return hasRaster || hasSamples || hasUpload;
  }
  function basemapAutoHideAfterResult(payload){
    const v = payload?.map?.basemap_auto_hide_after_result;
    return v === undefined || v === null ? true : !!v;
  }
  function setBasemapCheckbox(checked){
    const n = basemapToggleNode();
    if(n) n.checked = !!checked;
  }
  function autoSetBasemapForPayload(payload, opts){
    opts = opts || {};
    if(payload?.map?.show_basemap === true){ setBasemapCheckbox(true); return; }
    if(payload?.map?.show_basemap === false){ setBasemapCheckbox(false); return; }
    const hasResult = hasResultOverlay(payload);
    const autoHide = basemapAutoHideAfterResult(payload);
    const key = `${payload?.active_layout_key || 'empty'}::${payload?.result_key || 'empty'}::${hasResult ? 'result' : 'no-result'}::${autoHide ? 'auto' : 'manual'}`;
    if(!opts.force && editor.lastBasemapAutoKey === key) return;
    editor.lastBasemapAutoKey = key;
    // 制图前：显示天地图作空间参照；制图后：默认隐藏底图，让专题栅格成为主视觉。
    // 用户仍可在右侧复选框手动重新打开底图。
    setBasemapCheckbox(!(hasResult && autoHide));
  }
  function isBasemapEnabled(){ return !!basemapToggleNode()?.checked; }
  function applyBasemapVisibility(){
    if(!editor.map) return;
    const show = isBasemapEnabled();
    editor.baseLayers.forEach(layer => {
      try {
        const has = editor.map.hasLayer(layer);
        if(show && !has) editor.map.addLayer(layer);
        if(!show && has) editor.map.removeLayer(layer);
      } catch(e) {}
    });
    const pane = mapNode();
    if(pane) pane.style.background = show ? '#f8fafc' : '#ffffff';
  }

  function initMap(force){
    const mapEl = mapNode();
    if(!mapEl) return;
    if(!window.L){ showMapFallback('Leaflet 未加载，无法显示底图'); return; }
    if(editor.map && !force) { setTimeout(() => editor.map.invalidateSize(), 50); return; }
    clearMap();
    mapEl.innerHTML = '';
    const payload = editor.payload || {};
    const token = payload?.map?.tdt_token || '';
    if(payload?.active_layout_key && payload.active_layout_key !== editor.lastServerActiveLayoutKey){
      editor.lastServerActiveLayoutKey = payload.active_layout_key;
      try {
        const valid = new Set(getLayouts(payload).map(x => x.key));
        const stored = sessionStorage.getItem(activeLayoutPersistKey());
        // A changed server active key means a new task/result intentionally owns
        // the visible layer, e.g. GCP completion should switch to the GCP map.
        // For ordinary polling the key is unchanged and user selection is kept.
        sessionStorage.setItem(activeLayoutPersistKey(), payload.active_layout_key);
      } catch(e) {}
    }
    mapEl.style.background = '#ffffff';
    mapEl.classList.add('realtime-raster-query');
    editor.map = window.L.map(mapEl, {
      zoomControl:false, attributionControl:false, preferCanvas:true,
      scrollWheelZoom:true, dragging:true, doubleClickZoom:true, touchZoom:true, boxZoom:true, keyboard:true
    });
    const vec = window.L.tileLayer('https://t{s}.tianditu.gov.cn/vec_w/wmts?service=wmts&request=GetTile&version=1.0.0&LAYER=vec&style=default&tileMatrixSet=w&format=tiles&TileMatrix={z}&TileRow={y}&TileCol={x}&tk=' + token, {
      subdomains:['0','1','2','3','4','5','6','7'], maxZoom:18, attribution:'天地图 vec_w'
    });
    const cva = window.L.tileLayer('https://t{s}.tianditu.gov.cn/cva_w/wmts?service=wmts&request=GetTile&version=1.0.0&LAYER=cva&style=default&tileMatrixSet=w&format=tiles&TileMatrix={z}&TileRow={y}&TileCol={x}&tk=' + token, {
      subdomains:['0','1','2','3','4','5','6','7'], maxZoom:18, attribution:'天地图 cva_w'
    });
    editor.baseLayers = [vec, cva];
    autoSetBasemapForPayload(payload);

    const saved = loadState(payload);
    if(saved?.mapView?.center && saved?.mapView?.zoom != null){
      editor.map.setView([saved.mapView.center.lat, saved.mapView.center.lng], saved.mapView.zoom);
    } else if(payload?.result?.bounds){
      if(isFormalDeliverablePayload(payload)){
        editor.map.fitBounds(payload.result.bounds, { paddingTopLeft:[140,150], paddingBottomRight:[330,120] });
      } else {
        editor.map.fitBounds(payload.result.bounds, { padding:[20,20] });
      }
    } else {
      const sampleBounds = boundsFromSamplePoints(uploadedSamplePoints());
      if(sampleBounds) editor.map.fitBounds(sampleBounds, { padding:[28,28] });
      else if(payload?.map?.default_bounds) editor.map.fitBounds(payload.map.default_bounds, { padding:[20,20] });
      else editor.map.setView(DEFAULT_CENTER, DEFAULT_ZOOM);
    }
    applyResultOverlay(payload?.result || null);
    preloadAllLayoutOverlays(payload);
    applyBasemapVisibility();
    editor.map.on('zoomend moveend resize', function(){ updateScaleBars(); saveState(); });
    installRasterHover();
    setMapInteractionMode(editor.interactionMode || 'query');
    setTimeout(() => editor.map && editor.map.invalidateSize(), 80); setTimeout(() => editor.map && editor.map.invalidateSize(), 350);
  }

  function frameFromInstruction(payload, fallback){
    const prefs = payload?.layout_preferences || payload?.map_layout_prefs || {};
    const st = stageSize();
    const pct = Number(prefs.map_frame_percent || prefs.mapFramePercent || 0);
    let box = { ...(fallback || responsiveDefaultMapFrame()) };
    if(Number.isFinite(pct) && pct > 0){
      const ratio = Math.max(0.2, Math.min(0.96, pct/100));
      const w = Math.max(360, Math.round(st.w * Math.sqrt(ratio)));
      const h = Math.max(280, Math.round(st.h * Math.sqrt(ratio)));
      box.width = Math.min(w, st.w - 8);
      box.height = Math.min(h, st.h - 8);
    }
    const pos = String(prefs.map_position || '').toLowerCase();
    if(pos === 'center'){
      box.x = Math.round((st.w - box.width)/2); box.y = Math.round((st.h - box.height)/2);
    } else if(pos === 'top_left') { box.x = 16; box.y = 16; }
    else if(pos === 'top_right') { box.x = st.w - box.width - 16; box.y = 16; }
    else if(pos === 'bottom_left') { box.x = 16; box.y = st.h - box.height - 16; }
    else if(pos === 'bottom_right') { box.x = st.w - box.width - 16; box.y = st.h - box.height - 16; }
    return normalizeMapFrame(box);
  }

  function buildState(payload){
    const saved = loadState(payload) || {};
    editor.payload = payload;
    const instructed = frameFromInstruction(payload, saved.mapFrame || responsiveDefaultMapFrame());
    editor.mapFrame = normalizeMapFrame(instructed);
    if(isFormalDeliverablePayload(payload)){
      const manualItems = Array.isArray(saved.items) ? saved.items.filter(x => !String(x?.id || '').startsWith('text_auto_') && !['north_auto','scale_auto','legend_auto'].includes(String(x?.id || ''))) : [];
      editor.items = [...defaultItems(payload), ...manualItems];
    } else {
      editor.items = saved.items || defaultItems(payload);
    }
    editor.selectedId = null;
    editor.drag = null;
    editor.interactionMode = 'query';
  }

  function beginDrag(kind, targetId, dir, e){
    const item = targetId === 'mapframe' ? null : itemById(targetId);
    if(item && item.locked) return;
    editor.drag = {
      kind, targetId, dir,
      startX: e.clientX, startY: e.clientY,
      baseMapFrame: { ...editor.mapFrame },
      baseItem: item ? { ...item } : null,
    };
    e.preventDefault();
    e.stopPropagation();
  }

  function onMove(e){
    const d = editor.drag;
    if(!d) return;
    const dx = e.clientX - d.startX, dy = e.clientY - d.startY;
    const guideV = qs('#snap-guide-v'), guideH = qs('#snap-guide-h');
    if(guideV) guideV.style.display = 'none';
    if(guideH) guideH.style.display = 'none';
    if(d.targetId === 'mapframe'){
      let { x, y, width, height } = d.baseMapFrame;
      const stage = stageSize();
      if(d.kind === 'move'){
        x = clamp(d.baseMapFrame.x + dx, 0, stage.w - width);
        y = clamp(d.baseMapFrame.y + dy, 0, stage.h - height);
      } else {
        if(d.dir.includes('e')) width = clamp(d.baseMapFrame.width + dx, 240, stage.w - x);
        if(d.dir.includes('s')) height = clamp(d.baseMapFrame.height + dy, 180, stage.h - y);
        if(d.dir.includes('w')) { const nx = clamp(d.baseMapFrame.x + dx, 0, d.baseMapFrame.x + d.baseMapFrame.width - 240); width = d.baseMapFrame.width + (d.baseMapFrame.x - nx); x = nx; }
        if(d.dir.includes('n')) { const ny = clamp(d.baseMapFrame.y + dy, 0, d.baseMapFrame.y + d.baseMapFrame.height - 180); height = d.baseMapFrame.height + (d.baseMapFrame.y - ny); y = ny; }
      }
      editor.mapFrame = { x,y,width,height };
      renderMapFrame();
      selectById('mapframe');
      return;
    }
    const item = itemById(d.targetId);
    if(!item || !d.baseItem) return;
    let { x, y, width, height } = d.baseItem;
    const stage = stageSize();
    if(d.kind === 'move'){
      x = clamp(d.baseItem.x + dx, 0, stage.w - width);
      y = clamp(d.baseItem.y + dy, 0, stage.h - height);
    } else {
      if(d.dir.includes('e')) width = clamp(d.baseItem.width + dx, 24, stage.w - x);
      if(d.dir.includes('s')) height = clamp(d.baseItem.height + dy, 24, stage.h - y);
      if(d.dir.includes('w')) { const nx = clamp(d.baseItem.x + dx, 0, d.baseItem.x + d.baseItem.width - 24); width = d.baseItem.width + (d.baseItem.x - nx); x = nx; }
      if(d.dir.includes('n')) { const ny = clamp(d.baseItem.y + dy, 0, d.baseItem.y + d.baseItem.height - 24); height = d.baseItem.height + (d.baseItem.y - ny); y = ny; }
      if(item.type === 'circle'){ const s = Math.max(width,height); width = s; height = s; }
    }
    const stForSnap = stageSize();
    const cx = x + width/2, cy = y + height/2;
    if(Math.abs(cx - stForSnap.w/2) < 8){ x = Math.round((stForSnap.w - width)/2); const gv = qs('#snap-guide-v'); if(gv){ gv.style.left = `${Math.round(stForSnap.w/2)}px`; gv.style.display = 'block'; } }
    if(Math.abs(cy - stForSnap.h/2) < 8){ y = Math.round((stForSnap.h - height)/2); const gh = qs('#snap-guide-h'); if(gh){ gh.style.top = `${Math.round(stForSnap.h/2)}px`; gh.style.display = 'block'; } }
    Object.assign(item, {x,y,width,height});
    renderItems();
    selectById(item.id);
  }
  function endDrag(){ if(editor.drag){ editor.drag = null; const gv=qs('#snap-guide-v'), gh=qs('#snap-guide-h'); if(gv) gv.style.display='none'; if(gh) gh.style.display='none'; saveState(); } }

  function addItem(type, options){
    options = options || {};
    const uid = `${type}_${Date.now()}_${Math.floor(Math.random()*9999)}`;
    const stage = stageSize();
    const base = { x:Math.max(40, Math.floor(stage.w*0.3)), y:Math.max(90, Math.floor(stage.h*0.22)), width:160, height:60, zIndex:60 + editor.items.length, fill:'#ffffff', stroke:'#17345a', textColor:'#17345a', fontSize:22, opacity:1, preset:'default' };
    let item = null;
    if(type === 'text') item = { ...base, id:uid, type, text:(options.text || '新建文字'), width:190, height:56, preset:(options.preset || 'banner') };
    if(type === 'image') item = { ...base, id:uid, type, src:(options.src || placeholderSvg('图片')), width:220, height:140 };
    if(type === 'rect') item = { ...base, id:uid, type, width:180, height:100, fill:'rgba(255,255,255,0.25)' };
    if(type === 'circle') item = { ...base, id:uid, type, width:120, height:120, fill:'rgba(255,255,255,0.25)' };
    if(type === 'line') item = { ...base, id:uid, type, width:180, height:6, fill:'transparent' };
    if(type === 'arrow') item = { ...base, id:uid, type, width:180, height:28, fill:'transparent' };
    if(type === 'north_arrow') item = { ...base, id:uid, type, width:84, height:84, fill:'transparent', preset:(options.preset || 'arcgis') };
    if(type === 'scale_bar') item = { ...base, id:uid, type, width:280, height:(options.preset === 'line-ticks' ? 48 : 72), preset:(options.preset || 'line-ticks') };
    if(type === 'legend') item = { ...base, id:uid, type, width:130, height:320, preset:(options.preset || 'card'), legendTitle: editor.payload?.result?.legend_title || '值' };
    if(!item) return;
    editor.items.push(item);
    renderItems();
    selectById(item.id);
  }
  function deleteSelected(){
    if(!editor.selectedId || editor.selectedId === 'mapframe') return;
    editor.items = editor.items.filter(x => x.id !== editor.selectedId);
    editor.selectedId = null;
    renderItems();
    selectById(null);
  }

  function applyPresetOrAdd(type, preset){
    const item = itemById(editor.selectedId);
    if(item && item.type === type){
      item.preset = preset || item.preset || 'default';
      if(type === 'scale_bar'){
        item.height = preset === 'line-ticks' || preset === 'qgis-line' ? Math.min(item.height || 54, 58) : Math.max(item.height || 68, 68);
      }
      renderItems();
      selectById(item.id);
      saveState();
      return;
    }
    addItem(type, { preset });
  }

  function updateStyleSelectionHighlights(){
    const item = itemById(editor.selectedId);
    qsa('.preset-btn[data-add-type]').forEach(btn => {
      const sameType = !!item && item.type === btn.dataset.addType;
      const samePreset = !!item && String(item.preset || '') === String(btn.dataset.preset || '');
      btn.classList.toggle('active', sameType && samePreset);
      btn.classList.toggle('compatible', sameType);
    });
  }


  function getPaletteColorsByValue(value){
    const btn = qs(`.palette-swatch-btn[data-palette="${value}"]`);
    if(!btn) return null;
    const colorsRaw = btn.dataset.colors || '';
    if(!colorsRaw) return null;
    try { return JSON.parse(colorsRaw); } catch(e) { return null; }
  }

  function setPaletteValue(value, opts){
    opts = opts || {};
    if(!value) return;
    const sel = qs('#result-ramp-select');
    if(sel) sel.value = value;
    qsa('.palette-swatch-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.palette === value));
    const active = qs(`.palette-swatch-btn[data-palette="${value}"]`);
    const label = qs('#palette-summary-label');
    const bar = qs('.palette-summary-bar');
    if(active && label){
      const nameNode = active.querySelector('.palette-swatch-label');
      label.textContent = nameNode ? nameNode.textContent : value;
    }
    if(active && bar){
      const sw = active.querySelector('.palette-swatch-bar');
      if(sw) bar.style.background = sw.style.background;
    }
    if(opts.user){
      editor.pendingPalette = value;
      if(editor.payload && editor.payload.result){ editor.payload.result.palette = value; }
      try {
        sessionStorage.setItem(palettePersistKey(), value);
        const tif = editor.payload?.result?.tif_path || '';
        if(tif) sessionStorage.setItem(paletteTifKey(), tif);
      } catch(e) {}
    }
    if(opts.server){
      editor.lastServerPalette = value;
      editor.pendingPalette = null;
      try {
        sessionStorage.removeItem(palettePersistKey());
        sessionStorage.removeItem(paletteTifKey());
      } catch(e) {}
    }
  }

  async function applyResultStyle(){
    const result = editor.payload?.result;
    if(!result || !result.tif_path) return;
    const palette = editor.pendingPalette || qs('#result-ramp-select')?.value || result.palette;
    const paletteColors = getPaletteColorsByValue(palette) || result.palette_colors || null;
    const opacity = Number(qs('#result-opacity-range')?.value || result.opacity || 1.0);
    const reverse = !!qs('#result-reverse-input input:checked');
    const resp = await fetch('/__render_overlay', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ tif_path: result.tif_path, palette, palette_colors: paletteColors, opacity, reverse, admin_region: result.admin_region || '', clcd_mask_path: result.clcd_mask_path || '', noncropland_mode: result.noncropland_mode || '', noncropland_color: result.noncropland_color || '' }) });
    const data = await resp.json();
    if(!data.ok){ alert(data.error || '结果层样式更新失败'); return; }
    editor.payload.result = { ...result, ...data, legend_title: result.legend_title || '值', kind: result.kind, sample_points: result.sample_points, sample_point_count: result.sample_point_count };
    setPaletteValue(data.palette || palette, { user: true });
    try { sessionStorage.setItem(styleBundlePersistKey(), JSON.stringify(editor.payload.result)); } catch(e) {}
    applyResultOverlay(editor.payload.result);
    // 等前端图层实际替换后再在对话历史中提示“完成”。
    setTimeout(() => notifyLayerVisible(editor.payload.result, 'style'), 80);
  }

  function handleLocalImageSelection(file){
    if(!file) return;
    const reader = new FileReader();
    reader.onload = function(ev){ addItem('image', { src: ev.target?.result || placeholderSvg(file.name || '图片') }); };
    reader.readAsDataURL(file);
  }

  function currentExportDpi(){
    const v = Number(qs('#export-dpi-select')?.value || 300);
    return [150,300,600].includes(v) ? v : 300;
  }
  async function exportLayout(format){
    const stage = qs('#layout-stage');
    if(!stage) return;
    if(format === 'tif'){
      const tif = editor.payload?.result?.tif_path;
      if(!tif){ alert('当前没有可导出的 GeoTIFF 结果。'); return; }
      const link = document.createElement('a');
      link.href = `/__local_file?path=${encodeURIComponent(tif)}`;
      link.download = tif.split(/[\\/]/).pop() || `result_${Date.now()}.tif`;
      document.body.appendChild(link); link.click(); link.remove();
      return;
    }
    const prev = editor.selectedId;
    selectById(null);
    hideContextMenu();
    stage.classList.add('exporting');
    const dpi = currentExportDpi();
    const scale = Math.max(1, dpi / 150);
    try {
      if(format === 'pdf'){
        const canvas = window.html2canvas ? await window.html2canvas(stage, {backgroundColor:'#ffffff', scale, useCORS:true}) : null;
        const popup = window.open('', '_blank');
        if(canvas){
          const img = canvas.toDataURL('image/png');
          popup.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>布局导出 PDF</title><style>@page{size:auto;margin:10mm;}body{margin:0;background:#fff;text-align:center;}img{max-width:100%;height:auto;}</style></head><body><img src="${img}"></body></html>`);
        } else {
          const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"], style')).map(n => n.outerHTML).join('\n');
          popup.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>布局导出 PDF</title>${styles}<style>body{margin:0;background:#fff;padding:12px;box-sizing:border-box;} #layout-stage{margin:0 auto;}</style></head><body>${stage.outerHTML}</body></html>`);
        }
        popup.document.close(); popup.focus(); setTimeout(() => popup.print(), 500);
      } else {
        if(!window.html2canvas){ alert('未加载 html2canvas，暂时无法导出 PNG。'); return; }
        const canvas = await window.html2canvas(stage, {backgroundColor:'#ffffff', scale, useCORS:true});
        const link = document.createElement('a');
        link.download = `layout_export_${dpi}dpi_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      }
    } finally {
      stage.classList.remove('exporting');
      if(prev) selectById(prev);
    }
  }

  function bindUi(){
    if(editor.uiBound) return;
    editor.uiBound = true;
    document.addEventListener('click', function(e){
      const imageBtn = e.target.closest('.tool-btn[data-add-type="image"]');
      if(imageBtn){
        e.preventDefault();
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.style.display = 'none';
        input.addEventListener('change', function(ev){
          const file = ev.target && ev.target.files && ev.target.files[0];
          if(file) handleLocalImageSelection(file);
          try { input.remove(); } catch(e) {}
        }, { once: true });
        document.body.appendChild(input);
        input.click();
        return;
      }
      const addBtn = e.target.closest('.tool-btn[data-add-type], .preset-btn[data-add-type]');
      if(addBtn){
        e.preventDefault();
        const type = addBtn.dataset.addType;
        const preset = addBtn.dataset.preset || undefined;
        if(addBtn.classList.contains('preset-btn')) applyPresetOrAdd(type, preset);
        else addItem(type, { preset });
        return;
      }
      const paletteBtn = e.target.closest('.palette-swatch-btn[data-palette]');
      if(paletteBtn){
        e.preventDefault();
        setPaletteValue(paletteBtn.dataset.palette, { user: true });
        const dd = qs('#palette-dropdown');
        if(dd) dd.removeAttribute('open');
        applyResultStyle();
        return;
      }
      const tabBtn = e.target.closest('.layout-tab-btn[data-layout-key]');
      if(tabBtn){ e.preventDefault(); switchLayout(tabBtn.dataset.layoutKey); return; }
      const sourceChoiceBtn = e.target.closest('#mapping-source-user-btn, #mapping-source-domestic-btn, #mapping-source-gee-btn');
      if(sourceChoiceBtn){
        sourceChoiceBtn.classList.add('choice-pending');
        const strong = sourceChoiceBtn.querySelector('strong');
        if(strong && !strong.dataset.oldText){
          strong.dataset.oldText = strong.textContent || '';
          strong.textContent = '已选择，正在启动任务…';
        }
      }
      const layerRow = e.target.closest('.layer-row[data-layer-id]');
      if(layerRow){ e.preventDefault(); selectById(layerRow.dataset.layerId); hideContextMenu(); return; }
      const menuBtn = e.target.closest('[data-menu-action]');
      if(menuBtn){
        e.preventDefault();
        const a = menuBtn.dataset.menuAction;
        if(a === 'duplicate') duplicateSelected();
        if(a === 'delete') deleteSelected();
        if(a === 'top') moveSelectedLayer('top');
        if(a === 'bottom') moveSelectedLayer('bottom');
        if(a === 'align-h') alignSelected('h');
        if(a === 'align-v') alignSelected('v');
        if(a === 'lock') toggleLockSelected();
        hideContextMenu(); return;
      }
      if(e.target.closest('#btn-layer-up')){ e.preventDefault(); moveSelectedLayer('up'); return; }
      if(e.target.closest('#btn-layer-down')){ e.preventDefault(); moveSelectedLayer('down'); return; }
      if(e.target.closest('#btn-layer-top')){ e.preventDefault(); moveSelectedLayer('top'); return; }
      if(e.target.closest('#btn-layer-bottom')){ e.preventDefault(); moveSelectedLayer('bottom'); return; }
      if(e.target.closest('#btn-align-h')){ e.preventDefault(); alignSelected('h'); return; }
      if(e.target.closest('#btn-align-v')){ e.preventDefault(); alignSelected('v'); return; }
      if(e.target.closest('#btn-duplicate-item')){ e.preventDefault(); duplicateSelected(); return; }
      if(e.target.closest('#btn-save-template')){ e.preventDefault(); saveTemplate(); return; }
      if(e.target.closest('#btn-load-template')){ e.preventDefault(); loadTemplate(); return; }
      if(e.target.closest('#btn-delete-item')){ e.preventDefault(); deleteSelected(); return; }
      if(e.target.closest('#btn-export-png')){ e.preventDefault(); exportLayout('png'); return; }
      if(e.target.closest('#btn-export-pdf')){ e.preventDefault(); exportLayout('pdf'); return; }
      if(e.target.closest('#btn-export-tif')){ e.preventDefault(); exportLayout('tif'); return; }
      if(e.target.closest('#apply-result-style-btn')){ e.preventDefault(); applyResultStyle(); return; }
    }, true);
    document.addEventListener('contextmenu', function(e){
      const itemEl = e.target.closest('.layout-item');
      const mapEl = e.target.closest('#map-frame-item');
      if(itemEl && itemEl.closest('#overlay-layer')){ selectById(itemEl.dataset.id); e.preventDefault(); showContextMenu(e.clientX, e.clientY); return; }
      if(mapEl){
        if(editor.interactionMode !== 'layout') return;
        selectById('mapframe'); e.preventDefault(); showContextMenu(e.clientX, e.clientY); return;
      }
      hideContextMenu();
    }, true);
    document.addEventListener('change', function(e){
      if(e.target && e.target.closest && e.target.closest('#layout-select')){
        const key = e.target.value;
        if(key) switchLayout(key);
      }
      if(e.target && e.target.closest && e.target.closest('#basemap-toggle-input')){ applyBasemapVisibility(); }
      if(e.target && e.target.closest && e.target.closest('#sample-points-toggle-input')){ applyResultOverlay(editor.payload?.result || null); }
      if(e.target && e.target.id === 'result-ramp-select'){
        setPaletteValue(e.target.value, { user: true });
        applyResultStyle();
      }
    }, true);
    document.addEventListener('mousedown', function(e){
      const modeBtn = e.target.closest('.map-mode-btn');
      if(modeBtn){
        e.preventDefault();
        e.stopPropagation();
        setMapInteractionMode(modeBtn.dataset.mapMode === 'layout' ? 'layout' : 'query');
        return;
      }
      const resize = e.target.closest('.resize-handle');
      const itemEl = e.target.closest('.layout-item');
      if(itemEl && itemEl.closest('#overlay-layer')){
        selectById(itemEl.dataset.id);
        beginDrag(resize ? 'resize' : 'move', itemEl.dataset.id, resize?.dataset.dir || '', e);
        return;
      }
      if(resize && resize.closest('#map-frame-item')){
        if(editor.interactionMode !== 'layout') return;
        selectById('mapframe');
        beginDrag('resize', 'mapframe', resize.dataset.dir || 'se', e);
        return;
      }
      if(e.target.closest('.map-frame-handle')){
        if(editor.interactionMode !== 'layout') return;
        selectById('mapframe');
        beginDrag('move', 'mapframe', '', e);
        return;
      }
      if(e.target.closest('#map-frame-item')){
        if(editor.interactionMode !== 'layout') return;
        selectById('mapframe'); return;
      }
      if(!e.target.closest('.side-panel')) { selectById(null); hideContextMenu(); }
    }, true);
    window.addEventListener('mousemove', onMove, true);
    window.addEventListener('mouseup', endDrag, true);
  }

  function mountOrUpdate(){
    const stage = stageNode();
    const payload = parsePayload();
    syncStageHeight();
    if(!stage || !payload) return;
    if(stage.clientWidth < 240 || stage.clientHeight < 240) return;
    bindUi();

    if(payload?.active_layout_key && payload.active_layout_key !== editor.lastServerActiveLayoutKey){
      editor.lastServerActiveLayoutKey = payload.active_layout_key;
      try {
        const valid = new Set(getLayouts(payload).map(x => x.key));
        const stored = sessionStorage.getItem(activeLayoutPersistKey());
        // A changed server active key means a new task/result intentionally owns
        // the visible layer, e.g. GCP completion should switch to the GCP map.
        // For ordinary polling the key is unchanged and user selection is kept.
        sessionStorage.setItem(activeLayoutPersistKey(), payload.active_layout_key);
      } catch(e) {}
    }

    const preferredKey = preferredActiveLayoutKey(payload);
    const layout = getLayoutByKey(payload, preferredKey);
    if(layout){
      payload.active_layout_key = preferredKey;
      payload.title = layout.title || payload.title;
      payload.result = layout.result || null;
      payload.result_key = layout.result_key || 'empty';
      mergePersistedStyleBundle(payload);
    }

    editor.activeLayoutKey = preferredKey;
    preloadAllLayoutOverlays(payload);

    const serverPalette = (payload?.result?.palette) || '';
    const resultTif = payload?.result?.tif_path || '';
    let storedPalette = null;
    let storedPaletteTif = null;
    try {
      storedPalette = sessionStorage.getItem(palettePersistKey());
      storedPaletteTif = sessionStorage.getItem(paletteTifKey());
    } catch(e) {}
    if(storedPalette && storedPaletteTif && resultTif && storedPaletteTif !== resultTif){
      storedPalette = null;
      try {
        sessionStorage.removeItem(palettePersistKey());
        sessionStorage.removeItem(paletteTifKey());
      } catch(e) {}
    }
    const currentDomPalette = qs('#result-ramp-select')?.value || '';
    const desiredPalette = editor.pendingPalette || storedPalette || currentDomPalette || serverPalette || '';

    const nextKey = `${preferredKey || 'empty'}::${payload?.result_key || 'empty'}`;
    if(editor.key !== nextKey){
      editor.key = nextKey;
      buildState(payload);
      renderMapFrame();
      renderItems();
      initMap(true);
      if(desiredPalette && desiredPalette !== qs('#result-ramp-select')?.value) setPaletteValue(desiredPalette, {});
      markActiveTab();
      updateRightPanelForActiveLayout();
      setMapInteractionMode('query');
      return;
    }

    editor.payload = payload;
    editor.mapFrame = normalizeMapFrame(editor.mapFrame);
    renderMapFrame();
    if(!editor.map) initMap(true);
    else applyResultOverlay(payload?.result || null);
    if(desiredPalette) setPaletteValue(desiredPalette, {});
    markActiveTab();
    updateRightPanelForActiveLayout();
  }

  function boot(){
    syncStageHeight();
    mountOrUpdate();
    refreshStyleCatalog();
    if(window.__somLayoutLoop) return;
    window.__somLayoutLoop = window.setInterval(mountOrUpdate, 500);
    if(!window.__somStyleLoop) window.__somStyleLoop = window.setInterval(refreshStyleCatalog, 15000);
  }

  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, {once:true});
  else boot();
  window.addEventListener('load', boot, {once:true});
  window.addEventListener('resize', function(){ syncStageHeight(); mountOrUpdate(); });
})();
