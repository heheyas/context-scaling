// ==================== Client-side config ====================
// This demo talks to a local FastAPI backend (see demo/app.py). All model
// inference — Structured Prompt generation via a text LLM AND image
// rendering via the QwenImage DiT — happens on the server side, so the
// browser never sees an API key or an SP system prompt.
//
// These constants are kept only so the rest of the (studio-derived)
// script continues to reference them without changes.
const DEFAULT_SYSTEM_PROMPT = "";
const DEFAULT_API_KEYS = [];


// ==================== Constants ====================
const BBOX_COLORS = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
            '#2980b9', '#27ae60', '#d35400', '#8e44ad', '#17a2b8',
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd'
        ];

const HISTORY_COLORS = ['hc-0', 'hc-1', 'hc-2', 'hc-3', 'hc-4', 'hc-5'];

// ==================== Stable JSON Stringify ====================
// Preserve object key order based on previous JSON (including nested objects in arrays).
// This avoids JS's default integer-like key reordering (e.g. "123" jumping to the front).
function _joinPath(pathKeys) {
    return pathKeys.join('\0');
}

// 直接扫描 JSON 原始字符串来提取 key 顺序，避免 JSON.parse 自动把数字 key 排到前面
function _extractKeyOrdersFromJSON(jsonStr) {
    if (!jsonStr || !jsonStr.trim()) return {};
    try {
        const keyOrders = {};
        const pathStack = []; // {type:'obj'|'arr', keys:[], currentKey:null, index:0}
        let i = 0;
        const len = jsonStr.length;

        function skipWS() { while (i < len && /\s/.test(jsonStr[i])) i++; }

        function readStr() {
            if (jsonStr[i] !== '"') return null;
            i++; // skip "
            let s = '';
            while (i < len && jsonStr[i] !== '"') {
                if (jsonStr[i] === '\\') { s += jsonStr[i] + jsonStr[i + 1]; i += 2; }
                else { s += jsonStr[i]; i++; }
            }
            i++; // skip closing "
            try { return JSON.parse('"' + s + '"'); } catch { return s; }
        }

        function curPath() {
            const parts = [];
            for (const f of pathStack) {
                if (f.type === 'arr') parts.push(String(f.index));
                else if (f.currentKey !== null) parts.push(f.currentKey);
            }
            return parts;
        }

        while (i < len) {
            skipWS();
            if (i >= len) break;
            const ch = jsonStr[i];
            if (ch === '{') {
                const p = curPath();
                const frame = { type: 'obj', keys: [], currentKey: null };
                pathStack.push(frame);
                keyOrders[_joinPath(p)] = frame.keys;
                i++;
            } else if (ch === '}') {
                pathStack.pop();
                i++;
                if (pathStack.length > 0 && pathStack[pathStack.length - 1].type === 'obj') {
                    pathStack[pathStack.length - 1].currentKey = null;
                }
            } else if (ch === '[') {
                pathStack.push({ type: 'arr', index: 0 });
                i++;
            } else if (ch === ']') {
                pathStack.pop();
                i++;
                if (pathStack.length > 0 && pathStack[pathStack.length - 1].type === 'obj') {
                    pathStack[pathStack.length - 1].currentKey = null;
                }
            } else if (ch === ',') {
                i++;
                if (pathStack.length > 0) {
                    const top = pathStack[pathStack.length - 1];
                    if (top.type === 'arr') top.index++;
                    else top.currentKey = null;
                }
            } else if (ch === ':') {
                i++;
            } else if (ch === '"') {
                const str = readStr();
                skipWS();
                if (i < len && jsonStr[i] === ':') {
                    // It's an object key
                    const top = pathStack[pathStack.length - 1];
                    if (top && top.type === 'obj') {
                        top.keys.push(str);
                        top.currentKey = str;
                    }
                }
                // else it's a string value — do nothing
            } else {
                // number / boolean / null
                const m = jsonStr.substring(i).match(/^(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/);
                if (m) i += m[0].length; else i++;
            }
        }
        return keyOrders;
    } catch (e) {
        return {};
    }
}

function _stableStr(val, keyOrders, pathKeys, level) {
    if (val === null || val === undefined) return 'null';
    if (typeof val === 'boolean') return val ? 'true' : 'false';
    if (typeof val === 'number') return isFinite(val) ? String(val) : 'null';
    if (typeof val === 'string') return JSON.stringify(val);

    const indentStr = '  '.repeat(level);
    const childIndent = '  '.repeat(level + 1);

    if (Array.isArray(val)) {
        if (val.length === 0) return '[]';
        const items = val.map((item, i) =>
            childIndent + _stableStr(item, keyOrders, [...pathKeys, String(i)], level + 1)
        );
        return '[\n' + items.join(',\n') + '\n' + indentStr + ']';
    }

    // Object
    const allKeys = Object.keys(val);
    if (allKeys.length === 0) return '{}';

    // Get reference key order for this path
    const parentPath = _joinPath(pathKeys);
    const refKeys = keyOrders[parentPath] || [];

    // Order: reference keys first (in reference order), then remaining keys (appended)
    const orderedKeys = [];
    refKeys.forEach(k => { if (allKeys.includes(k)) orderedKeys.push(k); });
    allKeys.forEach(k => { if (!orderedKeys.includes(k)) orderedKeys.push(k); });

    const items = orderedKeys.map(k =>
        childIndent + JSON.stringify(k) + ': ' + _stableStr(val[k], keyOrders, [...pathKeys, k], level + 1)
    );
    return '{\n' + items.join(',\n') + '\n' + indentStr + '}';
}

function stableJSONStringify(obj, referenceStr) {
    try {
        const keyOrders = _extractKeyOrdersFromJSON(referenceStr);
        return _stableStr(obj, keyOrders, [], 0);
    } catch (e) {
        return JSON.stringify(obj, null, 2);
    }
}

// ==================== Global State ====================
const state = {
    // Current image: { src: string (dataURL/URL), width: number, height: number } | null
    currentImage: null,

    // Structured Prompt (JSON string)
    structuredPrompt: '',

    // User Prompt
    userPrompt: '',

    // Generation parameters
    params: {
        forceT2I: false,
        ditEndpoint: '',
        cfgScale: 4.0,
        steps: 25,
        height: 1024,
        width: 1024,
        seed: 42,
    },

    // History: array of { type, inputImage, outputImage, userPrompt, structuredPrompt, params, timestamp, colorIndex }
    history: [],
    selectedHistoryIndex: -1,
    generationCount: 0,

    // SP undo stack
    undoStack: [],

    // UI flags
    bboxVisible: false,
    isGenerating: false,
    pendingGeneration: null,  // callback for diff confirmation
    spRawMode: false,         // true = raw textarea, false = structured view
    activeBboxIndex: -1,      // highlighted bbox element index
    bboxEditIndex: -1,        // bbox in edit mode (double-click to activate)
    focusedBboxIndex: -1,     // 被点击锁定的 bbox index（持久显示）
    // BBox drag state
    bboxDrag: null, // { elemIndex, handle, startX, startY, startBbox:[x1,y1,x2,y2], containerRect }

    // Parsed SP data
    parsedSP: null,           // parsed JSON object

    // Parsed elements from SP (elements + scene.elements with bbox)
    parsedElements: [],

    // Image modal zoom state
    zoomState: { scale: 1, translateX: 0, translateY: 0 },
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', init);

function init() {
    setupDragDrop();
    setupAutoSync();
    setupKeyboardShortcuts();
}

function setupDragDrop() {
    const imageArea = document.getElementById('imageArea');
    const dropzone = document.getElementById('dropzone');

    ['dragenter', 'dragover'].forEach(evt => {
        imageArea.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            if (dropzone) dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        imageArea.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            if (dropzone) dropzone.classList.remove('dragover');
        });
    });

    imageArea.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            loadImageFile(files[0]);
        }
    });

    // Also support paste
    document.addEventListener('paste', e => {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) loadImageFile(file);
                break;
            }
        }
    });
}

function setupAutoSync() {
    // Sync userPrompt from input
    document.getElementById('userPromptInput').addEventListener('input', function () {
        state.userPrompt = this.value;
    });

    // Sync SP from textarea
    document.getElementById('spTextarea').addEventListener('input', function () {
        state.structuredPrompt = this.value;
        updateSPButtons();
        parseSPAndUpdateBboxes();
    });

    // Sync params on change
    ['paramForceT2I', 'paramDitEndpoint', 'paramCfgScale', 'paramSteps', 'paramHeight', 'paramWidth', 'paramSeed'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', syncParamsFromUI);
    });
}

function setupKeyboardShortcuts() {
    // Ctrl+Z when focused on SP textarea => undo
    document.getElementById('spTextarea').addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.key === 'z') {
            e.preventDefault();
            handleSPUndo();
        }
    });

    // Ctrl+Enter in user prompt => generate SP
    document.getElementById('userPromptInput').addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            handleGenerateSP();
        }
    });
}

// ==================== Params Sync ====================
function syncParamsFromUI() {
    state.params.forceT2I = document.getElementById('paramForceT2I').checked;
    state.params.ditEndpoint = (document.getElementById('paramDitEndpoint').value || '').trim();
    state.params.cfgScale = parseFloat(document.getElementById('paramCfgScale').value) || 4.0;
    state.params.steps = parseInt(document.getElementById('paramSteps').value) || 25;
    state.params.height = parseInt(document.getElementById('paramHeight').value) || 1024;
    state.params.width = parseInt(document.getElementById('paramWidth').value) || 1024;
    state.params.seed = parseInt(document.getElementById('paramSeed').value) || 42;
}

function syncParamsToUI() {
    document.getElementById('paramForceT2I').checked = state.params.forceT2I || false;
    document.getElementById('paramDitEndpoint').value = state.params.ditEndpoint || '';
    document.getElementById('paramCfgScale').value = state.params.cfgScale !== undefined ? state.params.cfgScale : 4.0;
    document.getElementById('paramSteps').value = state.params.steps;
    document.getElementById('paramHeight').value = state.params.height;
    document.getElementById('paramWidth').value = state.params.width;
    document.getElementById('paramSeed').value = state.params.seed;
}

// ==================== Image Panel ====================
function triggerImageUpload() {
    document.getElementById('imageFileInput').click();
}

function handleImageFileSelected(event) {
    const file = event.target.files[0];
    if (file) loadImageFile(file);
    event.target.value = '';
}

function loadImageFile(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            state.currentImage = {
                src: e.target.result,
                width: img.naturalWidth,
                height: img.naturalHeight,
            };
            updateImagePanel();
            autoUpdateHW();
            showToast('图片已加载', 'success');
        };
        img.onerror = function () {
            showToast('图片加载失败', 'error');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function loadImageFromSrc(src) {
    const img = new Image();
    img.onload = function () {
        state.currentImage = {
            src: src,
            width: img.naturalWidth,
            height: img.naturalHeight,
        };
        updateImagePanel();
    };
    img.src = src;
}

function autoUpdateHW() {
    if (!state.currentImage) return;
    // Round to nearest 32
    state.params.height = Math.round(state.currentImage.height / 32) * 32;
    state.params.width = Math.round(state.currentImage.width / 32) * 32;
    syncParamsToUI();
}

function handleImageDelete() {
    state.currentImage = null;
    state.bboxVisible = false;
    state.activeBboxIndex = -1;
    updateImagePanel();
    showToast('图片已删除', 'info');
}

function handleImageEnlarge() {
    if (!state.currentImage) return;
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('modalImage');
    img.src = state.currentImage.src;

    // Reset zoom
    state.zoomState = { scale: 1, translateX: 0, translateY: 0 };
    applyModalZoom();

    modal.classList.add('visible');

    // Setup zoom listeners
    const container = document.getElementById('modalImageContainer');
    container.addEventListener('wheel', onModalWheel, { passive: false });
    container.addEventListener('mousedown', onModalPanStart);
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('visible');
    const container = document.getElementById('modalImageContainer');
    container.removeEventListener('wheel', onModalWheel);
    container.removeEventListener('mousedown', onModalPanStart);
    state.zoomState = { scale: 1, translateX: 0, translateY: 0 };
}

function onModalWheel(e) {
    e.preventDefault();
    const z = state.zoomState;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.5, Math.min(20, z.scale * factor));

    // Zoom towards mouse position
    const container = document.getElementById('modalImageContainer');
    const rect = container.getBoundingClientRect();
    const mx = e.clientX - rect.left - rect.width / 2;
    const my = e.clientY - rect.top - rect.height / 2;

    const sf = newScale / z.scale;
    z.translateX = mx - sf * (mx - z.translateX);
    z.translateY = my - sf * (my - z.translateY);
    z.scale = newScale;
    applyModalZoom();
}

function onModalPanStart(e) {
    if (e.button !== 0) return;
    if (state.zoomState.scale <= 1.01) return;
    e.preventDefault();
    const z = state.zoomState;
    const startX = e.clientX, startY = e.clientY;
    const startTX = z.translateX, startTY = z.translateY;

    const onMove = (ev) => {
        z.translateX = startTX + (ev.clientX - startX);
        z.translateY = startTY + (ev.clientY - startY);
        applyModalZoom();
    };
    const onUp = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
}

function applyModalZoom() {
    const img = document.getElementById('modalImage');
    const z = state.zoomState;
    img.style.transform = `translate(${z.translateX}px, ${z.translateY}px) scale(${z.scale})`;
    img.style.cursor = z.scale > 1.01 ? 'grab' : 'default';
}

function updateImagePanel() {
    const dropzone = document.getElementById('dropzone');
    const display = document.getElementById('imageDisplay');
    const btnZoom = document.getElementById('btnImageZoom');
    const btnDelete = document.getElementById('btnImageDelete');
    const btnBboxShow = document.getElementById('btnBboxShow');
    const btnBboxHide = document.getElementById('btnBboxHide');

    if (state.currentImage) {
        dropzone.style.display = 'none';
        display.style.display = 'flex';
        document.getElementById('displayImage').src = state.currentImage.src;
        btnZoom.style.display = '';
        btnDelete.style.display = '';
        updateBboxButtons();
        if (state.bboxVisible) {
            requestAnimationFrame(() => renderBboxes());
        }
            } else {
        dropzone.style.display = '';
        display.style.display = 'none';
        btnZoom.style.display = 'none';
        btnDelete.style.display = 'none';
        btnBboxShow.style.display = 'none';
        btnBboxHide.style.display = 'none';
        clearBboxes();
    }
}

function updateBboxButtons() {
    const hasElems = state.parsedElements.length > 0;
    const btnBboxShow = document.getElementById('btnBboxShow');
    const btnBboxHide = document.getElementById('btnBboxHide');
    if (!state.currentImage || !hasElems) {
        btnBboxShow.style.display = 'none';
        btnBboxHide.style.display = 'none';
        return;
    }
    if (state.bboxVisible) {
        btnBboxShow.style.display = 'none';
        btnBboxHide.style.display = '';
    } else {
        btnBboxShow.style.display = '';
        btnBboxHide.style.display = 'none';
    }
}

// ==================== BBox Visualization ====================
function showAllBboxes() {
    if (state.parsedElements.length === 0) {
        parseSPAndUpdateBboxes();
        if (state.parsedElements.length === 0) return;
    }
    state.focusedBboxIndex = -1;
    state.bboxVisible = true;
    state.activeBboxIndex = -1;
    state.bboxEditIndex = -1;
    renderBboxes(-1); // -1 means render all
    updateBboxButtons();
    document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
}

function hideAllBboxes() {
    state.focusedBboxIndex = -1;
    state.bboxVisible = false;
    state.activeBboxIndex = -1;
    state.bboxEditIndex = -1;
    clearBboxes();
    updateBboxButtons();
    document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
}

function handleImageClick(event) {
    // 检测点击位置是否有 element 的 bbox
    if (state.parsedElements.length > 0 && state.currentImage) {
        const imgEl = document.getElementById('displayImage');
        const imgRect = imgEl.getBoundingClientRect();
        const clickX = ((event.clientX - imgRect.left) / imgRect.width) * 1000;
        const clickY = ((event.clientY - imgRect.top) / imgRect.height) * 1000;

        // 找所有包含点击坐标的 bbox
        const overlapping = [];
        state.parsedElements.forEach((elem, i) => {
            if (!elem.bbox) return;
            const [x1, y1, x2, y2] = elem.bbox;
            if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                overlapping.push(i);
            }
        });

        if (overlapping.length > 0) {
            // 在重叠列表中循环切换
            let nextIndex;
            if (state.focusedBboxIndex >= 0 && overlapping.includes(state.focusedBboxIndex)) {
                const currentPos = overlapping.indexOf(state.focusedBboxIndex);
                nextIndex = overlapping[(currentPos + 1) % overlapping.length];
            } else {
                nextIndex = overlapping[0];
            }
            // 聚焦该 element 的 bbox
            focusBbox(nextIndex);
            // 滚动到对应的 element 卡片
            if (state.parsedElements[nextIndex]) {
                const card = document.querySelector(`.sp-element[data-elem-id="${state.parsedElements[nextIndex].id}"]`);
                if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
            return;
        }
    }

    // 点击空白区域 => 清除锁定的 bbox
    state.focusedBboxIndex = -1;
    state.activeBboxIndex = -1;
    state.bboxEditIndex = -1;
    if (state.bboxVisible) {
        clearBboxes();
        state.bboxVisible = false;
        updateBboxButtons();
    }
    document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
}

// filterIndex: 如果 >= 0，只渲染该索引的 bbox；否则渲染全部
function renderBboxes(filterIndex) {
    clearBboxes();
    if (!state.currentImage || state.parsedElements.length === 0) return;

    const imgEl = document.getElementById('displayImage');
    const container = document.getElementById('bboxContainer');

    if (!imgEl || !imgEl.offsetWidth) return;

    const imgRect = imgEl.getBoundingClientRect();
    const parentRect = imgEl.parentElement.getBoundingClientRect();

    container.style.left = (imgRect.left - parentRect.left) + 'px';
    container.style.top = (imgRect.top - parentRect.top) + 'px';
    container.style.width = imgRect.width + 'px';
    container.style.height = imgRect.height + 'px';
    container.style.display = 'block';

    state.parsedElements.forEach((elem, i) => {
        if (!elem.bbox) return;
        // 如果指定了 filterIndex，只渲染该元素
        if (filterIndex !== undefined && filterIndex >= 0 && i !== filterIndex) return;

        const [x1, y1, x2, y2] = elem.bbox;
        const color = getColorForElemId(elem.id);

        const isActive = (i === state.bboxEditIndex);
        const box = document.createElement('div');
        box.className = 'bbox-box' + (isActive ? ' active' : '');
        box.dataset.elemIndex = i;
        box.style.borderColor = color;
        box.style.left = (x1 / 1000 * 100) + '%';
        box.style.top = (y1 / 1000 * 100) + '%';
        box.style.width = ((x2 - x1) / 1000 * 100) + '%';
        box.style.height = ((y2 - y1) / 1000 * 100) + '%';

        // Semi-transparent background
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        box.style.background = `rgba(${r},${g},${b},0.12)`;

        // Label
        const label = document.createElement('div');
        label.className = 'bbox-label';
        label.style.background = color;
        label.textContent = `${elem.id}: ${elem.caption || ''}`.substring(0, 30);
        box.appendChild(label);

        // Drag to move / click to highlight (mousedown on box body)
        box.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('bbox-handle')) return;
            e.preventDefault();
            e.stopPropagation();
            // 仅在编辑模式下允许拖拽移动，否则仅处理点击
            if (state.bboxEditIndex !== i) {
                handleBboxClick(i, e);
                return;
            }
            state.bboxClickStart = { x: e.clientX, y: e.clientY, index: i };
            initBboxDrag(e, i, 'move');
        });

        // Resize handles (8 directions) — only visible when active
        const handles = ['nw', 'ne', 'sw', 'se', 'n', 's', 'w', 'e'];
        handles.forEach(h => {
            const handle = document.createElement('div');
            handle.className = `bbox-handle h-${h}`;
            handle.style.borderColor = color;
            handle.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                highlightBbox(i);
                initBboxDrag(e, i, h);
            });
            box.appendChild(handle);
        });

        container.appendChild(box);
    });
}

// 点击 bbox 时处理重叠切换
function handleBboxClick(clickedIndex, e) {
    const container = document.getElementById('bboxContainer');
    const containerRect = container.getBoundingClientRect();
    const clickX = ((e.clientX - containerRect.left) / containerRect.width) * 1000;
    const clickY = ((e.clientY - containerRect.top) / containerRect.height) * 1000;

    // 找所有包含点击坐标的 bbox
    const overlapping = [];
    state.parsedElements.forEach((elem, i) => {
        if (!elem.bbox) return;
        const [x1, y1, x2, y2] = elem.bbox;
        if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
            overlapping.push(i);
        }
    });

    if (overlapping.length <= 1) {
        // 单个 bbox：聚焦并显示该 bbox（不进入编辑模式）
        focusBbox(clickedIndex);
        return;
    }

    // 在重叠列表中循环切换
    const currentIdx = state.focusedBboxIndex >= 0 ? state.focusedBboxIndex : state.activeBboxIndex;
    const currentPos = overlapping.indexOf(currentIdx);
    const nextPos = (currentPos + 1) % overlapping.length;
    const nextIdx = overlapping[nextPos];
    // 重叠切换时必须用 focus，确保左侧图像 bbox 与右侧选中项一致
    focusBbox(nextIdx);
}

function clearBboxes() {
    const container = document.getElementById('bboxContainer');
    container.innerHTML = '';
    container.style.display = 'none';
}

function parseSPAndUpdateBboxes() {
    state.parsedElements = [];
    state.parsedSP = null;
    if (!state.structuredPrompt.trim()) {
        if (state.bboxVisible) clearBboxes();
        updateBboxButtons();
        return;
    }
    try {
        const data = JSON.parse(state.structuredPrompt);
        state.parsedSP = data;
        state.parsedElements = extractElements(data);
    } catch (e) {
        // Not valid JSON yet
    }
    if (state.bboxVisible && state.currentImage) {
        renderBboxes();
    }
    updateBboxButtons();
}

// 基于元素 ID 的稳定颜色映射（不因数组顺序变化而改变）
function getColorForElemId(id) {
    const numId = parseInt(id) || 0;
    return BBOX_COLORS[numId % BBOX_COLORS.length];
}

        function parseBbox(positionStr) {
            if (!positionStr) return null;
            const match = positionStr.match(/<bbox>([^<]+)<\/bbox>/) || positionStr.match(/([\d\s]+)/);
            if (!match) return null;
            const coords = match[1].trim().split(/\s+/).map(Number);
            if (coords.length !== 4) return null;
            return coords;
        }

        function extractElements(data) {
            const elements = [];
            
            if (data.elements && Array.isArray(data.elements)) {
                data.elements.forEach(elem => {
                    if (elem.position) {
                        const bbox = parseBbox(elem.position);
                        if (bbox) {
                    elements.push({ id: elem.id, caption: elem.caption || '', bbox, source: 'elements' });
                        }
                    }
                });
            }
            
            if (data.scene && data.scene.elements && Array.isArray(data.scene.elements)) {
                data.scene.elements.forEach(elem => {
                    if (elem.position) {
                        const bbox = parseBbox(elem.position);
                        if (bbox) {
                    elements.push({ id: elem.id, caption: elem.caption || '', bbox, source: 'scene' });
                        }
                    }
                });
            }
            
            return elements;
        }

// ==================== SP Panel ====================
function triggerSPImport() {
    document.getElementById('spFileInput').click();
}

function handleSPFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
        pushSPUndo();
        state.structuredPrompt = e.target.result;
        parseSPAndUpdateBboxes();
        updateSPPanel();
        showToast('Structured Prompt 已导入', 'success');
    };
    reader.readAsText(file);
    event.target.value = '';
}

function handleSPDelete() {
    if (!state.structuredPrompt.trim()) return;
    pushSPUndo();
    state.structuredPrompt = '';
    state.parsedSP = null;
    updateSPPanel();
    clearBboxes();
    state.parsedElements = [];
    state.bboxVisible = false;
    showToast('Structured Prompt 已清空', 'info');
}

function handleSPUndo() {
    if (state.undoStack.length === 0) {
        showToast('没有可撤销的操作', 'info');
                return;
    }
    state.structuredPrompt = state.undoStack.pop();
    parseSPAndUpdateBboxes();
    updateSPPanel();
    showToast('已撤销', 'success');
}

function pushSPUndo() {
    state.undoStack.push(state.structuredPrompt);
    if (state.undoStack.length > 50) state.undoStack.shift();
}

function toggleSPRawMode() {
    state.spRawMode = !state.spRawMode;
    const btn = document.getElementById('btnSPRaw');
    if (state.spRawMode) {
        btn.classList.add('active');
    } else {
        btn.classList.remove('active');
        // Sync textarea back to state when switching to structured view
        const ta = document.getElementById('spTextarea');
        if (ta.style.display !== 'none') {
            state.structuredPrompt = ta.value;
            parseSPAndUpdateBboxes();
        }
    }
    updateSPPanel();
}

function updateSPPanel() {
    const spEmpty = document.getElementById('spEmpty');
    const spTextarea = document.getElementById('spTextarea');
    const spStructured = document.getElementById('spStructured');
    const btnUndo = document.getElementById('btnSPUndo');
    const btnDelete = document.getElementById('btnSPDelete');

    if (state.structuredPrompt.trim()) {
        spEmpty.style.display = 'none';
        if (state.spRawMode) {
            spTextarea.style.display = 'block';
            spStructured.style.display = 'none';
            spTextarea.value = state.structuredPrompt;
        } else {
            spTextarea.style.display = 'none';
            spStructured.style.display = 'block';
            renderSPStructured();
        }
        btnDelete.style.display = '';
    } else {
        // SP 为空时显示 textarea 供直接输入
        spEmpty.style.display = 'none';
        spTextarea.style.display = 'block';
        spStructured.style.display = 'none';
        spTextarea.value = '';
        btnDelete.style.display = 'none';
    }
    btnUndo.style.display = state.undoStack.length > 0 ? '' : 'none';
}

function updateSPButtons() {
    const btnUndo = document.getElementById('btnSPUndo');
    const btnDelete = document.getElementById('btnSPDelete');
    btnUndo.style.display = state.undoStack.length > 0 ? '' : 'none';
    btnDelete.style.display = state.structuredPrompt.trim() ? '' : 'none';
}

// Set SP content programmatically (e.g., from LLM result)
function setSPContent(text) {
    pushSPUndo();
    state.structuredPrompt = text;
    parseSPAndUpdateBboxes();
    updateSPPanel();
}

// ==================== SP Structured View ====================
// Key fields displayed in the first section
const SP_KEY_FIELDS = ['intent', 'style', 'atmosphere', 'lighting', 'photography', 'relationships'];

function formatDisplayValue(value) {
    return value === '' ? '""' : value;
}

// 根据 reference JSON 字符串中的 key 出现顺序，返回有序 key 列表
let _renderKeyOrders = {};
function getStableKeys(obj, pathKeys) {
    const allKeys = Object.keys(obj);
    const pathStr = _joinPath(pathKeys);
    const refKeys = _renderKeyOrders[pathStr] || [];
    const ordered = [];
    refKeys.forEach(k => { if (allKeys.includes(k)) ordered.push(k); });
    allKeys.forEach(k => { if (!ordered.includes(k)) ordered.push(k); });
    return ordered;
}

function renderSPStructured() {
    const container = document.getElementById('spStructured');
    container.innerHTML = '';

    // 先确保 parsedElements 已就绪
    if (state.parsedElements.length === 0 && state.structuredPrompt.trim()) {
        parseSPAndUpdateBboxes();
    }

    // 从原始 JSON 字符串提取 key 顺序（避免 Object.keys 自动排序数字 key）
    _renderKeyOrders = _extractKeyOrdersFromJSON(state.structuredPrompt);

    let data;
    try {
        data = JSON.parse(state.structuredPrompt);
        state.parsedSP = data;
    } catch (e) {
        container.innerHTML = '<div style="padding:12px;color:var(--danger);font-size:12px;">JSON 解析失败，请切换到原始模式编辑</div>';
        return;
    }

    // ===== Section 1: Global Properties =====
    const keySection = createSection('Global', '🔑', undefined, 'global');
    const keyBody = keySection.querySelector('.sp-section-body');

    SP_KEY_FIELDS.forEach(key => {
        if (data[key] === undefined) return;
        const val = data[key];
        if (typeof val === 'object' && !Array.isArray(val)) {
            keyBody.appendChild(createDictField(key, val, [key]));
        } else if (key === 'relationships' && Array.isArray(val)) {
            keyBody.appendChild(createRelationshipsField(val));
        } else {
            keyBody.appendChild(createStringField(key, String(val), [key]));
        }
    });

    // Other top-level keys not in SP_KEY_FIELDS, elements, scene
    getStableKeys(data, []).forEach(key => {
        if (SP_KEY_FIELDS.includes(key) || key === 'elements' || key === 'scene') return;
        const val = data[key];
        if (typeof val === 'object' && !Array.isArray(val)) {
            keyBody.appendChild(createDictField(key, val, [key]));
        } else {
            const displayVal = typeof val === 'string' ? val : JSON.stringify(val);
            keyBody.appendChild(createStringField(key, displayVal, [key]));
        }
    });

    container.appendChild(keySection);

    // ===== Section 2: Elements =====
    const elemArr = data.elements || [];
    const elemSection = createSection('Elements', '🎯', elemArr.length, 'elements');
    const elemBody = elemSection.querySelector('.sp-section-body');
    elemArr.forEach((elem, idx) => {
        elemBody.appendChild(createElementCard(elem, idx, ['elements', idx]));
    });
    container.appendChild(elemSection);

    // ===== Section 3: Scene =====
    const scene = data.scene || {};
    const sceneElems = scene.elements || [];
    const sceneSection = createSection('Scene', '🏞️', sceneElems.length, 'scene');
    const sceneBody = sceneSection.querySelector('.sp-section-body');

    // Scene setting
    if (scene.setting) {
        sceneBody.appendChild(createStringField('setting', scene.setting, ['scene', 'setting']));
    }

    // Scene other keys
    getStableKeys(scene, ['scene']).forEach(key => {
        if (key === 'setting' || key === 'elements') return;
        const val = scene[key];
        if (typeof val === 'object' && !Array.isArray(val)) {
            sceneBody.appendChild(createDictField(key, val, ['scene', key]));
        } else {
            sceneBody.appendChild(createStringField(key, String(val), ['scene', key]));
        }
    });

    // Scene elements
    sceneElems.forEach((elem, idx) => {
        sceneBody.appendChild(createElementCard(elem, idx, ['scene', 'elements', idx], true));
    });
    container.appendChild(sceneSection);
}

function createSection(title, icon, count, sectionType) {
    const section = document.createElement('div');
    section.className = 'sp-section';

    const header = document.createElement('div');
    header.className = 'sp-section-header';
    header.onclick = function(e) {
        if (e.target.closest('.sp-section-add-btn') || e.target.closest('.sp-section-clear-btn')) return;
        section.classList.toggle('collapsed');
    };

    const titleSpan = document.createElement('span');
    titleSpan.className = 'sp-section-title';
    titleSpan.innerHTML = `${icon} ${title}${count !== undefined ? ` <span class='sp-section-badge'>${count}</span>` : ''}`;

    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'sp-section-actions';

    // 统一的 "+" 添加按钮
        const addBtn = document.createElement('button');
        addBtn.className = 'sp-section-add-btn';
    addBtn.textContent = '+';
    addBtn.title = '添加';
        addBtn.onclick = function(e) {
            e.stopPropagation();
        showSectionAddDropdown(addBtn, sectionType);
        };
        actionsDiv.appendChild(addBtn);

    // 🗑️ clear all button
    const clearBtn = document.createElement('button');
    clearBtn.className = 'sp-section-clear-btn';
    clearBtn.innerHTML = '🗑';
    clearBtn.title = '清空本区';
    clearBtn.onclick = function(e) {
        e.stopPropagation();
        clearSectionFields(sectionType);
    };
    actionsDiv.appendChild(clearBtn);

    const arrow = document.createElement('span');
    arrow.className = 'sp-section-arrow';
    arrow.textContent = '▾';
    actionsDiv.appendChild(arrow);

    header.appendChild(titleSpan);
    header.appendChild(actionsDiv);
    section.appendChild(header);

    const body = document.createElement('div');
    body.className = 'sp-section-body';
    section.appendChild(body);

    return section;
}

function createStringField(key, value, path) {
    const field = document.createElement('div');
    field.className = 'sp-field';

    const keySpan = document.createElement('span');
    keySpan.className = 'sp-field-key';
    keySpan.textContent = key;
    keySpan.title = '双击编辑 key';
    keySpan.dataset.path = JSON.stringify(path.slice(0, -1));
    keySpan.dataset.oldKey = key;
    keySpan.ondblclick = function() { startKeyEdit(this); };
    field.appendChild(keySpan);

    const valSpan = document.createElement('span');
    valSpan.className = 'sp-field-value';
    const renderedVal = formatDisplayValue(value);
    valSpan.textContent = renderedVal;
    if (value === '') {
        valSpan.dataset.emptyPlaceholder = '1';
    } else {
        delete valSpan.dataset.emptyPlaceholder;
    }
    valSpan.title = '双击编辑';
    valSpan.dataset.path = JSON.stringify(path);
    valSpan.ondblclick = function() { startFieldEdit(this); };
    field.appendChild(valSpan);

    const actions = document.createElement('div');
    actions.className = 'sp-field-actions';
    const delBtn = document.createElement('button');
    delBtn.className = 'sp-field-btn';
    delBtn.textContent = '✕';
    delBtn.title = '删除';
    delBtn.onclick = function() { deleteSPField(path); };
    actions.appendChild(delBtn);
    field.appendChild(actions);

    return field;
}

function createDictField(key, dictObj, basePath) {
    const field = document.createElement('div');
    field.className = 'sp-field sp-field-dict';

    const keySpan = document.createElement('span');
    keySpan.className = 'sp-field-key';
    keySpan.textContent = key;
    keySpan.title = '双击编辑 key';
    keySpan.dataset.path = JSON.stringify(basePath.slice(0, -1).length ? basePath.slice(0, -1) : []);
    keySpan.dataset.oldKey = key;
    keySpan.ondblclick = function() { startKeyEdit(this); };
    field.appendChild(keySpan);

    const block = document.createElement('div');
    block.className = 'sp-dict-block';
    getStableKeys(dictObj, basePath).forEach(subKey => {
        const subVal = dictObj[subKey];
        const row = document.createElement('div');
        row.className = 'sp-dict-row';
        const subPath = [...basePath, subKey];

        const dKey = document.createElement('span');
        dKey.className = 'sp-dict-key';
        dKey.textContent = subKey;
        dKey.title = '双击编辑 key';
        dKey.dataset.path = JSON.stringify(basePath);
        dKey.dataset.oldKey = subKey;
        dKey.ondblclick = function() { startKeyEdit(this); };
        row.appendChild(dKey);

        const dVal = document.createElement('span');
        dVal.className = 'sp-dict-value';
        if (typeof subVal === 'string') {
            dVal.textContent = formatDisplayValue(subVal);
            if (subVal === '') {
                dVal.dataset.emptyPlaceholder = '1';
            } else {
                delete dVal.dataset.emptyPlaceholder;
            }
        } else {
            dVal.textContent = JSON.stringify(subVal);
            delete dVal.dataset.emptyPlaceholder;
        }
        dVal.title = '双击编辑';
        dVal.dataset.path = JSON.stringify(subPath);
        dVal.ondblclick = function() { startFieldEdit(this); };
        row.appendChild(dVal);

        // Delete sub-key button
        const delSub = document.createElement('button');
        delSub.className = 'sp-dict-del';
        delSub.textContent = '✕';
        delSub.title = '删除';
        delSub.onclick = function(e) { e.stopPropagation(); deleteSPField(subPath); };
        row.appendChild(delSub);

        block.appendChild(row);
    });

    // Inline add row for dict — 字典内只添加 key-value（字符串）
    const addRow = document.createElement('div');
    addRow.className = 'sp-dict-add-row';
    const addBtnInline = document.createElement('button');
    addBtnInline.className = 'sp-inline-add-btn';
    addBtnInline.textContent = '+';
    addBtnInline.title = '添加 key-value';
    addBtnInline.onclick = function(e) {
        e.stopPropagation();
        promptAddDictEntry(basePath, addBtnInline);
    };
    addRow.appendChild(addBtnInline);
    block.appendChild(addRow);

    field.appendChild(block);

    const actions = document.createElement('div');
    actions.className = 'sp-field-actions';
    const delBtn = document.createElement('button');
    delBtn.className = 'sp-field-btn';
    delBtn.textContent = '✕';
    delBtn.title = '删除';
    delBtn.onclick = function() { deleteSPField(basePath); };
    actions.appendChild(delBtn);
    field.appendChild(actions);

    return field;
}

function createRelationshipsField(relArr) {
    const field = document.createElement('div');
    field.className = 'sp-field';

    const keySpan = document.createElement('span');
    keySpan.className = 'sp-field-key';
    keySpan.textContent = 'relationships';
    field.appendChild(keySpan);

    const list = document.createElement('ul');
    list.className = 'sp-rel-list';
    relArr.forEach((rel, i) => {
        const li = document.createElement('li');
        li.className = 'sp-rel-item';

        const bullet = document.createElement('span');
        bullet.className = 'sp-rel-bullet';
        bullet.textContent = '•';
        li.appendChild(bullet);

        const text = document.createElement('span');
        text.className = 'sp-rel-text';
        text.textContent = rel;
        text.title = '双击编辑';
        text.dataset.path = JSON.stringify(['relationships', i]);
        text.ondblclick = function() { startFieldEdit(this); };
        li.appendChild(text);

        list.appendChild(li);
    });
    field.appendChild(list);

    const actions = document.createElement('div');
    actions.className = 'sp-field-actions';
    const delBtn = document.createElement('button');
    delBtn.className = 'sp-field-btn';
    delBtn.textContent = '✕';
    delBtn.title = '删除';
    delBtn.onclick = function() { deleteSPField(['relationships']); };
    actions.appendChild(delBtn);
    field.appendChild(actions);

    return field;
}

function createElementCard(elem, idx, basePath, isScene) {
    // 使用 ID 稳定颜色映射
    const color = getColorForElemId(elem.id);
    // 动态查找 globalIdx，确保即使 parsedElements 延迟填充也能正确工作
    const elemId = elem.id;
    function _getGlobalIdx() { return findParsedElementIndex(elemId, isScene); }

    const card = document.createElement('div');
    card.className = 'sp-element';
    card.dataset.elemId = elem.id;

    // Header
    const header = document.createElement('div');
    header.className = 'sp-element-header';

    const colorDot = document.createElement('span');
    colorDot.className = 'sp-element-color';
    colorDot.style.background = color;
    header.appendChild(colorDot);

    const idSpan = document.createElement('span');
    idSpan.className = 'sp-element-id';
    idSpan.textContent = 'ID:' + elem.id;
    header.appendChild(idSpan);

    const captionSpan = document.createElement('span');
    captionSpan.className = 'sp-element-caption';
    captionSpan.textContent = elem.caption || '';
    header.appendChild(captionSpan);

    const elemActions = document.createElement('div');
    elemActions.className = 'sp-element-actions';

    const elemAddBtn = document.createElement('button');
    elemAddBtn.className = 'sp-field-btn sp-elem-add-btn';
    elemAddBtn.textContent = '+';
    elemAddBtn.title = '添加字段';
    elemAddBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        e.preventDefault();
        showAddDropdown(elemAddBtn, basePath);
    });
    elemActions.appendChild(elemAddBtn);

    const elemDelBtn = document.createElement('button');
    elemDelBtn.className = 'sp-field-btn';
    elemDelBtn.textContent = '✕';
    elemDelBtn.title = '删除元素';
    elemDelBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        e.preventDefault();
        deleteSPField(basePath);
    });
    elemActions.appendChild(elemDelBtn);

    header.appendChild(elemActions);

    header.addEventListener('click', function(e) {
        if (e.target.closest('.sp-field-btn')) return;
        card.classList.toggle('collapsed');
        // 点击 element → 立即显示对应 bbox
        focusBbox(_getGlobalIdx());
    });

    // Mouse enter => 只显示该 element 的 bbox（hover 预览），保存之前状态
    card.addEventListener('mouseenter', () => {
        const gIdx = _getGlobalIdx();
        if (!state.currentImage || gIdx < 0) return;
        card._savedBboxState = {
            bboxVisible: state.bboxVisible,
            activeBboxIndex: state.activeBboxIndex,
            focusedBboxIndex: state.focusedBboxIndex,
            bboxEditIndex: state.bboxEditIndex,
        };
        state.activeBboxIndex = gIdx;
        state.bboxEditIndex = -1;
        state.bboxVisible = true;
        renderBboxes(gIdx);
        updateBboxButtons();
        document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
        card.classList.add('highlight');
    });
    card.addEventListener('mouseleave', () => {
        if (!state.currentImage) return;
        const saved = card._savedBboxState;
        card._savedBboxState = null;
        if (!saved) return;
        // 如果 hover 期间用户点击锁定了新的 bbox，保持新状态
        if (state.focusedBboxIndex >= 0 && state.focusedBboxIndex !== saved.focusedBboxIndex) {
            state.activeBboxIndex = state.focusedBboxIndex;
            state.bboxEditIndex = state.bboxEditIndex; // 保持编辑状态
            renderBboxes(state.focusedBboxIndex);
            updateBboxButtons();
            document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
            if (state.parsedElements[state.focusedBboxIndex]) {
                const fc = document.querySelector(`.sp-element[data-elem-id="${state.parsedElements[state.focusedBboxIndex].id}"]`);
                if (fc) fc.classList.add('highlight');
            }
            return;
        }
        // 恢复之前的状态
        state.bboxVisible = saved.bboxVisible;
        state.activeBboxIndex = saved.activeBboxIndex;
        state.focusedBboxIndex = saved.focusedBboxIndex;
        state.bboxEditIndex = saved.bboxEditIndex;
        if (saved.bboxVisible) {
            if (saved.focusedBboxIndex >= 0) {
                renderBboxes(saved.focusedBboxIndex);
            } else {
                renderBboxes(-1);
            }
        } else {
            clearBboxes();
        }
            updateBboxButtons();
            document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
        if (saved.focusedBboxIndex >= 0 && state.parsedElements[saved.focusedBboxIndex]) {
            const fc = document.querySelector(`.sp-element[data-elem-id="${state.parsedElements[saved.focusedBboxIndex].id}"]`);
            if (fc) fc.classList.add('highlight');
        }
    });

    card.appendChild(header);

    // Body
    const body = document.createElement('div');
    body.className = 'sp-element-body';

    const skipKeys = ['id'];
    getStableKeys(elem, basePath).forEach(key => {
        if (skipKeys.includes(key)) return;
        const val = elem[key];
        const fieldPath = [...basePath, key];

        if (key === 'position') {
            const posField = document.createElement('div');
            posField.className = 'sp-field';
            const posKeySpan = document.createElement('span');
            posKeySpan.className = 'sp-field-key';
            posKeySpan.textContent = 'position';
            posField.appendChild(posKeySpan);
            const posBadge = document.createElement('span');
            posBadge.className = 'sp-pos-badge';
            posBadge.title = '点击定位';
            posBadge.innerHTML = '<span class="pos-icon">📍</span> ' + escapeHtml(extractBboxText(val));
            posBadge.onclick = function(ev) { ev.stopPropagation(); focusBbox(_getGlobalIdx()); };
            posField.appendChild(posBadge);
            // 编辑按钮 — 进入 bbox 编辑模式（可拖拽/缩放）
            const editBtn = document.createElement('button');
            editBtn.className = 'sp-field-btn sp-bbox-edit-btn';
            editBtn.textContent = '✏️';
            editBtn.title = '编辑 BBox 位置';
            editBtn.onclick = function(ev) {
                ev.stopPropagation();
                const gIdx = _getGlobalIdx();
                if (gIdx < 0) {
                    showToast('BBox 数据未就绪，请稍后重试', 'warning');
                    return;
                }
                activateBboxEdit(gIdx);
            };
            posField.appendChild(editBtn);
            body.appendChild(posField);
        } else if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
            body.appendChild(createDictField(key, val, fieldPath));
            } else {
            body.appendChild(createStringField(key, typeof val === 'string' ? val : JSON.stringify(val), fieldPath));
        }
    });

    card.appendChild(body);
    return card;
}

function extractBboxText(posStr) {
    if (!posStr) return '';
    const match = posStr.match(/<bbox>([^<]+)<\/bbox>/);
    return match ? match[1].trim() : posStr;
}

function findParsedElementIndex(elemId, isScene) {
    // Find element index in state.parsedElements by id
    for (let i = 0; i < state.parsedElements.length; i++) {
        if (state.parsedElements[i].id === elemId) return i;
    }
    return -1;
}

function highlightBbox(globalIdx) {
    state.activeBboxIndex = globalIdx;
    // Update bbox visual highlighting (使用 data-elemIndex 匹配)
    const container = document.getElementById('bboxContainer');
    container.querySelectorAll('.bbox-box').forEach(box => {
        const idx = parseInt(box.dataset.elemIndex);
        box.classList.remove('active', 'focused');
        if (idx === globalIdx) {
            if (state.bboxEditIndex === globalIdx) {
            box.classList.add('active');
        } else {
                box.classList.add('focused');
            }
        }
    });
    // Also highlight SP element card
    document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
    if (globalIdx >= 0 && state.parsedElements[globalIdx]) {
        const card = document.querySelector(`.sp-element[data-elem-id="${state.parsedElements[globalIdx].id}"]`);
        if (card) card.classList.add('highlight');
    }
}

function focusBbox(globalIdx) {
    if (globalIdx < 0 || !state.currentImage) return;
    // 点击 element => 锁定只显示该 element 的 bbox（仅查看，不带编辑手柄）
    state.focusedBboxIndex = globalIdx;
    state.activeBboxIndex = globalIdx;
    state.bboxEditIndex = -1;
    state.bboxVisible = true;
    renderBboxes(globalIdx);
    updateBboxButtons();
    // highlight SP card
    document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
    if (state.parsedElements[globalIdx]) {
        const card = document.querySelector(`.sp-element[data-elem-id="${state.parsedElements[globalIdx].id}"]`);
        if (card) card.classList.add('highlight');
    }
}

function activateBboxEdit(globalIdx) {
    if (globalIdx < 0 || !state.currentImage) return;
    // 双击 => 激活 bbox 编辑模式（显示拖拽/缩放手柄）
    state.focusedBboxIndex = globalIdx;
    state.activeBboxIndex = globalIdx;
    state.bboxEditIndex = globalIdx;
    state.bboxVisible = true;
    renderBboxes(globalIdx);
    updateBboxButtons();
    // highlight SP card
    document.querySelectorAll('.sp-element').forEach(el => el.classList.remove('highlight'));
    if (state.parsedElements[globalIdx]) {
        const card = document.querySelector(`.sp-element[data-elem-id="${state.parsedElements[globalIdx].id}"]`);
        if (card) card.classList.add('highlight');
    }
    showToast('已进入 BBox 编辑模式，可拖拽移动和缩放', 'info');
}

// ==================== SP Field Editing ====================
function startFieldEdit(el) {
    if (el.contentEditable === 'true') return;
    if (el.dataset.emptyPlaceholder === '1') {
        el.textContent = '';
        delete el.dataset.emptyPlaceholder;
    }
    el.contentEditable = 'true';
    el.focus();

    // Select all text
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);

    // Save on blur or Enter
    const finish = () => {
        el.contentEditable = 'false';
        el.removeEventListener('blur', onBlur);
        el.removeEventListener('keydown', onKey);
        commitFieldEdit(el);
    };

    const onBlur = () => finish();
    const onKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            finish();
        }
        if (e.key === 'Escape') {
            el.contentEditable = 'false';
            el.removeEventListener('blur', onBlur);
            el.removeEventListener('keydown', onKey);
            // Revert — re-render
            renderSPStructured();
        }
    };

    el.addEventListener('blur', onBlur);
    el.addEventListener('keydown', onKey);
}

function commitFieldEdit(el) {
    const path = JSON.parse(el.dataset.path);
    const newValue = el.textContent.trim();
    if (!state.parsedSP) return;

    pushSPUndo();

    // Navigate to parent and set value
    let obj = state.parsedSP;
    for (let i = 0; i < path.length - 1; i++) {
        obj = obj[path[i]];
        if (!obj) return;
    }
    const lastKey = path[path.length - 1];
    obj[lastKey] = newValue;

    // Sync back to state
    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    renderSPStructured();
}

// ==================== Key Editing ====================
function startKeyEdit(el) {
    if (el.contentEditable === 'true') return;
    el.contentEditable = 'true';
    el.focus();

    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);

    const finish = () => {
        el.contentEditable = 'false';
        el.removeEventListener('blur', onBlur);
        el.removeEventListener('keydown', onKey);
        commitKeyEdit(el);
    };

    const onBlur = () => finish();
    const onKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            finish();
        }
        if (e.key === 'Escape') {
            el.contentEditable = 'false';
            el.removeEventListener('blur', onBlur);
            el.removeEventListener('keydown', onKey);
            renderSPStructured();
        }
    };

    el.addEventListener('blur', onBlur);
    el.addEventListener('keydown', onKey);
}

function commitKeyEdit(el) {
    const oldKey = el.dataset.oldKey;
    const newKey = el.textContent.trim();
    if (!newKey || newKey === oldKey || !state.parsedSP) return;

    const parentPath = JSON.parse(el.dataset.path);

    pushSPUndo();

    // Navigate to parent
    let obj = state.parsedSP;
    for (let i = 0; i < parentPath.length; i++) {
        obj = obj[parentPath[i]];
        if (!obj) return;
    }

    if (typeof obj === 'object' && !Array.isArray(obj) && oldKey in obj) {
        // Rename: preserve order by rebuilding object
        const entries = Object.entries(obj);
        const newEntries = entries.map(([k, v]) => k === oldKey ? [newKey, v] : [k, v]);
        // Clear and repopulate
        for (const k of Object.keys(obj)) delete obj[k];
        for (const [k, v] of newEntries) obj[k] = v;
    }

    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    renderSPStructured();
    showToast(`字段已重命名: ${oldKey} → ${newKey}`, 'success');
}

// ==================== Add Dropdown ====================
let _addDropdownCleanup = null;
let _addDropdownAnchor = null;
let _savedAnchorRect = null; // 保存按钮位置，防止 DOM 变化后丢失

function showSectionAddDropdown(anchorEl, sectionType) {
    hideAddDropdown();
    hideInputPopup();
    _addDropdownAnchor = anchorEl;
    _savedAnchorRect = anchorEl.getBoundingClientRect();
    const dd = document.getElementById('spAddDropdown');
    dd.innerHTML = '';

    const options = [];
    if (sectionType === 'global') {
        options.push({ label: '普通字段', icon: '📝', action: () => promptAddField([], 'string', anchorEl) });
        options.push({ label: '字典字段', icon: '📦', action: () => promptAddField([], 'dict', anchorEl) });
    } else if (sectionType === 'elements') {
        options.push({ label: '添加元素', icon: '🎯', action: () => addSPElement('') });
    } else if (sectionType === 'scene') {
        options.push({ label: '普通字段', icon: '📝', action: () => promptAddField(['scene'], 'string', anchorEl) });
        options.push({ label: '字典字段', icon: '📦', action: () => promptAddField(['scene'], 'dict', anchorEl) });
        options.push({ label: '添加元素', icon: '🎯', action: () => addSPSceneElement('') });
    }

    options.forEach(opt => {
        const item = document.createElement('div');
        item.className = 'sp-add-option';
        item.innerHTML = `<span>${opt.icon}</span> ${opt.label}`;
        item.onclick = function(e) {
            e.stopPropagation();
            hideAddDropdown();
            opt.action();
        };
        dd.appendChild(item);
    });

    // Position near anchor
    const rect = anchorEl.getBoundingClientRect();
    dd.style.top = (rect.bottom + 4) + 'px';
    dd.style.left = Math.min(rect.left, window.innerWidth - 160) + 'px';
    dd.style.display = 'block';

    // Close on outside click
    _addDropdownCleanup = (e) => {
        if (!dd.contains(e.target) && e.target !== anchorEl) {
            hideAddDropdown();
        }
    };
    setTimeout(() => document.addEventListener('mousedown', _addDropdownCleanup), 0);
}

function showAddDropdown(anchorEl, basePath) {
    hideAddDropdown();
    hideInputPopup();
    _addDropdownAnchor = anchorEl;
    _savedAnchorRect = anchorEl.getBoundingClientRect();
    const dd = document.getElementById('spAddDropdown');
    dd.innerHTML = '';

    const options = [
        { label: '普通字段', icon: '📝', type: 'string' },
        { label: '字典字段', icon: '📦', type: 'dict' },
    ];

    options.forEach(opt => {
        const item = document.createElement('div');
        item.className = 'sp-add-option';
        item.innerHTML = `<span>${opt.icon}</span> ${opt.label}`;
        item.onclick = function(e) {
            e.stopPropagation();
            const savedAnchor = anchorEl;  // capture before hiding
            hideAddDropdown();
            promptAddField(basePath, opt.type, savedAnchor);
        };
        dd.appendChild(item);
    });

    // Position near anchor
    const rect = anchorEl.getBoundingClientRect();
    dd.style.top = (rect.bottom + 4) + 'px';
    dd.style.left = Math.min(rect.left, window.innerWidth - 160) + 'px';
    dd.style.display = 'block';

    // Close on outside click
    _addDropdownCleanup = (e) => {
        if (!dd.contains(e.target) && e.target !== anchorEl) {
            hideAddDropdown();
        }
    };
    setTimeout(() => document.addEventListener('mousedown', _addDropdownCleanup), 0);
}

function hideAddDropdown() {
    const dd = document.getElementById('spAddDropdown');
    if (dd) dd.style.display = 'none';
    if (_addDropdownCleanup) {
        document.removeEventListener('mousedown', _addDropdownCleanup);
        _addDropdownCleanup = null;
    }
}

// ==================== Floating Input Popup ====================
let _inputPopupEl = null;
let _inputPopupCleanup = null;
let _lastPopupAnchor = null;

function resolvePopupAnchor(anchorEl) {
    const candidates = [anchorEl, _addDropdownAnchor, _lastPopupAnchor];
    for (const candidate of candidates) {
        if (!candidate || !document.body.contains(candidate)) continue;
        const rect = candidate.getBoundingClientRect();
        if (rect.width > 0 || rect.height > 0) return candidate;
    }
    return null;
}

function showInputPopup(anchorEl, title, fields, onSubmit) {
    hideInputPopup();
    hideAddDropdown();
    const resolvedAnchor = resolvePopupAnchor(anchorEl);

    const popup = document.createElement('div');
    popup.className = 'sp-input-popup';

    // Title
    const titleEl = document.createElement('div');
    titleEl.className = 'popup-title';
    titleEl.textContent = title;
    popup.appendChild(titleEl);

    // Fields
    const inputs = {};
    fields.forEach(f => {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'popup-field';

        const label = document.createElement('label');
        label.textContent = f.label;
        fieldDiv.appendChild(label);

        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = f.placeholder || '';
        input.dataset.fieldKey = f.key;
        input.addEventListener('input', () => {
            input.classList.remove('input-error');
        });
        fieldDiv.appendChild(input);

        inputs[f.key] = input;
        popup.appendChild(fieldDiv);
    });

    // Actions
    const actions = document.createElement('div');
    actions.className = 'popup-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'popup-btn';
    cancelBtn.textContent = '取消';
    cancelBtn.onclick = (e) => { e.stopPropagation(); hideInputPopup(); };
    actions.appendChild(cancelBtn);

    const okBtn = document.createElement('button');
    okBtn.className = 'popup-btn popup-btn-primary';
    okBtn.textContent = '确定';
    okBtn.onclick = (e) => {
        e.stopPropagation();
        const values = {};
        let valid = true;
        fields.forEach(f => {
            values[f.key] = inputs[f.key].value.trim();
            if (f.required && !values[f.key]) {
                inputs[f.key].classList.add('input-error');
                valid = false;
            }
        });
        if (!valid) return;
        hideInputPopup();
        onSubmit(values);
    };
    actions.appendChild(okBtn);
    popup.appendChild(actions);

    // Position near anchor (with viewport clamping and fallback)
    document.body.appendChild(popup);
    _inputPopupEl = popup;

    // 优先用 resolvedAnchor 的实时位置，否则用之前保存的 _savedAnchorRect
    let rect;
    if (resolvedAnchor) {
        rect = resolvedAnchor.getBoundingClientRect();
    } else if (_savedAnchorRect && (_savedAnchorRect.width > 0 || _savedAnchorRect.height > 0)) {
        rect = _savedAnchorRect;
    } else {
        rect = { top: 0, left: 0, width: 0, height: 0, bottom: 0 };
    }
    const popupRect = popup.getBoundingClientRect();
    const popupW = popupRect.width || 260;
    const popupH = popupRect.height || 150;

    let top, left;
    // If anchor has zero size (detached from DOM or hidden), center in viewport
    if (rect.width === 0 && rect.height === 0 && rect.top === 0 && rect.left === 0) {
        top = Math.max(10, (window.innerHeight - popupH) / 2);
        left = Math.max(10, (window.innerWidth - popupW) / 2);
    } else {
        top = rect.bottom + 4;
        left = rect.left;
    }

    // Clamp to viewport bounds
    if (top + popupH > window.innerHeight - 10) {
        top = Math.max(10, rect.top - popupH - 4);
    }
    if (left + popupW > window.innerWidth - 10) {
        left = Math.max(10, window.innerWidth - popupW - 10);
    }
    if (top < 10) top = 10;
    if (left < 10) left = 10;

    popup.style.top = top + 'px';
    popup.style.left = left + 'px';

    // Focus first input
    const firstInput = popup.querySelector('input');
    if (firstInput) setTimeout(() => firstInput.focus(), 50);

    // Keyboard shortcuts
    popup.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            okBtn.click();
        }
        if (e.key === 'Escape') {
            hideInputPopup();
        }
    });

    // Close on outside click
    _inputPopupCleanup = (e) => {
        if (!popup.contains(e.target) && (!resolvedAnchor || e.target !== resolvedAnchor)) {
            hideInputPopup();
        }
    };
    setTimeout(() => document.addEventListener('mousedown', _inputPopupCleanup), 0);
}

function hideInputPopup() {
    if (_inputPopupEl) {
        _inputPopupEl.remove();
        _inputPopupEl = null;
    }
    if (_inputPopupCleanup) {
        document.removeEventListener('mousedown', _inputPopupCleanup);
        _inputPopupCleanup = null;
    }
}

// 弹出悬浮输入框让用户填写 key 和 value
function promptAddField(basePath, type, anchorOverride) {
    const anchor = anchorOverride || _addDropdownAnchor || document.body;
    _lastPopupAnchor = anchor;
    if (type === 'string') {
        showInputPopup(anchor, '添加普通字段', [
            { key: 'fieldKey', label: 'Key', placeholder: '字段名', required: true },
            { key: 'fieldValue', label: 'Value', placeholder: '字段值', required: false },
        ], (values) => {
            addFieldToPath(basePath, 'string', values.fieldKey, values.fieldValue);
        });
    } else if (type === 'dict') {
        showInputPopup(anchor, '添加字典字段', [
            { key: 'dictName', label: '字典名', placeholder: '字典名', required: true },
            { key: 'innerKey', label: '内部 Key', placeholder: 'key', required: true },
            { key: 'innerValue', label: '内部 Value', placeholder: 'value', required: false },
        ], (values) => {
            addFieldToPath(basePath, 'dict', values.dictName, null, values.innerKey, values.innerValue);
        });
    }
}

// 直接向字典内添加一对 key-value（通过悬浮输入框）
function promptAddDictEntry(basePath, anchorEl) {
    _lastPopupAnchor = anchorEl || document.body;
    if (anchorEl) _savedAnchorRect = anchorEl.getBoundingClientRect();
    showInputPopup(anchorEl || document.body, '添加字段', [
        { key: 'fieldKey', label: 'Key', placeholder: 'key', required: true },
        { key: 'fieldValue', label: 'Value', placeholder: 'value', required: false },
    ], (values) => {
        addFieldToPath(basePath, 'string', values.fieldKey, values.fieldValue);
    });
}

function addFieldToPath(basePath, type, fieldKey, fieldValue, innerKey, innerValue) {
    if (!state.parsedSP) return;

    // Navigate to target object
    let target = state.parsedSP;
    for (const key of basePath) {
        if (target[key] === undefined) target[key] = {};
        target = target[key];
    }

    if (typeof target !== 'object' || Array.isArray(target)) {
        showToast('无法在此位置添加字段', 'warning');
        return;
    }

    if (fieldKey in target) {
        showToast(`字段 "${fieldKey}" 已存在`, 'warning');
        return;
    }

    pushSPUndo();

    if (type === 'dict') {
        // 创建字典，带初始 key-value
        const dictObj = {};
        if (innerKey) dictObj[innerKey] = innerValue || '';
        target[fieldKey] = dictObj;
    } else {
        target[fieldKey] = fieldValue || '';
    }

    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    updateSPPanel();
    const isEmptyValue = type === 'dict' ? (!innerValue && !!innerKey) : !fieldValue;
    const emptySuffix = isEmptyValue ? '（值为空字符串）' : '';
    showToast(`已添加${type === 'dict' ? '字典' : ''}字段: ${fieldKey}${emptySuffix}`, 'success');
}

// ==================== Section Clear ====================
function clearSectionFields(sectionType) {
    if (!state.parsedSP) return;
    pushSPUndo();

    if (sectionType === 'global') {
        // Remove all top-level keys except elements and scene
        Object.keys(state.parsedSP).forEach(key => {
            if (key !== 'elements' && key !== 'scene') {
                delete state.parsedSP[key];
            }
        });
        showToast('已清空 Global 字段', 'info');
    } else if (sectionType === 'elements') {
        // 删除整个 elements key
        delete state.parsedSP.elements;
        showToast('已删除 Elements', 'info');
    } else if (sectionType === 'scene') {
        // 删除整个 scene key
        delete state.parsedSP.scene;
        showToast('已删除 Scene', 'info');
    }

    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    updateSPPanel();
}

function addSPElement(caption) {
    if (!state.parsedSP) return;
    if (!state.parsedSP.elements) state.parsedSP.elements = [];

    pushSPUndo();

    // 自动递增 ID
    let maxId = 0;
    state.parsedSP.elements.forEach(e => {
        const id = parseInt(e.id) || 0;
        if (id > maxId) maxId = id;
    });
    if (state.parsedSP.scene && state.parsedSP.scene.elements) {
        state.parsedSP.scene.elements.forEach(e => {
            const id = parseInt(e.id) || 0;
            if (id > maxId) maxId = id;
        });
    }

    const newElem = {
        id: maxId + 1,
        caption: caption || '',
        position: '<bbox>0 0 500 500</bbox>',
    };

    state.parsedSP.elements.push(newElem);
    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    updateSPPanel();
    showToast(`已添加 Element (ID:${newElem.id})`, 'success');
}

function addSPSceneElement(caption) {
    if (!state.parsedSP) return;
    if (!state.parsedSP.scene) state.parsedSP.scene = {};
    if (!state.parsedSP.scene.elements) state.parsedSP.scene.elements = [];

    pushSPUndo();

    let maxId = 0;
    if (state.parsedSP.elements) {
        state.parsedSP.elements.forEach(e => {
            const id = parseInt(e.id) || 0;
            if (id > maxId) maxId = id;
        });
    }
    state.parsedSP.scene.elements.forEach(e => {
        const id = parseInt(e.id) || 0;
        if (id > maxId) maxId = id;
    });

    const newElem = {
        id: maxId + 1,
        caption: caption || '',
        position: '<bbox>0 0 500 500</bbox>',
    };

    state.parsedSP.scene.elements.push(newElem);
    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    updateSPPanel();
    showToast(`已添加 Scene Element (ID:${newElem.id})`, 'success');
}

// ==================== Deleting Fields ====================
function deleteSPField(path) {
    if (!state.parsedSP) return;
    if (typeof path === 'string') path = JSON.parse(path);

    pushSPUndo();

    let obj = state.parsedSP;
    for (let i = 0; i < path.length - 1; i++) {
        obj = obj[path[i]];
        if (!obj) return;
    }
    const lastKey = path[path.length - 1];

    if (Array.isArray(obj)) {
        obj.splice(lastKey, 1);
    } else {
        delete obj[lastKey];
    }

    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    parseSPAndUpdateBboxes();
    updateSPPanel();
    showToast('已删除', 'info');
}

// ==================== Generation Logic (per DEFINITION.md) ====================

/**
 * 生成SP按钮逻辑:
 * - 当Image不存在时: User Prompt => SP (Text System Prompt)
 * - 当Image存在时:
 *   - H&W会被覆盖
 *   - 没有User Prompt => 根据图像生成SP (Image System Prompt)
 *   - 只有User Prompt (无SP) => 根据图像生成SP (Image System Prompt)
 *   - 有User Prompt和SP => 根据User Prompt修改SP (Editing System Prompt)
 */
async function handleGenerateSP() {
    if (state.isGenerating) return;

    const hasImage = !!state.currentImage;
    const hasUserPrompt = !!state.userPrompt.trim();
    const hasSP = !!state.structuredPrompt.trim();

    if (!hasImage) {
        // ---- Text → SP ----
        if (!hasUserPrompt) {
            showToast('请输入 User Prompt', 'warning');
                return;
            }
            
        setGenerating(true, 'sp');
        showToast('正在根据 User Prompt 生成 Structured Prompt...', 'info');

        try {
            const result = await callLLMForSP({
                type: 'text',
                userPrompt: state.userPrompt,
            });
            if (result) {
                setSPContent(result);
                showToast('Structured Prompt 生成完成', 'success');
            }
            addHistoryEntry('sp');
        } catch (err) {
            showToast('生成失败: ' + err.message, 'error');
        } finally {
            setGenerating(false);
        }
    } else {
        // ---- Image exists ----
        autoUpdateHW();  // H&W被覆盖

        if (hasUserPrompt && hasSP) {
            // Case 3: 有 User Prompt 和 SP → 修改 SP
            setGenerating(true, 'sp');
            showToast('正在根据 User Prompt 修改 Structured Prompt...', 'info');

            try {
                const result = await callLLMForSP({
                    type: 'edit',
                    image: state.currentImage.src,
                    userPrompt: state.userPrompt,
                    structuredPrompt: state.structuredPrompt,
                });
                if (result) {
                    setSPContent(result);
                    showToast('Structured Prompt 修改完成', 'success');
                }
                addHistoryEntry('sp');
            } catch (err) {
                showToast('修改失败: ' + err.message, 'error');
            } finally {
                setGenerating(false);
            }
        } else {
            // Case 1 & 2: 根据图像生成 SP
            setGenerating(true, 'sp');
            showToast('正在根据图像生成 Structured Prompt...', 'info');

            try {
                const result = await callLLMForSP({
                    type: 'image',
                    image: state.currentImage.src,
                });
                if (result) {
                    setSPContent(result);
                    showToast('Structured Prompt 生成完成', 'success');
                }
                addHistoryEntry('sp');
            } catch (err) {
                showToast('生成失败: ' + err.message, 'error');
            } finally {
                setGenerating(false);
            }
        }
    }
}

/**
 * 生成Image按钮逻辑:
 * - 当Image不存在时 (T2I):
 *   - 有SP => 根据SP生成
 *   - 无SP => 根据User Prompt生成
 * - 当Image存在时 ((I+T)2I):
 *   - 有SP => 根据SP修改图像
 *   - 无SP => 根据User Prompt修改图像
 *
 * 生成前: 与上轮对比 User & SP，高亮修改，点击确认才提交
 */
async function handleGenerateImage() {
    if (state.isGenerating) return;

    const hasImage = !!state.currentImage && !state.params.forceT2I;
    const hasSP = !!state.structuredPrompt.trim();

    if (!hasSP) {
        showToast('Please generate the Structured Prompt first.', 'warning');
        return;
    }

    syncParamsFromUI();

    const prompt = state.structuredPrompt;
    const message = hasImage
        ? '正在根据 Structured Prompt 修改图像...'
        : '正在根据 Structured Prompt 生成图像...';

    // 如果有历史记录，展示 diff 对比
    if (state.history.length > 0) {
        const lastEntry = state.history[state.history.length - 1];
        const upChanged = (lastEntry.userPrompt || '') !== (state.userPrompt || '');
        const spChanged = (lastEntry.structuredPrompt || '') !== (state.structuredPrompt || '');

        if (upChanged || spChanged) {
            showDiffDialog(lastEntry, () => {
                doGenerateImage(prompt, message, hasImage);
            });
            return;
        }
    }

    doGenerateImage(prompt, message, hasImage);
}

async function doGenerateImage(prompt, message, hasImage) {
    setGenerating(true, 'image');
    showToast(message, 'info');

    try {
        const result = await callDiTForImage({
            type: hasImage ? 'i2i' : 't2i',
            prompt: prompt,
            image: hasImage ? state.currentImage.src : null,
            params: { ...state.params },
        });

        if (result) {
            // result: image data URL or URL — 替换当前图片
            const img = new Image();
            img.onload = function () {
                state.currentImage = {
                    src: result,
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                };
                updateImagePanel();
                addHistoryEntry('image');
                showToast('图像生成成功！', 'success');
            };
            img.onerror = function () {
                showToast('生成的图像加载失败', 'error');
            };
            img.src = result;
            } else {
            // DiT 未实现时: 直接将当前状态存入历史（Debug 模式）
            addHistoryEntry('image');
            showToast('已记录当前状态到历史 (Debug)', 'success');
        }
    } catch (err) {
        showToast('图像生成失败: ' + err.message, 'error');
    } finally {
        setGenerating(false);
    }
}

function setGenerating(isGenerating, type) {
    state.isGenerating = isGenerating;
    const spBtn = document.getElementById('genSPBtn');
    const imgBtn = document.getElementById('genImageBtn');

    if (isGenerating) {
        spBtn.disabled = true;
        imgBtn.disabled = true;
        if (type === 'sp') {
            spBtn.classList.add('loading');
            spBtn.innerHTML = '<span class="spinner"></span>生成中...';
        } else if (type === 'image') {
            imgBtn.classList.add('loading');
            imgBtn.innerHTML = '<span class="spinner"></span>生成中...';
        }
            } else {
        spBtn.disabled = false;
        imgBtn.disabled = false;
        spBtn.classList.remove('loading');
        imgBtn.classList.remove('loading');
        spBtn.textContent = '生成SP';
        imgBtn.textContent = '生成Image';
    }
}

// ==================== Stub Functions (To Be Implemented) ====================

/**
 * 调用 LLM 生成 / 修改 Structured Prompt
 *
 * @param {Object} options
 * @param {string} options.type - 'text' | 'image' | 'edit'
 *
 * type='text': 纯文本 → SP
 *   options.userPrompt: 用户输入
 *   使用 Text System Prompt (DEFAULT_SYSTEM_PROMPT)
 *
 * type='image': 图片 → SP
 *   options.image: 图片 data URL
 *   使用 Image System Prompt
 *
 * type='edit': 根据指令修改 SP
 *   options.image: 图片 data URL
 *   options.userPrompt: 修改指令
 *   options.structuredPrompt: 当前 SP
 *   使用 Editing System Prompt
 *
 * @returns {string|null} 生成的 SP (formatted JSON string) 或 null
 */
async function callLLMForSP(options) {
    // Call the local FastAPI backend that runs the PE model in-process.
    // Only `type: 'text'` (natural-language → Structured Prompt) is supported
    // in this demo — image-to-SP and SP-editing modes need a VLM which the
    // demo does not ship. See demo/backend/pe.py for the model wrapper.
    if (options.type !== 'text') {
        showToast('该模式（image/edit）暂不支持，只支持 text → SP', 'warning');
        return null;
    }
    const width  = (Math.round((state.params.width  || 1024) / 32) * 32) || 1024;
    const height = (Math.round((state.params.height || 1024) / 32) * 32) || 1024;
    try {
        const resp = await fetch('/api/generate_sp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_prompt: options.userPrompt || '',
                width:  width,
                height: height,
            }),
        });
        if (!resp.ok) {
            const errText = await resp.text().catch(() => resp.statusText);
            throw new Error(`HTTP ${resp.status}: ${errText}`);
        }
        const data = await resp.json();
        if (!data.sp) throw new Error('Response missing sp');
        // Backend returns a JSON object; stringify for display in the SP editor.
        return typeof data.sp === 'string' ? data.sp : JSON.stringify(data.sp, null, 2);
    } catch (err) {
        console.error('[LLM] generate_sp failed:', err);
        showToast('SP 生成失败: ' + err.message, 'error');
        return null;
    }
}

// ==================== DiT Generation (QwenImage Serving API) ====================

/**
 * 将 JSON 对象转为 compact single-quote 格式字符串。
 * 服务端 prompt 字段期望此格式的 Structured Prompt。
 */
function compactSingleQuoteJSON(data) {
    const PH_S = '@@SQ@@', PH_D = '@@DQ@@';
    const reS = /@@SQ@@/g, reD = /@@DQ@@/g;
    function protect(obj) {
        if (obj === null || obj === undefined) return obj;
        if (Array.isArray(obj)) return obj.map(protect);
        if (typeof obj === 'object') {
            const r = {};
            for (const [k, v] of Object.entries(obj)) {
                const pk = typeof k === 'string' ? k.replace(/'/g, PH_S).replace(/"/g, PH_D) : k;
                r[pk] = protect(v);
            }
            return r;
        }
        if (typeof obj === 'string') return obj.replace(/'/g, PH_S).replace(/"/g, PH_D);
        return obj;
    }
    const jsonStr = JSON.stringify(protect(data));
    return jsonStr.replace(/"/g, "'").replace(reS, "\\'").replace(reD, '"');
}

/**
 * Call the local FastAPI backend which runs the QwenImage DiT in-process.
 * See demo/backend/dit.py for the model wrapper.
 *
 * @param {Object} options
 * @param {string} options.prompt - text prompt (SP JSON string or plain NL)
 * @param {Object} options.params - { cfgScale, steps, height, width, seed }
 *
 * @returns {string|null} generated image data URL, or null on failure.
 */
async function callDiTForImage(options) {
    const { prompt, params } = options;

    const width  = (Math.round((params.width  || 1024) / 32) * 32) || 1024;
    const height = (Math.round((params.height || 1024) / 32) * 32) || 1024;

    // SP JSON → compact single-quote format expected by the DiT; plain text as-is.
    let formattedPrompt;
    try {
        const parsed = typeof prompt === 'string' ? JSON.parse(prompt) : prompt;
        formattedPrompt = compactSingleQuoteJSON(parsed);
    } catch (_) {
        formattedPrompt = prompt;
    }

    const body = {
        prompt: formattedPrompt,
        height: height,
        width: width,
        num_steps: params.steps || 25,
        seed: (params.seed != null && params.seed !== -1) ? params.seed : 42,
        cfg_scale: params.cfgScale || 4.0,
        negative_prompt: "",
    };

    try {
        const resp = await fetch('/api/generate_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const errText = await resp.text().catch(() => resp.statusText);
            throw new Error(`HTTP ${resp.status}: ${errText}`);
        }
        const data = await resp.json();
        if (!data.image_base64) throw new Error('Response missing image_base64');
        return 'data:image/png;base64,' + data.image_base64;
    } catch (err) {
        console.error('[DiT] generate_image failed:', err);
        showToast('生图失败: ' + err.message, 'error');
        return null;
    }
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ==================== Diff Dialog ====================
function showDiffDialog(lastEntry, onConfirm) {
    state.pendingGeneration = onConfirm;
    const diffContent = document.getElementById('diffContent');

    let html = '';

    // User Prompt diff
    html += '<div class="diff-section-title">User Prompt 变更</div>';
    html += '<div class="diff-block">';
    html += generateDiffHTML(lastEntry.userPrompt || '', state.userPrompt || '');
    html += '</div>';

    // SP diff
    html += '<div class="diff-section-title">Structured Prompt 变更</div>';
    html += '<div class="diff-block">';
    html += generateDiffHTML(lastEntry.structuredPrompt || '', state.structuredPrompt || '');
    html += '</div>';

    diffContent.innerHTML = html;
    document.getElementById('diffModal').classList.add('visible');
}

function generateDiffHTML(oldText, newText) {
    if (oldText === newText) {
        return '<div class="diff-no-change">无变更</div>';
    }

    const oldLines = oldText.split('\n');
    const newLines = newText.split('\n');
    let html = '';
    const maxLen = Math.max(oldLines.length, newLines.length);

    for (let i = 0; i < maxLen; i++) {
        const oldLine = i < oldLines.length ? oldLines[i] : undefined;
        const newLine = i < newLines.length ? newLines[i] : undefined;

        if (oldLine === newLine) {
            html += `<div class="diff-line diff-same">&nbsp;${escapeHtml(oldLine)}</div>`;
            } else {
            if (oldLine !== undefined) {
                html += `<div class="diff-line diff-removed">-${escapeHtml(oldLine)}</div>`;
            }
            if (newLine !== undefined) {
                html += `<div class="diff-line diff-added">+${escapeHtml(newLine)}</div>`;
            }
        }
    }

    return html;
}

function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function cancelGeneration() {
    state.pendingGeneration = null;
    document.getElementById('diffModal').classList.remove('visible');
}

function confirmGeneration() {
    document.getElementById('diffModal').classList.remove('visible');
    if (state.pendingGeneration) {
        const fn = state.pendingGeneration;
        state.pendingGeneration = null;
        fn();
    }
}

// ==================== History ====================
function addHistoryEntry(type) {
    const entry = {
        type: type,                       // 'sp' or 'image'
        inputImage: state.currentImage ? state.currentImage.src : null,
        userPrompt: state.userPrompt || '',
        structuredPrompt: state.structuredPrompt || '',
        params: { ...state.params },
        timestamp: Date.now(),
        colorIndex: state.generationCount % HISTORY_COLORS.length,
    };

    state.history.push(entry);
    state.generationCount++;
    renderHistory();
}

function renderHistory() {
    const scroll = document.getElementById('historyScroll');
    const empty = document.getElementById('historyEmpty');
    const clearBtn = document.getElementById('historyClearBtn');

    // 先清除已有 DOM 元素
    scroll.querySelectorAll('.history-item, .history-separator').forEach(el => el.remove());

    if (state.history.length === 0) {
        empty.style.display = '';
        clearBtn.style.display = 'none';
                return;
            }

    empty.style.display = 'none';
    clearBtn.style.display = '';

    let prevColorIndex = -1;
    let prevOutputImage = null;

    state.history.forEach((entry, index) => {
        // Separator between different generation groups
        if (index > 0 && entry.colorIndex !== prevColorIndex && prevColorIndex !== -1) {
            const sep = document.createElement('div');
            sep.className = 'history-separator';
            scroll.appendChild(sep);
        }
        prevColorIndex = entry.colorIndex;

        const item = document.createElement('div');
        item.className = `history-item ${HISTORY_COLORS[entry.colorIndex]}`;
        item.onclick = () => showHistoryDetail(index);
        item.title = `#${index + 1} ${entry.type === 'sp' ? '生成SP' : '生成Image'}`;

        // Thumbnail
        if (entry.inputImage) {
            const isRepeated = prevOutputImage && entry.inputImage === prevOutputImage;
            const thumb = document.createElement('img');
            thumb.className = 'history-thumb' + (isRepeated ? ' repeated' : '');
            thumb.src = entry.inputImage;
            item.appendChild(thumb);
        }

        // Type label
        const typeLabel = document.createElement('span');
        typeLabel.className = 'history-type-label';
        typeLabel.textContent = entry.type === 'sp' ? 'SP' : 'IMG';
        item.appendChild(typeLabel);

        // Prompt badge
        if (entry.userPrompt) {
            const badge = document.createElement('span');
            badge.className = 'history-badge badge-p';
            badge.textContent = 'P';
            badge.title = entry.userPrompt.substring(0, 100);
            item.appendChild(badge);
        }

        // SP badge
        if (entry.structuredPrompt) {
            const sBadge = document.createElement('span');
            sBadge.className = 'history-badge badge-s';
            sBadge.textContent = 'S';
            sBadge.title = 'Structured Prompt';
            item.appendChild(sBadge);
        }

        // Config badge
        const cfgBadge = document.createElement('span');
        cfgBadge.className = 'history-badge badge-c';
        cfgBadge.textContent = 'C';
        cfgBadge.title = `CFG:${entry.params.cfgScale || entry.params.cfgText || entry.params.cfg || '-'} Step:${entry.params.steps} ${entry.params.width}×${entry.params.height} Seed:${entry.params.seed}`;
        item.appendChild(cfgBadge);

        scroll.appendChild(item);

        // Track for "repeated" detection
        if (entry.type === 'image' && entry.inputImage) {
            prevOutputImage = entry.inputImage;
        }
    });

    // Scroll to end
    requestAnimationFrame(() => {
        scroll.scrollLeft = scroll.scrollWidth;
    });
}

function clearHistory() {
    if (state.history.length === 0) return;
    if (!confirm('确定要清空所有历史记录并重置所有输入吗？')) return;

    // 清空历史
    state.history = [];
    state.generationCount = 0;
    renderHistory();

    // 清空图片
    state.currentImage = null;
    state.bboxVisible = false;
    state.activeBboxIndex = -1;
    state.bboxEditIndex = -1;
    state.focusedBboxIndex = -1;
    state.parsedElements = [];
    clearBboxes();
    updateImagePanel();

    // 清空 Structured Prompt
    state.structuredPrompt = '';
    state.parsedSP = null;
    state.undoStack = [];
    updateSPPanel();

    // 清空 User Prompt
    state.userPrompt = '';
    const promptInput = document.getElementById('userPromptInput');
    if (promptInput) promptInput.value = '';

    showToast('已清空所有历史记录和输入', 'info');
}

function showHistoryDetail(index) {
    state.selectedHistoryIndex = index;
    const entry = state.history[index];
    const body = document.getElementById('historyDetailBody');

    let html = '';

    // Meta
    html += `<div class="detail-section">`;
    html += `<div class="detail-section-title">基本信息</div>`;
    html += `<div class="detail-section-content">`;
    html += `类型: ${entry.type === 'sp' ? '生成 Structured Prompt' : '生成图像'}\n`;
    html += `时间: ${new Date(entry.timestamp).toLocaleString()}`;
    html += `</div></div>`;

    // Image
    if (entry.inputImage) {
        html += `<div class="detail-section">`;
        html += `<div class="detail-section-title">输入图片</div>`;
        html += `<img class="detail-image" src="${entry.inputImage}" />`;
        html += `</div>`;
    }

    // User Prompt
    if (entry.userPrompt) {
        html += `<div class="detail-section">`;
        html += `<div class="detail-section-title">User Prompt</div>`;
        html += `<div class="detail-section-content">${escapeHtml(entry.userPrompt)}</div>`;
        html += `</div>`;
    }

    // Structured Prompt
    if (entry.structuredPrompt) {
        html += `<div class="detail-section">`;
        html += `<div class="detail-section-title">Structured Prompt</div>`;
        html += `<div class="detail-section-content mono">${escapeHtml(entry.structuredPrompt)}</div>`;
        html += `</div>`;
    }

    // Params
    const cfgVal = entry.params.cfgScale || entry.params.cfgText || entry.params.cfg || '-';
    html += `<div class="detail-section">`;
    html += `<div class="detail-section-title">生成参数</div>`;
    html += `<div class="detail-section-content">`;
    html += `CFG: ${cfgVal}   Step: ${entry.params.steps}   Size: ${entry.params.width} × ${entry.params.height}   Seed: ${entry.params.seed}\n`;
    html += `</div></div>`;

    body.innerHTML = html;
    document.getElementById('historyModal').classList.add('visible');
}

function closeHistoryModal() {
    document.getElementById('historyModal').classList.remove('visible');
}

function restoreFromHistory() {
    const index = state.selectedHistoryIndex;
    if (index < 0 || index >= state.history.length) return;

    const entry = state.history[index];

    // Restore image
    if (entry.inputImage) {
        loadImageFromSrc(entry.inputImage);
                } else {
        state.currentImage = null;
        updateImagePanel();
    }

    // Restore User Prompt
    state.userPrompt = entry.userPrompt || '';
    document.getElementById('userPromptInput').value = state.userPrompt;

    // Restore SP
    setSPContent(entry.structuredPrompt || '');

    // Restore params
    state.params = { ...entry.params };
    syncParamsToUI();

    closeHistoryModal();
    showToast('已恢复历史记录 #' + (index + 1), 'success');
}

function restoreImageFromHistory() {
    const index = state.selectedHistoryIndex;
    if (index < 0 || index >= state.history.length) return;
    const entry = state.history[index];

    if (entry.inputImage) {
        loadImageFromSrc(entry.inputImage);
        autoUpdateHW();
    } else {
        state.currentImage = null;
        updateImagePanel();
    }
    closeHistoryModal();
    showToast('已恢复图片', 'success');
}

function restorePromptFromHistory() {
    const index = state.selectedHistoryIndex;
    if (index < 0 || index >= state.history.length) return;
    const entry = state.history[index];

    state.userPrompt = entry.userPrompt || '';
    document.getElementById('userPromptInput').value = state.userPrompt;
    setSPContent(entry.structuredPrompt || '');
    closeHistoryModal();
    showToast('已恢复 Prompt', 'success');
}

function restoreParamsFromHistory() {
    const index = state.selectedHistoryIndex;
    if (index < 0 || index >= state.history.length) return;
    const entry = state.history[index];

    state.params = { ...entry.params };
    syncParamsToUI();
    closeHistoryModal();
    showToast('已恢复生成参数', 'success');
}

// ==================== Toast Notifications ====================
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toast-out 0.3s ease forwards';
        setTimeout(() => {
            if (toast.parentNode) toast.remove();
        }, 300);
    }, 3000);
}

// ==================== BBox Drag & Resize ====================
function initBboxDrag(e, elemIndex, handle) {
    const container = document.getElementById('bboxContainer');
    const containerRect = container.getBoundingClientRect();
    const elem = state.parsedElements[elemIndex];
    if (!elem || !elem.bbox) return;

    state.bboxDrag = {
        elemIndex,
        handle,
        startX: e.clientX,
        startY: e.clientY,
        startBbox: [...elem.bbox],
        containerRect,
    };

    document.addEventListener('mousemove', onBboxDrag);
    document.addEventListener('mouseup', endBboxDrag);
}

function onBboxDrag(e) {
    const drag = state.bboxDrag;
    if (!drag) return;

    const { elemIndex, handle, startX, startY, startBbox, containerRect } = drag;
    const dx = ((e.clientX - startX) / containerRect.width) * 1000;
    const dy = ((e.clientY - startY) / containerRect.height) * 1000;

    let [x1, y1, x2, y2] = startBbox;

    if (handle === 'move') {
        const w = x2 - x1, h = y2 - y1;
        x1 = Math.max(0, Math.min(1000 - w, x1 + dx));
        y1 = Math.max(0, Math.min(1000 - h, y1 + dy));
        x2 = x1 + w;
        y2 = y1 + h;
    } else {
        // Resize
        if (handle.includes('n')) y1 = Math.max(0, Math.min(y2 - 10, y1 + dy));
        if (handle.includes('s')) y2 = Math.max(y1 + 10, Math.min(1000, y2 + dy));
        if (handle.includes('w')) x1 = Math.max(0, Math.min(x2 - 10, x1 + dx));
        if (handle.includes('e')) x2 = Math.max(x1 + 10, Math.min(1000, x2 + dx));
    }

    // Round
    x1 = Math.round(x1); y1 = Math.round(y1);
    x2 = Math.round(x2); y2 = Math.round(y2);

    // Update visual position
    const box = document.querySelector(`.bbox-box[data-elem-index="${elemIndex}"]`);
    if (box) {
        box.style.left = (x1 / 1000 * 100) + '%';
        box.style.top = (y1 / 1000 * 100) + '%';
        box.style.width = ((x2 - x1) / 1000 * 100) + '%';
        box.style.height = ((y2 - y1) / 1000 * 100) + '%';
    }

    // Store temp bbox for commit
    drag.currentBbox = [x1, y1, x2, y2];
}

function endBboxDrag(e) {
    document.removeEventListener('mousemove', onBboxDrag);
    document.removeEventListener('mouseup', endBboxDrag);

    const drag = state.bboxDrag;
    if (!drag) return;

    const newBbox = drag.currentBbox || drag.startBbox;
    const elemIndex = drag.elemIndex;
    state.bboxDrag = null;

    // Check if actually changed
    const old = drag.startBbox;
    const moved = old[0] !== newBbox[0] || old[1] !== newBbox[1] || old[2] !== newBbox[2] || old[3] !== newBbox[3];

    if (!moved) {
        // Was a click, not a drag — handle bbox click with overlap cycling
        handleBboxClick(elemIndex, e);
                            return;
                        }

    // Update parsed element
    state.parsedElements[elemIndex].bbox = newBbox;

    // Update SP JSON
    updateSPPosition(elemIndex, newBbox);
}

function updateSPPosition(elemIndex, newBbox) {
    if (!state.parsedSP) return;
    pushSPUndo();

    const elem = state.parsedElements[elemIndex];
    const newPosStr = `<bbox>${newBbox.join(' ')}</bbox>`;

    // Find and update in elements array
    let updated = false;
    if (state.parsedSP.elements) {
        for (const e of state.parsedSP.elements) {
            if (e.id === elem.id && e.position) {
                e.position = newPosStr;
                updated = true;
                break;
            }
        }
    }
    if (!updated && state.parsedSP.scene && state.parsedSP.scene.elements) {
        for (const e of state.parsedSP.scene.elements) {
            if (e.id === elem.id && e.position) {
                e.position = newPosStr;
                break;
            }
        }
    }

    state.structuredPrompt = stableJSONStringify(state.parsedSP, state.structuredPrompt);
    // Re-render structured view if visible
    if (!state.spRawMode) {
        renderSPStructured();
    } else {
        document.getElementById('spTextarea').value = state.structuredPrompt;
    }
}

// ==================== Window Events ====================
window.addEventListener('resize', () => {
    if (state.bboxVisible && state.currentImage) {
        renderBboxes();
    }
});
