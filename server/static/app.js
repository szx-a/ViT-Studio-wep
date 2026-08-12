// ==================== Tab 切换 ====================
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'dataset') refreshDatasets();
    if (btn.dataset.tab === 'train') refreshTrainDatasetList();
  });
});

// ==================== Tab 1: 图片识别 ====================
let selectedFile = null;
const zone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const btnPredict = document.getElementById('btnPredict');
const resultCard = document.getElementById('resultCard');
const resultList = document.getElementById('resultList');
const topK = document.getElementById('topK');
const predictProgress = document.getElementById('predictProgress');
let resultChart = null;

zone.addEventListener('click', () => fileInput.click());
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewBox.classList.remove('hidden');
  zone.querySelector('p').textContent = file.name;
  btnPredict.disabled = false;
  resultCard.classList.add('hidden');
}

btnPredict.addEventListener('click', async () => {
  if (!selectedFile) return;
  btnPredict.disabled = true;
  btnPredict.textContent = '⏳ 识别中...';
  predictProgress.classList.remove('hidden');

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('top_k', topK.value || '5');

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await res.json();

    resultList.innerHTML = data.results.map(r =>
      `<div class="result-item">
        <span class="result-rank">#${r.rank}</span>
        <span class="result-name">
          <strong>${r.class_name_zh}</strong>
          <small style="color:#888;margin-left:6px">${r.class_name}</small>
        </span>
        <span class="result-conf">${(r.confidence*100).toFixed(1)}%</span>
      </div>`
    ).join('');

    drawResultChart(data.results);
    resultCard.classList.remove('hidden');
  } catch (e) {
    resultList.innerHTML = '<p style="color:#e74c3c">识别失败: ' + e.message + '</p>';
  }
  btnPredict.disabled = false;
  btnPredict.textContent = '🔍 识别';
  predictProgress.classList.add('hidden');
});

function drawResultChart(results) {
  const ctx = document.getElementById('resultChart');
  if (resultChart) resultChart.destroy();
  resultChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: results.map(r => r.class_name_zh + ' / ' + r.class_name),
      datasets: [{
        label: '置信度',
        data: results.map(r => r.confidence * 100),
        backgroundColor: results.map((_, i) =>
          `hsla(${240 - i*30}, 70%, 65%, 0.7)`),
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { max: 100, ticks: { callback: v => v + '%', color: '#999' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#ccc', font: { size: 11 } } }
      }
    }
  });
}

// ==================== Tab 2: 数据集 ====================
async function refreshDatasets() {
  const res = await fetch('/api/datasets');
  const data = await res.json();
  const sel = document.getElementById('categorySelect');
  sel.innerHTML = '<option value="">-- 选择类别 --</option>';
  data.datasets.forEach(cat => {
    sel.innerHTML += `<option value="${cat.name}">${cat.name} (${cat.count})</option>`;
  });
  showImages(data.datasets);
}

document.getElementById('categorySelect').addEventListener('change', async function() {
  if (!this.value) { document.getElementById('imageGrid').innerHTML = '请先选择一个类别'; return; }
  const res = await fetch('/api/datasets');
  const data = await res.json();
  showImages(data.datasets, this.value);
});

function showImages(datasets, filterCategory) {
  const grid = document.getElementById('imageGrid');
  if (!datasets.length) { grid.innerHTML = '<p style="color:#666">暂无数据，请创建类别并上传图片</p>'; return; }

  let html = '';
  datasets.forEach(cat => {
    if (filterCategory && cat.name !== filterCategory) return;
    cat.images.forEach(img => {
      html += `<div class="grid-item">
        <img src="/datasets/${cat.name}/${img}" alt="${img}" onclick="deleteImage('${cat.name}','${img}')">
        <button class="del-btn" onclick="event.stopPropagation();deleteImage('${cat.name}','${img}')">✕</button>
      </div>`;
    });
  });
  grid.innerHTML = html || '<p style="color:#666">该类别暂无图片</p>';
}

async function deleteImage(cat, img) {
  if (!confirm(`删除 ${cat}/${img}?`)) return;
  const form = new FormData();
  form.append('category', cat);
  form.append('filename', img);
  await fetch('/api/datasets/image', { method: 'DELETE', body: form });
  refreshDatasets();
}

document.getElementById('btnCreateCategory').addEventListener('click', async () => {
  const name = document.getElementById('newCategory').value.trim();
  if (!name) return;
  const form = new FormData();
  form.append('category', name);
  await fetch('/api/datasets/category', { method: 'POST', body: form });
  document.getElementById('newCategory').value = '';
  document.getElementById('datasetMsg').textContent = '✓ 类别已创建';
  refreshDatasets();
});

document.getElementById('btnUploadTrigger').addEventListener('click', () => {
  document.getElementById('datasetFileInput').click();
});

document.getElementById('datasetFileInput').addEventListener('change', async function() {
  const sel = document.getElementById('categorySelect');
  if (!sel.value) { alert('请先选择类别'); return; }
  const msg = document.getElementById('datasetMsg');
  for (const file of this.files) {
    const form = new FormData();
    form.append('file', file);
    form.append('category', sel.value);
    await fetch('/api/datasets/upload', { method: 'POST', body: form });
  }
  msg.textContent = `✓ 已上传 ${this.files.length} 张图片`;
  refreshDatasets();
});

// ==================== Tab 3: 训练中心 ====================
let lossChart = null, accChart = null;
let pollTimer = null;

async function refreshTrainDatasetList() {
  const res = await fetch('/api/datasets');
  const data = await res.json();
  const sel = document.getElementById('trainDataset');
  sel.innerHTML = data.datasets.map(c => `<option value="${c.name}">${c.name} (${c.count}张)</option>`).join('');
}

document.getElementById('btnStartTrain').addEventListener('click', async () => {
  const ds = document.getElementById('trainDataset').value;
  if (!ds) { alert('请选择数据集'); return; }

  const form = new FormData();
  form.append('dataset_name', ds);
  form.append('epochs', document.getElementById('trainEpochs').value);
  form.append('batch_size', document.getElementById('trainBatch').value);
  form.append('lr', document.getElementById('trainLR').value);
  form.append('freeze_backbone', document.getElementById('trainFreeze').checked);

  const res = await fetch('/api/train/start', { method: 'POST', body: form });
  const data = await res.json();

  if (data.status === 'error') {
    alert(data.message); return;
  }

  document.getElementById('batchProgressBar').style.display = 'block';
  document.getElementById('batchProgressLabel').style.display = 'block';
  initCharts();
  startPolling();
});

document.getElementById('btnStopTrain').addEventListener('click', async () => {
  await fetch('/api/train/stop', { method: 'POST' });
  document.getElementById('trainMessage').textContent = '正在停止...';
});

function initCharts() {
  const lossCtx = document.getElementById('lossChart').getContext('2d');
  const accCtx = document.getElementById('accChart').getContext('2d');
  if (lossChart) lossChart.destroy();
  if (accChart) accChart.destroy();

  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: '训练 Loss', data: [], borderColor: '#7c8cf8', tension: 0.3 },
      { label: '验证 Loss', data: [], borderColor: '#4caf50', tension: 0.3 }
    ]},
    options: chartOptions('Loss')
  });

  accChart = new Chart(accCtx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: '训练 Acc', data: [], borderColor: '#7c8cf8', tension: 0.3 },
      { label: '验证 Acc', data: [], borderColor: '#4caf50', tension: 0.3 }
    ]},
    options: chartOptions('Accuracy')
  });
}

function chartOptions(label) {
  return {
    responsive: true,
    plugins: { legend: { labels: { color: '#999' } } },
    scales: {
      x: { title: { display: true, text: 'Epoch', color: '#999' }, ticks: { color: '#999' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { title: { display: true, text: label, color: '#999' }, ticks: { color: '#999' }, grid: { color: 'rgba(255,255,255,0.05)' } }
    }
  };
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      const s = await res.json();

      document.getElementById('trainStatus').textContent = '— ' + s.message;
      document.getElementById('trainMessage').textContent = s.message;

      // Epoch 进度
      if (s.total_epochs > 0) {
        const pct = (s.epoch / s.total_epochs * 100).toFixed(0);
        document.getElementById('progressFill').style.width = pct + '%';
      }

      // Batch 进度
      if (s.total_batches > 0) {
        const bpct = (s.batch / s.total_batches * 100).toFixed(0);
        document.getElementById('batchProgressFill').style.width = bpct + '%';
        document.getElementById('batchProgressLabel').textContent =
          `Batch ${s.batch}/${s.total_batches}  (Epoch ${s.epoch}/${s.total_epochs})`;
      }

      // Update charts
      if (s.history && s.history.length) {
        const epochs = s.history.map(h => h.epoch);
        lossChart.data.labels = epochs;
        lossChart.data.datasets[0].data = s.history.map(h => h.train_loss);
        lossChart.data.datasets[1].data = s.history.map(h => h.val_loss);
        lossChart.update();

        accChart.data.labels = epochs;
        accChart.data.datasets[0].data = s.history.map(h => h.train_acc);
        accChart.data.datasets[1].data = s.history.map(h => h.val_acc);
        accChart.update();
      }

      if (!s.running) {
        clearInterval(pollTimer);
        pollTimer = null;
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('batchProgressFill').style.width = '100%';
      }
    } catch (e) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }, 1000);
}
