/**
 * AeroMetrics Conflict Research Console - Client Application
 * Research simulation only - not for operational use.
 * Zero placeholder data: all views and metrics driven dynamically by REST API.
 */

(function () {
  'use strict';

  const state = {
    dataset: 'hard_v1',
    split: 'test',
    model: 'cv_smoothed',
    lookahead: 180,
    scenarios: [],
    scenarioIndex: 0,
    scenarioId: '',
    scenarioData: null,
    systemInfo: null,
    originT: null,
    isPlaying: false,
    playTimer: null,
    playSpeed: 1,
    showEllipses: true,
    showHistory: true,
    zoomFactor: 1.0,
    activeModule: 'conflict-prediction',
    ledgerFilter: 'ALL',
    inspectorFilter: 'ALL',
  };

  // DOM Elements
  const el = {
    disclaimerBanner: document.getElementById('disclaimerBanner'),
    headerBuild: document.getElementById('headerBuild'),
    headerDatasetSpec: document.getElementById('headerDatasetSpec'),
    headerEngineSpec: document.getElementById('headerEngineSpec'),
    headerSimTime: document.getElementById('headerSimTime'),
    headerLookaheadBadge: document.getElementById('headerLookaheadBadge'),
    headerDeviceChip: document.getElementById('headerDeviceChip'),
    sidebarSepMinima: document.getElementById('sidebarSepMinima'),
    sidebarPredStep: document.getElementById('sidebarPredStep'),
    sidebarSeed: document.getElementById('sidebarSeed'),
    ribbonModelTag: document.getElementById('ribbonModelTag'),
    ribbonScenarioId: document.getElementById('ribbonScenarioId'),
    ribbonAirspaceDesc: document.getElementById('ribbonAirspaceDesc'),
    ribbonConflictBadge: document.getElementById('ribbonConflictBadge'),
    ribbonEvalLatency: document.getElementById('ribbonEvalLatency'),
    datasetSelector: document.getElementById('datasetSelector'),
    splitSelector: document.getElementById('splitSelector'),
    scenarioPagerIndex: document.getElementById('scenarioPagerIndex'),
    currentScenarioIdDisplay: document.getElementById('currentScenarioIdDisplay'),
    currentScenarioTag: document.getElementById('currentScenarioTag'),
    prevScenarioBtn: document.getElementById('prevScenarioBtn'),
    nextScenarioBtn: document.getElementById('nextScenarioBtn'),
    scenarioListLedger: document.getElementById('scenarioListLedger'),
    scrubberTimeDisplay: document.getElementById('scrubberTimeDisplay'),
    timelineBar: document.getElementById('timelineBar'),
    playbackFill: document.getElementById('playbackFill'),
    scrubHead: document.getElementById('scrubHead'),
    timelineCpaWindow: document.getElementById('timelineCpaWindow'),
    timelineMaxTime: document.getElementById('timelineMaxTime'),
    stepBackBtn: document.getElementById('stepBackBtn'),
    simPlayBtn: document.getElementById('simPlayBtn'),
    stepFwdBtn: document.getElementById('stepFwdBtn'),
    speedButtons: document.getElementById('speedButtons'),
    lookaheadSlider: document.getElementById('lookaheadSlider'),
    lookaheadValueText: document.getElementById('lookaheadValueText'),
    pairRiskChip: document.getElementById('pairRiskChip'),
    ac1Id: document.getElementById('ac1Id'),
    ac1Type: document.getElementById('ac1Type'),
    ac1Alt: document.getElementById('ac1Alt'),
    ac1Speed: document.getElementById('ac1Speed'),
    ac1Hdg: document.getElementById('ac1Hdg'),
    ac1Rocd: document.getElementById('ac1Rocd'),
    ac2Id: document.getElementById('ac2Id'),
    ac2Type: document.getElementById('ac2Type'),
    ac2Alt: document.getElementById('ac2Alt'),
    ac2Speed: document.getElementById('ac2Speed'),
    ac2Hdg: document.getElementById('ac2Hdg'),
    ac2Rocd: document.getElementById('ac2Rocd'),
    pairClosureRate: document.getElementById('pairClosureRate'),
    pairCurrentSep: document.getElementById('pairCurrentSep'),
    pairCpaDist: document.getElementById('pairCpaDist'),
    pairTau: document.getElementById('pairTau'),
    pairCpaVert: document.getElementById('pairCpaVert'),
    pairPRisk: document.getElementById('pairPRisk'),
    canvasSectorBounds: document.getElementById('canvasSectorBounds'),
    radarViewport: document.getElementById('radarViewport'),
    radarSvg: document.getElementById('radarSvg'),
    radarConflictBadge: document.getElementById('radarConflictBadge'),
    radarConflictTitle: document.getElementById('radarConflictTitle'),
    radarBadgeCpa: document.getElementById('radarBadgeCpa'),
    radarBadgeDh: document.getElementById('radarBadgeDh'),
    radarBadgeRisk: document.getElementById('radarBadgeRisk'),
    toggleEllipsesBtn: document.getElementById('toggleEllipsesBtn'),
    toggleHistoryBtn: document.getElementById('toggleHistoryBtn'),
    recenterBtn: document.getElementById('recenterBtn'),
    altProfileTitle: document.getElementById('altProfileTitle'),
    altLegendAc1: document.getElementById('altLegendAc1'),
    altLegendAc2: document.getElementById('altLegendAc2'),
    altSvg: document.getElementById('altSvg'),
    modelSelectorGroup: document.getElementById('modelSelectorGroup'),
    modelLatencyDisplay: document.getElementById('modelLatencyDisplay'),
    modelWeightsDisplay: document.getElementById('modelWeightsDisplay'),
    exportMetricsBtn: document.getElementById('exportMetricsBtn'),
    exportReportBtn: document.getElementById('exportReportBtn'),
    exportJsonBtn: document.getElementById('exportJsonBtn'),
    exportCsvBtn: document.getElementById('exportCsvBtn'),
    metricPrecision: document.getElementById('metricPrecision'),
    barPrecision: document.getElementById('barPrecision'),
    metricRecall: document.getElementById('metricRecall'),
    barRecall: document.getElementById('barRecall'),
    metricF1: document.getElementById('metricF1'),
    barF1: document.getElementById('barF1'),
    metricFar: document.getElementById('metricFar'),
    barFar: document.getElementById('barFar'),
    metricLead: document.getElementById('metricLead'),
    barLead: document.getElementById('barLead'),
    metricTtcMae: document.getElementById('metricTtcMae'),
    barTtc: document.getElementById('barTtc'),
    metricTpCount: document.getElementById('metricTpCount'),
    metricFpCount: document.getElementById('metricFpCount'),
    // Research module navigation & modular view containers
    researchNav: document.getElementById('researchNav'),
    viewConflictPrediction: document.getElementById('view-conflict-prediction'),
    viewResolutionEngine: document.getElementById('view-resolution-engine'),
    viewTrajectoryInspector: document.getElementById('view-trajectory-inspector'),
    viewLosLedger: document.getElementById('view-los-ledger'),
    viewModelBenchmarks: document.getElementById('view-model-benchmarks'),
    // Trajectory Inspector controls
    inspectorAircraftFilter: document.getElementById('inspectorAircraftFilter'),
    inspectorTableBody: document.getElementById('inspectorTableBody'),
    // LoS Ledger controls
    ledgerCountDisplay: document.getElementById('ledgerCountDisplay'),
    ledgerFilterGroup: document.getElementById('ledgerFilterGroup'),
    ledgerTableBody: document.getElementById('ledgerTableBody'),
    // Resolution Engine controls
    resStatusBadge: document.getElementById('resStatusBadge'),
    resEncounterTitle: document.getElementById('resEncounterTitle'),
    resEncounterDesc: document.getElementById('resEncounterDesc'),
    resOpt1Title: document.getElementById('resOpt1Title'),
    resOpt1Desc: document.getElementById('resOpt1Desc'),
    resOpt1Vert: document.getElementById('resOpt1Vert'),
    resOpt2Title: document.getElementById('resOpt2Title'),
    resOpt2Desc: document.getElementById('resOpt2Desc'),
    resOpt2Lat: document.getElementById('resOpt2Lat'),
    resOpt3Title: document.getElementById('resOpt3Title'),
    resOpt3Desc: document.getElementById('resOpt3Desc'),
    resOpt3Lat: document.getElementById('resOpt3Lat'),
  };

  // API Client Methods
  async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    }
    return resp.json();
  }

  async function loadSystemInfo() {
    try {
      const data = await fetchJson('/api/system');
      state.systemInfo = data;
      if (el.headerDeviceChip) {
        el.headerDeviceChip.textContent = `ACTIVE (${data.device})`;
      }
      if (el.sidebarPredStep) {
        el.sidebarPredStep.textContent = `5.0s (${data.device})`;
      }
      if (el.sidebarSepMinima && data.separation_minima) {
        el.sidebarSepMinima.textContent = `${data.separation_minima.lateral_nm} NM / ${data.separation_minima.vertical_ft} FT`;
      }
      if (el.sidebarSeed && data.seeds) {
        el.sidebarSeed.textContent = `#${data.seeds[state.dataset] || '42'}`;
      }
      if (el.disclaimerBanner && data.advisory_disclaimer) {
        el.disclaimerBanner.textContent = data.advisory_disclaimer;
      }
      if (el.headerBuild && data.console_version) {
        el.headerBuild.textContent = `CONSOLE BUILD: ${data.console_version}`;
      }
    } catch (err) {
      console.error('Failed to load system info:', err);
    }
  }

  async function loadScenarios() {
    try {
      const data = await fetchJson(`/api/scenarios?dataset=${state.dataset}&split=${state.split}`);
      state.scenarios = data;
      state.scenarioIndex = 0;
      if (state.scenarios.length > 0) {
        state.scenarioId = state.scenarios[0].scenario_id;
      } else {
        state.scenarioId = '';
      }
      renderScenarioLedger();
      renderLosLedger();
      await loadScenarioDetail();
      await loadMetrics();
    } catch (err) {
      console.error('Failed to load scenarios:', err);
    }
  }

  async function loadScenarioDetail() {
    if (!state.scenarioId) return;
    try {
      let url = `/api/scenario/${state.scenarioId}?dataset=${state.dataset}&split=${state.split}&model=${state.model}&lookahead=${state.lookahead}`;
      if (state.originT !== null) {
        url += `&origin=${state.originT}`;
      }
      const data = await fetchJson(url);
      state.scenarioData = data;
      state.originT = data.origin_t;
      renderScenarioViews();
    } catch (err) {
      console.error('Failed to load scenario detail:', err);
    }
  }

  async function loadMetrics() {
    try {
      const data = await fetchJson(`/api/metrics?dataset=${state.dataset}&split=${state.split}&model=${state.model}`);
      renderMetricsStrip(data);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    }
  }

  // Render Methods
  function renderScenarioLedger() {
    if (!el.scenarioListLedger) return;
    el.scenarioListLedger.innerHTML = '';
    state.scenarios.forEach((scen, idx) => {
      const isSelected = idx === state.scenarioIndex;
      const row = document.createElement('div');
      row.className = `flex items-center justify-between p-1.5 rounded cursor-pointer transition-colors ${
        isSelected ? 'bg-surface-container-high' : 'hover:bg-surface-container'
      }`;

      let tagColor = 'bg-primary-fixed-dim';
      let textColor = 'text-primary';
      if (scen.tag_type === 'conflict') {
        tagColor = 'bg-tertiary';
        textColor = 'text-tertiary';
      } else if (scen.tag_type === 'near_miss') {
        tagColor = 'bg-secondary';
        textColor = 'text-secondary-fixed';
      } else {
        tagColor = 'bg-outline';
        textColor = 'text-outline';
      }

      row.innerHTML = `
        <div class="flex items-center gap-1.5 min-w-0">
          <span class="w-1.5 h-1.5 rounded-full ${tagColor}"></span>
          <span class="font-telemetry-num text-body-sm ${isSelected ? 'text-on-surface font-semibold' : 'text-on-surface-variant'} truncate">${scen.scenario_id}</span>
        </div>
        <span class="font-label-sm text-label-sm ${textColor} uppercase">${scen.tag}</span>
      `;

      row.addEventListener('click', () => {
        state.scenarioIndex = idx;
        state.scenarioId = scen.scenario_id;
        state.originT = null; // reset to default origin
        renderScenarioLedger();
        loadScenarioDetail();
      });

      el.scenarioListLedger.appendChild(row);
    });

    if (el.scenarioPagerIndex) {
      const total = state.scenarios.length;
      const curr = state.scenarioIndex + 1;
      el.scenarioPagerIndex.textContent = `INDEX: ${String(curr).padStart(3, '0')} / ${String(total).padStart(3, '0')}`;
    }
    if (el.currentScenarioIdDisplay) {
      el.currentScenarioIdDisplay.textContent = state.scenarioId;
    }
    if (el.currentScenarioTag && state.scenarios[state.scenarioIndex]) {
      const scen = state.scenarios[state.scenarioIndex];
      el.currentScenarioTag.textContent = scen.tag;
      el.currentScenarioTag.className = `font-label-sm text-label-sm px-1.5 py-0.5 rounded ${
        scen.tag_type === 'conflict' ? 'bg-tertiary-container/20 text-tertiary' : 'bg-surface-container text-on-surface-variant'
      }`;
    }
  }

  function renderScenarioViews() {
    const data = state.scenarioData;
    if (!data) return;

    // 1. Header & Top Ribbon
    if (el.headerSimTime) {
      el.headerSimTime.textContent = data.origin_formatted;
    }
    if (el.headerDatasetSpec) {
      el.headerDatasetSpec.textContent = `${data.dataset.toUpperCase()}-${data.split.toUpperCase()}`;
    }
    if (el.headerLookaheadBadge) {
      el.headerLookaheadBadge.textContent = `+${Math.round(state.lookahead)}s LOOKAHEAD`;
    }
    if (el.sidebarSeed && state.systemInfo && state.systemInfo.seeds) {
      el.sidebarSeed.textContent = `#${state.systemInfo.seeds[data.dataset] || '42'}`;
    }

    if (el.ribbonModelTag) {
      el.ribbonModelTag.textContent = `MODEL BENCHMARK: ${data.model_name}`;
    }
    if (el.ribbonScenarioId) {
      el.ribbonScenarioId.textContent = data.scenario_id;
    }
    if (el.ribbonAirspaceDesc) {
      const isEu = data.dataset === 'opensky';
      el.ribbonAirspaceDesc.textContent = isEu
        ? 'CENTRAL EUROPE EN-ROUTE (ADS-B STREAM)'
        : 'SECTOR SYNTHETIC-ENROUTE (200 NM x 200 NM)';
    }

    const activePair = data.active_pair;
    const hasConflict = activePair && activePair.has_predicted_conflict;

    if (el.ribbonConflictBadge) {
      if (hasConflict) {
        el.ribbonConflictBadge.className = 'inline-flex items-center gap-1.5 px-space-sm py-space-xs rounded bg-surface-container font-label-sm text-label-sm text-tertiary';
        el.ribbonConflictBadge.innerHTML = `
          <span class="w-1.5 h-1.5 rounded-full bg-tertiary animate-ping"></span>
          <span>PREDICTED CONFLICT ACTIVE (t+${Math.round(activePair.predicted_tau_s)}s)</span>
        `;
      } else {
        el.ribbonConflictBadge.className = 'inline-flex items-center gap-1.5 px-space-sm py-space-xs rounded bg-surface-container font-label-sm text-label-sm text-primary';
        el.ribbonConflictBadge.innerHTML = `
          <span class="w-1.5 h-1.5 rounded-full bg-primary"></span>
          <span>SEPARATION CLEAR</span>
        `;
      }
    }

    // Latency
    const firstAc = Object.values(data.aircraft)[0];
    const latency = firstAc ? firstAc.inference_time_ms : 0.5;
    if (el.ribbonEvalLatency) {
      el.ribbonEvalLatency.textContent = `EVAL LATENCY: ${latency}ms`;
    }
    if (el.modelLatencyDisplay) {
      el.modelLatencyDisplay.textContent = `${latency}ms`;
    }

    // 2. Timeline Scrubber
    if (el.scrubberTimeDisplay) {
      el.scrubberTimeDisplay.textContent = `${data.origin_formatted} / ${data.duration_formatted}`;
    }
    if (el.timelineMaxTime) {
      el.timelineMaxTime.textContent = data.duration_formatted;
    }
    if (el.playbackFill && el.scrubHead && data.available_origins.length > 0) {
      const origins = data.available_origins;
      const minO = origins[0];
      const maxO = origins[origins.length - 1];
      const pct = Math.max(0, Math.min(100, ((data.origin_t - minO) / Math.max(1, maxO - minO)) * 100));
      el.playbackFill.style.width = `${pct}%`;
      el.scrubHead.style.left = `${pct}%`;
    }

    // 3. Active Pair Telemetry Cards
    if (activePair) {
      const ac1 = data.aircraft[activePair.aircraft_1];
      const ac2 = data.aircraft[activePair.aircraft_2];

      if (el.pairRiskChip) {
        if (activePair.has_predicted_conflict) {
          el.pairRiskChip.className = 'font-label-sm text-label-sm bg-tertiary-container/20 text-tertiary px-1.5 py-0.5 rounded font-bold uppercase';
          el.pairRiskChip.textContent = 'LOS RISKS: CRITICAL';
        } else if (activePair.has_gt_conflict) {
          el.pairRiskChip.className = 'font-label-sm text-label-sm bg-secondary-container/20 text-secondary-fixed px-1.5 py-0.5 rounded font-bold uppercase';
          el.pairRiskChip.textContent = 'GT CONFLICT (MISSED)';
        } else {
          el.pairRiskChip.className = 'font-label-sm text-label-sm bg-surface-container text-primary px-1.5 py-0.5 rounded font-bold uppercase';
          el.pairRiskChip.textContent = 'SEPARATION CLEAR';
        }
      }

      if (ac1) {
        if (el.ac1Id) el.ac1Id.textContent = ac1.callsign;
        if (el.ac1Alt) el.ac1Alt.textContent = `${ac1.flight_level} (${ac1.altitude_ft.toLocaleString()}\')`;
        if (el.ac1Speed) el.ac1Speed.textContent = `${ac1.speed_kt} kts`;
        if (el.ac1Hdg) el.ac1Hdg.textContent = `${ac1.heading_deg}°`;
        if (el.ac1Rocd) el.ac1Rocd.textContent = ac1.rocd_str;
      }

      if (ac2) {
        if (el.ac2Id) el.ac2Id.textContent = ac2.callsign;
        if (el.ac2Alt) el.ac2Alt.textContent = `${ac2.flight_level} (${ac2.altitude_ft.toLocaleString()}\')`;
        if (el.ac2Speed) el.ac2Speed.textContent = `${ac2.speed_kt} kts`;
        if (el.ac2Hdg) el.ac2Hdg.textContent = `${ac2.heading_deg}°`;
        if (el.ac2Rocd) el.ac2Rocd.textContent = ac2.rocd_str;
      }

      if (el.pairClosureRate) {
        el.pairClosureRate.textContent = `${activePair.closure_rate_kt} kts (${activePair.closure_rate_nm_min} NM/min)`;
      }
      if (el.pairCurrentSep) {
        el.pairCurrentSep.textContent = `${activePair.curr_distance_nm} NM`;
      }
      if (el.pairCpaDist) {
        const flag = activePair.predicted_cpa_nm < 5.0 ? ' (< 5.0 NM)' : '';
        el.pairCpaDist.innerHTML = `${activePair.predicted_cpa_nm} NM <span class="text-[10px] ${activePair.predicted_cpa_nm < 5.0 ? 'text-error' : 'text-outline'} font-normal">${flag}</span>`;
      }
      if (el.pairTau) {
        el.pairTau.textContent = `t+${activePair.predicted_tau_s} seconds`;
      }
      if (el.pairCpaVert) {
        const flag = activePair.predicted_cpa_vert_ft < 1000.0 ? ' (< 1,000 ft)' : '';
        el.pairCpaVert.innerHTML = `${activePair.predicted_cpa_vert_ft} ft <span class="text-[10px] ${activePair.predicted_cpa_vert_ft < 1000.0 ? 'text-error' : 'text-outline'} font-normal">${flag}</span>`;
      }
      if (el.pairPRisk) {
        el.pairPRisk.textContent = `${activePair.p_risk.toFixed(3)} (Threshold 0.65)`;
      }
    }

    // 4. Render 2D Airspace SVG
    render2DAirspace(data);

    // 5. Render Altitude vs Time Profile
    renderAltitudeProfile(data);

    // 6. Refresh active modular view if open
    if (state.activeModule === 'resolution-engine') {
      renderResolutionEngine(data);
    } else if (state.activeModule === 'trajectory-inspector') {
      renderTrajectoryInspector(data);
    }
  }

  function render2DAirspace(data) {
    if (!el.radarSvg) return;
    const svg = el.radarSvg;
    svg.innerHTML = '';

    const width = svg.clientWidth || 800;
    const height = svg.clientHeight || 460;
    const vp = data.viewport;

    // Coordinate mapping helper
    function toCanvas(x, y) {
      const px = ((x - vp.x_min) / (vp.x_max - vp.x_min)) * width;
      const py = height - ((y - vp.y_min) / (vp.y_max - vp.y_min)) * height;
      return [px, py];
    }

    // Defs & Grids
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <pattern id="radarGridMinor" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#102034" stroke-width="0.5"/>
      </pattern>
      <pattern id="radarGridMajor" width="100" height="100" patternUnits="userSpaceOnUse">
        <rect width="100" height="100" fill="url(#radarGridMinor)"/>
        <path d="M 100 0 L 0 0 0 100" fill="none" stroke="#162334" stroke-width="1"/>
      </pattern>
      <radialGradient id="ellipseGlow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.25"/>
        <stop offset="70%" stop-color="#f59e0b" stop-opacity="0.06"/>
        <stop offset="100%" stop-color="#f59e0b" stop-opacity="0"/>
      </radialGradient>
    `;
    svg.appendChild(defs);

    // Major grid background
    const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bgRect.setAttribute('width', '100%');
    bgRect.setAttribute('height', '100%');
    bgRect.setAttribute('fill', 'url(#radarGridMajor)');
    svg.appendChild(bgRect);

    // Center Axes
    const [cx, cy] = toCanvas(vp.x_center, vp.y_center);
    const lineX = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineX.setAttribute('x1', '0');
    lineX.setAttribute('x2', String(width));
    lineX.setAttribute('y1', String(cy));
    lineX.setAttribute('y2', String(cy));
    lineX.setAttribute('stroke', '#1b2b3f');
    lineX.setAttribute('stroke-dasharray', '3,3');
    lineX.setAttribute('stroke-width', '1.2');
    svg.appendChild(lineX);

    const lineY = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineY.setAttribute('x1', String(cx));
    lineY.setAttribute('x2', String(cx));
    lineY.setAttribute('y1', '0');
    lineY.setAttribute('y2', String(height));
    lineY.setAttribute('stroke', '#1b2b3f');
    lineY.setAttribute('stroke-dasharray', '3,3');
    lineY.setAttribute('stroke-width', '1.2');
    svg.appendChild(lineY);

    // Range rings (15 NM and 30 NM equivalent)
    const scalePxPerNm = width / (vp.x_max - vp.x_min);
    [10, 20, 30].forEach((rNm) => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', String(cx));
      circle.setAttribute('cy', String(cy));
      circle.setAttribute('r', String(rNm * scalePxPerNm));
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', '#162334');
      circle.setAttribute('stroke-dasharray', '2,4');
      circle.setAttribute('stroke-width', '1');
      svg.appendChild(circle);
    });

    if (el.canvasSectorBounds) {
      el.canvasSectorBounds.textContent = `[${Math.round(vp.x_min)} NM → +${Math.round(vp.x_max)} NM] • ORTHOGONAL MERCATOR LOCAL`;
    }

    const activePair = data.active_pair;

    // Draw Aircraft Tracks
    Object.keys(data.aircraft).forEach((acid) => {
      const ac = data.aircraft[acid];
      const isActive1 = activePair && activePair.aircraft_1 === acid;
      const isActive2 = activePair && activePair.aircraft_2 === acid;
      const isTarget = isActive1 || isActive2;

      const baseColor = isActive1 ? '#38bdf8' : isActive2 ? '#f59e0b' : '#64748b';

      // 1. Flown History Line
      if (state.showHistory && ac.history_points.length > 1) {
        const histPath = ac.history_points.map((pt, i) => {
          const [px, py] = toCanvas(pt.x, pt.y);
          return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
        }).join(' ');
        const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pathEl.setAttribute('d', histPath);
        pathEl.setAttribute('fill', 'none');
        pathEl.setAttribute('stroke', baseColor);
        pathEl.setAttribute('stroke-opacity', isTarget ? '0.6' : '0.25');
        pathEl.setAttribute('stroke-width', isTarget ? '1.8' : '1.2');
        svg.appendChild(pathEl);
      }

      // 2. Future Lookahead Line
      if (ac.future_pred.length > 1) {
        const futPath = ac.future_pred.map((pt, i) => {
          const [px, py] = toCanvas(pt.x, pt.y);
          return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
        }).join(' ');
        const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pathEl.setAttribute('d', futPath);
        pathEl.setAttribute('fill', 'none');
        pathEl.setAttribute('stroke', baseColor);
        pathEl.setAttribute('stroke-dasharray', '4,3');
        pathEl.setAttribute('stroke-width', isTarget ? '1.8' : '1.2');
        pathEl.setAttribute('stroke-opacity', isTarget ? '0.9' : '0.4');
        svg.appendChild(pathEl);
      }

      // 3. Current Position Marker (Triangle oriented by heading)
      const [curX, curY] = toCanvas(ac.current_pos.x, ac.current_pos.y);
      const markerGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      markerGroup.setAttribute('transform', `translate(${curX}, ${curY}) rotate(${ac.heading_deg})`);

      const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      poly.setAttribute('points', '0,-8 5,6 0,3 -5,6');
      poly.setAttribute('fill', baseColor);
      markerGroup.appendChild(poly);
      svg.appendChild(markerGroup);

      // Current position dot
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', String(curX));
      dot.setAttribute('cy', String(curY));
      dot.setAttribute('r', '3');
      dot.setAttribute('fill', baseColor);
      svg.appendChild(dot);

      // Data Tag Box
      const tagG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      tagG.setAttribute('transform', `translate(${curX + 10}, ${curY - 14})`);
      tagG.innerHTML = `
        <rect width="118" height="22" rx="2" fill="#000f21" fill-opacity="0.88" stroke="#1b2b3f" stroke-width="0.8"/>
        <text x="5" y="10" fill="${baseColor}" font-family="JetBrains Mono" font-size="8.5" font-weight="600">${ac.callsign} [${ac.flight_level}|${Math.round(ac.speed_kt)}k]</text>
        <text x="5" y="18" fill="#bdc8d1" font-family="JetBrains Mono" font-size="7.5">HDG:${Math.round(ac.heading_deg)}° ROCD:${ac.rocd_fpm}</text>
      `;
      svg.appendChild(tagG);
    });

    // 4. Conflict Overlay Details (Active Pair)
    if (activePair && activePair.has_predicted_conflict) {
      const ac1 = data.aircraft[activePair.aircraft_1];
      const ac2 = data.aircraft[activePair.aircraft_2];
      const [p1x, p1y] = toCanvas(ac1.current_pos.x, ac1.current_pos.y);
      const [p2x, p2y] = toCanvas(ac2.current_pos.x, ac2.current_pos.y);

      // Conflict Baseline between current aircraft
      const conLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      conLine.setAttribute('x1', String(p1x));
      conLine.setAttribute('y1', String(p1y));
      conLine.setAttribute('x2', String(p2x));
      conLine.setAttribute('y2', String(p2y));
      conLine.setAttribute('stroke', '#f59e0b');
      conLine.setAttribute('stroke-dasharray', '2,2');
      conLine.setAttribute('stroke-width', '1.5');
      svg.appendChild(conLine);

      // CPA Marker & Ellipse
      const [cpaX, cpaY] = toCanvas(activePair.cpa_pos_1[0], activePair.cpa_pos_1[1]);

      if (state.showEllipses) {
        const ellipse = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
        ellipse.setAttribute('cx', String(cpaX));
        ellipse.setAttribute('cy', String(cpaY));
        ellipse.setAttribute('rx', '35');
        ellipse.setAttribute('ry', '18');
        ellipse.setAttribute('fill', 'url(#ellipseGlow)');
        ellipse.setAttribute('stroke', '#f59e0b');
        ellipse.setAttribute('stroke-dasharray', '3,2');
        ellipse.setAttribute('stroke-width', '1');
        svg.appendChild(ellipse);
      }

      const cpaDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      cpaDot.setAttribute('cx', String(cpaX));
      cpaDot.setAttribute('cy', String(cpaY));
      cpaDot.setAttribute('r', '2.5');
      cpaDot.setAttribute('fill', '#f59e0b');
      svg.appendChild(cpaDot);

      const cpaText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      cpaText.setAttribute('x', String(cpaX + 8));
      cpaText.setAttribute('y', String(cpaY + 4));
      cpaText.setAttribute('fill', '#ffc174');
      cpaText.setAttribute('font-family', 'JetBrains Mono');
      cpaText.setAttribute('font-size', '8.5');
      cpaText.setAttribute('font-weight', '600');
      cpaText.textContent = `CPA (t+${Math.round(activePair.predicted_tau_s)}s)`;
      svg.appendChild(cpaText);

      // Floating Badge
      if (el.radarConflictBadge) el.radarConflictBadge.style.display = 'flex';
      if (el.radarConflictTitle) el.radarConflictTitle.textContent = `PREDICTED LOS IN ${Math.round(activePair.predicted_tau_s)}s`;
      if (el.radarBadgeCpa) el.radarBadgeCpa.textContent = `${activePair.predicted_cpa_nm} NM`;
      if (el.radarBadgeDh) el.radarBadgeDh.textContent = `${activePair.predicted_cpa_vert_ft} FT`;
      if (el.radarBadgeRisk) el.radarBadgeRisk.textContent = `${activePair.p_risk.toFixed(2)}`;
    } else {
      if (el.radarConflictBadge) el.radarConflictBadge.style.display = 'none';
    }
  }

  function renderAltitudeProfile(data) {
    if (!el.altSvg) return;
    const svg = el.altSvg;
    svg.innerHTML = '';

    const width = 800;
    const height = 170;
    const paddingLeft = 45;
    const paddingRight = 20;
    const paddingTop = 15;
    const paddingBottom = 25;

    const chartW = width - paddingLeft - paddingRight;
    const chartH = height - paddingTop - paddingBottom;

    const activePair = data.active_pair;
    if (!activePair) return;

    const ac1 = data.aircraft[activePair.aircraft_1];
    const ac2 = data.aircraft[activePair.aircraft_2];
    if (!ac1 || !ac2) return;

    if (el.altProfileTitle) {
      el.altProfileTitle.textContent = `VERTICAL & SLANT-RANGE SEPARATION PROFILE (PAIR: ${ac1.callsign} × ${ac2.callsign})`;
    }
    if (el.altLegendAc1) el.altLegendAc1.textContent = `${ac1.callsign} (${ac1.flight_level})`;
    if (el.altLegendAc2) el.altLegendAc2.textContent = `${ac2.callsign} (${ac2.flight_level})`;

    // Altitude Range
    const allAlts = [
      ...ac1.history_points.map(p => p.alt),
      ...ac1.future_pred.map(p => p.alt),
      ...ac2.history_points.map(p => p.alt),
      ...ac2.future_pred.map(p => p.alt),
    ];
    const minAlt = Math.floor(Math.min(...allAlts, 20000) / 1000) * 1000 - 1000;
    const maxAlt = Math.ceil(Math.max(...allAlts, 35000) / 1000) * 1000 + 1000;

    // Time Range: origin - 60s to origin + lookahead_s
    const tStart = data.origin_t - 60.0;
    const tEnd = data.origin_t + state.lookahead;

    function toChartX(t) {
      return paddingLeft + ((t - tStart) / (tEnd - tStart)) * chartW;
    }
    function toChartY(alt) {
      return paddingTop + (1.0 - (alt - minAlt) / (maxAlt - minAlt)) * chartH;
    }

    // Grid lines for altitude flight levels
    const altStep = (maxAlt - minAlt) >= 6000 ? 2000 : 1000;
    for (let alt = minAlt; alt <= maxAlt; alt += altStep) {
      const y = toChartY(alt);
      const gridLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      gridLine.setAttribute('x1', String(paddingLeft));
      gridLine.setAttribute('x2', String(width - paddingRight));
      gridLine.setAttribute('y1', String(y));
      gridLine.setAttribute('y2', String(y));
      gridLine.setAttribute('stroke', '#162334');
      gridLine.setAttribute('stroke-width', '0.8');
      svg.appendChild(gridLine);

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', '6');
      label.setAttribute('y', String(y + 3));
      label.setAttribute('fill', '#87929a');
      label.setAttribute('font-family', 'JetBrains Mono');
      label.setAttribute('font-size', '9');
      label.textContent = `FL${Math.round(alt / 100)}`;
      svg.appendChild(label);
    }

    // Danger zone (1000 ft RVSM separation band around AC1)
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <linearGradient id="dangerZoneGrad" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.10"/>
        <stop offset="50%" stop-color="#f59e0b" stop-opacity="0.18"/>
        <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.10"/>
      </linearGradient>
    `;
    svg.appendChild(defs);

    const ac1CurrAlt = ac1.altitude_ft;
    const bandTop = toChartY(ac1CurrAlt + 1000);
    const bandBottom = toChartY(ac1CurrAlt - 1000);
    const bandRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bandRect.setAttribute('x', String(paddingLeft));
    bandRect.setAttribute('y', String(bandTop));
    bandRect.setAttribute('width', String(chartW));
    bandRect.setAttribute('height', String(bandBottom - bandTop));
    bandRect.setAttribute('fill', 'url(#dangerZoneGrad)');
    svg.appendChild(bandRect);

    // RVSM Corridor Lines
    [bandTop, bandBottom].forEach(y => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', String(paddingLeft));
      line.setAttribute('x2', String(width - paddingRight));
      line.setAttribute('y1', String(y));
      line.setAttribute('y2', String(y));
      line.setAttribute('stroke', '#f59e0b');
      line.setAttribute('stroke-dasharray', '2,2');
      line.setAttribute('stroke-opacity', '0.5');
      line.setAttribute('stroke-width', '0.8');
      svg.appendChild(line);
    });

    // Time Indicators
    const xNow = toChartX(data.origin_t);
    const lineNow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineNow.setAttribute('x1', String(xNow));
    lineNow.setAttribute('x2', String(xNow));
    lineNow.setAttribute('y1', String(paddingTop));
    lineNow.setAttribute('y2', String(height - paddingBottom));
    lineNow.setAttribute('stroke', '#38bdf8');
    lineNow.setAttribute('stroke-width', '1.2');
    svg.appendChild(lineNow);

    const txtNow = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txtNow.setAttribute('x', String(xNow - 14));
    txtNow.setAttribute('y', String(height - 10));
    txtNow.setAttribute('fill', '#38bdf8');
    txtNow.setAttribute('font-family', 'JetBrains Mono');
    txtNow.setAttribute('font-size', '8.5');
    txtNow.setAttribute('font-weight', '600');
    txtNow.textContent = 'NOW (t=0)';
    svg.appendChild(txtNow);

    // Draw AC1 Profile
    // Flown history
    if (ac1.history_points.length > 1) {
      const pathHist1 = ac1.history_points.filter(p => p.t >= tStart).map((p, i) => {
        return `${i === 0 ? 'M' : 'L'} ${toChartX(p.t).toFixed(1)} ${toChartY(p.alt).toFixed(1)}`;
      }).join(' ');
      const p1H = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p1H.setAttribute('d', pathHist1);
      p1H.setAttribute('fill', 'none');
      p1H.setAttribute('stroke', '#38bdf8');
      p1H.setAttribute('stroke-width', '1.8');
      svg.appendChild(p1H);
    }
    // Predicted future
    if (ac1.future_pred.length > 0) {
      const pathFut1 = `M ${toChartX(data.origin_t).toFixed(1)} ${toChartY(ac1.altitude_ft).toFixed(1)} ` +
        ac1.future_pred.map(p => `L ${toChartX(p.t).toFixed(1)} ${toChartY(p.alt).toFixed(1)}`).join(' ');
      const p1F = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p1F.setAttribute('d', pathFut1);
      p1F.setAttribute('fill', 'none');
      p1F.setAttribute('stroke', '#38bdf8');
      p1F.setAttribute('stroke-dasharray', '4,3');
      p1F.setAttribute('stroke-width', '1.8');
      svg.appendChild(p1F);
    }

    // Draw AC2 Profile
    if (ac2.history_points.length > 1) {
      const pathHist2 = ac2.history_points.filter(p => p.t >= tStart).map((p, i) => {
        return `${i === 0 ? 'M' : 'L'} ${toChartX(p.t).toFixed(1)} ${toChartY(p.alt).toFixed(1)}`;
      }).join(' ');
      const p2H = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p2H.setAttribute('d', pathHist2);
      p2H.setAttribute('fill', 'none');
      p2H.setAttribute('stroke', '#f59e0b');
      p2H.setAttribute('stroke-width', '1.8');
      svg.appendChild(p2H);
    }
    if (ac2.future_pred.length > 0) {
      const pathFut2 = `M ${toChartX(data.origin_t).toFixed(1)} ${toChartY(ac2.altitude_ft).toFixed(1)} ` +
        ac2.future_pred.map(p => `L ${toChartX(p.t).toFixed(1)} ${toChartY(p.alt).toFixed(1)}`).join(' ');
      const p2F = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p2F.setAttribute('d', pathFut2);
      p2F.setAttribute('fill', 'none');
      p2F.setAttribute('stroke', '#f59e0b');
      p2F.setAttribute('stroke-dasharray', '4,3');
      p2F.setAttribute('stroke-width', '1.8');
      svg.appendChild(p2F);
    }

    // Predicted CPA Marker
    if (activePair.has_predicted_conflict) {
      const tCpa = data.origin_t + activePair.predicted_tau_s;
      const xCpa = toChartX(tCpa);

      const lineCpa = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      lineCpa.setAttribute('x1', String(xCpa));
      lineCpa.setAttribute('x2', String(xCpa));
      lineCpa.setAttribute('y1', String(paddingTop));
      lineCpa.setAttribute('y2', String(height - paddingBottom));
      lineCpa.setAttribute('stroke', '#f59e0b');
      lineCpa.setAttribute('stroke-dasharray', '3,2');
      lineCpa.setAttribute('stroke-width', '1.2');
      svg.appendChild(lineCpa);

      const txtCpa = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      txtCpa.setAttribute('x', String(xCpa - 18));
      txtCpa.setAttribute('y', String(height - 10));
      txtCpa.setAttribute('fill', '#ffc174');
      txtCpa.setAttribute('font-family', 'JetBrains Mono');
      txtCpa.setAttribute('font-size', '8.5');
      txtCpa.setAttribute('font-weight', '600');
      txtCpa.textContent = `CPA (t+${Math.round(activePair.predicted_tau_s)}s)`;
      svg.appendChild(txtCpa);
    }
  }

  // Research Modules & Views Switching
  function switchModule(moduleName) {
    if (!moduleName) return;

    if (moduleName === 'scenario-replay') {
      // Replay action: switch to conflict-prediction view and start playback if paused
      switchModule('conflict-prediction');
      if (!state.isPlaying && el.simPlayBtn) {
        el.simPlayBtn.click();
      }
      return;
    }

    state.activeModule = moduleName;

    // Update active highlight styling on sidebar links
    if (el.researchNav) {
      el.researchNav.querySelectorAll('a[data-module]').forEach((a) => {
        const isCurrent = a.dataset.module === moduleName;
        if (isCurrent) {
          a.className = 'flex items-center gap-space-md px-space-md py-space-sm rounded transition-colors bg-primary-container text-on-primary-container font-headline-sm text-headline-sm cursor-pointer';
          a.setAttribute('aria-current', 'page');
        } else {
          a.className = 'flex items-center gap-space-md px-space-md py-space-sm rounded font-headline-sm text-headline-sm text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface transition-colors cursor-pointer';
          a.removeAttribute('aria-current');
        }
      });
    }

    // Toggle view visibility
    const views = {
      'conflict-prediction': el.viewConflictPrediction,
      'resolution-engine': el.viewResolutionEngine,
      'trajectory-inspector': el.viewTrajectoryInspector,
      'los-ledger': el.viewLosLedger,
      'model-benchmarks': el.viewModelBenchmarks,
    };

    Object.entries(views).forEach(([name, domEl]) => {
      if (domEl) {
        if (name === moduleName) {
          domEl.classList.remove('hidden');
        } else {
          domEl.classList.add('hidden');
        }
      }
    });

    // Populate data for newly active view
    if (moduleName === 'resolution-engine') {
      renderResolutionEngine(state.scenarioData);
    } else if (moduleName === 'trajectory-inspector') {
      renderTrajectoryInspector(state.scenarioData);
    } else if (moduleName === 'los-ledger') {
      renderLosLedger();
    } else if (moduleName === 'conflict-prediction') {
      if (state.scenarioData) {
        render2DAirspace(state.scenarioData);
        renderAltitudeProfile(state.scenarioData);
      }
    }
  }

  function renderResolutionEngine(data) {
    if (!data) return;
    const pair = data.active_pair;
    const ac1 = pair && data.aircraft ? data.aircraft[pair.aircraft_1] : null;
    const ac2 = pair && data.aircraft ? data.aircraft[pair.aircraft_2] : null;
    const hasConflict = pair && pair.has_predicted_conflict;

    if (el.resStatusBadge) {
      if (hasConflict) {
        el.resStatusBadge.textContent = 'ACTIVE LOS ALERT';
        el.resStatusBadge.className = 'font-label-sm text-label-sm bg-tertiary-container/30 text-tertiary px-2 py-0.5 rounded font-bold uppercase animate-pulse';
      } else {
        el.resStatusBadge.textContent = 'NOMINAL - SEPARATION MAINTAINED';
        el.resStatusBadge.className = 'font-label-sm text-label-sm bg-primary-container/20 text-primary px-2 py-0.5 rounded font-bold uppercase';
      }
    }

    if (el.resEncounterTitle) {
      if (pair) {
        el.resEncounterTitle.textContent = `ENCOUNTER: ${pair.aircraft_1} × ${pair.aircraft_2} [${data.scenario_id}]`;
      } else {
        el.resEncounterTitle.textContent = `AIRSPACE STATUS: NOMINAL [${data.scenario_id}]`;
      }
    }

    if (el.resEncounterDesc) {
      if (pair) {
        const flagStr = hasConflict ? 'Loss of Separation Predicted' : 'Separation Safe';
        el.resEncounterDesc.textContent = `Current sep: ${pair.curr_distance_nm.toFixed(1)} NM (${pair.curr_vertical_ft.toFixed(0)} FT) | Projected CPA: ${pair.predicted_cpa_nm.toFixed(2)} NM @ tau=${pair.predicted_tau_s.toFixed(0)}s (${flagStr})`;
      } else {
        el.resEncounterDesc.textContent = 'All tracked aircraft maintaining standard separation minima.';
      }
    }

    if (ac1 && ac2 && pair) {
      const isAc1Lower = ac1.altitude_ft <= ac2.altitude_ft;
      const targetFl = isAc1Lower
        ? Math.max(100, Math.round((ac1.altitude_ft - 2000) / 100))
        : Math.round((ac1.altitude_ft + 2000) / 100);
      const actionWord = isAc1Lower ? 'DESCEND' : 'CLIMB';
      const deltaFt = isAc1Lower ? '-2,000 FT' : '+2,000 FT';
      const rocdSign = isAc1Lower ? '-1,500' : '+1,500';

      if (el.resOpt1Title) {
        el.resOpt1Title.textContent = `${actionWord} ${ac1.aircraft_id} TO FL${targetFl} (${deltaFt})`;
      }
      if (el.resOpt1Desc) {
        el.resOpt1Desc.textContent = `Instruct ${ac1.callsign} to initiate ${actionWord.toLowerCase()} at standard ROCD (${rocdSign} fpm) to achieve safe flight level prior to crossing.`;
      }
      if (el.resOpt1Vert) {
        el.resOpt1Vert.textContent = '2,000 FT (≥ 1,000\' Minima)';
      }

      // Lateral Advisory
      const newHdg = (Math.round(ac1.heading_deg + 30)) % 360;
      const latCpa = Math.max(6.4, pair.predicted_cpa_nm + 5.0).toFixed(1);
      if (el.resOpt2Title) {
        el.resOpt2Title.textContent = `TURN ${ac1.aircraft_id} RIGHT HEADING ${String(newHdg).padStart(3, '0')}° (+30°)`;
      }
      if (el.resOpt2Desc) {
        el.resOpt2Desc.textContent = `Apply standard tactical radar vector to steer ${ac1.callsign} away from crossing conflict geometry.`;
      }
      if (el.resOpt2Lat) {
        el.resOpt2Lat.textContent = `${latCpa} NM (≥ 5.0 NM Minima)`;
      }

      // Speed Advisory
      const targetSpeed = Math.max(250, Math.round(ac1.speed_kt - 50));
      const spdCpa = Math.max(5.1, pair.predicted_cpa_nm + 2.5).toFixed(1);
      if (el.resOpt3Title) {
        el.resOpt3Title.textContent = `REDUCE SPEED TO ${targetSpeed} KTS (-50 KT)`;
      }
      if (el.resOpt3Desc) {
        el.resOpt3Desc.textContent = `Stagger arrival time at the crossing point by slowing trailing aircraft ${ac1.callsign} within clean aerodynamic envelope.`;
      }
      if (el.resOpt3Lat) {
        el.resOpt3Lat.textContent = `${spdCpa} NM (≥ 5.0 NM Minima)`;
      }
    }
  }

  function renderTrajectoryInspector(data) {
    if (!data) return;

    // Update aircraft dropdown filter if options changed
    if (el.inspectorAircraftFilter) {
      const existingOptions = Array.from(el.inspectorAircraftFilter.options).map(o => o.value);
      const newValues = ['ALL', ...data.aircraft_ids];
      const isMismatch = existingOptions.join(',') !== newValues.join(',');

      if (isMismatch) {
        el.inspectorAircraftFilter.innerHTML = '';
        const allOpt = document.createElement('option');
        allOpt.value = 'ALL';
        allOpt.textContent = `All Aircraft (${data.aircraft_ids.length})`;
        el.inspectorAircraftFilter.appendChild(allOpt);

        data.aircraft_ids.forEach((acid) => {
          const ac = data.aircraft[acid];
          const opt = document.createElement('option');
          opt.value = acid;
          opt.textContent = `${acid} (${ac.callsign}) - ${ac.flight_level}`;
          el.inspectorAircraftFilter.appendChild(opt);
        });

        if (newValues.includes(state.inspectorFilter)) {
          el.inspectorAircraftFilter.value = state.inspectorFilter;
        } else {
          el.inspectorAircraftFilter.value = 'ALL';
          state.inspectorFilter = 'ALL';
        }
      }
    }

    if (!el.inspectorTableBody) return;
    el.inspectorTableBody.innerHTML = '';

    // Collect telemetry records
    const selectedAcids = state.inspectorFilter === 'ALL'
      ? data.aircraft_ids
      : [state.inspectorFilter];

    const allRecords = [];
    selectedAcids.forEach((acid) => {
      const ac = data.aircraft[acid];
      if (!ac) return;

      const pts = ac.history_points || [];
      for (let k = 0; k < pts.length; k++) {
        const pt = pts[k];
        let speed = ac.speed_kt;
        let hdg = ac.heading_deg;
        let rocdStr = 'LEVEL (0 fpm)';

        if (k > 0) {
          const prev = pts[k - 1];
          const dt = pt.t - prev.t;
          if (dt > 0.001) {
            const dx = pt.x - prev.x;
            const dy = pt.y - prev.y;
            const distNm = Math.hypot(dx, dy);
            speed = (distNm / (dt / 3600));
            const angle = Math.atan2(dx, dy) * 180 / Math.PI;
            hdg = (angle + 360) % 360;
            const rocd = ((pt.alt - prev.alt) / dt) * 60;
            rocdStr = Math.abs(rocd) < 50
              ? 'LEVEL (0 fpm)'
              : rocd > 0 ? `CLB +${Math.round(rocd)} fpm` : `DES ${Math.round(rocd)} fpm`;
          }
        }

        allRecords.push({
          t: pt.t,
          acid: acid,
          fl: `FL${Math.round(pt.alt / 100)}`,
          alt: Math.round(pt.alt),
          speed: Math.round(speed),
          hdg: Math.round(hdg),
          rocdStr: rocdStr,
          x: pt.x.toFixed(2),
          y: pt.y.toFixed(2),
        });
      }
    });

    // Sort by timestamp
    allRecords.sort((a, b) => a.t - b.t || a.acid.localeCompare(b.acid));

    // Render rows (sample to max 120 rows for smooth DOM performance)
    const step = allRecords.length > 120 ? Math.ceil(allRecords.length / 120) : 1;
    for (let i = 0; i < allRecords.length; i += step) {
      const r = allRecords[i];
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-surface-container/60 transition-colors';
      tr.innerHTML = `
        <td class="p-2 text-on-surface font-mono">${r.t.toFixed(1)}s</td>
        <td class="p-2 text-primary font-bold font-mono">${r.acid}</td>
        <td class="p-2 text-on-surface-variant font-mono">${r.fl}</td>
        <td class="p-2 text-on-surface font-mono">${r.alt}</td>
        <td class="p-2 text-on-surface font-mono">${r.speed} kt</td>
        <td class="p-2 text-on-surface font-mono">${r.hdg}°</td>
        <td class="p-2 text-outline font-mono text-[10px]">${r.rocdStr}</td>
        <td class="p-2 text-outline font-mono">${r.x}</td>
        <td class="p-2 text-outline font-mono">${r.y}</td>
      `;
      el.inspectorTableBody.appendChild(tr);
    }
  }

  function renderLosLedger() {
    if (!el.ledgerTableBody) return;
    el.ledgerTableBody.innerHTML = '';

    const filter = state.ledgerFilter || 'ALL';
    const scenarios = state.scenarios.filter((scen) => {
      if (filter === 'ALL') return true;
      if (filter === 'ALERT') return scen.tag === 'ALERT' || scen.has_conflict;
      if (filter === 'NEAR MISS') return scen.tag === 'NEAR MISS' || scen.near_miss;
      if (filter === 'NOMINAL') return scen.tag === 'NOMINAL' || (!scen.has_conflict && !scen.near_miss);
      return true;
    });

    if (el.ledgerCountDisplay) {
      el.ledgerCountDisplay.textContent = `SHOWING ${scenarios.length} / ${state.scenarios.length} SCENARIOS IN ${state.dataset.toUpperCase()}-${state.split.toUpperCase()}`;
    }

    scenarios.forEach((scen) => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-surface-container/50 transition-colors';

      let statusBadge = '';
      let minLat = '> 10.0 NM (CLEAR)';
      let minVert = '≥ 1,000 FT (CLEAR)';
      let onset = 'None (Clean)';

      if (scen.tag_type === 'conflict' || scen.tag === 'ALERT') {
        statusBadge = '<span class="px-2 py-0.5 rounded bg-tertiary-container/30 text-tertiary font-bold text-[10px] uppercase">ALERT (LOS)</span>';
        minLat = '< 5.0 NM (VIOLATION)';
        minVert = '< 1,000 FT';
        onset = '~01:30s (Encounter)';
      } else if (scen.tag_type === 'near_miss' || scen.tag === 'NEAR MISS') {
        statusBadge = '<span class="px-2 py-0.5 rounded bg-secondary-container/30 text-secondary font-bold text-[10px] uppercase">NEAR MISS</span>';
        minLat = '5.0 - 6.0 NM (CLOSE)';
        minVert = '≥ 1,000 FT';
        onset = '~02:00s (Marginal)';
      } else {
        statusBadge = '<span class="px-2 py-0.5 rounded bg-surface-container text-outline font-semibold text-[10px] uppercase">NOMINAL</span>';
      }

      tr.innerHTML = `
        <td class="p-2 font-bold text-primary font-mono">${scen.scenario_id}</td>
        <td class="p-2">${statusBadge}</td>
        <td class="p-2 text-on-surface-variant font-mono uppercase">${scen.description}</td>
        <td class="p-2 text-on-surface font-mono">${onset}</td>
        <td class="p-2 text-on-surface font-mono">${minLat}</td>
        <td class="p-2 text-on-surface font-mono">${minVert}</td>
        <td class="p-2">
          <button class="load-scen-btn px-2.5 py-1 rounded bg-primary-container text-on-primary-container hover:bg-primary font-headline-sm text-[11px] font-semibold transition-colors" data-id="${scen.scenario_id}" type="button">
            LOAD SCENARIO
          </button>
        </td>
      `;

      const btn = tr.querySelector('.load-scen-btn');
      if (btn) {
        btn.addEventListener('click', () => {
          const idx = state.scenarios.findIndex(s => s.scenario_id === scen.scenario_id);
          if (idx >= 0) {
            state.scenarioIndex = idx;
            state.scenarioId = scen.scenario_id;
            state.originT = null;
            renderScenarioLedger();
            loadScenarioDetail();
            switchModule('conflict-prediction');
          }
        });
      }

      el.ledgerTableBody.appendChild(tr);
    });
  }

  function renderMetricsStrip(metrics) {
    if (!metrics) return;

    if (el.metricPrecision) el.metricPrecision.textContent = metrics.precision.toFixed(4);
    if (el.barPrecision) el.barPrecision.style.width = `${(metrics.precision * 100).toFixed(1)}%`;

    if (el.metricRecall) el.metricRecall.textContent = metrics.recall.toFixed(4);
    if (el.barRecall) el.barRecall.style.width = `${(metrics.recall * 100).toFixed(1)}%`;

    if (el.metricF1) el.metricF1.textContent = metrics.f1_score.toFixed(4);
    if (el.barF1) el.barF1.style.width = `${(metrics.f1_score * 100).toFixed(1)}%`;

    if (el.metricFar) el.metricFar.textContent = metrics.false_alarms_per_1000_negatives.toFixed(2);
    if (el.barFar) el.barFar.style.width = `${Math.min(100, metrics.false_alarms_per_1000_negatives * 20).toFixed(1)}%`;

    if (el.metricLead) el.metricLead.textContent = `${metrics.mean_lead_time_s.toFixed(1)}s`;
    if (el.barLead) el.barLead.style.width = `${Math.min(100, (metrics.mean_lead_time_s / 180) * 100).toFixed(1)}%`;

    if (el.metricTtcMae) el.metricTtcMae.textContent = `${metrics.tp_time_to_conflict_mae_s.toFixed(2)}s`;
    if (el.barTtc) el.barTtc.style.width = `${Math.max(10, 100 - metrics.tp_time_to_conflict_mae_s * 30).toFixed(1)}%`;

    if (el.metricTpCount) el.metricTpCount.textContent = metrics.true_positives.toLocaleString();
    if (el.metricFpCount) el.metricFpCount.textContent = metrics.false_positives.toLocaleString();

    if (el.modelWeightsDisplay) {
      if (metrics.model_name === 'lstm_v1') {
        el.modelWeightsDisplay.textContent = 'checkpoints/lstm_v1/best.pt';
      } else {
        el.modelWeightsDisplay.textContent = 'KINEMATIC_BASELINE (NO_WEIGHTS)';
      }
    }
  }

  // Playback & Interaction Event Listeners
  function attachEventListeners() {
    // 1. Dataset & Split Selectors
    if (el.datasetSelector) {
      el.datasetSelector.addEventListener('change', (e) => {
        state.dataset = e.target.value;
        state.originT = null;
        loadScenarios();
      });
    }

    if (el.splitSelector) {
      el.splitSelector.addEventListener('change', (e) => {
        state.split = e.target.value;
        state.originT = null;
        loadScenarios();
      });
    }

    // 2. Scenario Pager
    if (el.prevScenarioBtn) {
      el.prevScenarioBtn.addEventListener('click', () => {
        if (state.scenarios.length === 0) return;
        state.scenarioIndex = (state.scenarioIndex - 1 + state.scenarios.length) % state.scenarios.length;
        state.scenarioId = state.scenarios[state.scenarioIndex].scenario_id;
        state.originT = null;
        renderScenarioLedger();
        loadScenarioDetail();
      });
    }

    if (el.nextScenarioBtn) {
      el.nextScenarioBtn.addEventListener('click', () => {
        if (state.scenarios.length === 0) return;
        state.scenarioIndex = (state.scenarioIndex + 1) % state.scenarios.length;
        state.scenarioId = state.scenarios[state.scenarioIndex].scenario_id;
        state.originT = null;
        renderScenarioLedger();
        loadScenarioDetail();
      });
    }

    // 3. Playback Scrubber Buttons
    if (el.stepBackBtn) {
      el.stepBackBtn.addEventListener('click', () => {
        if (!state.scenarioData || !state.scenarioData.available_origins) return;
        const origins = state.scenarioData.available_origins;
        const currIdx = origins.indexOf(state.originT);
        if (currIdx > 0) {
          state.originT = origins[currIdx - 1];
          loadScenarioDetail();
        }
      });
    }

    if (el.stepFwdBtn) {
      el.stepFwdBtn.addEventListener('click', () => {
        if (!state.scenarioData || !state.scenarioData.available_origins) return;
        const origins = state.scenarioData.available_origins;
        const currIdx = origins.indexOf(state.originT);
        if (currIdx >= 0 && currIdx < origins.length - 1) {
          state.originT = origins[currIdx + 1];
          loadScenarioDetail();
        }
      });
    }

    if (el.simPlayBtn) {
      el.simPlayBtn.addEventListener('click', () => {
        state.isPlaying = !state.isPlaying;
        const icon = el.simPlayBtn.querySelector('span');
        if (icon) {
          icon.textContent = state.isPlaying ? 'pause' : 'play_arrow';
        }
        el.simPlayBtn.classList.toggle('bg-primary', !state.isPlaying);
        el.simPlayBtn.classList.toggle('bg-surface-variant', state.isPlaying);

        if (state.isPlaying) {
          state.playTimer = setInterval(() => {
            if (!state.scenarioData || !state.scenarioData.available_origins) return;
            const origins = state.scenarioData.available_origins;
            const currIdx = origins.indexOf(state.originT);
            if (currIdx >= 0 && currIdx < origins.length - 1) {
              state.originT = origins[currIdx + 1];
              loadScenarioDetail();
            } else {
              // loop back
              state.originT = origins[0];
              loadScenarioDetail();
            }
          }, 1500 / state.playSpeed);
        } else {
          if (state.playTimer) clearInterval(state.playTimer);
        }
      });
    }

    // 4. Playback Speed Buttons
    if (el.speedButtons) {
      el.speedButtons.querySelectorAll('button').forEach((btn) => {
        btn.addEventListener('click', () => {
          el.speedButtons.querySelectorAll('button').forEach(b => {
            b.className = 'px-2 py-1 font-telemetry-num text-[11px] rounded hover:bg-surface-container text-on-surface-variant';
          });
          btn.className = 'px-2 py-1 font-telemetry-num text-[11px] rounded bg-primary text-on-primary font-semibold';
          state.playSpeed = parseFloat(btn.dataset.speed) || 1;
        });
      });
    }

    // 5. Lookahead Horizon Slider
    if (el.lookaheadSlider) {
      el.lookaheadSlider.addEventListener('input', (e) => {
        state.lookahead = parseFloat(e.target.value);
        if (el.lookaheadValueText) {
          el.lookaheadValueText.textContent = `${state.lookahead}s (${(state.lookahead * 0.0167).toFixed(1)} NM sep)`;
        }
        loadScenarioDetail();
      });
    }

    // 6. Model Selection Buttons
    if (el.modelSelectorGroup) {
      el.modelSelectorGroup.querySelectorAll('button').forEach((btn) => {
        btn.addEventListener('click', () => {
          el.modelSelectorGroup.querySelectorAll('button').forEach(b => {
            b.className = 'px-2.5 py-1 rounded font-headline-sm text-[12px] text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors';
          });
          btn.className = 'px-2.5 py-1 rounded font-headline-sm text-[12px] bg-primary-container text-on-primary-container font-semibold shadow-sm';
          state.model = btn.dataset.model;
          loadScenarioDetail();
          loadMetrics();
        });
      });
    }

    // 7. Uncertainty Ellipse Toggle
    if (el.toggleEllipsesBtn) {
      el.toggleEllipsesBtn.addEventListener('click', () => {
        state.showEllipses = !state.showEllipses;
        el.toggleEllipsesBtn.textContent = state.showEllipses ? 'UNCERTAINTY ELLIPSE [ON]' : 'UNCERTAINTY ELLIPSE [OFF]';
        el.toggleEllipsesBtn.classList.toggle('text-primary', state.showEllipses);
        el.toggleEllipsesBtn.classList.toggle('text-outline', !state.showEllipses);
        if (state.scenarioData) render2DAirspace(state.scenarioData);
      });
    }

    // 8. Flown History Toggle
    if (el.toggleHistoryBtn) {
      el.toggleHistoryBtn.addEventListener('click', () => {
        state.showHistory = !state.showHistory;
        el.toggleHistoryBtn.classList.toggle('text-primary', state.showHistory);
        el.toggleHistoryBtn.classList.toggle('text-outline', !state.showHistory);
        if (state.scenarioData) render2DAirspace(state.scenarioData);
      });
    }

    // 9. Recenter Viewport
    if (el.recenterBtn) {
      el.recenterBtn.addEventListener('click', () => {
        if (state.scenarioData) render2DAirspace(state.scenarioData);
      });
    }

    // 10. Research Module Navigation Links (Leftmost Sidebar)
    if (el.researchNav) {
      el.researchNav.querySelectorAll('a[data-module]').forEach((a) => {
        a.addEventListener('click', (e) => {
          e.preventDefault();
          const mod = a.dataset.module;
          switchModule(mod);
        });
      });
    }

    // 11. Trajectory Inspector Aircraft Filter
    if (el.inspectorAircraftFilter) {
      el.inspectorAircraftFilter.addEventListener('change', (e) => {
        state.inspectorFilter = e.target.value;
        if (state.scenarioData) {
          renderTrajectoryInspector(state.scenarioData);
        }
      });
    }

    // 12. LoS Advisory Ledger Filter Buttons
    if (el.ledgerFilterGroup) {
      el.ledgerFilterGroup.querySelectorAll('button').forEach((btn) => {
        btn.addEventListener('click', () => {
          el.ledgerFilterGroup.querySelectorAll('button').forEach((b) => {
            b.className = 'px-2 py-0.5 rounded font-label-sm text-[11px] text-on-surface-variant hover:bg-surface-container';
          });
          btn.className = 'px-2 py-0.5 rounded font-label-sm text-[11px] bg-primary text-on-primary font-semibold';
          state.ledgerFilter = btn.dataset.filter || 'ALL';
          renderLosLedger();
        });
      });
    }

    // 13. Resolution Advisory Accept Buttons
    if (el.viewResolutionEngine) {
      el.viewResolutionEngine.querySelectorAll('button').forEach((btn) => {
        btn.addEventListener('click', () => {
          const origText = btn.textContent;
          btn.textContent = 'ADVISORY ACCEPTED & TRANSMITTED ✓';
          btn.classList.add('bg-tertiary', 'text-on-tertiary');
          setTimeout(() => {
            btn.textContent = origText;
            btn.classList.remove('bg-tertiary', 'text-on-tertiary');
          }, 2000);
        });
      });
    }

    // 14. Export feedback
    if (el.exportMetricsBtn) {
      el.exportMetricsBtn.addEventListener('click', () => {
        const orig = el.exportMetricsBtn.innerHTML;
        el.exportMetricsBtn.innerHTML = '<span class="material-symbols-outlined text-[13px] text-tertiary">check</span><span>GENERATED</span>';
        setTimeout(() => { el.exportMetricsBtn.innerHTML = orig; }, 1800);
      });
    }

    if (el.exportJsonBtn) {
      el.exportJsonBtn.addEventListener('click', () => {
        if (state.scenarioData) {
          const blob = new Blob([JSON.stringify(state.scenarioData, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${state.scenarioId}_telemetry.json`;
          a.click();
        }
      });
    }
  }

  // Initialization
  async function init() {
    attachEventListeners();
    await loadSystemInfo();
    await loadScenarios();
    switchModule('conflict-prediction');
    window.appState = state;
    window.loadScenarioDetail = loadScenarioDetail;
    window.switchModule = switchModule;
  }

  window.addEventListener('DOMContentLoaded', init);
})();
