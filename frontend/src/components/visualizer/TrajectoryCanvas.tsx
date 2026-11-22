import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ShieldAlert, Radio, Sparkles } from 'lucide-react';
import { RunTrajectoryData } from '../../types';
import { fetchRunTrajectory } from '../../services/api';

interface AgentState {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
  token: number;
  targetLm: number;
}

interface LandmarkState {
  id: number;
  x: number;
  y: number;
}

interface TrajectoryCanvasProps {
  numAgents?: number;
  numLandmarks?: number;
  isPerturbed?: boolean;
  perturbationScope?: 'partner_full' | 'partner' | 'all';
  runId?: string;
  isRepairedView?: boolean;
}

export const TrajectoryCanvas: React.FC<TrajectoryCanvasProps> = ({
  numAgents = 2,
  numLandmarks = 3,
  isPerturbed = false,
  perturbationScope = 'partner_full',
  runId,
  isRepairedView = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [step, setStep] = useState<number>(0);
  const [showGhostPerception, setShowGhostPerception] = useState<boolean>(true);
  const [showCommChannel, setShowCommChannel] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<'perturbed' | 'repaired'>(
    isRepairedView ? 'repaired' : 'perturbed'
  );
  const [trajData, setTrajData] = useState<RunTrajectoryData | null>(null);

  useEffect(() => {
    setViewMode(isRepairedView ? 'repaired' : 'perturbed');
  }, [isRepairedView]);

  useEffect(() => {
    let active = true;
    if (!runId) {
      setTrajData(null);
      return;
    }
    fetchRunTrajectory(runId)
      .then((data) => {
        if (active) {
          setTrajData(data);
        }
      })
      .catch(() => {
        if (active) setTrajData(null);
      });
    return () => {
      active = false;
    };
  }, [runId]);

  const effectiveAgents = trajData?.num_agents || numAgents;
  const effectiveLandmarks = trajData?.num_landmarks || numLandmarks;
  const maxSteps = trajData?.steps?.length || 50;

  const AGENT_COLORS = ['#38bdf8', '#f43f5e', '#a855f7', '#fb923c', '#10b981', '#eab308'];

  // Dynamically build balanced landmark ring for any count 2..8
  const buildFallbackLandmarks = (count: number): LandmarkState[] => {
    const res: LandmarkState[] = [];
    for (let i = 0; i < count; i++) {
      const ang = (2 * Math.PI * i) / Math.max(count, 1) + 0.35;
      const rad = i % 2 === 0 ? 0.62 : 0.48;
      res.push({
        id: i,
        x: Math.cos(ang) * rad,
        y: Math.sin(ang) * rad,
      });
    }
    return res;
  };

  const landmarks: LandmarkState[] = trajData?.landmarks?.length
    ? trajData.landmarks
    : buildFallbackLandmarks(effectiveLandmarks);

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setStep((prev) => (prev + 1) % maxSteps);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying, maxSteps]);

  const getAgentsAtStep = (stepIdx: number): AgentState[] => {
    const safeIdx = Math.max(0, Math.min(stepIdx, maxSteps - 1));
    const progress = safeIdx / Math.max(maxSteps - 1, 1);
    const brokenMode = isPerturbed && viewMode === 'perturbed';

    if (trajData?.steps && trajData.steps[safeIdx]) {
      return trajData.steps[safeIdx].agents.map((ag) => {
        let ax = ag.x;
        let ay = ag.y;
        let token = ag.emitted_token;

        // Apply visible behavioral drift when showing Perturbed (unrepaired) state
        if (brokenMode) {
          const driftMag = Math.sin(progress * Math.PI) * 0.42;
          if (perturbationScope === 'partner_full') {
            ax = ag.x + (ag.id % 2 === 0 ? -driftMag : driftMag);
            ay = ag.y + (ag.id % 2 === 0 ? driftMag * 0.7 : -driftMag * 0.7);
            token = (ag.emitted_token + 3) % 5; // scrambled coordination signal
          } else if (perturbationScope === 'partner') {
            ax = ag.x + (ag.id % 2 === 0 ? -driftMag : driftMag);
            token = (ag.emitted_token + 2) % 5;
          } else {
            ax = -ag.x * 0.55;
            ay = -ag.y * 0.55;
            token = 4;
          }
        }

        return {
          id: ag.id,
          x: ax,
          y: ay,
          vx: ag.vx,
          vy: ag.vy,
          color: AGENT_COLORS[ag.id % AGENT_COLORS.length],
          token,
          targetLm: ag.target_lm,
        };
      });
    }

    // Fallback geometric rollout for any agent/landmark count
    const agents: AgentState[] = [];
    for (let i = 0; i < effectiveAgents; i++) {
      const angle = (i * 2 * Math.PI) / Math.max(effectiveAgents, 1) + 0.6 * Math.PI;
      const targetLm = landmarks[i % landmarks.length];
      const startX = Math.cos(angle) * 0.78;
      const startY = Math.sin(angle) * 0.78;
      const ease = 1 - Math.pow(1 - Math.min(progress * 1.3, 1.0), 2);
      let curX = startX + (targetLm.x - startX) * ease;
      let curY = startY + (targetLm.y - startY) * ease;

      if (brokenMode) {
        const drift = Math.sin(progress * Math.PI) * 0.4;
        if (perturbationScope === 'all') {
          curX = -curX * 0.5;
          curY = -curY * 0.5;
        } else {
          curX += i % 2 === 0 ? -drift : drift;
          if (perturbationScope === 'partner_full') {
            curY += i % 2 === 0 ? drift * 0.6 : -drift * 0.6;
          }
        }
      }

      agents.push({
        id: i,
        x: curX,
        y: curY,
        vx: (targetLm.x - curX) * 0.1,
        vy: (targetLm.y - curY) * 0.1,
        color: AGENT_COLORS[i % AGENT_COLORS.length],
        token: brokenMode ? (i * 3 + safeIdx) % 5 : (i + Math.floor(safeIdx / 10)) % 5,
        targetLm: i % landmarks.length,
      });
    }
    return agents;
  };

  // Render on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const scale = width / 2.4;
    const cx = width / 2;
    const cy = height / 2;

    const toCanvasX = (x: number) => cx + x * scale;
    const toCanvasY = (y: number) => cy - y * scale;

    ctx.fillStyle = '#0b0f19';
    ctx.fillRect(0, 0, width, height);

    // Subtle grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let x = -1; x <= 1; x += 0.5) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(x), 0);
      ctx.lineTo(toCanvasX(x), height);
      ctx.stroke();
    }
    for (let y = -1; y <= 1; y += 0.5) {
      ctx.beginPath();
      ctx.moveTo(0, toCanvasY(y));
      ctx.lineTo(width, toCanvasY(y));
      ctx.stroke();
    }

    const agents = getAgentsAtStep(step);
    const brokenMode = isPerturbed && viewMode === 'perturbed';

    // Draw inverted Ghost Landmarks when perturbationScope === 'all'
    if (brokenMode && showGhostPerception && perturbationScope === 'all') {
      landmarks.forEach((lm) => {
        const glx = toCanvasX(-lm.x);
        const gly = toCanvasY(-lm.y);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(glx, gly, 13, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`L${lm.id + 1}*`, glx, gly - 16);
      });
    }

    // Draw Landmarks with live occupancy highlighting
    landmarks.forEach((lm) => {
      const lx = toCanvasX(lm.x);
      const ly = toCanvasY(lm.y);
      const isCovered = agents.some(
        (ag) => Math.hypot(ag.x - lm.x, ag.y - lm.y) < 0.18
      );

      ctx.strokeStyle = isCovered ? '#10b981' : '#475569';
      ctx.lineWidth = isCovered ? 2.5 : 1.5;
      ctx.setLineDash(isCovered ? [] : [4, 4]);
      ctx.beginPath();
      ctx.arc(lx, ly, 24, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = isCovered ? 'rgba(16, 185, 129, 0.25)' : '#334155';
      ctx.beginPath();
      ctx.arc(lx, ly, 12, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = isCovered ? '#34d399' : '#94a3b8';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`L${lm.id + 1}`, lx, ly);
    });

    // Draw Agent Trajectory Tails (last 8 steps)
    const tailLen = 8;
    for (let t = Math.max(0, step - tailLen); t < step; t++) {
      const prevAgents = getAgentsAtStep(t);
      const nextAgents = getAgentsAtStep(t + 1);
      const alpha = (t - Math.max(0, step - tailLen) + 1) / (tailLen + 1);
      prevAgents.forEach((pa, idx) => {
        const na = nextAgents[idx];
        if (!na) return;
        ctx.strokeStyle = pa.color;
        ctx.globalAlpha = alpha * 0.45;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(pa.x), toCanvasY(pa.y));
        ctx.lineTo(toCanvasX(na.x), toCanvasY(na.y));
        ctx.stroke();
      });
    }
    ctx.globalAlpha = 1.0;

    // Draw Communication Channel Beams
    if (showCommChannel && agents.length > 1) {
      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const a1 = agents[i];
          const a2 = agents[j];
          ctx.strokeStyle = brokenMode
            ? 'rgba(244, 63, 94, 0.55)'
            : isPerturbed && viewMode === 'repaired'
            ? 'rgba(16, 185, 129, 0.55)'
            : 'rgba(56, 189, 248, 0.35)';
          ctx.lineWidth = brokenMode ? 2 : 1.8;
          ctx.setLineDash(brokenMode ? [3, 4] : [6, 5]);
          ctx.beginPath();
          ctx.moveTo(toCanvasX(a1.x), toCanvasY(a1.y));
          ctx.lineTo(toCanvasX(a2.x), toCanvasY(a2.y));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // Draw Ghost Perception Illusion (when perturbed)
    if (brokenMode && showGhostPerception && agents.length >= 2) {
      const a1 = agents[0];
      const a2 = agents[1];

      const relX = a2.x - a1.x;
      const relY = a2.y - a1.y;
      const ghostX = a1.x - relX;
      const ghostY = perturbationScope === 'partner' ? a2.y : a1.y - relY;

      const gx = toCanvasX(ghostX);
      const gy = toCanvasY(ghostY);

      ctx.fillStyle = 'rgba(244, 63, 94, 0.22)';
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(gx, gy, 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#fb7185';
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        perturbationScope === 'partner' ? 'Ghost A2 (X-Mirror)' : 'Ghost A2 (180° Illusion)',
        gx,
        gy - 21
      );

      ctx.strokeStyle = 'rgba(244, 63, 94, 0.65)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(toCanvasX(a1.x), toCanvasY(a1.y));
      ctx.lineTo(gx, gy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw real agents & emitted communication tokens
    agents.forEach((ag) => {
      const ax = toCanvasX(ag.x);
      const ay = toCanvasY(ag.y);

      ctx.fillStyle = ag.color;
      ctx.shadowColor = ag.color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(ax, ay, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = '#0f172a';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`A${ag.id + 1}`, ax, ay);

      const bubbleX = ax;
      const bubbleY = ay - 25;
      ctx.fillStyle = brokenMode ? '#3b0a18' : '#1e293b';
      ctx.strokeStyle = brokenMode ? '#f43f5e' : '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(bubbleX - 24, bubbleY - 9, 48, 18, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = brokenMode ? '#fda4af' : '#38bdf8';
      ctx.font = 'bold 9px monospace';
      ctx.fillText(brokenMode ? `err_${ag.token}` : `tok_${ag.token}`, bubbleX, bubbleY);
    });
  }, [step, isPerturbed, viewMode, perturbationScope, showGhostPerception, showCommChannel, effectiveAgents, effectiveLandmarks, maxSteps, trajData]);

  return (
    <div className="trajectory-canvas-wrap">
      <div className="canvas-header">
        <div className="canvas-title">
          <Radio size={16} className="text-emerald" />
          <span>
            2D Trajectory Simulation ({effectiveAgents} Agents / {effectiveLandmarks} Landmarks)
          </span>
          {trajData && (
            <span className="live-config-pill">
              <Sparkles size={11} /> LIVE RUN CONFIG &amp; METRICS
            </span>
          )}
        </div>

        <div className="canvas-options">
          {trajData && (trajData.latest_comm_effect !== null || trajData.latest_kl !== null) && (
            <span className="live-metrics-chip">
              &Delta;R<sub>comm</sub>:{' '}
              <strong>
                {trajData.latest_comm_effect !== null
                  ? `${trajData.latest_comm_effect >= 0 ? '+' : ''}${trajData.latest_comm_effect.toFixed(1)}`
                  : 'N/A'}
              </strong>{' '}
              | KL:{' '}
              <strong>
                {trajData.latest_kl !== null ? trajData.latest_kl.toFixed(3) : 'N/A'}
              </strong>
            </span>
          )}

          <label className="toggle-badge">
            <input
              type="checkbox"
              checked={showCommChannel}
              onChange={(e) => setShowCommChannel(e.target.checked)}
            />
            <span>Show Comm Channel</span>
          </label>

          {isPerturbed && (
            <>
              <label className="toggle-badge warning">
                <input
                  type="checkbox"
                  checked={showGhostPerception}
                  onChange={(e) => setShowGhostPerception(e.target.checked)}
                />
                <ShieldAlert size={13} />
                <span>Ghost Illusion ({perturbationScope})</span>
              </label>

              <div className="view-mode-toggle">
                <button
                  type="button"
                  className={`mode-btn ${viewMode === 'perturbed' ? 'active-danger' : ''}`}
                  onClick={() => setViewMode('perturbed')}
                >
                  Perturbed (Broken Comm)
                </button>
                <button
                  type="button"
                  className={`mode-btn ${viewMode === 'repaired' ? 'active-success' : ''}`}
                  onClick={() => setViewMode('repaired')}
                >
                  Post-LoRA Repair
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="canvas-viewport">
        <canvas ref={canvasRef} width={500} height={410} className="sim-canvas" />
      </div>

      <div className="canvas-controls">
        <button className="icon-btn-sm" onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button
          className="icon-btn-sm"
          onClick={() => {
            setIsPlaying(false);
            setStep(0);
          }}
        >
          <RotateCcw size={14} />
        </button>

        <input
          type="range"
          min={0}
          max={maxSteps - 1}
          value={step}
          onChange={(e) => {
            setIsPlaying(false);
            setStep(parseInt(e.target.value));
          }}
          className="step-slider"
        />
        <span className="step-counter">
          Step {step + 1}/{maxSteps}
        </span>
      </div>

      {/* Legend Bar */}
      <div className="chart-legend-row" style={{ marginTop: '4px' }}>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#38bdf8', borderRadius: '50%' }} />
          <span>Real Agent</span>
        </div>
        {isPerturbed && (
          <div className="legend-item">
            <span className="legend-dot" style={{ border: '1.5px dashed #f43f5e', background: 'rgba(244,63,94,0.25)', borderRadius: '50%' }} />
            <span>Ghost Illusion</span>
          </div>
        )}
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#10b981', borderRadius: '50%' }} />
          <span>Covered Landmark</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: isPerturbed && viewMode === 'perturbed' ? '#f43f5e' : '#38bdf8' }} />
          <span>{isPerturbed && viewMode === 'perturbed' ? 'Jammed / Misaligned Comm' : 'Active Comm Beam'}</span>
        </div>
      </div>
    </div>
  );
};
