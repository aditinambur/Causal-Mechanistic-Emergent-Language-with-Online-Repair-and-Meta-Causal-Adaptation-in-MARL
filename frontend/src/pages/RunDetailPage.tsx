import React, { useState } from 'react';
import {
  BarChart3,
  Terminal,
  Image,
  Layers,
  Wrench,
  ArrowLeft,
  Radio,
  Award,
  TrendingUp,
  Play
} from 'lucide-react';
import { Run, MetricRow, CheckpointInfo } from '../types';
import { MetricsChart } from '../components/metrics/MetricsChart';
import { TrajectoryCanvas } from '../components/visualizer/TrajectoryCanvas';
import { GifViewer } from '../components/visualizer/GifViewer';
import { TerminalLogViewer } from '../components/visualizer/TerminalLogViewer';
import { Badge } from '../components/common/Badge';

interface RunDetailPageProps {
  run: Run;
  runs: Run[];
  metrics: MetricRow[];
  checkpoints: CheckpointInfo[];
  logs: string;
  onSelectRun: (runId: string) => void;
  onBack: () => void;
  onOpenRepair: (runId: string) => void;
  onRefreshRunData: () => void;
}

export const RunDetailPage: React.FC<RunDetailPageProps> = ({
  run,
  runs,
  metrics,
  checkpoints,
  logs,
  onSelectRun,
  onBack,
  onOpenRepair,
  onRefreshRunData,
}) => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'trajectory' | 'gif' | 'logs' | 'checkpoints'>(
    run.status === 'running' && metrics.length === 0 ? 'logs' : 'metrics'
  );

  const numAgents = run.config?.num_agents || 2;
  const numLandmarks = run.config?.num_landmarks || 3;

  const bestCheckpoint =
    checkpoints.find((c) => c.is_best) ||
    checkpoints[0] ||
    null;

  const evalRewards = metrics
    .map((m) => m.eval_reward ?? m.reward)
    .filter((v): v is number => v !== undefined && v !== null);
  const commEffects = metrics
    .map((m) => m.comm_effect)
    .filter((v): v is number => v !== undefined && v !== null);

  const peakEvalReward = evalRewards.length > 0 ? Math.max(...evalRewards) : null;
  const peakCommEffect = commEffects.length > 0 ? Math.max(...commEffects) : null;
  const latestStep =
    metrics.length > 0
      ? metrics[metrics.length - 1].total_num_steps ?? metrics[metrics.length - 1].step ?? 0
      : 0;

  return (
    <div className="page-container">
      {/* Top Navigation & Run Switcher */}
      <div className="detail-header-bar">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Experiments
        </button>

        <div className="detail-header-controls">
          <div className="inline-run-switcher">
            <label htmlFor="detail-run-select">Inspect Run:</label>
            <select
              id="detail-run-select"
              className="form-select compact-select"
              value={run.run_id}
              onChange={(e) => onSelectRun(e.target.value)}
            >
              {runs.filter((r) => !r.archived || r.run_id === run.run_id).map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} ({r.config?.num_agents || 2}A/{r.config?.num_landmarks || 3}L)
                </option>
              ))}
            </select>
          </div>

          <button className="btn-primary-sm" onClick={() => onOpenRepair(run.run_id)}>
            <Wrench size={14} /> Open in Causal Repair Studio
          </button>
        </div>
      </div>

      {/* Main Run Banner */}
      <div className="run-banner">
        <div className="banner-left">
          <h2>{run.run_id}</h2>
          <span className="banner-path">{run.path}</span>
        </div>
        <div className="banner-right" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="meta-badge highlight">{numAgents} Agents / {numLandmarks} Landmarks</span>
          <Badge variant={run.status}>{run.status.toUpperCase()}</Badge>
        </div>
      </div>

      {/* Detail Tab Navigation */}
      <div className="detail-tabs">
        <button
          className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <BarChart3 size={16} /> Causal Metrics &amp; Charts
        </button>
        <button
          className={`tab-btn ${activeTab === 'trajectory' ? 'active' : ''}`}
          onClick={() => setActiveTab('trajectory')}
        >
          <Radio size={16} /> 2D Trajectory Simulation
        </button>
        <button
          className={`tab-btn ${activeTab === 'gif' ? 'active' : ''}`}
          onClick={() => setActiveTab('gif')}
        >
          <Image size={16} /> MPE Render GIF
        </button>
        <button
          className={`tab-btn ${activeTab === 'checkpoints' ? 'active' : ''}`}
          onClick={() => setActiveTab('checkpoints')}
        >
          <Layers size={16} /> Checkpoints ({checkpoints.length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          <Terminal size={16} /> Console Logs
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content-wrapper">
        {activeTab === 'metrics' && (
          <div className="tab-pane">
            <MetricsChart metrics={metrics} />
          </div>
        )}

        {activeTab === 'trajectory' && (
          <div className="tab-pane">
            <TrajectoryCanvas
              numAgents={numAgents}
              numLandmarks={numLandmarks}
              isPerturbed={false}
              runId={run.run_id}
            />
          </div>
        )}

        {activeTab === 'gif' && (
          <div className="tab-pane">
            <GifViewer
              runId={run.run_id}
              hasGif={run.has_gif}
              onGifGenerated={onRefreshRunData}
            />
          </div>
        )}

        {activeTab === 'checkpoints' && (
          <div className="tab-pane">
            {checkpoints.length === 0 ? (
              <div className="checkpoints-table-card">
                <p className="p-4 text-gray">No checkpoints found in models/ directory.</p>
              </div>
            ) : (
              <div className="checkpoints-split-layout">
                {/* Left Column: All Saved Checkpoints */}
                <div className="checkpoints-archive-column">
                  <div className="checkpoints-section-header">
                    <h4>
                      <Layers size={16} className="inline mr-2 text-primary" />
                      All Saved Policy Checkpoints ({checkpoints.length})
                    </h4>
                  </div>

                  <div className="checkpoints-archive-list">
                    {checkpoints.map((cp) => (
                      <div
                        key={cp.name}
                        className={`checkpoint-row-item ${cp.is_best ? 'is-best-row' : ''}`}
                      >
                        <div className="cp-row-left">
                          <div className="cp-header">
                            <strong>{cp.name}</strong>
                            {cp.is_best && (
                              <span className="best-pill">
                                <Award size={11} /> BEST
                              </span>
                            )}
                          </div>
                          <span className="cp-step">
                            {cp.step > 0 ? `Step ${cp.step.toLocaleString()}` : 'Best Eval Snapshot'}
                          </span>
                        </div>

                        <button
                          className="btn-outline-sm"
                          onClick={() => onOpenRepair(run.run_id)}
                        >
                          <Wrench size={13} /> Repair
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right Column: Spotlight Best Checkpoint Card */}
                <div className="best-checkpoint-spotlight">
                  {bestCheckpoint && (
                    <div className="best-spotlight-card">
                      <div className="spotlight-badge-row">
                        <span className="spotlight-star-badge">
                          <Award size={14} /> ★ BEST POLICY CHECKPOINT
                        </span>
                      </div>

                      <h3 className="spotlight-name">{bestCheckpoint.name}</h3>
                      <p className="spotlight-subtitle">
                        {bestCheckpoint.step > 0
                          ? `Saved at ${bestCheckpoint.step.toLocaleString()} environment steps`
                          : `Highest evaluation return checkpoint (latest logged step: ${latestStep.toLocaleString()})`}
                      </p>

                      <div className="spotlight-metrics-grid">
                        <div className="meta-stat-box">
                          <span className="meta-stat-label">Peak Eval Return</span>
                          <span className="meta-stat-val text-emerald">
                            {peakEvalReward !== null ? peakEvalReward.toFixed(1) : 'N/A'}
                          </span>
                        </div>
                        <div className="meta-stat-box">
                          <span className="meta-stat-label">Peak Comm Gain (&Delta;R)</span>
                          <span className="meta-stat-val text-primary">
                            {peakCommEffect !== null
                              ? `${peakCommEffect >= 0 ? '+' : ''}${peakCommEffect.toFixed(1)}`
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="meta-stat-box">
                          <span className="meta-stat-label">Team Topology</span>
                          <span className="meta-stat-val">
                            {numAgents}A / {numLandmarks}L
                          </span>
                        </div>
                        <div className="meta-stat-box">
                          <span className="meta-stat-label">Total Checkpoints</span>
                          <span className="meta-stat-val text-amber">{checkpoints.length}</span>
                        </div>
                      </div>

                      <div className="spotlight-actions">
                        <button
                          className="btn-primary w-full"
                          onClick={() => onOpenRepair(run.run_id)}
                        >
                          <Wrench size={15} /> Repair This Checkpoint in Causal Studio
                        </button>
                        <button
                          className="btn-secondary w-full"
                          onClick={() => setActiveTab('trajectory')}
                        >
                          <Play size={15} /> Simulate 2D Trajectory
                        </button>
                      </div>

                      <div className="spotlight-footnote">
                        <TrendingUp size={13} className="text-emerald" />
                        <span>
                          Recommended for causal intervention testing &amp; LoRA perturbation recovery.
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="tab-pane">
            <TerminalLogViewer logs={logs} title={`Training Logs - ${run.run_id}`} />
          </div>
        )}
      </div>
    </div>
  );
};
