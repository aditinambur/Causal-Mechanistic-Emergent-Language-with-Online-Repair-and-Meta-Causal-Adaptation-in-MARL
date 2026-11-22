import React, { useState, useEffect } from 'react';
import {
  Wrench,
  ShieldAlert,
  Play,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Zap,
  Sliders,
  Scale,
  Coins,
  Activity,
  Layers,
  AlertTriangle,
  Cpu,
  RotateCcw
} from 'lucide-react';
import { RepairConfig, CheckpointInfo, RepairVerdict } from '../../types';
import { fetchRepairVerdict } from '../../services/api';
import { CheckpointSelector } from './CheckpointSelector';
import { RepairMetricsVisualizer } from './RepairMetricsVisualizer';
import { TerminalLogViewer } from '../visualizer/TerminalLogViewer';

interface RepairDashboardProps {
  runId?: string;
  checkpoints: CheckpointInfo[];
  repairLogs: string;
  onStartRepair: (config: RepairConfig) => Promise<void>;
  isRepairing: boolean;
  mirrorScope?: 'partner_full' | 'partner' | 'all';
  onMirrorScopeChange?: (scope: 'partner_full' | 'partner' | 'all') => void;
}

const PERTURBATION_IMPACTS: Record<'partner_full' | 'partner' | 'all', { title: string; math: string; effect: string }> = {
  partner_full: {
    title: 'Ghost Perception — 180° Partner Coordinate Inversion',
    math: '(x_rel, y_rel) ↦ (-x_rel, -y_rel)',
    effect: 'Agents see landmarks accurately, but teammate positions appear mirrored across each agent. Emitted communication tokens convey opposite spatial occupancy, triggering coordination collapse.'
  },
  partner: {
    title: 'Horizontal Ghost Perception — X-Axis Partner Mirror',
    math: '(x_rel, y_rel) ↦ (-x_rel, y_rel)',
    effect: 'Negates teammate horizontal offsets only, creating left/right mirror illusions that break spatial separation and token consensus.'
  },
  all: {
    title: 'Full Frame Inversion — Global Sensor & Landmark Flip',
    math: '(x, y) ↦ (-x, -y) for landmarks & teammates',
    effect: 'Flips both landmark targets and teammate relative frames, inducing severe global disorientation and catastrophic reward collapse.'
  }
};

export const RepairDashboard: React.FC<RepairDashboardProps> = ({
  runId,
  checkpoints,
  repairLogs,
  onStartRepair,
  isRepairing,
  mirrorScope: controlledScope,
  onMirrorScopeChange,
}) => {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>(
    checkpoints[0]?.name || 'checkpoint_best'
  );
  const [internalScope, setInternalScope] = useState<'partner_full' | 'partner' | 'all'>('partner_full');
  const mirrorScope = controlledScope ?? internalScope;
  const setMirrorScope = (val: 'partner_full' | 'partner' | 'all') => {
    if (onMirrorScopeChange) onMirrorScopeChange(val);
    setInternalScope(val);
  };

  const [repairTarget, setRepairTarget] = useState<'auto' | 'embedding' | 'comm' | 'lora' | 'full' | 'noncomm'>('auto');

  // Trigger & Frugality State
  const [triggerMode, setTriggerMode] = useState<'meta_causal' | 'and_gate' | 'reward_only'>('meta_causal');
  const [costLambda, setCostLambda] = useState<number>(0.02);
  const [causalAttributionThreshold, setCausalAttributionThreshold] = useState<number>(0.20);
  const [severeRewardRatio, setSevereRewardRatio] = useState<number>(0.50);
  const [loraRank, setLoraRank] = useState<number>(4);
  const [holdoutEpisodes, setHoldoutEpisodes] = useState<number>(6);
  const [measureEpisodes, setMeasureEpisodes] = useState<number>(6);
  const [repairIters, setRepairIters] = useState<number>(15);

  const [verdict, setVerdict] = useState<RepairVerdict | null>(null);

  useEffect(() => {
    if (checkpoints.length > 0 && !checkpoints.some((c) => c.name === selectedCheckpoint)) {
      setSelectedCheckpoint(checkpoints[0].name);
    }
  }, [checkpoints, selectedCheckpoint]);

  useEffect(() => {
    if (!runId) return;
    fetchRepairVerdict(runId)
      .then((data) => setVerdict(data))
      .catch(() => setVerdict(null));
  }, [runId, repairLogs, isRepairing]);

  const handleLaunch = (e: React.FormEvent) => {
    e.preventDefault();
    onStartRepair({
      checkpoint_name: selectedCheckpoint,
      mirror_scope: mirrorScope,
      repair_target: repairTarget,
      controller: triggerMode === 'reward_only' ? 'reward_only' : 'causal',
      trigger_mode: triggerMode === 'and_gate' ? 'and_gate' : 'meta_causal',
      cost_lambda: costLambda,
      causal_attribution_threshold: causalAttributionThreshold,
      severe_reward_ratio: severeRewardRatio,
      lora_rank: loraRank,
      holdout_episodes: holdoutEpisodes,
      measure_episodes: measureEpisodes,
      repair_iters: repairIters,
      seed: 1,
    });
  };

  const hasAccepted = verdict?.accepted === true || repairLogs.includes('ACCEPTED') || repairLogs.includes('repair accepted');
  const hasRejected = verdict?.accepted === false || repairLogs.includes('all repair attempts rejected');
  const hasHeldoutConfirm = verdict?.heldout_validation === 'CONFIRMED' || repairLogs.includes('HELD-OUT CONFIRMS');
  const hasHeldoutFailed = verdict?.heldout_validation === 'FAILED' || repairLogs.includes('HELD-OUT DOES NOT CONFIRM');
  const isFrugalAbstention = verdict?.decision_type === 'FRUGAL_ABSTENTION' || repairLogs.includes('Frugal Abstention');
  const isEmergency = verdict?.decision_type === 'EMERGENCY_TRIGGER' || repairLogs.includes('Emergency Trigger');
  const isCausalConfirmed = verdict?.decision_type === 'CAUSAL_CONFIRMED' || repairLogs.includes('Causal Trigger Confirmed');

  // Compute configured or actual runtime LoRA/target parameter footprint
  const TOTAL_ACTOR_PARAMS = verdict?.total_params || 10192;
  const getConfiguredParams = () => {
    if (verdict?.trainable_params && verdict.trainable_params > 0) {
      return verdict.trainable_params;
    }
    if (repairTarget === 'embedding') return 320;
    if (repairTarget === 'comm') return 640;
    if (repairTarget === 'noncomm') return 325;
    if (repairTarget === 'full') return TOTAL_ACTOR_PARAMS;
    // LoRA / Auto LoRA-first
    if (loraRank === 2) return 646;
    if (loraRank === 8) return 2586;
    return 1293;
  };
  const activeTrainableParams = getConfiguredParams();
  const activeParamPct = Math.min(100, (activeTrainableParams / TOTAL_ACTOR_PARAMS) * 100);
  const frozenParams = Math.max(0, TOTAL_ACTOR_PARAMS - activeTrainableParams);

  // Build ladder visualization steps
  const defaultLadder = ['embedding', 'comm', 'lora', 'full'];
  const ladderMeta: Record<string, { label: string; params: string }> = {
    embedding: { label: 'Embedding', params: '320w' },
    comm: { label: 'Comm Attn', params: '640w' },
    lora: { label: `LoRA r=${loraRank}`, params: `${loraRank === 2 ? 646 : loraRank === 8 ? 2586 : 1293}w` },
    full: { label: 'Full Actor', params: '10,192w' },
  };
  const escalationMap = new Map(
    (verdict?.escalation_steps || []).map((s) => [s.target, s])
  );

  const impactInfo = PERTURBATION_IMPACTS[mirrorScope];

  return (
    <div className="repair-dashboard-container">
      <div className="repair-hero">
        <div className="hero-text">
          <h2>
            <Wrench className="inline mr-2 text-primary" size={24} /> Phase 2/3: Causal Breakdown &amp; Online Repair {runId ? `— ${runId}` : ''}
          </h2>
          <p>
            Inject controlled coordinate perturbations, detect causal communication breakdown with Pareto Frugality (Cost-Utility ROI), execute LoRA parameter-efficient adaptation, and verify generalization on held-out layouts.
          </p>
        </div>
      </div>

      <div className="repair-layout-grid">
        {/* Left Column: Configuration Form */}
        <div className="repair-config-column">
          <form className="repair-form-card" onSubmit={handleLaunch}>
            <h4 className="card-section-title">
              <Sliders size={16} /> 1. Perturbation &amp; Architecture
            </h4>

            {/* Checkpoint selector */}
            <CheckpointSelector
              checkpoints={checkpoints}
              selectedCheckpoint={selectedCheckpoint}
              onSelectCheckpoint={setSelectedCheckpoint}
            />

            {/* Perturbation Modality */}
            <div className="form-group mt-4">
              <label>
                <ShieldAlert size={14} className="inline mr-1 text-rose" /> Perturbation Modality (mirror_scope)
              </label>
              <select
                value={mirrorScope}
                onChange={(e) => setMirrorScope(e.target.value as any)}
                className="form-select"
              >
                <option value="partner_full">Ghost Perception (partner_full) — 180° Partner Coordinate Inversion</option>
                <option value="partner">Horizontal Ghost Perception (partner) — X-Axis Partner Mirror</option>
                <option value="all">Full Frame Inversion (all) — Global Sensor &amp; Landmark Flip</option>
              </select>

              <div className="perturbation-impact-card">
                <div className="impact-header">
                  <strong>{impactInfo.title}</strong>
                  <code className="impact-math">{impactInfo.math}</code>
                </div>
                <p>{impactInfo.effect}</p>
              </div>
            </div>

            {/* Repair Target Slicing */}
            <div className="form-group">
              <label>
                <Layers size={14} className="inline mr-1 text-emerald" /> Repair Parameter Target
              </label>
              <select
                value={repairTarget}
                onChange={(e) => setRepairTarget(e.target.value as any)}
                className="form-select"
              >
                <option value="auto">auto (LoRA-First Escalation Ladder: lora &rarr; comm &rarr; full)</option>
                <option value="lora">lora (Parameter-Efficient: 1,293 weights, &lt;1.5% MARL params)</option>
                <option value="comm">comm only (message_head + token_embedding + attention)</option>
                <option value="embedding">embedding only (~320 weights, discrete dictionary)</option>
                <option value="full">full (10,192 weights — entire actor policy network)</option>
                <option value="noncomm">noncomm (Control Baseline: motor actions only)</option>
              </select>
            </div>

            {/* Trainable LoRA Weight Footprint Preview Card */}
            <div className="param-footprint-card">
              <div className="footprint-header">
                <span>
                  <Cpu size={14} className="inline mr-1 text-emerald" />
                  Trainable Weight Footprint ({verdict?.accepted_target || verdict?.repair_target || repairTarget})
                </span>
                <strong className="text-emerald">
                  {activeTrainableParams.toLocaleString()} / {TOTAL_ACTOR_PARAMS.toLocaleString()} weights ({activeParamPct.toFixed(1)}%)
                </strong>
              </div>
              <div className="footprint-bar-track">
                <div
                  className="footprint-bar-fill"
                  style={{ width: `${Math.max(4, activeParamPct)}%` }}
                />
              </div>
              <div className="footprint-subrow">
                <span>Retrained: <strong>{activeTrainableParams.toLocaleString()}</strong></span>
                <span>Frozen Base: <strong>{frozenParams.toLocaleString()}</strong></span>
                <span>Budget: <strong>{repairIters} PPO iters</strong></span>
              </div>
            </div>

            <h4 className="card-section-title mt-4">
              <Scale size={16} /> 2. Detection Trigger &amp; Pareto Frugality
            </h4>

            <div className="form-group">
              <label>Decision Mechanism</label>
              <div className="radio-pill-group">
                <button
                  type="button"
                  className={`radio-pill ${triggerMode === 'meta_causal' ? 'active' : ''}`}
                  onClick={() => setTriggerMode('meta_causal')}
                >
                  <Coins size={14} className="text-amber" />
                  <div>
                    <strong>Meta-Causal Trigger (ROI &amp; Frugality)</strong>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Gated by Causal Attribution Ratio + Compute Cost ROI</div>
                  </div>
                </button>
                <button
                  type="button"
                  className={`radio-pill ${triggerMode === 'and_gate' ? 'active' : ''}`}
                  onClick={() => setTriggerMode('and_gate')}
                >
                  <Zap size={14} className="text-primary" />
                  <div>
                    <strong>Classical AND-Gated Trigger</strong>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Reward drop &ge; 10% AND comm_effect collapse</div>
                  </div>
                </button>
                <button
                  type="button"
                  className={`radio-pill ${triggerMode === 'reward_only' ? 'active' : ''}`}
                  onClick={() => setTriggerMode('reward_only')}
                >
                  <Activity size={14} className="text-gray" />
                  <div>
                    <strong>Naive Reward-Only Ablation</strong>
                    <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Triggers blindly on reward drop &ge; 20% (Ablation arm)</div>
                  </div>
                </button>
              </div>
            </div>

            {triggerMode === 'meta_causal' && (
              <div className="frugal-controls-box">
                <div className="slider-group">
                  <div className="slider-header">
                    <span>Compute Cost Penalty (&lambda;)</span>
                    <strong className="text-amber">{costLambda.toFixed(3)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0.00"
                    max="0.08"
                    step="0.005"
                    value={costLambda}
                    onChange={(e) => setCostLambda(parseFloat(e.target.value))}
                    className="form-range"
                  />
                </div>

                <div className="slider-group mt-3">
                  <div className="slider-header">
                    <span>Causal Attribution Cutoff (&rho;<sub>causal</sub>)</span>
                    <strong className="text-primary">{causalAttributionThreshold.toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.50"
                    step="0.05"
                    value={causalAttributionThreshold}
                    onChange={(e) => setCausalAttributionThreshold(parseFloat(e.target.value))}
                    className="form-range"
                  />
                </div>

                <div className="slider-group mt-3">
                  <div className="slider-header">
                    <span>Catastrophic Emergency Threshold</span>
                    <strong className="text-rose">{(severeRewardRatio * 100).toFixed(0)}%</strong>
                  </div>
                  <input
                    type="range"
                    min="0.20"
                    max="0.80"
                    step="0.05"
                    value={severeRewardRatio}
                    onChange={(e) => setSevereRewardRatio(parseFloat(e.target.value))}
                    className="form-range"
                  />
                </div>
              </div>
            )}

            <h4 className="card-section-title mt-4">
              <Sliders size={16} /> 3. Verification &amp; Fine-Tuning Budget
            </h4>

            <div className="form-row">
              <div className="form-group flex-1">
                <label>LoRA Rank (r)</label>
                <select
                  value={loraRank}
                  onChange={(e) => setLoraRank(parseInt(e.target.value))}
                  className="form-select"
                >
                  <option value={2}>r = 2 (646 weights)</option>
                  <option value={4}>r = 4 (1,293 weights - recommended)</option>
                  <option value={8}>r = 8 (2,586 weights)</option>
                </select>
              </div>
              <div className="form-group flex-1">
                <label>Held-Out Episodes</label>
                <input
                  type="number"
                  min={2}
                  max={30}
                  value={holdoutEpisodes}
                  onChange={(e) => setHoldoutEpisodes(parseInt(e.target.value) || 6)}
                  className="form-input"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group flex-1">
                <label>Measure Episodes</label>
                <input
                  type="number"
                  min={2}
                  max={20}
                  value={measureEpisodes}
                  onChange={(e) => setMeasureEpisodes(parseInt(e.target.value) || 6)}
                  className="form-input"
                />
              </div>
              <div className="form-group flex-1">
                <label>Repair Iterations</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={repairIters}
                  onChange={(e) => setRepairIters(parseInt(e.target.value) || 15)}
                  className="form-input"
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn-primary w-full mt-3"
              disabled={isRepairing || !selectedCheckpoint}
            >
              <Play size={16} />
              {isRepairing ? 'Executing Online Repair...' : 'Execute Perturbation & Repair'}
            </button>
          </form>
        </div>

        {/* Right Column: Execution Live Status, Escalation Stepper & Demographics */}
        <div className="repair-results-column">
          <div className="verdict-summary-card">
            <h4 className="card-section-title">Online Repair Pipeline &amp; Surgical Ladder</h4>

            {/* 5-Stage Pipeline Stepper */}
            <div className="verdict-stepper">
              <div className={`step-item ${verdict?.baseline_reward !== null || isRepairing ? 'active' : ''}`}>
                <span className="step-num">1</span>
                <span>CRN Probe</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className={`step-item ${isFrugalAbstention ? 'warning' : verdict?.detector_fired !== null ? 'active' : ''}`}>
                <span className="step-num">2</span>
                <span>{isFrugalAbstention ? 'Abstained' : 'Causal Gate'}</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className={`step-item ${verdict?.repair_target ? 'active' : ''}`}>
                <span className="step-num">3</span>
                <span>Target ({verdict?.accepted_target || verdict?.repair_target || repairTarget})</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className={`step-item ${hasAccepted ? 'success' : hasRejected ? 'danger' : ''}`}>
                <span className="step-num">4</span>
                <span>{hasAccepted ? 'Accepted' : hasRejected ? 'Rollback' : 'Dual Gate'}</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className={`step-item ${hasHeldoutConfirm ? 'success' : hasHeldoutFailed ? 'danger' : ''}`}>
                <span className="step-num">5</span>
                <span>{hasHeldoutConfirm ? 'Held-Out Pass' : hasHeldoutFailed ? 'Held-Out Fail' : 'Verify (N=' + holdoutEpisodes + ')'}</span>
              </div>
            </div>

            {/* Multi-Step Escalation Ladder Pills (emb -> comm -> lora -> full) */}
            <div className="escalation-ladder-box">
              <div className="ladder-header">
                <span>Multi-Step Surgical Escalation Ladder</span>
                {verdict?.trainable_params && (
                  <span className="ladder-weights-pill">
                    {verdict.trainable_params.toLocaleString()} weights retrained ({((verdict.trainable_params / TOTAL_ACTOR_PARAMS) * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
              <div className="escalation-ladder-pills">
                {defaultLadder.map((stepKey, idx) => {
                  const attempt = escalationMap.get(stepKey);
                  const meta = ladderMeta[stepKey];
                  const isAcceptedStep = attempt?.status === 'ACCEPTED' || verdict?.accepted_target === stepKey;
                  const isRollbackStep = attempt?.status === 'ROLLBACK' || attempt?.status === 'REJECTED';
                  const isRunningStep = attempt?.status === 'RUNNING' || (isRepairing && verdict?.repair_target === stepKey);
                  const pillClass = isAcceptedStep
                    ? 'accepted'
                    : isRollbackStep
                    ? 'rollback'
                    : isRunningStep
                    ? 'running'
                    : 'idle';

                  return (
                    <React.Fragment key={stepKey}>
                      <div className={`ladder-pill ${pillClass}`}>
                        <span className="ladder-pill-title">{meta.label}</span>
                        <span className="ladder-pill-meta">
                          {attempt ? `${attempt.trainable_params.toLocaleString()}w` : meta.params}
                          {isAcceptedStep && ' ✓'}
                          {isRollbackStep && ' ↺ Rollback'}
                          {isRunningStep && ' ⋯'}
                        </span>
                      </div>
                      {idx < defaultLadder.length - 1 && <ArrowRight size={12} className="text-gray" />}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>

            {/* Decision Callout Banners */}
            {isFrugalAbstention && (
              <div className="decision-callout frugal">
                <div className="flex items-center gap-2 font-bold mb-1">
                  <Coins size={16} /> FRUGAL ABSTENTION (COMPUTE PRESERVED)
                </div>
                <div>{verdict?.decision_summary || "Reward degradation was not causally attributable to communication breakdown. Policy fine-tuning abstained."}</div>
              </div>
            )}

            {isEmergency && (
              <div className="decision-callout emergency">
                <div className="flex items-center gap-2 font-bold mb-1">
                  <AlertTriangle size={16} /> EMERGENCY COORDINATION COLLAPSE TRIGGER
                </div>
                <div>{verdict?.decision_summary || "Catastrophic multi-agent performance collapse detected (>50% reward drop). Emergency adaptation initiated."}</div>
              </div>
            )}

            {isCausalConfirmed && (
              <div className="decision-callout confirmed">
                <div className="flex items-center gap-2 font-bold mb-1">
                  <CheckCircle2 size={16} /> CAUSAL BREAKDOWN VERIFIED (POSITIVE ROI)
                </div>
                <div>{verdict?.decision_summary || "Interventional communication breakdown confirmed. Surgical adaptation initiated."}</div>
              </div>
            )}

            {hasRejected && (
              <div className="decision-callout emergency">
                <div className="flex items-center gap-2 font-bold mb-1">
                  <RotateCcw size={16} /> BIT-EXACT ROLLBACK EXECUTED
                </div>
                <div>Adaptation failed dual acceptance thresholds. Pre-repair weights, critic, and Adam momentum buffers restored.</div>
              </div>
            )}

            {/* Held-Out Confirmation Banner */}
            {hasHeldoutConfirm && (
              <div className="heldout-banner success">
                <CheckCircle2 size={18} />
                <div>
                  <strong>HELD-OUT GENERALIZATION CONFIRMED</strong>
                  <p>
                    Communication recovery verified on {holdoutEpisodes} fresh, unseen random seeds.
                    {typeof verdict?.heldout_reward_recovery === 'number' ? ` (Held-Out Recovery: ${verdict.heldout_reward_recovery.toFixed(1)}%)` : ''}
                  </p>
                </div>
              </div>
            )}

            {hasHeldoutFailed && (
              <div className="heldout-banner danger">
                <XCircle size={18} />
                <div>
                  <strong>HELD-OUT GENERALIZATION FAILED</strong>
                  <p>Repair cleared training layouts but failed on held-out seeds (potential layout overfitting).</p>
                </div>
              </div>
            )}
          </div>

          {/* Interactive Repair Graphs, Post-Training Demographics & Analytics */}
          <RepairMetricsVisualizer
            verdict={verdict}
            repairLogs={repairLogs}
            isRepairing={isRepairing}
          />

          {/* Real-time streaming log terminal */}
          <TerminalLogViewer
            logs={repairLogs}
            title="Phase 2/3 Online Repair Process Stream"
            autoScroll={true}
          />
        </div>
      </div>
    </div>
  );
};
