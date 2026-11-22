import React, { useState } from 'react';
import { Wrench, ArrowLeft, Layers } from 'lucide-react';
import { Run, CheckpointInfo, RepairConfig } from '../types';
import { RepairDashboard } from '../components/repair/RepairDashboard';
import { TrajectoryCanvas } from '../components/visualizer/TrajectoryCanvas';

interface CausalRepairPageProps {
  runs: Run[];
  selectedRun: Run | null;
  onSelectRun: (runId: string) => void;
  checkpoints: CheckpointInfo[];
  repairLogs: string;
  onStartRepair: (config: RepairConfig) => Promise<void>;
  isRepairing: boolean;
  onBack: () => void;
}

export const CausalRepairPage: React.FC<CausalRepairPageProps> = ({
  runs,
  selectedRun,
  onSelectRun,
  checkpoints,
  repairLogs,
  onStartRepair,
  isRepairing,
  onBack,
}) => {
  const [mirrorScope, setMirrorScope] = useState<'partner_full' | 'partner' | 'all'>('partner_full');

  const availableRuns = runs.filter((r) => !r.archived || r.run_id === selectedRun?.run_id);
  const numAgents = selectedRun?.config?.num_agents || 3;
  const numLandmarks = selectedRun?.config?.num_landmarks || 4;

  return (
    <div className="page-container">
      {/* Top Bar with Embedded Target Experiment Selector */}
      <div className="detail-header-bar causal-top-bar">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Run Details
        </button>

        <div className="causal-run-selector-bar">
          <label htmlFor="causal-target-run-select" className="causal-selector-label">
            <Layers size={15} className="text-primary" /> Target Experiment:
          </label>
          <select
            id="causal-target-run-select"
            className="form-select compact-select"
            value={selectedRun?.run_id || ''}
            onChange={(e) => onSelectRun(e.target.value)}
            disabled={availableRuns.length === 0}
          >
            {availableRuns.length === 0 ? (
              <option value="">No trained experiments available</option>
            ) : (
              availableRuns.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} — {r.config?.num_agents || 2}A/{r.config?.num_landmarks || 3}L ({r.checkpoints_count} ckpts)
                </option>
              ))
            )}
          </select>
        </div>
      </div>

      {!selectedRun ? (
        <div className="empty-state">
          <Wrench size={40} className="text-gray" style={{ margin: '0 auto 10px' }} />
          <h3>No Experiment Selected</h3>
          <p>Choose an experiment from the Target Experiment dropdown above or train a new model first.</p>
        </div>
      ) : (
        <>
          {/* Perturbation Trajectory Demo Canvas synced with selected mirrorScope */}
          <div className="repair-simulation-banner">
            <TrajectoryCanvas
              numAgents={numAgents}
              numLandmarks={numLandmarks}
              isPerturbed={true}
              perturbationScope={mirrorScope}
              runId={selectedRun.run_id}
              isRepairedView={false}
            />
          </div>

          {/* Repair Execution Dashboard */}
          <RepairDashboard
            runId={selectedRun.run_id}
            checkpoints={checkpoints}
            repairLogs={repairLogs}
            onStartRepair={onStartRepair}
            isRepairing={isRepairing}
            mirrorScope={mirrorScope}
            onMirrorScopeChange={setMirrorScope}
          />
        </>
      )}
    </div>
  );
};
