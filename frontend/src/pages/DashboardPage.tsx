import React from 'react';
import { 
  Zap, 
  Layers, 
  CheckCircle2, 
  Play, 
  Wrench
} from 'lucide-react';
import { Run, SystemInfo } from '../types';
import { MetricStatCard } from '../components/metrics/MetricStatCard';
import { ExperimentDropdown } from '../components/experiments/ExperimentDropdown';
import { TrajectoryCanvas } from '../components/visualizer/TrajectoryCanvas';

interface DashboardPageProps {
  runs: Run[];
  systemInfo: SystemInfo | null;
  selectedRunId: string | null;
  onFocusRun?: (runId: string) => void;
  onSelectRun: (runId: string) => void;
  onInspectRepair: (runId: string) => void;
  onToggleArchive: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
  onNavigateToSimulate: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  runs,
  systemInfo,
  selectedRunId,
  onFocusRun,
  onSelectRun,
  onInspectRepair,
  onToggleArchive,
  onDeleteRun,
  onNavigateToSimulate,
}) => {
  const activeRuns = runs.filter((r) => !r.archived);
  const completedRuns = activeRuns.filter((r) => r.status === 'completed');
  const repairedRuns = activeRuns.filter((r) => r.has_repair);
  const selectedRun =
    activeRuns.find((r) => r.run_id === selectedRunId) ||
    activeRuns[0] ||
    null;

  const heroAgents = selectedRun?.config?.num_agents || 3;
  const heroLandmarks = selectedRun?.config?.num_landmarks || 4;

  return (
    <div className="page-container">
      {/* Top Stat Cards */}
      <div className="stats-grid">
        <MetricStatCard
          title="LoRA Surgery Footprint"
          value="1,293 Weights"
          subtext="<1.5% System Params Tuned (Rank 4)"
          icon={Zap}
          color="text-emerald"
        />
        <MetricStatCard
          title="Active Experiments"
          value={activeRuns.length}
          subtext={`${completedRuns.length} Completed Successfully`}
          icon={Layers}
          color="text-primary"
        />
        <MetricStatCard
          title="Phase 2/3 Repaired"
          value={repairedRuns.length}
          subtext="Evaluated under Perturbations"
          icon={Wrench}
          color="text-amber"
        />
        <MetricStatCard
          title="System Engine"
          value={systemInfo?.status === 'busy' ? 'RUNNING' : 'IDLE'}
          subtext={systemInfo?.active_run_id ? `Task: ${systemInfo.active_run_id}` : 'Ready for New Simulation'}
          icon={CheckCircle2}
          color={systemInfo?.status === 'busy' ? 'text-amber' : 'text-emerald'}
        />
      </div>

      {/* Trajectory Simulation & Quick Action Card */}
      <div className="dashboard-hero-section">
        <div className="hero-simulation-box">
          <TrajectoryCanvas
            numAgents={heroAgents}
            numLandmarks={heroLandmarks}
            isPerturbed={false}
            runId={selectedRun?.run_id}
          />
        </div>

        <div className="hero-action-box">
          <h3>Multi-Agent Emergent Communication</h3>
          <p>
            Train multi-agent teams on MPE simple_spread to learn discrete token communication, measure causal influence via counterfactual do-operators, and test surgical online repair.
          </p>

          <div className="hero-buttons">
            <button className="btn-primary" onClick={onNavigateToSimulate}>
              <Play size={16} /> Launch New Simulation
            </button>
          </div>
        </div>
      </div>

      {/* Recent Experiments Dropdown */}
      <div className="section-container">
        <div className="section-header">
          <h3>Recent MARL Experiments</h3>
        </div>
        <ExperimentDropdown
          runs={runs}
          selectedRunId={selectedRun?.run_id || selectedRunId}
          onSelectRunId={(id) => (onFocusRun ? onFocusRun(id) : onSelectRun(id))}
          onViewDetails={onSelectRun}
          onInspectRepair={onInspectRepair}
          onToggleArchive={onToggleArchive}
          onDeleteRun={onDeleteRun}
          label="Choose Recent Experiment"
          showArchived={false}
          showDetailsCard={true}
        />
      </div>
    </div>
  );
};
