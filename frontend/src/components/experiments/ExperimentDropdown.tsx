import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Layers, Sparkles } from 'lucide-react';
import { Run } from '../../types';
import { RunList } from './RunList';
import { Badge } from '../common/Badge';

interface ExperimentDropdownProps {
  runs: Run[];
  selectedRunId: string | null;
  onSelectRunId?: (runId: string) => void;
  onViewDetails: (runId: string) => void;
  onInspectRepair: (runId: string) => void;
  onToggleArchive: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
  label?: string;
  showArchived?: boolean;
  showDetailsCard?: boolean;
  defaultOpen?: boolean;
}

export const ExperimentDropdown: React.FC<ExperimentDropdownProps> = ({
  runs,
  selectedRunId,
  onViewDetails,
  onInspectRepair,
  onToggleArchive,
  onDeleteRun,
  label = 'MARL Experiments',
  showArchived = false,
  defaultOpen = false,
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(defaultOpen);

  const visibleRuns = runs.filter((r) => (showArchived ? r.archived : !r.archived));
  const activeRun =
    visibleRuns.find((r) => r.run_id === selectedRunId) ||
    visibleRuns[0] ||
    null;

  const numAgents = activeRun?.config?.num_agents || 2;
  const numLandmarks = activeRun?.config?.num_landmarks || 3;

  return (
    <div className="experiment-dropdown-container">
      {/* Clickable Dropdown Header Trigger */}
      <button
        type="button"
        className={`dropdown-trigger-bar ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <div className="trigger-left">
          <Layers size={16} className="text-primary" />
          <span className="trigger-title">{label}</span>
          <span className="trigger-count-pill">{visibleRuns.length} Runs</span>
          {activeRun && (
            <span className="trigger-active-preview">
              Active: <strong>{activeRun.run_id}</strong> ({numAgents}A/{numLandmarks}L)
              <Badge variant={activeRun.status}>{activeRun.status.toUpperCase()}</Badge>
              {activeRun.has_repair && (
                <span className="meta-badge repair">
                  <Sparkles size={11} /> Repaired
                </span>
              )}
            </span>
          )}
        </div>

        <div className="trigger-right">
          <span className="trigger-toggle-hint">
            {isOpen ? 'Hide Experiments' : 'Open Dropdown'}
          </span>
          {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </button>

      {/* Collapsible Dropdown Body showing full original RunList UI */}
      {isOpen && (
        <div className="dropdown-drawer-body">
          <RunList
            runs={runs}
            selectedRunId={selectedRunId}
            onSelectRun={onViewDetails}
            onInspectRepair={onInspectRepair}
            onToggleArchive={onToggleArchive}
            onDeleteRun={onDeleteRun}
            showArchived={showArchived}
          />
        </div>
      )}
    </div>
  );
};
