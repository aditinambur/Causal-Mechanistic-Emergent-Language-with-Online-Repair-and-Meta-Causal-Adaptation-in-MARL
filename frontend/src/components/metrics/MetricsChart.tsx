import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { MetricRow } from '../../types';

interface MetricsChartProps {
  metrics: MetricRow[];
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ metrics }) => {
  const [selectedMetric, setSelectedMetric] = useState<string>('all');

  if (!metrics || metrics.length === 0) {
    return (
      <div className="empty-metrics-box">
        <p>No causal influence metrics recorded yet. Metrics update at each eval interval.</p>
      </div>
    );
  }

  // Detect which metrics actually have values
  const hasKl = metrics.some((m) => m.causal_influence_kl_mean !== undefined && m.causal_influence_kl_mean !== null);
  const hasCommEffect = metrics.some((m) => m.comm_effect !== undefined && m.comm_effect !== null);
  const hasValue = metrics.some((m) => m.causal_influence_value_sensitivity_mean !== undefined && m.causal_influence_value_sensitivity_mean !== null);
  const hasReward = metrics.some((m) => (m.eval_reward !== undefined && m.eval_reward !== null) || (m.reward !== undefined && m.reward !== null));

  // Format data points for charts
  const data = metrics.map((m, idx) => {
    const stepVal = m.total_num_steps ?? m.step ?? m.episode ?? (idx + 1) * 5;
    return {
      index: idx + 1,
      step: stepVal,
      kl_divergence:
        m.causal_influence_kl_mean !== undefined && m.causal_influence_kl_mean !== null
          ? Number(Number(m.causal_influence_kl_mean).toFixed(4))
          : null,
      value_sensitivity:
        m.causal_influence_value_sensitivity_mean !== undefined && m.causal_influence_value_sensitivity_mean !== null
          ? Number(Number(m.causal_influence_value_sensitivity_mean).toFixed(4))
          : null,
      comm_effect:
        m.comm_effect !== undefined && m.comm_effect !== null
          ? Number(Number(m.comm_effect).toFixed(2))
          : null,
      eval_reward:
        m.eval_reward !== undefined && m.eval_reward !== null
          ? Number(Number(m.eval_reward).toFixed(2))
          : m.reward !== undefined && m.reward !== null
          ? Number(Number(m.reward).toFixed(2))
          : null,
    };
  });

  const latest = data[data.length - 1] || {};
  const peakCommEffect = hasCommEffect
    ? Math.max(...data.filter((d) => d.comm_effect !== null).map((d) => d.comm_effect as number))
    : null;
  const peakReward = hasReward
    ? Math.max(...data.filter((d) => d.eval_reward !== null).map((d) => d.eval_reward as number))
    : null;

  const formatStepTick = (val: number) => {
    if (val >= 1000) return `${(val / 1000).toFixed(0)}k`;
    return `${val}`;
  };

  const tooltipStyle = {
    backgroundColor: '#0f172a',
    borderColor: '#334155',
    borderRadius: '8px',
    color: '#f8fafc',
    fontSize: '12px',
  };

  return (
    <div className="metrics-chart-container">
      {/* 4 Summary KPI Cards */}
      <div className="metrics-kpi-grid">
        <div className="meta-stat-box">
          <span className="meta-stat-label">Reward Gain (&Delta;R<sub>comm</sub>)</span>
          <span className={`meta-stat-val ${latest.comm_effect !== null && latest.comm_effect >= 0 ? 'text-emerald' : 'text-rose'}`}>
            {latest.comm_effect !== null ? `${latest.comm_effect >= 0 ? '+' : ''}${latest.comm_effect.toFixed(1)}` : 'N/A'}
          </span>
          {peakCommEffect !== null && (
            <span className="field-hint">Peak: {peakCommEffect >= 0 ? '+' : ''}{peakCommEffect.toFixed(1)}</span>
          )}
        </div>

        <div className="meta-stat-box">
          <span className="meta-stat-label">Policy Sensitivity (CIC<sub>KL</sub>)</span>
          <span className="meta-stat-val text-primary">
            {latest.kl_divergence !== null ? latest.kl_divergence.toFixed(4) : 'N/A'}
          </span>
          <span className="field-hint">Action steering divergence</span>
        </div>

        <div className="meta-stat-box">
          <span className="meta-stat-label">Value Sensitivity (&Delta;V)</span>
          <span className="meta-stat-val text-amber">
            {latest.value_sensitivity !== null ? latest.value_sensitivity.toFixed(3) : 'N/A'}
          </span>
          <span className="field-hint">Critic reliance on messages</span>
        </div>

        <div className="meta-stat-box">
          <span className="meta-stat-label">Evaluation Return</span>
          <span className="meta-stat-val" style={{ color: '#c084fc' }}>
            {latest.eval_reward !== null ? latest.eval_reward.toFixed(1) : 'N/A'}
          </span>
          {peakReward !== null && (
            <span className="field-hint">Best: {peakReward.toFixed(1)}</span>
          )}
        </div>
      </div>

      <div className="chart-header mt-4">
        <h4 className="chart-title">Causal Influence of Communication (CIC) &amp; Reward Gain Over Training</h4>

        <div className="chart-controls">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="metric-select"
          >
            <option value="all">Dual Charts: Reward Gain &amp; Sensitivities</option>
            {hasCommEffect && <option value="comm_effect">Reward Gain Only (&Delta;R_comm)</option>}
            {hasReward && <option value="reward">Evaluation Return Only</option>}
            {hasKl && <option value="kl">Policy Sensitivity Only (KL)</option>}
            {hasValue && <option value="value">Value Sensitivity Only (|V_real - V_zero|)</option>}
          </select>
        </div>
      </div>

      {selectedMetric === 'all' ? (
        <div className="dual-charts-grid">
          {/* Chart 1: Reward Gain & Task Return */}
          <div className="sub-chart-card">
            <h5 className="sub-chart-title">
              1. Communication Reward Gain (&Delta;R<sub>comm</sub>) &amp; Evaluation Return
            </h5>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height={290}>
                <LineChart data={data} margin={{ top: 10, right: 20, left: 6, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a3342" />
                  <XAxis
                    dataKey="step"
                    stroke="#64748b"
                    tickFormatter={formatStepTick}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis yAxisId="left" stroke="#10b981" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" stroke="#a855f7" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend verticalAlign="top" height={34} />
                  {hasCommEffect && (
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="comm_effect"
                      name="Reward Gain (ΔR_comm)"
                      stroke="#10b981"
                      strokeWidth={2.5}
                      dot={{ r: 3, fill: '#10b981' }}
                      activeDot={{ r: 6 }}
                    />
                  )}
                  {hasReward && (
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="eval_reward"
                      name="Eval Return"
                      stroke="#a855f7"
                      strokeWidth={2}
                      dot={{ r: 3, fill: '#a855f7' }}
                      activeDot={{ r: 5 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-definition-box">
              <div>
                <strong className="text-emerald">Reward Gain (&Delta;R<sub>comm</sub>):</strong> Extra team score earned purely from communication, measured by comparing identical layouts with messages ON vs. muted (<em>R<sub>comm</sub> &minus; R<sub>silent</sub></em>). Positive values prove messages help coordination.
              </div>
              <div>
                <strong style={{ color: '#c084fc' }}>Eval Return:</strong> Total team episode score (reward based on distance to landmarks minus collision penalties; closer to 0 / higher is better).
              </div>
            </div>
          </div>

          {/* Chart 2: Policy & Value Sensitivity */}
          <div className="sub-chart-card">
            <h5 className="sub-chart-title">
              2. Policy Sensitivity (CIC<sub>KL</sub>) &amp; Value Sensitivity (&Delta;V<sub>comm</sub>)
            </h5>
            <div className="chart-body">
              <ResponsiveContainer width="100%" height={290}>
                <LineChart data={data} margin={{ top: 10, right: 20, left: 6, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a3342" />
                  <XAxis
                    dataKey="step"
                    stroke="#64748b"
                    tickFormatter={formatStepTick}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis yAxisId="left" stroke="#38bdf8" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend verticalAlign="top" height={34} />
                  {hasKl && (
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="kl_divergence"
                      name="Policy Sensitivity (KL)"
                      stroke="#38bdf8"
                      strokeWidth={2.5}
                      dot={{ r: 3, fill: '#38bdf8' }}
                      activeDot={{ r: 6 }}
                    />
                  )}
                  {hasValue && (
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="value_sensitivity"
                      name="Value Sensitivity"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={{ r: 3, fill: '#f59e0b' }}
                      activeDot={{ r: 5 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-definition-box">
              <div>
                <strong className="text-primary">Policy Sensitivity (CIC<sub>KL</sub>):</strong> How strongly teammate messages change an agent&rsquo;s action probabilities (<em>D<sub>KL</sub>(&pi;(a|o,c) || &pi;(a|o,0))</em>). Higher means agents actively steer based on messages.
              </div>
              <div>
                <strong className="text-amber">Value Sensitivity (&Delta;V<sub>comm</sub>):</strong> How much the critic&rsquo;s expected return estimate shifts when messages are silenced (<em>|V(s,c) &minus; V(s,0)|</em>).
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="sub-chart-card">
          <div className="chart-body">
            <ResponsiveContainer width="100%" height={330}>
              <LineChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3342" />
                <XAxis
                  dataKey="step"
                  stroke="#64748b"
                  tickFormatter={formatStepTick}
                  label={{ value: 'Environment Steps', position: 'insideBottom', offset: -10, fill: '#64748b' }}
                />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend verticalAlign="top" height={36} />

                {selectedMetric === 'kl' && hasKl && (
                  <Line
                    type="monotone"
                    dataKey="kl_divergence"
                    name="Policy Sensitivity (KL)"
                    stroke="#38bdf8"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#38bdf8' }}
                    activeDot={{ r: 6 }}
                  />
                )}
                {selectedMetric === 'comm_effect' && hasCommEffect && (
                  <Line
                    type="monotone"
                    dataKey="comm_effect"
                    name="Reward Gain (ΔR_comm)"
                    stroke="#10b981"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#10b981' }}
                    activeDot={{ r: 6 }}
                  />
                )}
                {selectedMetric === 'value' && hasValue && (
                  <Line
                    type="monotone"
                    dataKey="value_sensitivity"
                    name="Value Sensitivity"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#f59e0b' }}
                    activeDot={{ r: 6 }}
                  />
                )}
                {selectedMetric === 'reward' && hasReward && (
                  <Line
                    type="monotone"
                    dataKey="eval_reward"
                    name="Evaluation Return"
                    stroke="#a855f7"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#a855f7' }}
                    activeDot={{ r: 6 }}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-definition-box">
            {selectedMetric === 'comm_effect' && (
              <div><strong className="text-emerald">Reward Gain (&Delta;R<sub>comm</sub>):</strong> Extra team score earned purely from communication on paired layouts (<em>R<sub>comm</sub> &minus; R<sub>silent</sub></em>).</div>
            )}
            {selectedMetric === 'reward' && (
              <div><strong style={{ color: '#c084fc' }}>Evaluation Return:</strong> Total cooperative episode reward (landmark coverage minus collision penalties; higher / closer to 0 is better).</div>
            )}
            {selectedMetric === 'kl' && (
              <div><strong className="text-primary">Policy Sensitivity (CIC<sub>KL</sub>):</strong> KL divergence between an agent&rsquo;s action distribution with messages ON vs. muted. Measures how much communication steers behavior.</div>
            )}
            {selectedMetric === 'value' && (
              <div><strong className="text-amber">Value Sensitivity (&Delta;V<sub>comm</sub>):</strong> Average shift in the centralized critic&rsquo;s state-value prediction when communication context is zeroed out.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
