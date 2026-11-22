import React, { useState } from 'react';
import { RefreshCw, Image, AlertCircle, Info } from 'lucide-react';
import { getRunGifUrl, triggerRenderGif } from '../../services/api';

interface GifViewerProps {
  runId: string;
  hasGif: boolean;
  onGifGenerated?: () => void;
}

export const GifViewer: React.FC<GifViewerProps> = ({
  runId,
  hasGif,
  onGifGenerated,
}) => {
  const [isRendering, setIsRendering] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [imgKey, setImgKey] = useState<number>(Date.now());

  const handleGenerate = async () => {
    setIsRendering(true);
    setError(null);
    try {
      await triggerRenderGif(runId);
      setImgKey(Date.now());
      if (onGifGenerated) onGifGenerated();
    } catch (err: any) {
      setError(err.message || 'Failed to render GIF');
    } finally {
      setIsRendering(false);
    }
  };

  return (
    <div className="gif-viewer-container">
      <div className="gif-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <h4>Environment Render Trajectory (render.gif)</h4>
        <button
          className="btn-outline-sm"
          onClick={handleGenerate}
          disabled={isRendering}
        >
          <RefreshCw size={13} className={isRendering ? 'animate-spin' : ''} />
          {isRendering ? 'Generating GIF...' : hasGif ? 'Re-render GIF' : 'Render GIF'}
        </button>
      </div>

      {error && (
        <div className="error-alert">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <div className="gif-body">
        {hasGif ? (
          <img
            key={imgKey}
            src={getRunGifUrl(runId)}
            alt={`Render trajectory for ${runId}`}
            className="render-gif-img"
          />
        ) : (
          <div className="empty-gif-box">
            <Image size={36} className="text-gray" />
            <p>No render.gif generated yet for this run.</p>
            <button className="btn-primary-sm" onClick={handleGenerate} disabled={isRendering}>
              Generate GIF Visualizer
            </button>
          </div>
        )}
      </div>

      {/* Explanation & Legend for Raw MPE Render */}
      <div className="gif-info-footer">
        <div className="gif-legend-row">
          <div className="legend-item">
            <span className="legend-dot" style={{ background: '#8b8bd8', borderRadius: '50%', width: '12px', height: '12px' }} />
            <span><strong>Purple Circles:</strong> Cooperative MARL Agents executing <code>actor.pt</code></span>
          </div>
          <div className="legend-item">
            <span className="legend-dot" style={{ background: '#333333', border: '1px solid #94a3b8', borderRadius: '50%', width: '8px', height: '8px' }} />
            <span><strong>Black Dots:</strong> Target Landmarks to spread out &amp; cover</span>
          </div>
        </div>
        <div className="gif-pipeline-note">
          <Info size={14} className="text-primary" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>
            <strong>How this is generated:</strong> Clicking <strong>Render GIF</strong> runs <code>render_mpe.py</code>, stepping the saved checkpoint through the MPE physics engine for 25 timesteps to save raw RGB frames (<code>render.gif</code>) and export live coordinates (<code>trajectory.json</code>) for the interactive 2D Simulation tab.
          </span>
        </div>
      </div>
    </div>
  );
};
