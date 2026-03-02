import React from 'react';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

export default function PredictionResult({ result }) {
    const { prediction, explanation } = result;

    const getStatusColor = (diagnosis) => {
        if (diagnosis === 'Healthy') return 'success';
        if (diagnosis === 'Encephalitis') return 'danger';
        return 'warning';
    };

    const getRiskColor = (risk) => {
        if (risk.includes('Low')) return 'success';
        if (risk.includes('High')) return 'danger';
        return 'warning';
    };

    const getUncertaintyInterpretation = (uncertainty) => {
        if (uncertainty < 15) return "High confidence – standard protocol applies.";
        if (uncertainty < 30) return "Moderate uncertainty – clinical review recommended.";
        return "High uncertainty – manual expert analysis required.";
    };

    const status = getStatusColor(prediction.diagnosis);
    const riskStatus = getRiskColor(prediction.risk_level);

    return (
        <div className="results-section">
            <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                    Predicted Diagnosis
                </h3>
                <div style={{ margin: '0.5rem 0' }}>
                    <span className={`status-badge ${status}`} style={{ marginRight: '0.5rem' }}>
                        {prediction.diagnosis}
                    </span>
                    <span className={`status-badge ${riskStatus}`} style={{ fontSize: '0.875rem', padding: '0.35rem 0.75rem', verticalAlign: 'middle' }}>
                        {prediction.risk_level}
                    </span>
                </div>
            </div>

            <div className="metrics-container">
                <div className="metric-box">
                    <div className="metric-label">Model Confidence</div>
                    <div className="metric-value">{prediction.confidence.toFixed(1)}%</div>
                    <div style={{ width: '100%', height: '4px', backgroundColor: 'var(--secondary)', borderRadius: '2px', marginTop: '0.5rem' }}>
                        <div style={{ width: `${prediction.confidence}%`, height: '100%', backgroundColor: 'var(--primary)', borderRadius: '2px' }} />
                    </div>
                </div>

                <div className="metric-box">
                    <div className="metric-label">Prediction Uncertainty</div>
                    <div className="metric-value" style={{ color: prediction.uncertainty > 20 ? 'var(--warning)' : 'var(--text-main)' }}>
                        {prediction.uncertainty.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                        {getUncertaintyInterpretation(prediction.uncertainty)}
                    </div>
                </div>
            </div>

            {prediction.class_probabilities && (
                <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'var(--background)', borderRadius: 'var(--radius-md)' }}>
                    <h4 style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Class Probability Breakdown</h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {Object.entries(prediction.class_probabilities).map(([className, prob]) => (
                            <div key={className} style={{ display: 'flex', alignItems: 'center' }}>
                                <div style={{ width: '150px', fontSize: '0.875rem' }}>{className}</div>
                                <div style={{ flex: 1, backgroundColor: 'var(--border)', height: '6px', borderRadius: '3px', margin: '0 1rem' }}>
                                    <div style={{ width: `${prob}%`, height: '100%', backgroundColor: getStatusColor(className) === 'danger' ? 'var(--danger)' : 'var(--success)', borderRadius: '3px' }} />
                                </div>
                                <div style={{ width: '40px', fontSize: '0.875rem', fontWeight: 600, textAlign: 'right' }}>{prob.toFixed(1)}%</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem', marginTop: '0.5rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Info size={16} color="var(--primary)" />
                    AI Explainability Maps
                </h3>

                <div className="heatmap-container">
                    <p style={{ fontSize: '0.875rem', fontWeight: 500, marginBottom: '0.5rem' }}>Grad-CAM: Imaging Focus Regions</p>
                    <img
                        src={`data:image/jpeg;base64,${explanation.heatmap}`}
                        alt="Grad-CAM Heatmap"
                        className="heatmap-img"
                    />
                </div>

                <div className="shap-chart">
                    <p style={{ fontSize: '0.875rem', fontWeight: 500, marginBottom: '1rem' }}>SHAP Values: Clinical Logic Impact</p>

                    {Object.entries(explanation.shap_values).map(([feature, value]) => {
                        const normalizedVal = Math.min(Math.abs(value) * 100, 100);
                        const isPositive = value > 0;

                        return (
                            <div key={feature} className="shap-bar">
                                <div className="shap-label" style={{ textTransform: 'capitalize' }}>
                                    {feature.replace(/_/g, ' ')}
                                </div>
                                <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                                    <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end', paddingRight: '0.25rem' }}>
                                        {!isPositive && (
                                            <div className="shap-track" style={{ width: `${normalizedVal}%`, backgroundColor: 'transparent' }}>
                                                <div className="shap-fill negative" style={{ width: '100%', float: 'right' }} />
                                            </div>
                                        )}
                                    </div>
                                    <div style={{ width: '2px', height: '16px', backgroundColor: 'var(--border)' }} />
                                    <div style={{ flex: 1, paddingLeft: '0.25rem' }}>
                                        {isPositive && (
                                            <div className="shap-track" style={{ width: `${normalizedVal}%`, backgroundColor: 'transparent' }}>
                                                <div className="shap-fill positive" style={{ width: '100%' }} />
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        );
                    })}

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            <div style={{ width: 8, height: 8, backgroundColor: 'var(--danger)', borderRadius: '2px' }} /> Increased Risk
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            <div style={{ width: 8, height: 8, backgroundColor: 'var(--primary)', borderRadius: '2px' }} /> Decreased Risk
                        </span>
                    </div>
                </div>

                {explanation.decision_summary && (
                    <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#f8fafc', borderLeft: '4px solid var(--primary)', borderRadius: '4px' }}>
                        <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Decision Summary</h4>
                        <ul style={{ paddingLeft: '1.5rem', margin: 0, fontSize: '0.875rem', color: 'var(--text-main)' }}>
                            {explanation.decision_summary.map((line, idx) => (
                                <li key={idx} style={{ marginBottom: '0.25rem' }}>{line}</li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}
