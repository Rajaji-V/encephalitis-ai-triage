import React, { useState } from 'react';
import axios from 'axios';
import PredictionResult from './PredictionResult';
import { Upload, Activity, ActivityIcon, PlusCircle, AlertCircle } from 'lucide-react';

export default function Dashboard() {
    const [formData, setFormData] = useState({
        age: '',
        gender: 'M',
        csf_protein: '',
        csf_glucose: '',
        symptom_severity: ''
    });
    const [image, setImage] = useState(null);
    const [imagePreview, setImagePreview] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleImageChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setImage(e.target.files[0]);
            setImagePreview(URL.createObjectURL(e.target.files[0]));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) {
            setError('Please upload an MRI/CT image.');
            return;
        }

        setError('');
        setLoading(true);

        const data = new FormData();
        data.append('image', image);
        Object.keys(formData).forEach(key => {
            data.append(key, formData[key]);
        });

        try {
            const response = await axios.post('http://localhost:8000/predict', data, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'An error occurred during prediction.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="dashboard-grid">
            <div className="card">
                <div className="card-header">
                    <Activity size={20} color="var(--primary)" />
                    <h2 className="card-title">Patient Data Entry</h2>
                </div>
                <div className="card-body">
                    {error && (
                        <div style={{ color: 'var(--danger)', backgroundColor: '#fef2f2', padding: '0.75rem', borderRadius: 'var(--radius-md)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <AlertCircle size={16} />
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        <div className="upload-zone" onClick={() => document.getElementById('image-upload').click()}>
                            <Upload className="upload-icon" size={32} />
                            <p style={{ fontWeight: 500 }}>Click to upload MRI / CT Image</p>
                            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>PNG, JPG, JPEG up to 10MB</p>
                            <input
                                id="image-upload"
                                type="file"
                                accept="image/*"
                                onChange={handleImageChange}
                                style={{ display: 'none' }}
                            />
                            {imagePreview && (
                                <img src={imagePreview} alt="Preview" className="image-preview" />
                            )}
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                            <div className="form-group">
                                <label className="form-label">Age</label>
                                <input required type="number" name="age" className="form-input" value={formData.age} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Gender</label>
                                <select name="gender" className="form-input" value={formData.gender} onChange={handleInputChange}>
                                    <option value="M">Male</option>
                                    <option value="F">Female</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">CSF Protein (mg/dL)</label>
                                <input required type="number" step="0.01" name="csf_protein" className="form-input" value={formData.csf_protein} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">CSF Glucose (mg/dL)</label>
                                <input required type="number" step="0.01" name="csf_glucose" className="form-input" value={formData.csf_glucose} onChange={handleInputChange} />
                            </div>
                            <div className="form-group" style={{ gridColumn: 'span 2' }}>
                                <label className="form-label">Symptom Severity (1-10)</label>
                                <input required type="number" min="1" max="10" name="symptom_severity" className="form-input" value={formData.symptom_severity} onChange={handleInputChange} />
                            </div>
                        </div>

                        <button type="submit" className="btn btn-primary" disabled={loading}>
                            {loading ? (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{ width: 16, height: 16, border: '2px solid white', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                                    Processing Analysis...
                                </div>
                            ) : 'Generate Diagnosis & Explainability'}
                        </button>
                    </form>
                </div>
            </div>

            <div className="card">
                <div className="card-header">
                    <PlusCircle size={20} color="var(--primary)" />
                    <h2 className="card-title">Diagnostic Results & AI Explainability</h2>
                </div>
                <div className="card-body">
                    {result ? (
                        <PredictionResult result={result} />
                    ) : (
                        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '4rem 0' }}>
                            <div style={{ display: 'inline-flex', padding: '1.5rem', borderRadius: '50%', backgroundColor: 'var(--secondary)', marginBottom: '1.5rem' }}>
                                <ActivityIcon size={48} color="var(--primary)" />
                            </div>
                            <p style={{ fontSize: '1.125rem', fontWeight: 500, color: 'var(--text-main)' }}>Awaiting Patient Data</p>
                            <p style={{ maxWidth: '300px', margin: '0.5rem auto 0' }}>Submit clinical data and imaging to view AI-generated diagnosis, confidence intervals, and interpretability maps.</p>
                        </div>
                    )}
                </div>
            </div>
            <style dangerouslySetInnerHTML={{
                __html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}} />
        </div>
    );
}
