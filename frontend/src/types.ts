export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";
export interface Health {
  status: string;
  service: string;
  analysis_mode: string;
  data_mode: "live";
  credentials_configured: boolean;
  satellite_connected: boolean;
  satellite_provider: string;
  model_configured: boolean;
  model_validated: boolean;
  screening_backend: string;
  live_ready: boolean;
  components: Record<string, boolean>;
  ollama_enabled: boolean;
  model_version: string;
}
export interface Scene { id: string; external_scene_id: string; satellite: string; acquisition_time: string; orbit_direction: string; polarization: string[]; bbox: number[]; status: string; source_metadata: Record<string, unknown>; preview_url?: string | null; }
export interface Detection { id: string; scene_id: string; geometry: GeoJSON.FeatureCollection; area_km2: number; mean_confidence: number; max_confidence: number; risk_level: RiskLevel; anomaly_type: string; verification_status: string; coordinates: number[]; model_version: string; acquisition_time: string; satellite: string; image_url: string; mask_url: string; warning: string; explanation?: string | null; }
export interface AnalysisStatus { job_id: string; status: string; progress: number; stage: string; error_message?: string | null; detection_id?: string | null; }
