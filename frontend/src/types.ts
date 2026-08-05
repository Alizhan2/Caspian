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
  model_validation_status: string;
  live_ready: boolean;
  components: Record<string, boolean>;
  ollama_enabled: boolean;
  model_version: string;
}
export interface Scene { id: string; external_scene_id: string; satellite: string; acquisition_time: string; orbit_direction: string; polarization: string[]; bbox: number[]; status: string; source_metadata: Record<string, unknown>; preview_url?: string | null; }
export interface Detection { id: string; scene_id: string; geometry: GeoJSON.FeatureCollection; area_km2: number; mean_confidence: number; max_confidence: number; risk_level: RiskLevel; anomaly_type: string; verification_status: string; coordinates: number[]; model_version: string; acquisition_time: string; satellite: string; image_url: string; mask_url: string; warning: string; explanation?: string | null; }
export interface AnalysisStatus { job_id: string; status: string; progress: number; stage: string; error_message?: string | null; detection_id?: string | null; }
export interface LabelSample { sample_id: string; scene_id: string; acquisition_time: string; region: string; region_name: string; bbox: number[]; review_status: "unreviewed" | "reviewed_positive" | "reviewed_negative"; reviewed_by?: string | null; reviewed_at?: string | null; polygon_count: number; polygons: number[][][]; weather_context?: { status: string; provider?: string; timestamp?: string; wind_speed_10m_ms?: number; wind_direction_10m_deg?: number; wind_gusts_10m_ms?: number | null } | null; temporal_context?: { status: string; previous_sample_id?: string; days_between?: number; mean_absolute_delta?: number; changed_pixel_fraction?: number; interpretation?: string } | null; preview_url: string; }
