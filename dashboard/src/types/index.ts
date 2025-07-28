/**
 * Type definitions for the MLOps Dashboard
 */

// API Types
export interface PredictionRequest {
  MedInc: number;
  HouseAge: number;
  AveRooms: number;
  AveBedrms: number;
  Population: number;
  AveOccup: number;
  Latitude: number;
  Longitude: number;
}

export interface PredictionResponse {
  prediction: number;
  model_version: string;
  confidence_interval?: [number, number];
  processing_time_ms: number;
  request_id: string;
  timestamp: string;
}

export interface ModelInfo {
  model_version: string;
  model_type: string;
  performance_metrics: {
    rmse: number;
    mae: number;
    r2: number;
  };
  training_date: string;
  gpu_accelerated: boolean;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  gpu_available: boolean;
  uptime: number;
  version: string;
}

// WebSocket Types
export interface GPUMetrics {
  utilization: number;
  memory_used: number;
  memory_total: number;
  temperature: number;
  power_usage: number;
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error';
  progress: number;
  current_epoch?: number;
  total_epochs?: number;
  current_loss?: number;
  model_type?: string;
  gpu_metrics?: GPUMetrics;
}

export interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  gpu_temperature: number;
  api_response_time: number;
  active_connections: number;
}

// Dashboard Types
export interface DashboardTab {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  component: React.ComponentType;
}

export interface NavigationItem {
  id: string;
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string | number;
}

// Training Configuration
export interface TrainingConfig {
  model_type: 'linear_regression' | 'random_forest' | 'xgboost' | 'neural_network' | 'lightgbm';
  hyperparameters: Record<string, unknown>;
  gpu_enabled: boolean;
  mixed_precision: boolean;
  early_stopping: boolean;
  validation_split: number;
}

export interface TrainingJob {
  id: string;
  model_type: string;
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error';
  progress: number;
  current_epoch?: number;
  total_epochs?: number;
  current_loss?: number;
  best_loss?: number;
  start_time?: string;
  end_time?: string;
  config: TrainingConfig;
  metrics?: TrainingMetrics;
}

export interface TrainingMetrics {
  train_loss: number[];
  val_loss: number[];
  train_accuracy?: number[];
  val_accuracy?: number[];
  learning_rate: number[];
  epochs: number[];
  timestamps: string[];
}

export interface ModelComparison {
  model_id: string;
  model_type: string;
  model_version: string;
  performance_metrics: {
    rmse: number;
    mae: number;
    r2: number;
    training_time: number;
  };
  hyperparameters: Record<string, unknown>;
  training_date: string;
  gpu_accelerated: boolean;
  status: 'training' | 'completed' | 'failed' | 'selected';
}

export interface HyperparameterPreset {
  id: string;
  name: string;
  model_type: string;
  description: string;
  parameters: Record<string, unknown>;
  recommended_for: string[];
}

// Chart Data Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc: number;
}

// Filter and Pagination Types
export interface FilterConfig {
  dateRange?: {
    start?: string;
    end?: string;
  };
  modelVersion?: string;
  status?: string;
  searchTerm?: string;
}

export interface PaginationConfig {
  page: number;
  limit: number;
  total?: number;
}

// Export Types
export interface ExportConfig {
  format: 'csv' | 'json' | 'xlsx';
  fields: string[];
  filters?: FilterConfig;
}

// Alert Types
export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  dismissed?: boolean;
}

// System Monitoring Types
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  memory_total: number;
  disk_usage: number;
  disk_total: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
  };
  load_average: number[];
}

export interface APIHealthMetrics {
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time_ms: number;
  requests_per_second: number;
  error_rate: number;
  active_connections: number;
  uptime_seconds: number;
}

export interface ErrorLog {
  id: string;
  timestamp: string;
  level: 'error' | 'warning' | 'info';
  message: string;
  endpoint?: string;
  error_type?: string;
  stack_trace?: string;
  request_id?: string;
}

export interface SystemAlert {
  id: string;
  type: 'system' | 'api' | 'model' | 'gpu';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  resolved: boolean;
  component: string;
  threshold_value?: number;
  current_value?: number;
}

export interface MonitoringDashboardData {
  system_metrics: SystemMetrics;
  api_health: APIHealthMetrics;
  gpu_metrics: GPUMetrics | null;
  error_logs: ErrorLog[];
  alerts: SystemAlert[];
  performance_trends: {
    timestamps: string[];
    cpu_usage: number[];
    memory_usage: number[];
    api_response_time: number[];
    error_rate: number[];
  };
}

// Database Types
export interface DatabasePredictionRecord {
  id: number;
  request_id: string;
  model_version: string;
  model_stage: string;
  input_features: Record<string, number>;
  prediction: number;
  confidence_lower?: number;
  confidence_upper?: number;
  confidence_score?: number;
  processing_time_ms: number;
  timestamp: string;
  user_agent?: string;
  ip_address?: string;
  batch_id?: string;
  status: string;
  error_message?: string;
}

export interface DatabasePredictionHistoryResponse {
  predictions: DatabasePredictionRecord[];
  total_count: number;
  page: number;
  limit: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface DatabaseStatsResponse {
  total_predictions: number;
  successful_predictions: number;
  failed_predictions: number;
  success_rate: number;
  average_processing_time_ms: number;
  date_range: {
    start_date?: string;
    end_date?: string;
  };
}

export interface DatabaseTrendsResponse {
  date_range: {
    start_date: string;
    end_date: string;
    days: number;
    interval: string;
  };
  trends: {
    volume_trends: Array<{
      timestamp: string;
      total_predictions: number;
      successful_predictions: number;
      failed_predictions: number;
    }>;
    success_rate_trends: Array<{
      timestamp: string;
      success_rate: number;
    }>;
    processing_time_trends: Array<{
      timestamp: string;
      avg_processing_time_ms: number;
    }>;
    model_usage: Array<{
      model_version: string;
      usage_count: number;
    }>;
  };
}