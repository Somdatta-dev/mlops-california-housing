/**
 * API Client for communicating with FastAPI backend
 */

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

export interface GPUMetrics {
  utilization: number;
  memory_used: number;
  memory_total: number;
  temperature: number;
  power_usage: number;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `HTTP error! status: ${response.status}`
        );
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Health check endpoint
  async getHealth(): Promise<HealthStatus> {
    return this.request<HealthStatus>('/health');
  }

  // Model info endpoint
  async getModelInfo(): Promise<ModelInfo> {
    return this.request<ModelInfo>('/model/info');
  }

  // Single prediction endpoint
  async predict(data: PredictionRequest): Promise<PredictionResponse> {
    return this.request<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Batch prediction endpoint
  async predictBatch(data: PredictionRequest[]): Promise<PredictionResponse[]> {
    return this.request<PredictionResponse[]>('/predict/batch', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get prediction history (if endpoint exists)
  async getPredictionHistory(
    limit: number = 50,
    offset: number = 0
  ): Promise<PredictionResponse[]> {
    return this.request<PredictionResponse[]>(
      `/predictions?limit=${limit}&offset=${offset}`
    );
  }

  // Database endpoints
  async getDatabasePredictions(params: {
    page?: number;
    limit?: number;
    model_version?: string;
    status?: string;
    batch_id?: string;
    start_date?: string;
    end_date?: string;
    search_term?: string;
  } = {}): Promise<DatabasePredictionHistoryResponse> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, value.toString());
      }
    });
    
    return this.request<DatabasePredictionHistoryResponse>(
      `/database/predictions?${searchParams.toString()}`
    );
  }

  async getDatabaseStats(params: {
    start_date?: string;
    end_date?: string;
  } = {}): Promise<DatabaseStatsResponse> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, value.toString());
      }
    });
    
    return this.request<DatabaseStatsResponse>(
      `/database/predictions/stats?${searchParams.toString()}`
    );
  }

  async exportPredictions(params: {
    format: 'csv' | 'json';
    model_version?: string;
    status?: string;
    batch_id?: string;
    start_date?: string;
    end_date?: string;
    search_term?: string;
    limit?: number;
  }): Promise<Blob> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, value.toString());
      }
    });

    const url = `${this.baseUrl}/database/predictions/export?${searchParams.toString()}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }

    return response.blob();
  }

  async getPredictionTrends(params: {
    days?: number;
    interval?: 'hour' | 'day';
  } = {}): Promise<DatabaseTrendsResponse> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && String(value) !== '') {
        searchParams.append(key, value.toString());
      }
    });
    
    return this.request<DatabaseTrendsResponse>(
      `/database/predictions/trends?${searchParams.toString()}`
    );
  }

  // System Monitoring endpoints
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.request<SystemMetrics>('/health/system');
  }

  async getAPIHealthMetrics(): Promise<APIHealthMetrics> {
    return this.request<APIHealthMetrics>('/health/api');
  }

  async getGPUMetrics(): Promise<GPUMetrics | null> {
    try {
      return await this.request<GPUMetrics>('/health/gpu');
    } catch (error) {
      // GPU might not be available
      return null;
    }
  }

  async getErrorLogs(params: {
    limit?: number;
    level?: 'error' | 'warning' | 'info';
    start_date?: string;
    end_date?: string;
    search?: string;
  } = {}): Promise<ErrorLog[]> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, value.toString());
      }
    });
    
    return this.request<ErrorLog[]>(`/monitoring/logs?${searchParams.toString()}`);
  }

  async getSystemAlerts(params: {
    resolved?: boolean;
    severity?: 'low' | 'medium' | 'high' | 'critical';
    type?: 'system' | 'api' | 'model' | 'gpu';
  } = {}): Promise<SystemAlert[]> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && String(value) !== '') {
        searchParams.append(key, value.toString());
      }
    });
    
    return this.request<SystemAlert[]>(`/monitoring/alerts?${searchParams.toString()}`);
  }

  async getMonitoringDashboard(): Promise<MonitoringDashboardData> {
    return this.request<MonitoringDashboardData>('/monitoring/dashboard');
  }

  async resolveAlert(alertId: string): Promise<void> {
    await this.request(`/monitoring/alerts/${alertId}/resolve`, {
      method: 'POST',
    });
  }

  async getPerformanceTrends(params: {
    hours?: number;
    interval?: 'minute' | 'hour';
  } = {}): Promise<{
    timestamps: string[];
    cpu_usage: number[];
    memory_usage: number[];
    api_response_time: number[];
    error_rate: number[];
  }> {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && String(value) !== '') {
        searchParams.append(key, value.toString());
      }
    });
    
    return this.request(`/monitoring/trends?${searchParams.toString()}`);
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default ApiClient;