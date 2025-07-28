/**
 * Custom hooks for API interactions
 */

import { useState, useEffect, useCallback } from 'react';
import type React from 'react';
import { apiClient, PredictionRequest, PredictionResponse, ModelInfo, HealthStatus, DatabasePredictionHistoryResponse, DatabaseStatsResponse, DatabaseTrendsResponse } from '@/services/api';

export interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

// Generic hook for API calls
export function useApi<T>(
  apiCall: () => Promise<T>,
  dependencies: React.DependencyList = []
): UseApiState<T> & { refetch: () => Promise<void> } {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  const fetchData = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const data = await apiCall();
      setState({ data, loading: false, error: null });
    } catch (error) {
      setState({
        data: null,
        loading: false,
        error: error instanceof Error ? error.message : 'An error occurred',
      });
    }
  }, [apiCall, ...dependencies]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    ...state,
    refetch: fetchData,
  };
}

// Hook for health status
export function useHealthStatus() {
  return useApi<HealthStatus>(() => apiClient.getHealth());
}

// Hook for model info
export function useModelInfo() {
  return useApi<ModelInfo>(() => apiClient.getModelInfo());
}

// Hook for predictions with manual trigger
export function usePrediction() {
  const [state, setState] = useState<UseApiState<PredictionResponse>>({
    data: null,
    loading: false,
    error: null,
  });

  const predict = useCallback(async (data: PredictionRequest) => {
    setState({ data: null, loading: true, error: null });
    
    try {
      const result = await apiClient.predict(data);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Prediction failed';
      setState({ data: null, loading: false, error: errorMessage });
      throw error;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return {
    ...state,
    predict,
    reset,
  };
}

// Hook for batch predictions
export function useBatchPrediction() {
  const [state, setState] = useState<UseApiState<PredictionResponse[]>>({
    data: null,
    loading: false,
    error: null,
  });

  const predictBatch = useCallback(async (data: PredictionRequest[]) => {
    setState({ data: null, loading: true, error: null });
    
    try {
      const result = await apiClient.predictBatch(data);
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Batch prediction failed';
      setState({ data: null, loading: false, error: errorMessage });
      throw error;
    }
  }, []);

  return {
    ...state,
    predictBatch,
  };
}

// Hook for prediction history
export function usePredictionHistory(limit: number = 50, offset: number = 0) {
  return useApi<PredictionResponse[]>(
    () => apiClient.getPredictionHistory(limit, offset),
    [limit, offset]
  );
}

// Hook for database predictions with filtering
export function useDatabasePredictions(params: {
  page?: number;
  limit?: number;
  model_version?: string;
  status?: string;
  batch_id?: string;
  start_date?: string;
  end_date?: string;
  search_term?: string;
} = {}) {
  return useApi<DatabasePredictionHistoryResponse>(
    () => apiClient.getDatabasePredictions(params),
    [JSON.stringify(params)]
  );
}

// Hook for database statistics
export function useDatabaseStats(params: {
  start_date?: string;
  end_date?: string;
} = {}) {
  return useApi<DatabaseStatsResponse>(
    () => apiClient.getDatabaseStats(params),
    [JSON.stringify(params)]
  );
}

// Hook for prediction trends
export function usePredictionTrends(params: {
  days?: number;
  interval?: 'hour' | 'day';
} = {}) {
  return useApi<DatabaseTrendsResponse>(
    () => apiClient.getPredictionTrends(params),
    [JSON.stringify(params)]
  );
}

// Hook for data export
export function useDataExport() {
  const [state, setState] = useState<UseApiState<Blob>>({
    data: null,
    loading: false,
    error: null,
  });

  const exportData = useCallback(async (params: {
    format: 'csv' | 'json';
    model_version?: string;
    status?: string;
    batch_id?: string;
    start_date?: string;
    end_date?: string;
    search_term?: string;
    limit?: number;
  }) => {
    setState({ data: null, loading: true, error: null });
    
    try {
      const blob = await apiClient.exportPredictions(params);
      setState({ data: blob, loading: false, error: null });
      
      // Trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `predictions_export_${new Date().toISOString().split('T')[0]}.${params.format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      return blob;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Export failed';
      setState({ data: null, loading: false, error: errorMessage });
      throw error;
    }
  }, []);

  return {
    ...state,
    exportData,
  };
}